# SDR-RFID - Seguimiento de placas con DeepSORT
#
# Adaptado de tracker.py al pipeline peruano:
#   - Cada lectura OCR es (texto, confianza, frame).
#   - Umbral IoU track<->deteccion subido de 0.15 a 0.3 y emparejamiento
#     GREEDY (todos los pares ordenados por IoU, se asigna de mayor a
#     menor): con varios vehiculos cercanos el "primero el mejor por
#     track" cruzaba casamientos.
#   - Nueva: finalizacion de sesiones. Cuando un track no se ve durante
#     max_age frames, se marca finalizado y main_v5 lo publica UNA vez
#     en Redis con el texto votado de toda la sesion.
#

from __future__ import annotations
from collections import Counter
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


# ---------------------------------------------------------------------------
# utilidades de texto
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Distancia de edicion de Levenshtein entre dos cadenas."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def clean_plate_text(text: str) -> str:
    """Solo alfanumericos en mayusculas, para comparar lecturas."""
    return "".join(c for c in text if c.isalnum()).upper()


def is_similar_plate(p1: str, p2: str) -> bool:
    """Dos lecturas corresponden a la misma placa real (tolerando errores OCR).
       < 4 chars: igualdad exacta; 4-5: dist <= 1; 6+: dist <= 2."""
    c1, c2 = clean_plate_text(p1), clean_plate_text(p2)
    if not c1 or not c2:
        return False
    if c1 == c2:
        return True
    if min(len(c1), len(c2)) < 4:
        return False
    dist = levenshtein_distance(c1, c2)
    if min(len(c1), len(c2)) >= 6:
        return dist <= 2
    return dist <= 1


def compute_iou(boxA, boxB) -> float:
    """Intersection-over-Union entre dos cajas [x1, y1, x2, y2]."""
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = aA + aB - inter
    return inter / union if union > 0.0 else 0.0


# ---------------------------------------------------------------------------
# sesion de una placa
# ---------------------------------------------------------------------------

class TrackedPlate:
    """Una placa (vehiculo) seguida a traves del video."""

    def __init__(self, track_id: int, bbox, confidence: float, frame_idx: int):
        self.track_id = track_id
        self.bbox = list(bbox)
        self.confidence = confidence
        self.first_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.detections_history = []   # [{"frame_idx", "bbox", "confidence", "text", "conf_ocr"}]
        self.ocr_history = []          # [{"text", "conf", "frame_idx"}]
        self.finalizado = False
        self.add_detection(frame_idx, bbox, confidence)

    def add_detection(self, frame_idx: int, bbox, confidence: float):
        self.bbox = list(bbox)
        self.confidence = confidence
        self.last_seen_frame = frame_idx
        # sin duplicar entrada si se llama dos veces en el mismo frame
        for entry in self.detections_history:
            if entry["frame_idx"] == frame_idx:
                entry["bbox"] = list(bbox)
                entry["confidence"] = confidence
                return
        self.detections_history.append({
            "frame_idx": frame_idx, "bbox": list(bbox), "confidence": confidence,
            "text": "", "conf_ocr": 0.0,
        })

    def add_ocr_result(self, frame_idx: int, text: str, conf: float):
        text = (text or "").strip()
        if not text:
            return
        # adjuntar la lectura a la deteccion de ese frame (evidencia)
        for entry in reversed(self.detections_history):
            if entry["frame_idx"] == frame_idx:
                entry["text"] = text
                entry["conf_ocr"] = float(conf)
                break
        self.ocr_history.append({"text": text, "conf": float(conf), "frame_idx": frame_idx})

    def mejor_lectura(self):
        """Voto mayoritario entre las lecturas de la sesion.
           Devuelve (texto, confianza_promedio, n_lecturas_del_ganador)."""
        textos = [o["text"] for o in self.ocr_history if o["text"]]
        if not textos:
            return "", 0.0, 0
        counts = Counter(textos)
        max_n = counts.most_common(1)[0][1]
        candidatos = [t for t, c in counts.most_common() if c == max_n]
        # desempate: mayor confianza promedio
        mejor = max(candidatos,
                    key=lambda t: float(np.mean([o["conf"] for o in self.ocr_history
                                                 if o["text"] == t])))
        conf_prom = float(np.mean([o["conf"] for o in self.ocr_history
                                   if o["text"] == mejor]))
        return mejor, conf_prom, max_n


# ---------------------------------------------------------------------------
# tracker principal
# ---------------------------------------------------------------------------

class PlateTracker:
    """Envuelve DeepSORT con voto de texto por sesion y fusion de tracks."""

    def __init__(self, max_age: int = 30, iou_minimo: float = 0.3):
        self.max_age = max_age
        self.iou_minimo = iou_minimo
        self.deepsort = DeepSort(
            max_age=max_age,
            n_init=1,
            embedder="mobilenet",
            embedder_gpu=False,
            bgr=True,
        )
        self.tracks: dict[int, TrackedPlate] = {}
        self.canonical_ids: dict[int, int] = {}   # fusiones de IDs

    # -- utilidad de IDs -----------------------------------------------------

    def get_canonical_id(self, track_id: int) -> int:
        """Resuelve cadenas transitivas de fusiones (12 -> 9 -> 7)."""
        curr = track_id
        while curr in self.canonical_ids:
            curr = self.canonical_ids[curr]
        return curr

    # -- actualizacion por frame ---------------------------------------------

    def update(self, detections: list, frame_idx: int, frame: np.ndarray) -> list:
        """Alimenta DeepSORT y devuelve [(session_id, deteccion), ...]
           de las detecciones confirmadas este frame."""
        raw_dets = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            raw_dets.append(([x1, y1, x2 - x1, y2 - y1], det.confidence, det.class_id))

        ds_tracks = self.deepsort.update_tracks(raw_dets, frame=frame)

        # pares (iou, track, det) que superan el umbral, ordenados de
        # mayor a menor IoU -> asignacion greedy sin cruces
        candidatos = []
        for track in ds_tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            for i, det in enumerate(detections):
                iou = compute_iou(det.bbox_xyxy, ltrb)
                if iou >= self.iou_minimo:
                    candidatos.append((iou, track, i))
        candidatos.sort(key=lambda c: -c[0])

        asignaciones = []
        dets_usados, tracks_usados = set(), set()
        for _iou, track, i in candidatos:
            ds_id = int(track.track_id)
            if i in dets_usados or ds_id in tracks_usados:
                continue
            dets_usados.add(i)
            tracks_usados.add(ds_id)

            session_id = self.get_canonical_id(ds_id)
            det = detections[i]
            det.track_id = session_id
            if session_id not in self.tracks:
                self.tracks[session_id] = TrackedPlate(session_id, det.bbox_xyxy,
                                                       det.confidence, frame_idx)
            else:
                self.tracks[session_id].add_detection(frame_idx, det.bbox_xyxy,
                                                    det.confidence)
            asignaciones.append((session_id, det))
        return asignaciones

    # -- OCR ------------------------------------------------------------------

    def add_ocr_result(self, track_id: int, frame_idx: int, text: str, conf: float):
        """Guarda la lectura y, si el mejor texto de dos sesiones se parece
           (DeepSORT perdio y recupero la identidad), las fusiona."""
        session_id = self.get_canonical_id(track_id)
        if session_id not in self.tracks:
            return
        self.tracks[session_id].add_ocr_result(frame_idx, text, conf)

        best_text, _, _ = self.tracks[session_id].mejor_lectura()
        if not best_text or len(clean_plate_text(best_text)) < 4:
            return

        for other_id in list(self.tracks.keys()):
            other_sid = self.get_canonical_id(other_id)
            if other_sid == session_id:
                continue
            other_best, _, _ = self.tracks[other_sid].mejor_lectura()
            if not other_best or len(clean_plate_text(other_best)) < 4:
                continue
            if is_similar_plate(best_text, other_best):
                # fusionar la sesion mas nueva dentro de la mas antigua
                if self.tracks[session_id].first_frame < self.tracks[other_sid].first_frame:
                    dest_id, src_id = session_id, other_sid
                else:
                    dest_id, src_id = other_sid, session_id
                self.tracks[dest_id].detections_history.extend(
                    self.tracks[src_id].detections_history)
                self.tracks[dest_id].ocr_history.extend(
                    self.tracks[src_id].ocr_history)
                self.tracks[dest_id].detections_history.sort(key=lambda x: x["frame_idx"])
                self.canonical_ids[src_id] = dest_id
                del self.tracks[src_id]
                print(f"[tracker] sesion #{src_id} fusionada en #{dest_id} "
                      f"por texto similar: '{best_text}' ~ '{other_best}'")
                break

    def get(self, track_id: int):
        session_id = self.get_canonical_id(track_id)
        return self.tracks.get(session_id)

    # -- finalizacion ----------------------------------------------------------

    def finalizar_muertos(self, frame_idx: int) -> list:
        """Sesiones no vistas hace >= max_age frames -> listas para publicar."""
        muertos = []
        for sid, tp in list(self.tracks.items()):
            if not tp.finalizado and frame_idx - tp.last_seen_frame >= self.max_age:
                tp.finalizado = True
                muertos.append(tp)
        return muertos

    def flush(self) -> list:
        """Al cerrar: finaliza TODO lo que quede (p. ej. Ctrl+C)."""
        restantes = [tp for tp in self.tracks.values() if not tp.finalizado]
        for tp in restantes:
            tp.finalizado = True
        return restantes