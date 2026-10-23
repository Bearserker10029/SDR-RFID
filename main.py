# SDR-RFID - Deteccion de placas peruanas con TRACKING (varios vehiculos)
#
# Diferencias con main_v4 (multiproceso, stateless):
#   - DeepSORT da una IDENTIDAD PERSISTENTE a cada vehiculo, asi que las
#     lecturas OCR se votan DENTRO de cada vehiculo y no se mezclan placas
#     parecidas de autos distintos.
#   - Redis recibe UNA publicacion por VEHICULO (al morir su track), no una
#     por frame. analisis.py sigue funcionando sin cambios.
#   - El OCR se ejecuta cada OCR_CADA_N_FRAMES por track (no en cada frame).
#
# Pipeline por frame:
#   1. YOLO detecta TODAS las placas visibles.
#   2. tracker.update() las empareja con identidades DeepSORT (greedy IoU).
#   3. Por cada track que le toca: warp + preproceso + RapidOCR -> voto.
#   4. Tracks muertos (no vistos en MAX_AGE frames): texto final por mayoria
#      y UN write a Redis + evidencia en disco.

import os
import re
import json
import time
import logging

import cv2
import numpy as np
import redis
from ultralytics import YOLO

from utils.placas_utils import ALLOW_TODO, placa_valida
from utils.ocr_engine import RapidOCRReader
from utils.tracker import PlateTracker

#  CONFIGURACION
MODELO = "./models/best_placas_v1.pt"
FUENTE = "./videos/video1.mp4"        # 0 = webcam; ruta = video; "rtsp://..." = camara IP
INTERVALO_S = 0.4                     # muestreo entre frames procesados
CONF_MINIMA = 0.3                     # confianza minima YOLO
CARPETA_SALIDA = "salidas"
MIN_CARACTERES = 4
SCALE = 3
OCR_TEXT_SCORE = 0.3
OCR_CONF_RESUELTO = 0.9              # track con lectura VALIDA y >= esto: no mas OCR
MOSTRAR_VIDEO = False

MAX_AGE = 60                  # frames sin ver un track antes de darlo por muerto
OCR_CADA_N_FRAMES = 2         # throttle: OCR por track cada N frames
OCR_EARLY_STOP = 0.85          # si una lectura es VALIDA y >= esta conf, no seguir

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

DIMS_LIVIANO = (300, 150)
DIMS_MENOR = (170, 200)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

class Deteccion:
    """Lo que PlateTracker espera de cada deteccion."""
    def __init__(self, bbox_xyxy, confidence, class_id=0):
        self.bbox_xyxy = [float(v) for v in bbox_xyxy]
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.track_id = None


def preprocesar(warped):
    """Gris + CLAHE + unsharp mask para maximizar nitidez de caracteres."""
    g = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    sv = cv2.GaussianBlur(cl, (0, 0), 3)
    return cv2.addWeighted(cl, 1.8, sv, -0.8, 0)


def unir_lecturas(detecciones, alto):
    """Concatena las detecciones del OCR ordenadas por linea y luego por x.
       Mismo formato para EasyOCR y RapidOCR: (bbox[[x,y]x4], texto, conf).
       En modo solo_rec la caja es ficticia: todo se trata como una linea."""
    bloques = []
    for bbox, txt, cf in detecciones:
        limpio = re.sub(r"[^A-Z0-9]", "", txt.upper())
        if not limpio:
            continue
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        cy = sum(ys) / len(ys)
        cx = sum(xs) / len(xs)
        h = max(ys) - min(ys)
        bloques.append((cy, cx, limpio, cf, h))
    if not bloques:
        return "", 0.0
    h_max = max(b[4] for b in bloques)
    bloques = [b for b in bloques if b[4] >= h_max * 0.45]
    bloques.sort(key=lambda b: b[0])
    lineas = [[bloques[0]]]
    for b in bloques[1:]:
        y_media = sum(x[0] for x in lineas[-1]) / len(lineas[-1])
        if abs(b[0] - y_media) < alto * 0.12:
            lineas[-1].append(b)
        else:
            lineas.append([b])
    partes, pesos = [], []
    for lin in lineas:
        for b in sorted(lin, key=lambda x: x[1]):
            partes.append(b[2])
            pesos.append(b[3] * b[4])
    return "".join(partes), sum(pesos) / len(pesos)


def leer_placa(warped, nit, ocr, solo_rec=False):
    """Mejor lectura entre versiones de preprocesamiento.
       - solo_rec=True: reconocimiento puro (rapido); se prueban nit y gris.
       - solo_rec=False: deteccion+reconocimiento; se agrega la version a color.
       Devuelve (texto, confianza) o None."""
    alto = warped.shape[0]
    gris = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    versiones = (nit, gris) if solo_rec else (nit, gris, warped)
    mejor = None
    for version in versiones:
        detecciones = ocr.readtext(version, detail=1, allowlist=ALLOW_TODO,
                                   text_score=OCR_TEXT_SCORE, solo_rec=solo_rec)
        texto, cf = unir_lecturas(detecciones, alto)
        if texto.startswith("PERU"):            # quitar banner
            texto = texto[4:]
        if not texto:
            continue
        if mejor is None or (placa_valida(texto), cf) > (placa_valida(mejor[0]), mejor[1]):
            mejor = (texto, cf)
        if placa_valida(texto) and cf >= OCR_EARLY_STOP:
            break
    return mejor


# ---------------------------------------------------------------------------
# preparacion del crop (warp por geometria segun aspecto de la caja)
# ---------------------------------------------------------------------------

def crop_con_margen(img, box, margen=0.08):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margen)); y1 = max(0, int(y1 - bh * margen))
    x2 = min(w, int(x2 + bw * margen)); y2 = min(h, int(y2 + bh * margen))
    return x1, y1, x2, y2


def es_geometria_menor(box):
    """Caja mas alta que ancha -> placa de moto (170x200, dos lineas)."""
    x1, y1, x2, y2 = box
    return (y2 - y1) > (x2 - x1) * 1.15


def warp_placa(img, box):
    """Warp a la geometria legal segun el aspecto de la caja.
       Devuelve (warped, nit)."""
    x1, y1, x2, y2 = crop_con_margen(img, box)
    dims = DIMS_MENOR if es_geometria_menor(box) else DIMS_LIVIANO
    Ww, Hh = dims
    src = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    dst = np.float32([[0, 0], [Ww * SCALE, 0], [Ww * SCALE, Hh * SCALE], [0, Hh * SCALE]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (Ww * SCALE, Hh * SCALE), flags=cv2.INTER_CUBIC)
    return warped, preprocesar(warped)


# ---------------------------------------------------------------------------
# Redis: una publicacion por vehiculo (track finalizado)
# ---------------------------------------------------------------------------

def guardar_sesion(rdb, sesion, evidencia):
    """Publica el resultado de UNA sesion de vehiculo en Redis.
       Esquema compatible con analisis.py."""
    texto, conf_prom, n_votos = sesion.mejor_lectura()
    if not texto or len(texto) < MIN_CARACTERES:
        return None
    valida = placa_valida(texto)
    ahora = time.time()
    rdb.hincrby(f"placa:{texto}", "veces", 1)
    rdb.hset(f"placa:{texto}", mapping={
        "conf_ocr": round(conf_prom, 3),
        "ultima_vez": ahora,
        "tipo": "pendiente",              # lo rellena analisis.py
        "estado_ocr": "valida" if valida else "dudosa",
        "track_id": sesion.track_id,
        "frames_vistos": len(sesion.detections_history),
        "votos": n_votos,
    })
    rdb.sadd("placas:activas", texto)
    rdb.lpush("sightings", json.dumps({
        "placa": texto,
        "conf_ocr": round(conf_prom, 3),
        "caja": [round(v, 1) for v in sesion.bbox],
        "ts": ahora,
        "track_id": sesion.track_id,
        "frames_vistos": len(sesion.detections_history),
    }))
    rdb.ltrim("sightings", 0, 4999)

    # evidencia en disco: ultimo warp + nit de la sesion
    if evidencia is not None:
        warped, nit = evidencia
        nit_color = cv2.cvtColor(nit, cv2.COLOR_GRAY2BGR)
        nombre = f"track{sesion.track_id:04d}_{texto}.jpg".replace("?", "x")
        cv2.imwrite(os.path.join(CARPETA_SALIDA, nombre),
                    np.hstack([warped, nit_color]))
    return {"placa": texto, "valida": valida, "conf": conf_prom,
            "votos": n_votos, "frames": len(sesion.detections_history)}


# ---------------------------------------------------------------------------
# captura
# ---------------------------------------------------------------------------

def extraer_fotogramas(fuente, intervalo_s):
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir la fuente: {fuente}")
    es_camara = isinstance(fuente, int) or str(fuente).startswith("rtsp")
    n, ultimo = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if es_camara:
            ahora = time.monotonic()
            if ahora - ultimo < intervalo_s:
                continue
            ultimo = ahora
            ts = ahora
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            paso = max(1, int(round(fps * intervalo_s)))
            if n % paso != 0:
                n += 1
                continue
            ts = n / fps
        yield n, ts, frame
        n += 1
    cap.release()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    detector = YOLO(MODELO)
    ocr = RapidOCRReader()
    tracker = PlateTracker(max_age=MAX_AGE)

    # silenciar INFO de rapidocr DESPUES de crear el engine (si no, no surte
    # efecto: rapidocr reconfigura su logger al instanciarse)
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    for nombre in ("RapidOCR", "rapidocr"):
        lg = logging.getLogger(nombre)
        lg.setLevel(logging.WARNING)

    try:
        rdb = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                          decode_responses=True)
        rdb.ping()
        print(f"[redis] conectado a {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
    except redis.ConnectionError:
        rdb = None
        print("[redis] SIN CONEXION - se procesa sin persistencia")

    ultima_ocr = {}        # session_id -> ultimo frame con OCR
    evidencia = {}         # session_id -> (warped, nit) mas reciente
    publicadas = 0

    try:
        for n_frame, ts, frame in extraer_fotogramas(FUENTE, INTERVALO_S):
            vis = frame.copy()
            t0 = time.monotonic()

            # 1. deteccion
            _t = time.monotonic()
            results = detector(frame, verbose=False)[0]
            t_yolo = time.monotonic() - _t
            detecciones = [Deteccion(b.xyxy[0], float(b.conf[0]), b.cls[0])
                           for b in results.boxes if float(b.conf[0]) >= CONF_MINIMA]

            # 2. tracking (identidad persistente)
            _t = time.monotonic()
            asignaciones = tracker.update(detecciones, n_frame, frame)
            t_track = time.monotonic() - _t

            # 3. OCR con throttle + skip de tracks resueltos
            _t = time.monotonic()
            for session_id, det in asignaciones:
                tp = tracker.get(session_id)
                best, conf_best, _ = tp.mejor_lectura() if tp else ("", 0.0, 0)
                ya_resuelto = placa_valida(best) and conf_best >= OCR_CONF_RESUELTO
                toca = n_frame - ultima_ocr.get(session_id, -10**9) >= OCR_CADA_N_FRAMES
                if ya_resuelto or not toca:
                    continue
                # solo_rec (rapido) para placas de una linea; det+rec para motos
                warped, nit = warp_placa(frame, det.bbox_xyxy)
                res = leer_placa(warped, nit, ocr,
                                 solo_rec=not es_geometria_menor(det.bbox_xyxy))
                if res:
                    texto, cf = res
                    tracker.add_ocr_result(session_id, n_frame, texto, cf)
                    ultima_ocr[session_id] = n_frame
                    evidencia[session_id] = (warped, nit)
            t_ocr = time.monotonic() - _t

            # 4. dibujar tracks activos
            for session_id, det in asignaciones:
                x1, y1, x2, y2 = map(int, det.bbox_xyxy)
                tp = tracker.get(session_id)
                best, _, _ = tp.mejor_lectura() if tp else ("", 0.0, 0)
                color = (0, 255, 0) if placa_valida(best) else (0, 200, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"#{session_id} {best}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 5. publicar sesiones muertas
            for sesion in tracker.finalizar_muertos(n_frame):
                if rdb is not None:
                    pub = guardar_sesion(rdb, sesion, evidencia.get(sesion.track_id))
                    if pub:
                        publicadas += 1
                        print(f"[redis] sesion #{sesion.track_id}: {pub}")
                ultima_ocr.pop(sesion.track_id, None)
                evidencia.pop(sesion.track_id, None)

            dt = (time.monotonic() - t0) * 1000
            if asignaciones:
                print(f"[frame {n_frame}] {len(asignaciones)} track(s) | "
                      f"yolo={t_yolo*1000:.0f} track={t_track*1000:.0f} "
                      f"ocr={t_ocr*1000:.0f} total={dt:.0f} ms")

            if MOSTRAR_VIDEO:
                ancho_vis = 960
                alto_vis = int(ancho_vis * vis.shape[0] / vis.shape[1])
                cv2.imshow("SDR-RFID v5 | tracking", cv2.resize(vis, (ancho_vis, alto_vis)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[video] detenido por el usuario (q)")
                    break

    except KeyboardInterrupt:
        print("\n[mp] Ctrl+C - finalizando sesiones activas...")

    if rdb is not None:
        for sesion in tracker.flush():
            pub = guardar_sesion(rdb, sesion, evidencia.get(sesion.track_id))
            if pub:
                publicadas += 1
                print(f"[redis] flush sesion #{sesion.track_id}: {pub}")

    cv2.destroyAllWindows()
    print(f"[fin] {publicadas} vehiculo(s) publicados en Redis")