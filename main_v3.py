# SDR-RFID - Deteccion de placas vehiculares peruanas (MULTIPROCESO)
#
# Pipeline:
#   1. YOLO detecta placas de TODOS los vehiculos visibles en el frame.
#   2. Por cada placa se prueban ambas geometrias de warp
#      (liviano 300x150 y menor 170x200) y se toma la mejor lectura OCR
#      (sin correcciones de caracteres).
#   3. Se guarda el avistamiento en Redis:
#        - hash  "placa:<placa>": ultima lectura, confianzas, contador de veces
#        - lista "sightings": historial cronologico de avistamientos
#   4. El TIPO de vehiculo NO se decide aqui: un submodulo aparte analizara
#      los acumulados en Redis (varios match de la misma placa) para inferir
#      liviano / menor / especial.
#
# Paralelismo: la deteccion + OCR corre en N_WORKERS procesos en paralelo
# (cada uno con sus propios modelos y conexion Redis); un hilo captura la
# fuente y el proceso principal solo dibuja.

import os
import re
import json
import time
import threading
import queue as pyqueue
from multiprocessing import Process, Queue

from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import redis
import torch

from utils.placas_utils import ALLOW_TODO, placa_valida

#  CONFIGURACION
MODELO = "./models/best_placas_v1.pt"
FUENTE = "fine_tuning/image/17.jpg"    # 0 = webcam; o ruta/URL rtsp
INTERVALO_S = 0.5                       # muestreo (techo; si OCR tarda, manda)
N_WORKERS = 2                           # procesos paralelos (RAM: ~1-2 GB c/u)
CARPETA_SALIDA = "salidas"
CONF_MINIMA = 0.3
MIN_CARACTERES = 4
SCALE = 3
MOSTRAR_VIDEO = True
ANCHO_FILA, ALTO_FILA = 1024, 512

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

DIMS_LIVIANO = (300, 150)
DIMS_MENOR   = (170, 200)

#  FUNCIONES AUXILIARES
def preprocesar(warped):
    g = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    sv = cv2.GaussianBlur(cl, (0, 0), 3)
    return cv2.addWeighted(cl, 1.8, sv, -0.8, 0)

def unir_lecturas(detecciones, alto):
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

def leer_placa(warped, nit, ocr):
    alto = warped.shape[0]
    mejor = None
    for version in (nit, cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), warped):
        detecciones = ocr.readtext(version, detail=1, allowlist=ALLOW_TODO)
        texto, cf = unir_lecturas(detecciones, alto)
        if texto.startswith("PERU"):
            texto = texto[4:]
        if not texto:
            continue
        if mejor is None or cf > mejor[1]:
            mejor = (texto, cf)
        if placa_valida(texto) and cf >= 0.9:
            break
    return mejor

def guardar_avistamiento(r, placa, conf_ocr, conf_yolo, caja):
    ahora = time.time()
    r.hincrby(f"placa:{placa}", "veces", 1)
    r.hset(f"placa:{placa}", mapping={
        "conf_ocr": round(conf_ocr, 3),
        "conf_yolo": round(conf_yolo, 3),
        "ultima_vez": ahora,
        "tipo": "pendiente",
    })
    r.sadd("placas:activas", placa)
    r.lpush("sightings", json.dumps({
        "placa": placa,
        "conf_ocr": round(conf_ocr, 3),
        "conf_yolo": round(conf_yolo, 3),
        "caja": list(caja),
        "ts": ahora,
    }))
    r.ltrim("sightings", 0, 4999)

def procesar_frame(img, n_frame, ts, rdb, detector, ocr):
    vis = img.copy()
    results = detector(img)[0]

    filas_evidencia = []
    resultados = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf < CONF_MINIMA:
            continue

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        src = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        placa_final, ocr_conf, warped_sel, nit_sel = None, 0.0, None, None
        for (Ww, Hh) in (DIMS_LIVIANO, DIMS_MENOR):
            dst = np.float32([[0, 0], [Ww*SCALE, 0], [Ww*SCALE, Hh*SCALE], [0, Hh*SCALE]])
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img, M, (Ww*SCALE, Hh*SCALE), flags=cv2.INTER_CUBIC)
            nit = preprocesar(warped)
            res = leer_placa(warped, nit, ocr)
            if res is None:
                continue
            texto, cf = res
            if placa_final is None or (placa_valida(texto), cf) > (placa_valida(placa_final), ocr_conf):
                placa_final, ocr_conf, warped_sel, nit_sel = texto, cf, warped, nit
            if placa_valida(texto) and cf >= 0.9:
                break

        if placa_final is None:
            placa_final, estado, ocr_conf = "???", "sin lectura", 0.0
        else:
            estado = "VALIDA" if placa_valida(placa_final) else "dudosa"
            cv2.putText(vis, placa_final, (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if rdb is not None and placa_final != "???" and len(placa_final) >= MIN_CARACTERES:
            guardar_avistamiento(rdb, placa_final, ocr_conf, conf, (x1, y1, x2, y2))

        resultados.append({
            "frame": n_frame, "ts": round(ts, 2), "placa": placa_final,
            "estado": estado, "conf_yolo": conf, "conf_ocr": ocr_conf,
            "caja": (x1, y1, x2, y2),
        })

        if warped_sel is not None:
            nit_color = cv2.cvtColor(nit_sel, cv2.COLOR_GRAY2BGR)
            filas_evidencia.append(np.hstack([warped_sel, nit_color]))
            nombre = f"f{n_frame:05d}_{placa_final}_{i}.jpg".replace("?", "x")
            cv2.imwrite(os.path.join(CARPETA_SALIDA, nombre), np.hstack([warped_sel, nit_color]))

    return vis, filas_evidencia, resultados

def extraer_fotogramas(fuente, intervalo_s, cola, stop):
    """Productor (hilo): lee la fuente y encola frames muestreados."""
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        stop.set()
        raise FileNotFoundError(f"No se pudo abrir la fuente: {fuente}")
    es_camara = isinstance(fuente, int) or str(fuente).startswith("rtsp")
    n = 0
    ultimo = 0.0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        if es_camara:
            ahora = time.monotonic()
            if ahora - ultimo < intervalo_s:
                continue
            ts = ahora
            ultimo = ahora
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            paso = max(1, int(round(fps * intervalo_s)))
            if n % paso != 0:
                n += 1
                continue
            ts = n / fps
        try:
            cola.put((n, ts, frame), timeout=1)
        except pyqueue.Full:
            continue                    # workers saturados: se descarta el frame
        n += 1
    cap.release()
    for _ in range(N_WORKERS):          # senal de fin para cada worker
        try:
            cola.put(None, timeout=5)
        except pyqueue.Full:
            pass

def worker(entrada, salida):
    """Proceso worker: carga modelos UNA vez y procesa hasta recibir None."""
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // N_WORKERS))
    detector = YOLO(MODELO)
    ocr = easyocr.Reader(["es"], gpu=False)
    try:
        rdb = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                          decode_responses=True)
        rdb.ping()
    except redis.ConnectionError:
        rdb = None
    while True:
        item = entrada.get()
        if item is None:
            break
        n_frame, ts, frame = item
        vis, filas, res = procesar_frame(frame, n_frame, ts, rdb, detector, ocr)
        salida.put((n_frame, ts, vis, filas, res))
    salida.put(None)                    # avisa que termino

#  ARRANQUE
if __name__ == "__main__":
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    q_in, q_out = Queue(maxsize=8), Queue(maxsize=8)
    stop = threading.Event()

    productor = threading.Thread(target=extraer_fotogramas,
                                 args=(FUENTE, INTERVALO_S, q_in, stop),
                                 daemon=True)
    workers = [Process(target=worker, args=(q_in, q_out), daemon=True)
               for _ in range(N_WORKERS)]

    t0 = time.monotonic()
    for w in workers:
        w.start()
    productor.start()

    terminados = 0
    procesados = 0
    try:
        while terminados < N_WORKERS:
            item = q_out.get()
            if item is None:
                terminados += 1
                continue
            n_frame, ts, vis, filas, res = item
            procesados += 1
            t_trans = time.monotonic() - t0
            print(f"[out frame {n_frame} | +{t_trans:.1f}s] "
                  f"{[r['placa'] for r in res]}")
            if MOSTRAR_VIDEO:
                ANCHO_VIS = 1280
                alto_vis = int(ANCHO_VIS * vis.shape[0] / vis.shape[1])
                vis_r = cv2.resize(vis, (ANCHO_VIS, alto_vis))
                cv2.imshow("SDR-RFID | detecciones", vis_r)
                if filas:
                    filas_norm = [cv2.resize(f, (ANCHO_FILA, ALTO_FILA)) for f in filas]
                    cv2.imshow("SDR-RFID | placas", np.vstack(filas_norm))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop.set()
                    break
    except KeyboardInterrupt:
        stop.set()

    cv2.destroyAllWindows()
    for w in workers:
        w.terminate()
    print(f"\n[mp] fin. {procesados} frame(s) procesados con {N_WORKERS} workers")
