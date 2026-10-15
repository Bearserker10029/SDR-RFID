# SDR-RFID - Deteccion de placas vehiculares peruanas
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

import os
import re
import json
import time
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import redis

from placas_utils import ALLOW_TODO, placa_valida

#  CONFIGURACION
MODELO = "./models/best_placas_v1.pt"

# Fuente: 0 = webcam local, 1 = segunda camara, o ruta/URL:
#   FUENTE = 0
#   FUENTE = "./video_prueba.mp4"
FUENTE = 0
INTERVALO_S = 1
CARPETA_SALIDA = "salidas"
CONF_MINIMA = 0.3          # filtrado por confianza YOLO
SCALE = 3                  # warp en alta resolucion
MOSTRAR_VIDEO = True
ANCHO_FILA, ALTO_FILA = 1024, 512
MIN_CARACTERES = 4

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Dimensiones legales de placa (Art. 26.3, en mm usadas como pixeles del warp)
DIMS_LIVIANO = (300, 150)
DIMS_MENOR   = (170, 200)

#  FUNCIONES AUXILIARES
def preprocesar(warped):
    """Gris + CLAHE + unsharp mask para maximizar nitidez de caracteres."""
    g = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    sv = cv2.GaussianBlur(cl, (0, 0), 3)
    return cv2.addWeighted(cl, 1.8, sv, -0.8, 0)

def unir_lecturas(detecciones, alto):
    """Concatena todas las detecciones del OCR en orden de arriba a abajo y, dentro de la misma linea, de izquierda a derecha."""
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

    # Descartar bloques en letra pequena: el codigo de zona ('D1')
    h_max = max(b[4] for b in bloques)
    bloques = [b for b in bloques if b[4] >= h_max * 0.45]

    # orden de lectura: agrupar por lineas y ordenar dentro de cada una
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
            pesos.append(b[3] * b[4])           # confianza ponderada por altura
    texto = "".join(partes)
    conf = sum(pesos) / len(pesos)              # confianza promedio ponderada por altura
    return texto, conf

def leer_placa(warped, nit):
    """Lectura OCR directa de la placa (sin correcciones de caracteres).
       Devuelve (texto, confianza_ocr) de la mejor version o None."""
    alto = warped.shape[0]
    mejor = None
    for version in (nit, cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), warped):
        detecciones = ocr.readtext(version, detail=1, allowlist=ALLOW_TODO)
        texto, cf = unir_lecturas(detecciones, alto)
        if texto.startswith("PERU"):            # quitar banner
            texto = texto[4:]
        if not texto:
            continue
        print(f"  [debug] lectura: '{texto}' (conf={cf:.2f})")
        if mejor is None or cf > mejor[1]:
            mejor = (texto, cf)
    return mejor

def guardar_avistamiento(r, placa, conf_ocr, conf_yolo, caja):
    """Persiste el avistamiento en Redis para el analisis acumulado
       (match cruzado con RFID e inferencia de tipo de vehiculo)."""
    ahora = time.time()
    r.hincrby(f"placa:{placa}", "veces", 1)                     # contador de avistamientos
    r.hset(f"placa:{placa}", mapping={
        "conf_ocr": round(conf_ocr, 3),
        "conf_yolo": round(conf_yolo, 3),
        "ultima_vez": ahora,
        "tipo": "pendiente",        # lo rellena el submodulo de analisis
    })
    r.sadd("placas:activas", placa)                             # indice de placas vistas
    evento = {
        "placa": placa,
        "conf_ocr": round(conf_ocr, 3),
        "conf_yolo": round(conf_yolo, 3),
        "caja": list(caja),
        "ts": ahora,
    }
    r.lpush("sightings", json.dumps(evento))                    # historial cronologico
    r.ltrim("sightings", 0, 4999)                               # mantener ultimos 5000
    veces = int(r.hget(f"placa:{placa}", "veces"))
    print(f"  [redis] placa:{placa} -> avistamiento #{veces} guardado")

def procesar_frame(img, n_frame, ts, rdb):
    """Detecta y lee placas de un fotograma. Devuelve la imagen anotada
       y la lista de resultados."""
    vis = img.copy()
    results = licence_plate_detector(img)[0]
    print(f"[frame {n_frame} | {ts:.1f}s] {len(results.boxes)} deteccion(es)")

    filas_evidencia = []
    resultados = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if conf < CONF_MINIMA:
            print(f"  [debug] deteccion {i} descartada: conf={conf:.2f} < {CONF_MINIMA}")
            continue

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Warp: se prueban AMBAS geometrias y se queda la mejor lectura OCR
        src = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

        placa_final, ocr_conf, warped_sel, nit_sel = None, 0.0, None, None
        for (Ww, Hh) in (DIMS_LIVIANO, DIMS_MENOR):
            dst = np.float32([[0, 0], [Ww*SCALE, 0], [Ww*SCALE, Hh*SCALE], [0, Hh*SCALE]])
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(img, M, (Ww*SCALE, Hh*SCALE), flags=cv2.INTER_CUBIC)
            nit = preprocesar(warped)
            res = leer_placa(warped, nit)
            if res is None:
                continue
            texto, cf = res

            if placa_final is None or (placa_valida(texto), cf) > (placa_valida(placa_final), ocr_conf):
                placa_final, ocr_conf, warped_sel, nit_sel = texto, cf, warped, nit

        if placa_final is None:
            placa_final, estado, ocr_conf = "???", "sin lectura", 0.0
        else:
            estado = "VALIDA" if placa_valida(placa_final) else "dudosa"
            print(f"  placa: '{placa_final}' (conf={ocr_conf:.2f}, {estado})")
            cv2.putText(vis, placa_final, (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Persistir avistamiento en Redis (el tipo se infiere luego, acumulando)
        if rdb is not None and placa_final != "???" and len(placa_final) >= MIN_CARACTERES:
            guardar_avistamiento(rdb, placa_final, ocr_conf, conf, (x1, y1, x2, y2))

        resultados.append({
            "frame": n_frame,
            "ts": round(ts, 2),
            "placa": placa_final,
            "estado": estado,
            "conf_yolo": conf,
            "conf_ocr": ocr_conf,
            "caja": (x1, y1, x2, y2),
        })

        # Evidencia (warp + nit lado a lado)
        if warped_sel is not None:
            nit_color = cv2.cvtColor(nit_sel, cv2.COLOR_GRAY2BGR)
            filas_evidencia.append(np.hstack([warped_sel, nit_color]))
            nombre = f"f{n_frame:05d}_{placa_final}_{i}.jpg".replace("?", "x")
            cv2.imwrite(os.path.join(CARPETA_SALIDA, nombre), np.hstack([warped_sel, nit_color]))

    return vis, filas_evidencia, resultados

def extraer_fotogramas(fuente, intervalo_s):
    """Generador: yield (n_frame, timestamp_s, frame) cada intervalo_s.
       - Archivo de video: el tiempo lo marca el fps del archivo.
       - Camara (indice int o rtsp://): el tiempo lo marca el reloj del
         sistema; se leen todos los frames pero se procesa 1 cada intervalo."""
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir la fuente: {fuente}")

    es_camara = isinstance(fuente, int) or str(fuente).startswith("rtsp")
    if es_camara:
        n = 0
        ultimo = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ahora = time.monotonic()
            if ahora - ultimo >= intervalo_s:
                ultimo = ahora
                yield n, ahora, frame
                n += 1
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        paso = max(1, int(round(fps * intervalo_s)))   # frames a saltar entre muestras
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n % paso == 0:
                yield n, n / fps, frame
            n += 1
    cap.release()

#  CARGA DE MODELOS
licence_plate_detector = YOLO(MODELO)
ocr = easyocr.Reader(["es"], gpu=False)

#  CONEXION A REDIS
try:
    rdb = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                      decode_responses=True)
    rdb.ping()
    print(f"[redis] conectado a {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
except redis.ConnectionError:
    rdb = None
    print("[redis] SIN CONEXION - se procesa sin persistencia")

os.makedirs(CARPETA_SALIDA, exist_ok=True)

#  PROCESAMIENTO (video, webcam o rtsp)
todos = []
for n_frame, ts, frame in extraer_fotogramas(FUENTE, INTERVALO_S):
    vis, filas, res = procesar_frame(frame, n_frame, ts, rdb)
    todos.extend(res)

    print(f"=== RESUMEN frame {n_frame} ===")
    for r in res:
        print(f"  {r['placa']} ({r['estado']}, yolo={r['conf_yolo']:.2f}, ocr={r['conf_ocr']:.2f})")

    if MOSTRAR_VIDEO:
        ANCHO_VIS = 1280
        alto_vis = int(ANCHO_VIS * vis.shape[0] / vis.shape[1])
        vis = cv2.resize(vis, (ANCHO_VIS, alto_vis))
        cv2.imshow("SDR-RFID | detecciones", vis)
        if filas:
            filas_norm = [cv2.resize(f, (ANCHO_FILA, ALTO_FILA)) for f in filas]
            cv2.imshow("SDR-RFID | placas", np.vstack(filas_norm))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[video] detenido por el usuario (q)")
            break

cv2.destroyAllWindows()

print(f"\n[video] fin. {len(todos)} lectura(s) en total")