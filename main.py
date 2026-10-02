# SDR-RFID - Deteccion de placas vehiculares peruanas
# Soporta formatos de la norma peruana:
#   Art. 17 - ordinaria livianos/pesados:  [A-Z][A-Z0-9]{2}-123   (ABC-123, A1A-234, ...)
#   Art. 16 - ordinaria vehiculos menores: [A-Z][A-Z0-9]{2} 1234  (ABC 1234, sin guion)
#   Art. 19 - especial livianos/pesados:   E[A-Z][A-Z0-9]-12C    (EAB-12C, E mas pequena)
# Proporcion de placa segun Art. 26.3 del Reglamento Nacional de Vehiculos:
#   - Livianos/pesados: 300 x 150 mm (2:1, horizontal)
#   - Vehiculos menores (motos): 170 x 200 mm (vertical)
# Dataset base: Peru License Plate v9 (Roboflow, CC BY 4.0)

import os
import re
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

# CONFIGURACION 
MODELO = "./models/best_placas_v1.pt"
IMAGEN_PRUEBA = "./fine_tuning/image/17.jpg"
CARPETA_SALIDA = "salidas"
CONF_MINIMA = 0.5          # filtrado por confianza

# Proporcion legal de placa (largo x alto en mm, usado como pixeles del warp)
W, H = 300, 150            # livianos y pesados
# W, H = 170, 200          # motos
SCALE = 3                  # warp en alta resolucion

ALLOW_LETRAS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALLOW_DIGITOS = "0123456789"
ALLOW_TODO = ALLOW_LETRAS + ALLOW_DIGITOS

# Patrones segun la norma (Art. 16, 17 y 19)
PATRON_LIVIANO = re.compile(r"^[A-Z][A-Z0-9]{2}\d{3}$")
PATRON_MENOR = re.compile(r"^[A-Z][A-Z0-9]{2}\d{4}$")
PATRON_ESPECIAL = re.compile(r"^E[A-Z][A-Z0-9]\d{2}[A-Z0-9]$")
PATRONES_VALIDOS = (PATRON_LIVIANO, PATRON_MENOR, PATRON_ESPECIAL)

def placa_valida(s):
    return any(p.match(s) for p in PATRONES_VALIDOS)

def tipo_placa(s):
    if PATRON_LIVIANO.match(s):
        return "ordinaria_liviano"
    if PATRON_MENOR.match(s):
        return "ordinaria_menor"
    if PATRON_ESPECIAL.match(s):
        return "especial"
    return "desconocido"

# Celdas
# tipo: L=letra, D=digito, A=alfanumerico
LAYOUT_6 = [("L", .06, .20), ("A", .22, .36), ("A", .38, .52),
            ("D", .60, .74), ("D", .75, .89), ("D", .90, 1.00)]     # ABC-123
LAYOUT_7 = [("L", .05, .17), ("A", .18, .30), ("A", .31, .43),
            ("D", .50, .62), ("D", .63, .75), ("D", .76, .88), ("D", .89, 1.00)]  # ABC 1234
Y1F, Y2F = .27, .84         # franja vertical de los caracteres
ALLOW_CELDA = {"L": ALLOW_LETRAS, "D": ALLOW_DIGITOS, "A": ALLOW_TODO}

os.makedirs(CARPETA_SALIDA, exist_ok=True)

#  CARGA DE MODELOS 
licence_plate_detector = YOLO(MODELO)
ocr = easyocr.Reader(["es"], gpu=False)

img = cv2.imread(IMAGEN_PRUEBA)
if img is None:
    raise FileNotFoundError(f"No se encontro la imagen: {IMAGEN_PRUEBA}")

img_original = img.copy()
results = licence_plate_detector(img)[0]

#  PROCESAMIENTO 
for i, box in enumerate(results.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])

    # 1. Filtrar detecciones de baja confianza
    if conf < CONF_MINIMA:
        continue

    # 2. Dibujar la caja sobre la imagen original
    cv2.rectangle(img_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_original, f"{conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 3. Esquinas de la placa en la imagen original
    src = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # 4. Correccion de perspectiva con proporcion legal (warpPerspective)
    dst = np.float32([[0, 0], [W*SCALE, 0], [W*SCALE, H*SCALE], [0, H*SCALE]])
    M = cv2.getPerspectiveTransform(src, dst)
    placa_frontal = cv2.warpPerspective(img, M, (W*SCALE, H*SCALE),
                                        flags=cv2.INTER_CUBIC)

    # 5. Pre-procesado: gris + CLAHE + unsharp mask
    gris = cv2.cvtColor(placa_frontal, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gris_clahe = clahe.apply(gris)
    suave = cv2.GaussianBlur(gris_clahe, (0, 0), 3)
    nitido = cv2.addWeighted(gris_clahe, 1.8, suave, -0.8, 0)

    # 6. OCR multi-version con allowlist
    candidatos = []
    for version in (nitido, gris_clahe, gris, placa_frontal):
        t = ocr.readtext(version, detail=0, allowlist=ALLOW_TODO)
        if t:
            c = re.sub(r"[^A-Z0-9]", "", "".join(t).upper())
            if len(c) >= 5:  # filtrar lecturas cortas (ruido)
                candidatos.append(c)
                print(f"  intento: {c}")

    # 7. Elegir layout segun cantidad de caracteres de la mejor lectura
    mejor = max(candidatos, key=len) if candidatos else ""
    n_chars = 7 if len(mejor) >= 7 else 6
    celdas = LAYOUT_7 if n_chars == 7 else LAYOUT_6
    Pw, Ph = W*SCALE, H*SCALE

    # 8. Lectura por celdas fijas con allowlist restringido por posicion
    lectura = ""
    debug = cv2.cvtColor(nitido, cv2.COLOR_GRAY2BGR)
    for tipo, fx1, fx2 in celdas:
        cx1, cx2 = int(fx1*Pw), int(fx2*Pw)
        cy1, cy2 = int(Y1F*Ph), int(Y2F*Ph)
        celda = nitido[cy1:cy2, cx1:cx2]
        t = ocr.readtext(celda, detail=0, allowlist=ALLOW_CELDA[tipo])
        lectura += t[0].upper() if t else "?"
        cv2.rectangle(debug, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
        cv2.putText(debug, lectura[-1], (cx1, cy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    print(f"  por celdas ({n_chars} chars): {lectura}")

    # 9. Validar contra los patrones de la norma; priorizar lectura por celdas
    candidatos_finales = [lectura] + candidatos
    validas = [c for c in candidatos_finales if placa_valida(c)]
    if validas:
        placa_final, estado = validas[0], "VALIDA ✓"
    elif candidatos_finales:
        placa_final, estado = max(candidatos_finales, key=len), "dudosa"
    else:
        placa_final, estado = "???", "sin lectura"

    print(f"Placa {i}: {placa_final} ({estado}, conf={conf:.2f}, tipo={tipo_placa(placa_final)})")

    # 10. Mostrar placa + version nitida + cuadricula de celdas
    nitido_color = cv2.cvtColor(nitido, cv2.COLOR_GRAY2BGR)
    fila_placa = np.hstack([placa_frontal, nitido_color])
    cv2.imshow(f"Placa {i} | {placa_final} ({estado})", fila_placa)
    cv2.imshow(f"Celdas {i} (layout {n_chars} chars)", debug)

    # 11. Guardar evidencia
    nombre = f"{placa_final}_{i}.jpg".replace(":", "-").replace("?", "x")
    cv2.imwrite(os.path.join(CARPETA_SALIDA, nombre), fila_placa)

#  VISUALIZACION FINAL 
cv2.imshow("Detecciones originales", img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()