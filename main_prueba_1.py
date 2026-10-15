# SDR-RFID - Deteccion de placas vehiculares peruanas (por INFERENCIA)
#
# Enfoque (basado en bryandejesusrt/Reconocimiento-de-placas-con-IA-Bytecoders):
#   1. YOLO detecta la placa.
#   2. El TIPO de placa se infiere del TAMAÑO fisico (Art. 26.3 del MTC):
#        - 300 x 150 mm  -> livianos/pesados
#        - 170 x 200 mm  -> vehiculos menores
#   3. Se hace UNA sola lectura OCR de toda la placa.
#   4. Se corrige posicion por posicion segun la estructura del tipo:
#        - zona de letras: si el OCR lee un digito, se mapea a letra (0->O, 1->I...)
#        - zona de digitos: si el OCR lee una letra, se mapea a digito (O->0, G->0...)
#   5. Se valida contra el patron del tipo inferido.

import os
import re
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

#  CONFIGURACION 
MODELO = "./models/best_placas_v1.pt"
IMAGEN_PRUEBA = "./salidas/f02700_FU_0.jpg"
CARPETA_SALIDA = "salidas"
CONF_MINIMA = 0.3          # filtrado por confianza
SCALE = 3                 # warp en alta resolucion

# Dimensiones legales de placa (Art. 26.3, en mm usadas como pixeles del warp)
DIMS_LIVIANO = (300, 150)
DIMS_MENOR   = (170, 200)

ALLOW_LETRAS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALLOW_DIGITOS = "0123456789"
ALLOW_TODO = ALLOW_LETRAS + ALLOW_DIGITOS

#  ESTRUCTURAS DE LA NORMA 
ESTRUCTURA_LIVIANO       = "LAADDD"     # ABC-123
ESTRUCTURA_MENOR         = "LLLDDDD"    # ABC 1234 (dos lineas)
ESTRUCTURA_ESPECIAL      = "LLADDA"     # EAB-12C

ESTRUCTURAS = {
    "liviano": ESTRUCTURA_LIVIANO,
    "menor": ESTRUCTURA_MENOR,
    "especial": ESTRUCTURA_ESPECIAL,
}

# Correciones cuando el OCR lee un CARACTER pero la posicion exige un DIGITO.
MAPA_LETRA_A_DIGITO = {
    'O': '0', 'Q': '0', 'D': '0', 'G': '0', 'C': '0',
    'I': '1', 'L': '1', 'T': '1',
    'Z': '2',
    'J': '3', 'E': '3',
    'A': '4',
    'S': '5',
    'B': '8',
}

# Correciones cuando el OCR lee un DIGITO pero la posicion exige una LETRA.
MAPA_DIGITO_A_LETRA = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '3': 'J',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '8': 'B',
}

MAPA_LETRA_A_LETRA = {
}

MAPA_DIGITO_A_DIGITO = {
}

def corregir_por_estructura(texto, estructura):
    """Corrige cada caracter segun la zona la estructura."""
    if len(texto) != len(estructura):
        return texto
    salida = []
    for ch, zona in zip(texto, estructura):
        if zona == 'D':
            salida.append(MAPA_LETRA_A_DIGITO.get(ch, ch))
        elif zona == 'L':
            salida.append(MAPA_DIGITO_A_LETRA.get(ch, MAPA_LETRA_A_LETRA.get(ch, ch)))
        else:  # 'A': alfanumerico
            salida.append(ch)
    return "".join(salida)

def placa_valida(s, tipo=None):
    """Valida contra el patron del tipo."""
    tipos = [tipo] if tipo else list(ESTRUCTURAS.keys())
    for t in tipos:
        est = ESTRUCTURAS[t]
        if len(s) != len(est):
            continue
        if all(z == 'A' or (z == 'L' and c in ALLOW_LETRAS) or (z == 'D' and c in ALLOW_DIGITOS)
               for c, z in zip(s, est)):
            return True
    return False

def tipo_placa(s):
    for t, est in ESTRUCTURAS.items():
        if len(s) == len(est) and placa_valida(s, t):
            return t
    return "desconocido"

#  FUNCIONES AUXILIARES 
def preprocesar(warped):
    """Gris + CLAHE + unsharp mask para maximizar nitidez de caracteres."""
    g = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    sv = cv2.GaussianBlur(cl, (0, 0), 3)
    return cv2.addWeighted(cl, 1.8, sv, -0.8, 0)

def hdr_fusion(img_bgr):
    """HDR por sintesis de exposiciones + fusion Mertens: a partir de UNA imagen se generan varias exposiciones (subexpuesta a
    sobreexpuesta), se fusionan con cv2.createMergeMertens y se comprime el rango dinamico a 8 bits. Resultado: detalle simultaneo en luces y sombras"""
    img_f = img_bgr.astype(np.float32) / 255.0
    exposiciones = [np.clip(img_f * f, 0, 1) for f in (0.5, 1.0, 2.0)]
    fusion = cv2.createMergeMertens().process(exposiciones)
    return (np.clip(fusion, 0, 1) * 255).astype(np.uint8)

def clasificar_por_tamano(x1, y1, x2, y2):
    """Clasifica SOLO por el tamaño de la caja detectada (inferencia geometrica).
       - Liviano/pesado: 300x150  -> relacion ancho/alto = 2.0
       - Menor (moto):   170x200 -> relacion ancho/alto = 0.85"""
    ancho = float(x2 - x1)
    alto = float(y2 - y1)
    if ancho <= 0 or alto <= 0:
        return "liviano"
    if alto > ancho * 1.15:
        return "menor"
    return "liviano"

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

def ajustar_longitud(crudo, estructura):
    """Si el OCR devuelve mas caracteres de los que caben en la estructura, probar quitando de a
       uno los caracteres sospechosos hasta que la longitud coincida.
       Prioridad de borrado: 'F' (tipico falso del guion), luego no alfanumericos
       problematicos, luego cualquiera. Devuelve el texto recortado o None."""
    n = len(estructura)
    if len(crudo) == n:
        return crudo
    if len(crudo) < n:
        return None
    exceso = len(crudo) - n

    prefieren_quitar = [i for i, c in enumerate(crudo) if c == 'F']
    demas = [i for i in range(len(crudo)) if i not in prefieren_quitar]
    from itertools import combinations

    # probar combinaciones:
    opciones = list(combinations(prefieren_quitar + demas, exceso))

    # ordenar para probar primero
    opciones.sort(key=lambda comb: sum(1 for i in comb if crudo[i] == 'F'), reverse=True)
    for comb in opciones:
        recortado = "".join(c for i, c in enumerate(crudo) if i not in comb)
        if placa_valida(corregir_por_estructura(recortado, estructura), None):
            return recortado
        
    # si ninguna combinacion valida, quitar de derecha a izquierda las 'F' y recortar
    idx = list(range(len(crudo)))
    while len(idx) > n:
        f_encontrada = next((i for i in reversed(idx) if crudo[i] == 'F'), idx[-1])
        idx.remove(f_encontrada)
    return "".join(crudo[i] for i in idx)

def inferir_placa(warped, nit, estructura):
    """INFERENCIA: lectura OCR completa, se quita el banner 'PERU', se ajusta la longitud quitando ruido y se corrige posicion a posicion segun la estructura.
       Devuelve (texto_crudo, texto_corregido, confianza_ocr, n_recortados)
       o None. n_recortados > 0 indica que hubo que borrar caracteres del OCR para cuadrar la longitud."""
    
    alto = warped.shape[0]
    hdr = hdr_fusion(warped)
    hdr_nit = preprocesar(hdr)
    for version in (nit, hdr_nit, cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), warped):
        detecciones = ocr.readtext(version, detail=1, allowlist=ALLOW_TODO)
        crudo, cf = unir_lecturas(detecciones, alto)
        if crudo.startswith("PERU"):                # quitar banner
            crudo = crudo[4:]
        print(f"  [debug] lectura unida: '{crudo}' (conf={cf:.2f})")
        original = crudo
        crudo = ajustar_longitud(crudo, estructura)
        if crudo is None:
            continue
        n_recortados = (len(original) - len(crudo)) if original else 0
        corregido = corregir_por_estructura(crudo, estructura)
        print(f"  [debug] tras ajuste:   '{crudo}' -> '{corregido}' (recortados={n_recortados})")
        if placa_valida(corregido, None):
            return crudo, corregido, cf, n_recortados
        return crudo, corregido, cf, n_recortados
    return None

#  CARGA DE MODELOS 
licence_plate_detector = YOLO(MODELO)
ocr = easyocr.Reader(["es"], gpu=False)

img = cv2.imread(IMAGEN_PRUEBA)
if img is None:
    raise FileNotFoundError(f"No se encontro la imagen: {IMAGEN_PRUEBA}")

img_original = img.copy()
results = licence_plate_detector(img)[0]

print(f"[debug] {len(results.boxes)} deteccion(es):")
for b in results.boxes:
    xx1, yy1, xx2, yy2 = map(int, b.xyxy[0])
    print(f"  conf={float(b.conf[0]):.2f} caja=({xx1},{yy1},{xx2},{yy2})")

os.makedirs(CARPETA_SALIDA, exist_ok=True)

#  PROCESAMIENTO 
for i, box in enumerate(results.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])

    if conf < CONF_MINIMA:
        print(f"  [debug] deteccion {i} descartada: conf={conf:.2f} < {CONF_MINIMA} (caja {x1},{y1},{x2},{y2})")
        continue

    cv2.rectangle(img_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_original, f"{conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 1. CLASIFICACION POR TAMAÑO
    tipo_geo = clasificar_por_tamano(x1, y1, x2, y2)
    print(f"  clasificacion (por tamano): {tipo_geo}")

    # 2-3. Warp + INFERENCIA
    src = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    if tipo_geo == "liviano":
        orden = [(DIMS_LIVIANO, ["liviano", "especial"]),
                 (DIMS_MENOR,   ["menor"])]
    else:
        orden = [(DIMS_MENOR,   ["menor"]),
                 (DIMS_LIVIANO, ["liviano", "especial"])]
        
    candidatos = []
    prio = 0
    for (Ww, Hh), tipos in orden:
        dst = np.float32([[0, 0], [Ww*SCALE, 0], [Ww*SCALE, Hh*SCALE], [0, Hh*SCALE]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (Ww*SCALE, Hh*SCALE), flags=cv2.INTER_CUBIC)
        nit = preprocesar(warped)
        for t in tipos:
            res = inferir_placa(warped, nit, ESTRUCTURAS[t])
            if res is None:
                continue
            crudo, corregido, cf, n_rec = res
            n_corr = sum(1 for a, b in zip(crudo, corregido) if a != b)
            valida = placa_valida(corregido, None)
            
            candidatos.append((0 if valida else 1, -cf, n_rec + n_corr, prio, t, warped, nit, res))
            prio += 1

    if not candidatos:
        placa_final, estado, ocr_conf, warped, nit = "???", "sin lectura", 0.0, None, None
    else:
        candidatos.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
        _, _, _, _, tipo_final, warped, nit, res = candidatos[0]
        crudo, corregido, ocr_conf, _ = res
        print(f"  OCR crudo: '{crudo}' -> corregido: '{corregido}' (conf={ocr_conf:.2f}, geo={tipo_final})")
        if placa_valida(corregido, None):
            placa_final, estado = corregido, "VALIDA"
        else:
            placa_final, estado = corregido, "dudosa"

    print(f"Placa {i}: {placa_final} ({estado}, conf={conf:.2f}, ocr={ocr_conf:.2f}, tipo={tipo_placa(placa_final)})")

    # 4. Mostrar y guardar evidencia
    if warped is not None:
        nit_color = cv2.cvtColor(nit, cv2.COLOR_GRAY2BGR)
        fila_placa = np.hstack([warped, nit_color])
        cv2.imshow(f"Placa {i} | {placa_final} ({estado})", fila_placa)
        nombre = f"{placa_final}_{i}.jpg".replace(":", "-").replace("?", "x")
        cv2.imwrite(os.path.join(CARPETA_SALIDA, nombre), fila_placa)

#  VISUALIZACION FINAL 
cv2.imshow("Detecciones originales", img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()