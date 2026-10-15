# SDR-RFID - Submodulo de analisis acumulado de placas (Redis)
#
# Pipeline:
#   1. Recorre "placas:activas".
#   2. Para cada placa con al menos MIN_MATCHES avistamientos:
#        - infiere el TIPO de vehiculo a partir de la lectura (liviano / menor / especial segun la estructura de la norma),
#        - verifica si la placa es VALIDA segun esa estructura,
#        - guarda el veredicto en Redis (tipo, valida, estado, ts_analisis),
#        - la mueve al set "placas:confirmadas" (lista corta para alertas),
#   3. Queda corriendo y re-analiza cada INTERVALO_S.

import time
import redis

from placas_utils import placa_valida, inferir_tipo

#  CONFIGURACION
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

MIN_MATCHES = 3            # avistamientos minimos para confirmar una placa
UMBRAL_MATCH = 0.8         # score minimo para agrupar placas similares
INTERVALO_S = 10           # segundos entre rondas
MODO_CONTINUO = False      # True = daemon; False = una sola pasada

def match_score(a, b):
    """Coincidencia por digito tolerando desplazamientos (caracter faltante
       o extra al inicio/fin). Se prueba alinear b contra a en todos los
       offsets y se toma el mejor:
       'ABC123' vs 'BC123'  -> 5/6 = 0.83  (falta 'A' al inicio)
       'ABC123' vs 'ABC12B' -> 5/6 = 0.83  (ultimo digito dudoso)
       'ABC123' vs 'XYZ789' -> 0.0"""
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    mejor = 0
    for des in range(-m + 1, n):        # b desplazado respecto a a
        aciertos = sum(1 for i in range(n)
                       if 0 <= i - des < m and a[i] == b[i - des])
        mejor = max(mejor, aciertos)
    sim_desplazamiento = mejor / max(n, m)
    fila = list(range(m + 1))           # Levenshtein (programacion dinamica)
    for i, ca in enumerate(a, 1):
        nueva = [i]
        for j, cb in enumerate(b, 1):
            nueva.append(min(fila[j] + 1, nueva[-1] + 1, fila[j - 1] + (ca != cb)))
        fila = nueva
    sim_edicion = 1 - fila[-1] / max(n, m)
    return max(sim_desplazamiento, sim_edicion)

def agrupar(lecturas, umbral):
    """Agrupa lecturas similares (greedy: primero las de mas veces).
       lecturas: lista [(placa, veces)]. Devuelve lista de grupos
       [{'centroide': p, 'miembros': [(p, v), ...], 'total': int}]."""
    pend = sorted(lecturas, key=lambda x: -x[1])
    grupos = []
    while pend:
        base, vbase = pend.pop(0)
        grupo = [(base, vbase)]
        resto = []
        for p, v in pend:
            if match_score(base, p) >= umbral:
                grupo.append((p, v))
            else:
                resto.append((p, v))
        pend = resto
        grupos.append(grupo)
    return grupos

def consultar_incidencias(r, placa):
    """Hook OE3: aqui se consulta la base de datos centralizada (requisitorias,
    papeletas, ordenes de captura) y se genera la alerta.
    Debe devolver una lista de incidencias (vacia si no hay)."""
    return []

def analizar_ronda(r):
    """Una pasada de analisis sobre las placas acumuladas (con fusion)."""
    placas = r.smembers("placas:activas")
    confirmadas, pendientes, rechazadas = 0, 0, 0

    candidatas = []
    for placa in placas:
        estado = r.hget(f"placa:{placa}", "estado")
        if estado == "confirmada":
            confirmadas += 1
            continue
        veces = int(r.hget(f"placa:{placa}", "veces") or 0)
        candidatas.append((placa, veces))

    for grupo in agrupar(candidatas, UMBRAL_MATCH):
        total = sum(v for _, v in grupo)
        # centroide = la de mas apoyo; desempate: valida primero, luego nombre
        centroide = max(grupo, key=lambda x: (placa_valida(x[0]), x[1], x[0]))[0]

        if total < MIN_MATCHES:
            pendientes += len(grupo)
            continue

        tipo = inferir_tipo(centroide)
        valida = placa_valida(centroide, tipo)
        ahora = time.time()

        # marcar fusionados
        for p, _ in grupo:
            if p != centroide:
                r.hset(f"placa:{p}", mapping={
                    "estado": "fusionada",
                    "canonica": centroide,
                    "ts_analisis": ahora,
                })

        r.hset(f"placa:{centroide}", mapping={
            "tipo": tipo,
            "valida": 1 if valida else 0,
            "estado": "confirmada" if valida else "rechazada",
            "match_total": total,
            "ts_analisis": ahora,
        })

        if len(grupo) > 1:
            print(f"  [grupo] {centroide} <= {[p for p, _ in grupo if p != centroide]} "
                  f"(match total={total})")

        if valida:
            confirmadas += 1
            r.sadd("placas:confirmadas", centroide)
            print(f"  [ok] {centroide}: tipo={tipo}, VALIDA ({total} match)")
            incidencias = consultar_incidencias(r, centroide)
            if incidencias:
                r.hset(f"placa:{centroide}", "incidencias", len(incidencias))
                r.sadd("placas:alerta", centroide)
                print(f"  [ALERTA] {centroide}: {len(incidencias)} incidencia(s)")
        else:
            rechazadas += 1
            print(f"  [descartada] {centroide}: tipo={tipo}, placa no valida ({total} match)")

    print(f"[resumen] confirmadas={confirmadas} rechazadas={rechazadas} "
          f"pendientes(<{MIN_MATCHES} match)={pendientes}")
    return confirmadas

def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                    decode_responses=True)
    r.ping()
    print(f"[redis] conectado a {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")

    if MODO_CONTINUO:
        print(f"[analisis] modo continuo cada {INTERVALO_S}s (Ctrl+C para salir)")
        try:
            while True:
                analizar_ronda(r)
                time.sleep(INTERVALO_S)
        except KeyboardInterrupt:
            print("[analisis] detenido")
    else:
        analizar_ronda(r)

if __name__ == "__main__":
    main()