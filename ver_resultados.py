import os, redis

r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0,
                decode_responses=True)

filas = []
for placa in r.smembers("placas:activas"):
    h = r.hgetall(f"placa:{placa}")
    filas.append((
        placa,
        h.get("estado", "sin analizar"),   # confirmada / rechazada / fusionada
        h.get("tipo", "-"),
        h.get("veces", "0"),
        h.get("conf_ocr", "-"),
        h.get("votos", "-"),
        h.get("frames_vistos", "-"),
        h.get("canonica", ""),             # si fue fusionada, en cuál
    ))

filas.sort(key=lambda f: -int(f[3]))
cab = ("PLACA", "ESTADO", "TIPO", "VECES", "CONF", "VOTOS", "FRAMES", "FUSIONADA EN")
print(("{:<10}{:<14}{:<10}{:<7}{:<8}{:<7}{:<8}{}").format(*cab))
print("-" * 80)
for f in filas:
    print(("{:<10}{:<14}{:<10}{:<7}{:<8}{:<7}{:<8}{}").format(*f))
print(f"\nTotal: {len(filas)} | confirmadas: {r.scard('placas:confirmadas')} "
      f"| alertas: {r.scard('placas:alerta')}")