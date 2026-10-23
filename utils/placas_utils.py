# SDR-RFID - Utilidades validacion de placas peruanas
#
# Placas según la norma (Art. 26.3 MTC).

ALLOW_LETRAS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALLOW_DIGITOS = "0123456789"
ALLOW_TODO = ALLOW_LETRAS + ALLOW_DIGITOS

#  ESTRUCTURAS DE LA NORMA
ESTRUCTURAS = {
    "liviano":  "LAADDD",    # ABC-123
    "menor":    "LLLDDDD",   # ABC 1234 (dos lineas)
    "especial": "LLADDA",    # EAB-12C
}

def placa_valida(s, tipo=None):
    """Valida contra el patron del tipo (o contra cualquiera si tipo=None)."""
    if tipo is not None and tipo not in ESTRUCTURAS:
        return False
    tipos = [tipo] if tipo else list(ESTRUCTURAS.keys())
    for t in tipos:
        est = ESTRUCTURAS[t]
        if len(s) != len(est):
            continue
        if all(z == 'A' or (z == 'L' and c in ALLOW_LETRAS) or (z == 'D' and c in ALLOW_DIGITOS)
               for c, z in zip(s, est)):
            return True
    return False

def inferir_tipo(s):
    """Tipo de vehiculo a partir de la placa (liviano / menor / especial)."""
    for t, est in ESTRUCTURAS.items():
        if len(s) == len(est) and placa_valida(s, t):
            return t
    return "desconocido"