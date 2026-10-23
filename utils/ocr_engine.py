# SDR-RFID - Capa de compatibilidad EasyOCR -> RapidOCR

from rapidocr import RapidOCR


class RapidOCRReader:
    """Drop-in replacement de easyocr.Reader para este proyecto."""

    def __init__(self, **params):
        # params admite overrides de RapidOCR, p. ej.:
        #   RapidOCRReader(params={"Rec.model_path": "..."})
        self._engine = RapidOCR(**params)

    def readtext(self, img, detail=1, allowlist=None, solo_rec=False, **kw):
        """Replica ocr.readtext() de easyocr.
           - img: numpy array BGR (o path str).
           - detail: se acepta por compatibilidad (el resultado siempre
             trae bbox + texto + confianza).
           - allowlist: ignorado (el filtrado se hace post-OCR).
           - kw: se pasan al engine (p. ej. text_score=0.3 para lecturas
             de confianza mas baja).
           Devuelve [(bbox, texto, confianza), ...] igual que easyocr.
           Soporta las 3 formas que ha tenido la API de rapidocr:
             v3+ : engine() -> RapidOCROutput(.boxes, .txts, .scores)
             vieja: engine() -> (result, elapse), item = [box, (texto, score)]
             vieja: engine() -> (result, elapse), item = [box, texto, score]
           - solo_rec=True: reconocimiento puro (use_det=False, use_cls=False,
             use_rec=True) para crops ya localizados: ~60-70% mas rapido en CPU.
             La caja devuelta es ficticia (el crop entero es la placa)."""
        if solo_rec:
            kw.update(use_det=False, use_cls=False, use_rec=True)
        salida = self._engine(img, **kw)

        if isinstance(salida, (tuple, list)):
            result, _elapse = salida
        else:
            txts = getattr(salida, "txts", None) or ()
            scores = getattr(salida, "scores", None) or ()
            boxes = getattr(salida, "boxes", None)
            if not txts:
                return []
            return [
                (_a_lista(boxes[i]) if boxes is not None and i < len(boxes)
                 else [[0, 0], [1, 0], [1, 1], [0, 1]], txt, float(sc))
                for i, (txt, sc) in enumerate(zip(txts, scores))
            ]
        if not result:
            return []
        normalizado = []
        for item in result:
            box = item[0]
            if isinstance(item[1], (tuple, list)):
                texto, score = item[1]
            else:
                texto, score = item[1], item[2]
            normalizado.append((_a_lista(box), texto, float(score)))
        return normalizado


def _a_lista(box):
    """Convierte la caja a lista plana [[x, y], ...] (numpy -> list)."""
    if hasattr(box, "tolist"):
        return box.tolist()
    return [list(p) for p in box]