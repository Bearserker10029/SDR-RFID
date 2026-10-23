from ultralytics import YOLO
import cv2

model = YOLO("models/best_placas_v1.pt")

# Probar con una camara
results = model("./fine_tuning/image/3.png")

# Dibuja las detecciones sobre la imagen
annotated = results[0].plot()

# Muestra y espera una tecla (queda fijo hasta que presiones algo)
cv2.imshow("Deteccion de placa", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()