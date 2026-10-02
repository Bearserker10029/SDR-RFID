from ultralytics import YOLO
import cv2

model = YOLO("models/best_placas_v1.pt")

# Probar con una camara
results = model(0, show=True)
cv2.waitKey(0)