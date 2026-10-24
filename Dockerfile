# SDR-RFID - imagen del detector + analisis
#
# Uso:
#   docker compose up --build          # levanta redis + detector + analisis
#   docker compose logs -f detector    # ver la salida del pipeline

ARG PY_VERSION=3.14
FROM python:${PY_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Librerias de sistema que onnxruntime/opencv necesitan aun en modo headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) torch + torchvision CPU (deep-sort-realtime usa torchvision para el embedder)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX} torch torchvision

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY . .

CMD ["python", "main.py"]
