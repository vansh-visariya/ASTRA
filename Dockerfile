FROM python:3.10-slim

LABEL maintainer="DevHacks Team"
LABEL description="Async Federated Learning Framework"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "main.py", "--config", "config.yaml", "--demo"]
