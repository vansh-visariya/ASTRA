FROM python:3.10-slim

LABEL maintainer="ASTRA"
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

CMD ["python", "-m", "uvicorn", "astra.app.server_api:app", "--host", "0.0.0.0", "--port", "8000"]
