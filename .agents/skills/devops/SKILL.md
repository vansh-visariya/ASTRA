# SKILL: DevOps & Infrastructure for FL Platform

## PURPOSE
Production deployment, containerization, orchestration, monitoring, logging, and CI/CD for the federated learning platform.

---

## ARCHITECTURE

```
infrastructure/
├── docker/
│   ├── Dockerfile.server        # FL gRPC server
│   ├── Dockerfile.dashboard     # FastAPI dashboard backend
│   ├── Dockerfile.client        # FL training client
│   └── docker-compose.yml       # Local dev stack
├── k8s/                         # Kubernetes manifests
│   ├── fl-server/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── hpa.yaml             # Horizontal Pod Autoscaler
│   ├── dashboard/
│   ├── postgres/
│   └── monitoring/
│       ├── prometheus.yaml
│       └── grafana.yaml
├── monitoring/
│   ├── prometheus.yml           # Scrape config
│   ├── alerts.yml               # Alerting rules
│   └── dashboards/              # Grafana JSON dashboards
├── scripts/
│   ├── generate_certs.sh        # TLS cert generation
│   ├── healthcheck.sh           # Deployment healthcheck
│   └── backup_models.sh         # Model storage backup
└── .github/
    └── workflows/
        ├── ci.yml               # Tests on PR
        └── cd.yml               # Deploy on main merge
```

---

## DOCKER

### FL Server Dockerfile
```dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Runtime stage ---
FROM python:3.11-slim

WORKDIR /app

# Non-root user for security
RUN groupadd -r fluser && useradd -r -g fluser fluser

COPY --from=builder /root/.local /home/fluser/.local
COPY --chown=fluser:fluser . .

# Generate proto stubs at build time
RUN python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./generated \
    --grpc_python_out=./generated \
    ./proto/fl_service.proto

USER fluser

# gRPC port, health check port
EXPOSE 50051 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import grpc; ch=grpc.insecure_channel('localhost:50051'); grpc.channel_ready_future(ch).result(timeout=3)"

ENV PYTHONPATH="/app/generated:/app"
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### docker-compose.yml (Local Dev)
```yaml
version: "3.9"

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: fl
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: fl_platform
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fl"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_PASSWORD}
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"

  fl-server:
    build:
      context: .
      dockerfile: docker/Dockerfile.server
    environment:
      DATABASE_URL: postgresql+asyncpg://fl:${POSTGRES_PASSWORD}@postgres/fl_platform
      REDIS_URL: redis://:${REDIS_PASSWORD}@redis:6379/0
      JWT_SECRET: ${JWT_SECRET}
      S3_ENDPOINT: http://minio:9000
      S3_BUCKET: fl-models
    ports:
      - "50051:50051"
    depends_on:
      postgres: { condition: service_healthy }
      redis: { condition: service_healthy }
    restart: unless-stopped

  dashboard:
    build:
      context: .
      dockerfile: docker/Dockerfile.dashboard
    environment:
      DATABASE_URL: postgresql+asyncpg://fl:${POSTGRES_PASSWORD}@postgres/fl_platform
      FL_SERVER_ADDR: fl-server:50051
      JWT_SECRET: ${JWT_SECRET}
    ports:
      - "8000:8000"
    depends_on:
      - fl-server
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/alerts.yml:/etc/prometheus/alerts.yml:ro
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  minio_data:
  grafana_data:
```

---

## METRICS & MONITORING

### Prometheus Metrics (Python)
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics at module level (singletons)
ROUNDS_TOTAL = Counter(
    "fl_rounds_total",
    "Total FL rounds completed",
    ["status"],   # status: "complete" | "failed"
)
ROUND_DURATION = Histogram(
    "fl_round_duration_seconds",
    "Duration of each FL round",
    buckets=[10, 30, 60, 120, 300, 600, 1200],
)
CLIENTS_ACTIVE = Gauge(
    "fl_clients_active",
    "Number of currently active clients",
)
CLIENTS_REGISTERED = Counter(
    "fl_clients_registered_total",
    "Total client registrations",
)
MODEL_LOSS = Gauge(
    "fl_global_model_loss",
    "Latest global validation loss",
)
MODEL_ACCURACY = Gauge(
    "fl_global_model_accuracy",
    "Latest global validation accuracy",
)
UPDATE_SIZE_BYTES = Histogram(
    "fl_update_size_bytes",
    "Size of client model updates in bytes",
    buckets=[1e5, 1e6, 10e6, 50e6, 100e6, 500e6],
)
BYZANTINE_DETECTED = Counter(
    "fl_byzantine_clients_detected_total",
    "Byzantine/anomalous clients detected",
)
DP_EPSILON_SPENT = Gauge(
    "fl_dp_epsilon_spent",
    "Differential privacy budget spent",
)

class MetricsContext:
    """Context manager for automatic round timing."""
    def __init__(self):
        self._start = None
    
    def __enter__(self):
        self._start = time.time()
        return self
    
    def __exit__(self, exc_type, *_):
        duration = time.time() - self._start
        ROUND_DURATION.observe(duration)
        status = "failed" if exc_type else "complete"
        ROUNDS_TOTAL.labels(status=status).inc()

# Usage:
# with MetricsContext():
#     await run_fl_round()
```

### Prometheus Scrape Config
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: "fl-server"
    static_configs:
      - targets: ["fl-server:8080"]
    metrics_path: "/metrics"
  
  - job_name: "dashboard"
    static_configs:
      - targets: ["dashboard:8000"]
    metrics_path: "/metrics"
```

### Alert Rules
```yaml
# monitoring/alerts.yml
groups:
  - name: fl_platform
    rules:
      - alert: FLRoundFailureRateHigh
        expr: rate(fl_rounds_total{status="failed"}[5m]) > 0.3
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "FL round failure rate > 30%"

      - alert: ClientCountLow
        expr: fl_clients_active < 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Fewer than 2 active clients"

      - alert: DPBudgetNearlyExhausted
        expr: fl_dp_epsilon_spent / fl_dp_epsilon_total > 0.9
        labels:
          severity: warning
        annotations:
          summary: "DP privacy budget >90% spent"
      
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
```

---

## STRUCTURED LOGGING

```python
import structlog
import logging

def configure_logging(log_level: str = "INFO", json_output: bool = True):
    """Call once at app startup."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
    )

logger = structlog.get_logger()

# Usage — always include context fields
async def run_round(round_id: int):
    log = logger.bind(round_id=round_id, component="round_manager")
    log.info("round_started")
    try:
        result = await _execute_round(round_id)
        log.info("round_complete", num_clients=result.num_clients, loss=result.loss)
    except Exception as e:
        log.exception("round_failed", error=str(e))
        raise
```

---

## CI/CD PIPELINE

### GitHub Actions CI (ci.yml)
```yaml
name: CI
on: [pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: fl
          POSTGRES_PASSWORD: test
          POSTGRES_DB: fl_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      
      - name: Install deps
        run: pip install -r requirements.txt -r requirements-dev.txt
      
      - name: Generate proto stubs
        run: |
          python -m grpc_tools.protoc -I./proto \
            --python_out=./generated --grpc_python_out=./generated \
            ./proto/fl_service.proto
      
      - name: Lint
        run: |
          ruff check .
          mypy . --ignore-missing-imports
      
      - name: Unit tests
        run: pytest tests/unit -v --cov=. --cov-report=xml
        env:
          DATABASE_URL: postgresql+asyncpg://fl:test@localhost/fl_test
          JWT_SECRET: test-secret-key-minimum-32-chars
      
      - name: Integration tests
        run: pytest tests/integration -v --timeout=120
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
          fail_ci_if_error: true
```

---

## ENVIRONMENT VARIABLES (.env template)
```bash
# REQUIRED — no defaults
JWT_SECRET=                   # Min 32 chars, random, keep secret
POSTGRES_PASSWORD=            # Strong password
REDIS_PASSWORD=               # Strong password
MINIO_USER=                   # MinIO access key
MINIO_PASSWORD=               # MinIO secret (min 8 chars)
GRAFANA_PASSWORD=             # Grafana admin password

# FL Server
GRPC_PORT=50051
HTTP_PORT=8080
MIN_CLIENTS_PER_ROUND=2
NUM_ROUNDS=100
DP_ENABLED=true
DP_EPSILON=10.0
DP_DELTA=1e-5

# Storage
S3_BUCKET=fl-models
S3_ENDPOINT=http://minio:9000  # Leave empty for AWS S3

# Logging
LOG_LEVEL=INFO
LOG_JSON=true                  # true for production, false for dev

# TLS (required in production)
TLS_CERT_PATH=certs/server.crt
TLS_KEY_PATH=certs/server.key
TLS_CA_PATH=certs/ca.crt
```

---

## TLS CERTIFICATE GENERATION (scripts/generate_certs.sh)
```bash
#!/bin/bash
set -euo pipefail

CERT_DIR="certs"
mkdir -p "$CERT_DIR"

# 1. CA key + cert
openssl genrsa -out "$CERT_DIR/ca.key" 4096
openssl req -new -x509 -days 3650 -key "$CERT_DIR/ca.key" \
  -out "$CERT_DIR/ca.crt" \
  -subj "/CN=FL-CA/O=FLPlatform"

# 2. Server key + CSR + cert (signed by CA)
openssl genrsa -out "$CERT_DIR/server.key" 2048
openssl req -new -key "$CERT_DIR/server.key" \
  -out "$CERT_DIR/server.csr" \
  -subj "/CN=fl-server/O=FLPlatform"
openssl x509 -req -days 365 \
  -in "$CERT_DIR/server.csr" \
  -CA "$CERT_DIR/ca.crt" -CAkey "$CERT_DIR/ca.key" \
  -CAcreateserial \
  -out "$CERT_DIR/server.crt"

echo "Certificates generated in $CERT_DIR/"
echo "Expires: $(openssl x509 -enddate -noout -in $CERT_DIR/server.crt)"
```

---

## COMMON DEVOPS BUGS & FIXES

| Bug | Symptom | Fix |
|-----|---------|-----|
| Secret in Dockerfile ENV | Secret in image layers | Use `--secret` build arg or runtime env |
| Running as root in container | Security risk | Add `useradd` + `USER` in Dockerfile |
| No `depends_on` health check | App starts before DB ready | Use `condition: service_healthy` |
| DB connection pool exhaustion | `QueuePool limit exceeded` | Set `pool_size`, `max_overflow` in SQLAlchemy |
| No `PYTHONUNBUFFERED=1` | Logs delayed, lost on crash | Always set in container env |
| Missing resource limits | OOM kills, noisy neighbor | Set `resources.limits` in K8s pods |
| prometheus_client not started | No /metrics endpoint | Call `start_http_server(port)` at startup |
| Logs not JSON in production | Hard to parse in ELK/Loki | Set `LOG_JSON=true` in prod |
| Checkpoint dir not mounted | Checkpoints lost on restart | Mount persistent volume for `/app/checkpoints` |

---

## TESTING CHECKLIST
- [ ] `docker compose up` → all services healthy in <60s
- [ ] gRPC server reachable from client container
- [ ] Prometheus scrapes metrics from all services
- [ ] Alert fires when `fl_clients_active < 2`
- [ ] CI pipeline fails on mypy type errors
- [ ] CI pipeline fails on test failure
- [ ] TLS cert generated + server accepts connections
- [ ] Log output is valid JSON in prod mode
- [ ] Model storage survives container restart (volume mounted)
- [ ] Graceful shutdown: SIGTERM → in-flight round completes