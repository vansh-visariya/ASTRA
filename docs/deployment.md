# Deployment Guide

How to run ASTRA in production.

## Quick Deploy (Docker Compose)

```bash
# Clone and configure
git clone https://github.com/vansh-visariya/ASTRA.git
cd ASTRA
cp .env.example .env
# Edit .env — set a strong SECRET_KEY, set ENV=prod

# Start all services
docker compose up -d
# API on :8000, Dashboard on :3000
```

## Manual Deployment

### Backend (FastAPI)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install gunicorn

# Production server with multiple workers
gunicorn astra.app.server_api:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000
```

### Dashboard (Next.js)

```bash
cd dashboard
npm install
npm run build
npm start  # Runs on port 3000
```

## Reverse Proxy (Nginx)

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate     /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;

    # API
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket
    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    # Dashboard
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }
}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SECRET_KEY` | **Yes** (prod) | JWT signing secret, 32+ chars |
| `ENV` | **Yes** | Set to `prod` |
| `GEMINI_API_KEY` | No | AI model recommendations |
| `DB_PATH` | No | SQLite path (default: `./astra.db`) |
| `ASTRA_DEFAULT_ADMIN_PASSWORD` | No | First-run admin password (default: `adminpass`) |
| `ASTRA_SEED` | No | RNG seed for reproducibility |

## Scaling

### SQLite → PostgreSQL

For concurrent write-heavy workloads, switch to PostgreSQL:

1. Set `DATABASE_URL=postgresql://user:pass@host:5432/astra` in `.env`
2. Run Alembic migrations: `alembic upgrade head`
3. The codebase uses SQLAlchemy models — switch the `create_engine` call in `config.py`

### Multiple Server Instances

For horizontal scaling behind a load balancer:

- Use Redis for WebSocket state sharing (included in `docker-compose.yml` production profile)
- Run multiple `gunicorn` instances
- Sticky sessions required for WebSocket connections

### GPU Support

For GPU-accelerated training:

```bash
# Use the CUDA-enabled PyTorch base image
# Build with: docker build -f Dockerfile.gpu -t astra-gpu .
```

Or install locally:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then pass `--device cuda:0` when starting clients:
```bash
python -m astra.client.cli --server http://host:8000 --client-id gpu_client --device cuda:0
```

## Monitoring

- Health check: `GET /health`
- Server status: `GET /api/server/status`
- System metrics: `GET /api/system/metrics`
- Swagger docs: `/docs`
- Dashboard: real-time metrics via WebSocket at `:3000`

## Troubleshooting

### Clients can't connect
- Check the server is running: `curl http://localhost:8000/health`
- Verify CORS settings in `server_api.py` if dashboard and API are on different hosts
- Check firewall allows port 8000

### WebSocket disconnects frequently
- Ensure the reverse proxy has `Upgrade` and `Connection` headers configured for `/ws`
- Increase proxy timeout settings (at least 60s)

### SQLite "database is locked"
- SQLite under heavy concurrent writes hits lock contention
- Solution: increase WAL timeout or migrate to PostgreSQL

### Model training produces NaN loss
- Reduce learning rate in `config.yaml`
- Check `sigma` and `clip_norm` in privacy settings
- Disable malicious simulation if enabled
