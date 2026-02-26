# Async Federated Learning Platform

A production-ready distributed federated learning platform with real networking, web dashboard, and model management.

## Quick Start

### Local Development (No Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start API Server
python -m networking.server_api

# Start Dashboard (separate terminal)
cd dashboard && npm install && npm run dev

# Run Client
python client_app/client_app.py --server http://localhost:8000 --client-id client_1
```

### With Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| API Server | 8000 | FastAPI + WebSocket |
| Dashboard | 3000 | React Web UI |
| Redis | 6379 | Caching (optional) |

## Features

### Core Engine (`core_engine/`)
- Async server with staleness weighting
- Hybrid robust aggregation
- Trust scoring & quarantine
- Differential Privacy
- Secure aggregation simulation
- Gradient compression (top-k)
- PEFT support for HuggingFace models

### Networking (`networking/`)
- FastAPI REST API
- WebSocket for live updates
- SQLite experiment tracking
- Client registration & management

### Model Registry (`model_registry/`)
- HuggingFace model loading
- Custom CNN/MLP architectures
- PEFT (LoRA) integration
- Local model upload

### Authentication (`api/`)
- JWT token-based auth
- Role-based access (Admin, Observer, Client)
- API key management

### Dashboard (`dashboard/`)
- Real-time metrics visualization
- Client monitoring panel
- Model management
- Experiment controls

## Configuration

### Environment Variables

```bash
ENV=dev                    # dev or prod
SECRET_KEY=your-secret    # JWT secret
SERVER_PORT=8000
```

### Docker Compose

```bash
# Development
ENV=dev docker-compose up

# Production (requires HTTPS)
ENV=prod docker-compose up -f docker-compose.prod.yml
```

## Client Commands

```bash
# Connect via WebSocket
python client_app/client_app.py --server http://SERVER_IP:8000 --client-id client_1

# With authentication
python client_app/client_app.py --server http://SERVER_IP:8000 --token YOUR_TOKEN
```

## API Endpoints

- `GET /api/server/status` - Server status
- `POST /api/experiments/start` - Start experiment
- `POST /api/experiments/{id}/control` - Control (pause/resume/stop)
- `GET /api/experiments/{id}/metrics` - Get metrics
- `GET /api/models` - List registered models
- `POST /api/models/register` - Register new model
- `WS /ws` - WebSocket for live updates

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Dashboard  │────▶│  API Server │◀────│   Clients  │
│   (React)   │     │  (FastAPI)  │     │  (Python)  │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Core FL   │
                   │   Engine    │
                   └─────────────┘
```

## Testing

```bash
# Unit tests
pytest core_engine/tests/ -v

# Integration test (3 local clients)
python tests/integration_local.py
```
