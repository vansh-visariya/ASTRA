# ASTRA — Async Scalable Training & Research Architecture

A distributed **Federated Learning** platform — clients train locally, the server aggregates updates without ever seeing raw data. Includes a dual-panel web dashboard (Admin + Client), Byzantine-robust aggregation, trust scoring, differential privacy, and HuggingFace model support.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Running a Client](#running-a-client)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)

---

## How It Works

1. **Admin** creates a training group and picks a model (built-in CNN/MLP or any HuggingFace model)
2. **Clients** browse groups on their dashboard and request to join
3. **Admin** approves join requests
4. **Client** activates membership → registered as an FL participant
5. **Training** — clients train locally on their own data and push gradient updates to the server
6. **Server** aggregates updates using the configured strategy (FedAvg, Trimmed Mean, Median, or Hybrid); triggers when **N updates arrive** or a **time window expires** — whichever comes first
7. **Dashboard** streams live metrics, trust scores, and training logs via WebSocket

---

## Features

| Category | What's included |
|---|---|
| **Aggregation** | FedAvg, Trimmed Mean, Coordinate Median, Hybrid (trust-weighted + staleness decay) |
| **Robustness** | Byzantine-tolerant trust scoring, soft quarantine, cosine-similarity anomaly detection |
| **Privacy** | DP-SGD (client-side or server-side), gradient clipping, Gaussian noise, moments accountant |
| **Compression** | Top-k sparsification, quantization |
| **Models** | SimpleCNN, CIFAR10CNN, SimpleMLP + any HuggingFace model with optional LoRA/PEFT |
| **Data splits** | IID, Dirichlet (non-IID), pathological |
| **Auth** | JWT + bcrypt, roles: `admin` / `client` / `observer` |
| **Dashboard** | Next.js 14 — separate Admin and Client panels, real-time WebSocket updates |
| **Notifications** | In-app notification system for join approvals, training events |
| **Model Recommendations** | Gemini API–powered suggestions (optional) |

---

## Project Structure

```
ASTRA/
├── src/astra/              # All Python source (importable as astra.*)
│   ├── app/                # API layer, orchestration, group lifecycle
│   │   ├── server_api.py   # FastAPI entry point (REST + WebSocket + Socket.IO)
│   │   ├── fl_server.py    # FLServer — wires core engine to API
│   │   ├── group_manager.py# Manages concurrent training groups
│   │   ├── database.py     # AstraDB — single SQLite file (astra.db)
│   │   ├── integration.py  # FLPlatformIntegration — auth/notifs/trust/recommender
│   │   ├── extended_endpoints.py  # Auth, join, notifications, trust REST routes
│   │   ├── notifications.py       # In-app notification service
│   │   ├── model_recommender.py   # Gemini-powered model suggestions
│   │   └── routes/         # system, groups, clients, models, experiments routers
│   │
│   ├── core/               # FL algorithms — no web dependencies
│   │   ├── server.py       # AsyncServer — staleness-weighted aggregation engine
│   │   ├── trust_manager.py# TrustManager — cosine-sim scoring + quarantine
│   │   ├── compression.py  # Top-k sparsification + quantization
│   │   ├── data_splitter.py# IID / Dirichlet / pathological splits
│   │   ├── inference.py    # Server-side and client-side inference
│   │   ├── exceptions.py   # Custom FL exceptions
│   │   ├── aggregation/    # aggregator.py, robust.py, heterogeneous.py
│   │   ├── models/         # model_zoo.py (CNN/MLP), hf_models.py (HF + PEFT)
│   │   ├── privacy/        # privacy.py (DP-SGD), malicious_simulator.py
│   │   └── utils/          # metrics.py, seed.py, logging_utils.py
│   │
│   ├── infra/              # Transport and storage
│   │   ├── connection_manager.py  # WebSocket broadcast / per-client send
│   │   ├── websocket_handler.py   # WS endpoint + Socket.IO handlers
│   │   ├── models.py       # Pydantic schemas (ClientUpdate, ExperimentConfig…)
│   │   ├── registry.py     # ModelRegistry — HF / custom / local .pt models
│   │   └── security/auth.py# AuthManager, TokenManager, JoinRequestManager
│   │
│   └── client/
│       └── client.py       # FederatedClient CLI — trains locally, pushes updates
│
├── dashboard/              # Next.js 14 web dashboard
│   ├── app/dashboard/      # Admin UI (groups, join requests, logs, metrics)
│   └── app/client/         # Client UI (browse groups, training, trust, notifs)
│
├── tests/                  # pytest unit tests (36 tests)
├── config.yaml             # Default training configuration
├── pyproject.toml          # Package config + pytest settings
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Multi-service deployment
└── conftest.py             # Adds src/ to sys.path for pytest

```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- (Optional) Docker & Docker Compose

### 1 — Backend setup

```bash
# Clone and enter the repo
git clone https://github.com/vansh-visariya/ASTRA.git
cd ASTRA

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / Mac

# Install Python dependencies
pip install -r requirements.txt

# Start the API server  (http://localhost:8000)
uvicorn astra.app.server_api:app --reload
```

> **Alternative (run as script):**
> ```bash
> python src/astra/app/server_api.py
> ```

### 2 — Dashboard setup

```bash
# In a new terminal
cd dashboard
npm install
npm run dev
# Dashboard → http://localhost:3000
```

### 3 — Register users

Open `http://localhost:3000`, go to the login page and sign up:
- Role **`admin`** → access the Admin dashboard
- Role **`client`** → access the Client dashboard

### Docker Compose

```bash
docker-compose up -d        # start all services
docker-compose logs -f      # stream logs
docker-compose down         # stop
```

---

## Running a Client

```bash
# Basic usage
python src/astra/client/client.py \
  --server http://localhost:8000 \
  --client-id client_1

# With a specific group and more rounds
python src/astra/client/client.py \
  --server http://localhost:8000 \
  --client-id client_1 \
  --group-id group_a
```

The client auto-registers, downloads the global model, trains locally, and pushes updates. It reconnects automatically on connection loss.

---

## API Reference

Interactive docs available at `http://localhost:8000/docs` (Swagger) once the server is running.

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/signup` | Register (`role`: admin / client / observer) |
| `POST` | `/api/auth/login` | Login → JWT token |

### Groups
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/groups` | List all groups |
| `POST` | `/api/groups` | Create group (admin) |
| `GET` | `/api/groups/{id}` | Group detail |
| `POST` | `/api/groups/{id}/start\|pause\|resume\|stop` | Lifecycle control (admin) |

### Join Requests
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/join/join-request` | Request to join (client) |
| `GET` | `/api/join/join-requests` | List pending (admin) |
| `POST` | `/api/join/join-requests/approve\|reject` | Approve / reject (admin) |
| `POST` | `/api/join/activate/{group_id}` | Activate after approval (client) |

### Other
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/notifications` | Get notifications |
| `POST` | `/api/notifications/{id}/read` | Mark read |
| `GET` | `/api/models` | List registered models |
| `POST` | `/api/models/register/hf` | Register a HuggingFace model |
| `GET` | `/health` | Health check |
| `WS` | `/ws?token=<jwt>` | WebSocket — live dashboard updates |

---

## Configuration

### Environment Variables

```bash
SECRET_KEY=your-secure-key            # JWT signing secret (REQUIRED in prod)
ENV=dev                               # dev | prod  (prod enforces SECRET_KEY)
NEXT_PUBLIC_API_URL=http://localhost:8000  # Dashboard → API URL
GEMINI_API_KEY=your-key               # Optional — Gemini model recommendations
```

### `config.yaml` — Key Settings

```yaml
server:
  aggregator_window: 10   # aggregate after N updates …
  poll_timeout: 1.0       # … or after T seconds (whichever first)
  async_lambda: 0.2       # staleness decay factor

robust:
  method: hybrid          # fedavg | trimmed_mean | median | hybrid

privacy:
  dp_enabled: true
  dp_mode: client         # client | server
  clip_norm: 1.0
  sigma: 1.2

communication:
  compression: topk
  topk_ratio: 0.1
```

See `config.yaml` for the full list (dataset, model, client, trust, malicious, logging settings).

---

## Testing

```bash
# Run all unit tests
pytest tests/ -v

# Expected: 36 passed
```

Tests cover: aggregation strategies, compression, differential privacy, reproducibility, trust manager.

---

## Deployment

| Service | Port | Description |
|---------|------|-------------|
| API Server | 8000 | FastAPI — REST + WebSocket + FL engine |
| Dashboard | 3000 | Next.js — Admin + Client UI |

### Docker Compose (recommended for prod)

```bash
docker-compose up -d
```

### Production checklist

- [ ] Set `ENV=prod` and a strong random `SECRET_KEY` (32+ chars)
- [ ] Set `NEXT_PUBLIC_API_URL` to your domain
- [ ] Configure CORS origins in `server_api.py`
- [ ] Put HTTPS in front (nginx / Caddy / Traefik)
- [ ] Migrate from SQLite to PostgreSQL for concurrent-write workloads

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| ML | PyTorch, HuggingFace Transformers, PEFT (LoRA) |
| Database | SQLite with WAL mode (`astra.db`) |
| Auth | PyJWT, bcrypt |
| Real-time | WebSocket, Socket.IO (fastapi-socketio) |
| Deployment | Docker, Docker Compose |

---

## License

This project is for research and educational purposes.
