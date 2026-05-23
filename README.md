<p align="center">
  <h1 align="center">ASTRA</h1>
  <p align="center"><strong>A</strong>sync <strong>S</strong>calable <strong>T</strong>raining & <strong>R</strong>esearch <strong>A</strong>rchitecture</p>
  <p align="center">Distributed Federated Learning — train models across devices without sharing data.</p>
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/quick_start-8000?style=flat&logo=fastapi&labelColor=white&color=009688" alt="Quick Start"></a>
  <a href="/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://github.com/vansh-visariya/ASTRA/actions"><img src="https://img.shields.io/badge/tests-43%2F43-brightgreen" alt="Tests"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/next.js-14-black?logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/pytorch-latest-EE4C2C?logo=pytorch" alt="PyTorch">
</p>

---

## Overview

ASTRA is a production-ready federated learning platform. It lets you train machine learning models across distributed clients **without centralizing raw data**. A dual-panel dashboard (Admin + Client) gives you full control over training groups, trust scoring, differential privacy, and real-time metrics.

**What makes it different:** ASTRA uses **hybrid async windowing** — aggregation triggers when _either_ N client updates arrive _or_ a time window expires. No more waiting on stragglers. Byzantine-robust aggregation, trust-based client scoring, and DP-SGD come built in.

---

## How It Works

```
Admin creates group → Clients request to join → Admin approves
  → Clients train locally and push model deltas
  → Server aggregates using FedAvg / Trimmed Mean / Median / Hybrid
  → Global model improves → Dashboard streams live metrics
```

1. **Admin** picks a model (built‑in CNN/MLP or any HuggingFace model) and creates a training group.
2. **Clients** browse available groups and request to join.
3. **Admin** approves pending join requests.
4. **Clients** activate their membership and begin training on their own data.
5. **Updates** flow asynchronously — the server aggregates every `window_size` updates or `time_limit` seconds (whichever comes first).
6. **Dashboard** shows live accuracy, loss, trust scores, and event logs via WebSocket.

---

## Features

| Category | Capability |
|---|---|
| **Aggregation** | FedAvg · Trimmed Mean · Coordinate Median · Hybrid (trust‑weighted + staleness decay) |
| **Robustness** | Byzantine‑tolerant trust scoring · Cosine‑similarity anomaly detection · Soft quarantine |
| **Privacy** | DP‑SGD (client‑side or server‑side) · Gaussian noise · Gradient clipping · Moments accountant |
| **Compression** | Top‑k sparsification · Quantization |
| **Models** | SimpleCNN · CIFAR10CNN · SimpleMLP · Any HuggingFace model · Optional LoRA / PEFT |
| **Data Splits** | IID · Dirichlet (non‑IID) · Pathological |
| **Auth** | JWT + bcrypt · Roles: `admin` \| `client` \| `observer` |
| **Dashboard** | Next.js 14 · Admin panel (groups, join requests, metrics) · Client panel (training, trust, notifications) |
| **Real‑time** | WebSocket + Socket.IO · Live metrics streaming | 
| **Recommendations** | Gemini API‑powered model suggestions (optional) |

---

## Project Structure

```
ASTRA/
├── src/astra/                     # Python package (importable as astra.*)
│   ├── __init__.py                #   Public exports: AsyncServer, FLClient, load_config…
│   ├── app/                       #   API layer — FastAPI, orchestration, DB, groups
│   │   ├── server_api.py          #     Entry point — REST + WebSocket + Socket.IO
│   │   ├── fl_server.py           #     FLServer — bridges core engine ↔ API
│   │   ├── group_manager.py       #     Manages concurrent TrainingGroups
│   │   ├── training_group.py      #     TrainingGroup + AsyncWindowConfig dataclass
│   │   ├── database.py            #     AstraDB — single SQLite file (astra.db), WAL mode
│   │   ├── state.py               #     FLServer singleton accessor
│   │   ├── integration.py         #     Wires auth + notifications + trust + recommender
│   │   ├── extended_endpoints.py  #     Auth / join / notification / trust REST routes
│   │   ├── notifications.py       #     In‑app notification service
│   │   ├── model_recommender.py   #     Gemini‑powered model suggestions
│   │   └── routes/                #     system · groups · clients · models · experiments
│   │
│   ├── core/                      #   FL algorithms — pure Python, zero web deps
│   │   ├── server.py              #     AsyncServer — staleness‑weighted aggregation engine
│   │   ├── fl_client.py           #     FLClient — local training loop
│   │   ├── trust_manager.py       #     Trust scoring via cosine similarity + quarantine
│   │   ├── config.py              #     Centralized YAML + env config loader
│   │   ├── compression.py         #     Top‑k sparsification + quantization
│   │   ├── data_splitter.py       #     IID / Dirichlet / pathological data splits
│   │   ├── inference.py           #     Server‑side and client‑side inference
│   │   ├── exceptions.py          #     Custom FL exception hierarchy
│   │   ├── aggregation/           #     aggregator · robust · heterogeneous
│   │   ├── models/                #     model_zoo (CNN/MLP) · hf_models (HF + PEFT)
│   │   ├── privacy/               #     DP‑SGD · MaliciousSimulator
│   │   └── utils/                 #     metrics · seed · logging
│   │
│   ├── infra/                     #   Transport, schemas, auth, model registry
│   │   ├── connection_manager.py  #     WebSocket registry — broadcast + per‑client send
│   │   ├── websocket_handler.py   #     WS endpoint + Socket.IO event handlers
│   │   ├── models.py              #     Pydantic schemas (ClientUpdate, ExperimentConfig…)
│   │   ├── registry.py            #     ModelRegistry — HF / custom / local .pt
│   │   └── security/auth.py       #     JWT + bcrypt authentication
│   │
│   └── client/
│       └── cli.py                 #   FederatedClient CLI — trains locally, pushes updates
│
├── dashboard/                     # Next.js 14 app
│   ├── Dockerfile
│   ├── next.config.js
│   ├── app/
│   │   ├── dashboard/             #   Admin UI — groups, join requests, logs, events
│   │   └── client/                #   Client UI — browse groups, training, trust, notifs
│   └── components/
│       └── AuthContext.tsx
│
├── tests/                         # 43 pytest tests
│   ├── conftest.py                #   Test fixtures (TestClient, sample_config)
│   ├── test_smoke.py              #   API smoke tests (health, status, groups, clients…)
│   ├── test_aggregator.py
│   ├── test_compression.py
│   ├── test_privacy.py
│   ├── test_reproducibility.py
│   └── test_trust_manager.py
│
├── .github/workflows/ci.yml       # CI — lint + typecheck + test
├── .pre-commit-config.yaml        # pre‑commit hooks (ruff + mypy)
├── config.yaml                    # Default training configuration
├── pyproject.toml                 # Package metadata + tool config (ruff, mypy, pytest)
├── Makefile                       # make test / lint / run-server / run-client …
├── Dockerfile                     # Root — API server
├── Dockerfile.server              # Production API server
├── Dockerfile.client              # FL client container
├── docker-compose.yml             # Multi‑service deployment
├── requirements.txt               # Python dependencies
├── conftest.py                    # Adds src/ to sys.path for pytest
├── LICENSE                        # MIT
├── CHANGELOG.md
└── AGENTS.md                      # AI assistant memory (architecture, gotchas, workflows)
```

---

## Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| Docker | (optional) |

### 1. Backend

```bash
git clone https://github.com/vansh-visariya/ASTRA.git
cd ASTRA

python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt

# Start the server
uvicorn astra.app.server_api:app --reload --port 8000
```

The API is now live at **http://localhost:8000**. Interactive Swagger docs at `/docs`.

### 2. Dashboard

```bash
cd dashboard
npm install
npm run dev
```

Dashboard at **http://localhost:3000**. Sign up with any role (`admin` or `client`).

### 3. Docker (all-in-one)

```bash
docker compose up -d
# → API on :8000, Dashboard on :3000, Redis on :6379
```

---

## Running a Client

```bash
# Start a client that connects to the server
python -m astra.client.cli \
  --server http://localhost:8000 \
  --client-id client_1

# With a specific group
python -m astra.client.cli \
  --server http://localhost:8000 \
  --client-id client_1 \
  --group-id group_a
```

Or use the installed CLI (after `pip install -e .`):

```bash
astra-client --server http://localhost:8000 --client-id client_1
```

The client auto‑registers, downloads the global model, trains locally, and pushes gradient updates. It reconnects automatically on connection loss.

---

## API Reference

Full Swagger UI at **http://localhost:8000/docs**.

### Auth
```
POST /api/auth/signup          Register (role: admin | client | observer)
POST /api/auth/login           Login → JWT token
```

### Groups
```
GET    /api/groups              List all groups
POST   /api/groups              Create group (admin)
GET    /api/groups/{id}         Group detail
POST   /api/groups/{id}/start   Start training (admin)
POST   /api/groups/{id}/pause   Pause training
POST   /api/groups/{id}/resume  Resume training
POST   /api/groups/{id}/stop    Stop training
```

### Join Requests
```
POST  /api/join/join-request               Request to join group (client)
GET   /api/join/join-requests              List pending (admin)
POST  /api/join/join-requests/approve      Approve request (admin)
POST  /api/join/join-requests/reject       Reject request (admin)
POST  /api/join/activate/{group_id}        Activate membership (client)
```

### Clients & Models
```
GET   /api/clients                         List connected clients
POST  /api/clients/register                Register a client
GET   /api/models                          List registered models
POST  /api/models/register/hf             Register a HuggingFace model
GET   /api/models/{group_id}/download      Download trained model weights
```

### System
```
GET   /health                              Health check
GET   /api/server/status                   Server runtime status
GET   /api/system/metrics                  System‑wide metrics
GET   /api/logs                            Event logs
WS    /ws?token=<jwt>                      Live dashboard updates
```

---

## Configuration

### Environment Variables

```bash
SECRET_KEY=your-secure-key           # JWT signing secret (required in production)
ENV=dev                              # dev | prod
GEMINI_API_KEY=your-key              # Optional — Gemini model recommendations
```

### `config.yaml` — Key Toggles

```yaml
server:
  aggregator_window: 10              # Aggregate after N updates…
  poll_timeout: 1.0                  # …or after T seconds (whichever first)
  async_lambda: 0.2                  # Staleness decay (higher = harder penalty)
  adaptive_lr: true                  # Auto‑reduce LR on instability
  momentum: 0.9

robust:
  method: hybrid                     # fedavg | trimmed_mean | median | hybrid
  trim_ratio: 0.1                    # For trimmed mean
  trust_power: 1.0                   # Trust exponent in hybrid weighting

privacy:
  dp_enabled: true
  dp_mode: client                    # client | server
  clip_norm: 1.0                     # Gradient clipping threshold
  sigma: 1.2                         # Gaussian noise multiplier

communication:
  compression: topk                  # topk | none
  topk_ratio: 0.1                    # Fraction of gradients to transmit

malicious:
  ratio: 0.30                        # Fraction of clients simulated as malicious
  behaviors: [label_flip, noise, sign_flip, scale, backdoor]
```

Full reference in [`config.yaml`](./config.yaml).

---

## Testing

```bash
pip install -e ".[dev]"              # install with dev deps (pytest, ruff, mypy)
pytest tests/ -v                     # 43 tests — aggregation, compression, DP, trust, API smoke
```

```
tests/test_aggregator.py ........   9 passed
tests/test_compression.py .......   7 passed
tests/test_privacy.py .........     9 passed
tests/test_reproducibility.py ....  4 passed
tests/test_trust_manager.py ......  7 passed
tests/test_smoke.py .......         7 passed
```

Run with coverage:
```bash
pytest tests/ -v --cov=astra --cov-report=term-missing
```

---

## Makefile Commands

```bash
make install        # pip install -e ".[dev]"
make test           # pytest tests/ -v
make test-cov       # pytest with coverage
make lint           # ruff check src/ tests/
make fmt            # ruff format src/ tests/
make typecheck      # mypy src/astra/
make run-server     # uvicorn astra.app.server_api:app --reload
make run-client     # run FL client
make clean          # remove all cache dirs
make docker-build   # build all images
make docker-up      # start all services
```

---

## Deployment

| Service | Port | Description |
|---------|------|-------------|
| API Server | 8000 | FastAPI — REST + WebSocket + FL engine |
| Dashboard | 3000 | Next.js 14 — Admin + Client UI |
| Redis | 6379 | (optional, `--profile production` for caching) |

### Production Checklist

- [ ] Set `ENV=prod` and a strong random `SECRET_KEY` (32+ chars)
- [ ] Configure CORS origins in [`server_api.py`](src/astra/app/server_api.py)
- [ ] Put a reverse proxy in front (nginx / Caddy / Traefik) with HTTPS
- [ ] Migrate from SQLite to PostgreSQL for concurrent‑write workloads (Alembic ready)
- [ ] Run `docker compose --profile production up -d` for Redis cache layer

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10+ · FastAPI · Uvicorn |
| **Frontend** | Next.js 14 · React 18 · TypeScript · Tailwind CSS · Recharts |
| **ML** | PyTorch · HuggingFace Transformers · PEFT (LoRA) |
| **Database** | SQLite with WAL mode (`astra.db`) |
| **Auth** | PyJWT · bcrypt (4.1+) |
| **Real‑time** | WebSocket · Socket.IO (`fastapi-socketio`) |
| **Privacy** | DP‑SGD · Gaussian noise · Moments accountant |
| **Deployment** | Docker · Docker Compose |
| **CI** | GitHub Actions (lint + typecheck + test) |
| **Linting** | Ruff · mypy · pre‑commit |

---

## Contributing

1. Fork the repo and create a branch.
2. Install dev dependencies: `pip install -e ".[dev]"` and set up `pre-commit install`.
3. Make changes, write tests, run `make lint test typecheck`.
4. Open a PR against `main`.

See [`AGENTS.md`](AGENTS.md) for architecture notes, data flow diagrams, and known gotchas.

---

## License

MIT — see [LICENSE](LICENSE).
