<p align="center">
  <h1 align="center">ASTRA</h1>
  <p align="center"><strong>A</strong>sync <strong>S</strong>calable <strong>T</strong>raining & <strong>R</strong>esearch <strong>A</strong>rchitecture</p>
  <p align="center">Distributed Federated Learning — train models across devices without sharing data.</p>
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/quick_start-8000?style=flat&logo=fastapi&labelColor=white&color=009688" alt="Quick Start"></a>
  <a href="/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/next.js-14-black?logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/pytorch-latest-EE4C2C?logo=pytorch" alt="PyTorch">
</p>

---

## Overview

**ASTRA** is a production-ready federated learning (FL) platform. Clients train machine learning models on their own hardware — the server never sees raw data — and submit pre-computed weight deltas via REST or the bundled Next.js dashboard. The server aggregates those deltas using hybrid async windowing, scores trust per-client, optionally applies server-side DP, and broadcasts the new global model back to every dashboard.

**What makes it different:** ASTRA uses **hybrid async windowing** — aggregation triggers when _either_ N client updates arrive _or_ a time window expires. No more waiting on stragglers. Byzantine-robust aggregation (FedAvg / Trimmed Mean / Coordinate Median / Hybrid), trust-based client scoring, and server-side DP-SGD come built in.

> **Scope:** ASTRA is a **server + aggregation + delivery** platform. The client training loop itself is **out of scope** — clients bring their own training code (PyTorch, JAX, anything) and submit the resulting weight delta. The dashboard's "Upload Delta" page is the easiest way to do this.

---

## How It Works

```
Admin creates group → Clients request to join → Admin approves
  → Client activates membership (REST or dashboard)
  → Client trains externally on its own data
  → Client computes delta = new_weights - old_weights (float32 little-endian)
  → Client POSTs base64-encoded delta to /api/clients/{id}/delta
  → Server validates, applies DP if configured, scores trust via cosine similarity
  → Server aggregates when N updates received OR T seconds elapsed (whichever first)
  → Server applies aggregated delta to global model, broadcasts model_update
  → Dashboard shows live accuracy, loss, trust scores, event logs via WebSocket
```

### Step-by-step

1. **Admin** registers a model in the registry (any PyTorch module, HuggingFace model, or dynamic import) and creates a training group.
2. **Clients** sign up, browse available groups, and request to join.
3. **Admin** approves pending join requests.
4. **Clients** activate their membership via `POST /api/join/activate/{group_id}` or the dashboard.
5. **Clients** train **on their own hardware** (out of scope for ASTRA), then POST the resulting delta to the server.
6. **Updates** flow asynchronously — the server aggregates every `window_size` updates or `time_limit` seconds (whichever comes first), using the configured aggregator.
7. **Dashboard** shows live accuracy, loss, trust scores, and event logs via WebSocket + REST.

---

## Features

| Category | Capability |
|---|---|
| **Aggregation** | FedAvg · Trimmed Mean · Coordinate Median · Hybrid (trust-weighted + staleness decay) |
| **Robustness** | Byzantine-tolerant trust scoring · Cosine-similarity anomaly detection · Soft quarantine |
| **Privacy** | DP-SGD (server-side) · Gaussian noise · Gradient clipping |
| **Compression** | Top-k sparsification · Quantization |
| **Models** | Registry-driven · Any PyTorch module via dynamic import · Any HuggingFace model · Optional LoRA / PEFT |
| **Client training** | Out of scope — clients train externally and submit pre-computed deltas |
| **Auth** | JWT + bcrypt · Roles: `admin` \| `client` \| `observer` |
| **Dashboard** | Next.js 14 · Admin panel (groups, join requests, metrics, logs) · Client panel (upload delta, trust, notifications) |
| **Real-time** | WebSocket + Socket.IO · Live metrics streaming |
| **Recommendations** | Gemini API-powered model suggestions (optional) |

---

## Project Structure

```
ASTRA/
├── src/astra/                     # Python package (importable as astra.*)
│   ├── __init__.py                #   Public exports: AsyncServer, TrustManager, load_config
│   ├── app/                       #   API layer — FastAPI, orchestration, DB, groups
│   │   ├── server_api.py          #     Entry point — REST + WebSocket + Socket.IO
│   │   ├── fl_server.py           #     FLServer — orchestrates AsyncServer + GroupManager
│   │   ├── group_manager.py       #     Manages concurrent TrainingGroups + hybrid windowing
│   │   ├── training_group.py      #     TrainingGroup + AsyncWindowConfig dataclasses
│   │   ├── database.py            #     AstraDB — single SQLite file (astra.db), WAL mode
│   │   ├── state.py               #     FLServer singleton accessor
│   │   ├── integration.py         #     Wires auth + notifications + trust + recommender
│   │   ├── extended_endpoints.py  #     Auth / join / notification / trust REST routes
│   │   ├── notifications.py       #     In-app notification service
│   │   ├── model_recommender.py   #     Gemini-powered model suggestions
│   │   └── routes/                #     system · groups · clients · models · experiments
│   │
│   ├── core/                      #   Server-side FL algorithms — pure Python, no web deps
│   │   ├── server.py              #     AsyncServer — staleness-weighted aggregation engine
│   │   ├── trust_manager.py       #     Trust scoring via cosine similarity + quarantine
│   │   ├── config.py              #     Centralized YAML + env config loader
│   │   ├── compression.py         #     Top-k sparsification + quantization
│   │   ├── inference.py           #     Server-side and client-side inference
│   │   ├── exceptions.py          #     Custom FL exception hierarchy
│   │   ├── aggregation/           #     aggregator · robust · heterogeneous
│   │   ├── models/                #     model_zoo · hf_models (HF + PEFT)
│   │   ├── privacy/               #     DP-SGD (server-side clip_and_noise)
│   │   └── utils/                 #     metrics · seed · logging
│   │
│   └── infra/                     #   Transport, schemas, auth, model registry
│       ├── connection_manager.py  #     WebSocket registry — broadcast + per-client send
│       ├── websocket_handler.py   #     WS endpoint (event channel only) + Socket.IO handlers
│       ├── models.py              #     Pydantic schemas (ClientUpdate, ExperimentConfig…)
│       ├── registry.py            #     ModelRegistry — HF / custom / local .pt
│       └── security/auth.py       #     JWT + bcrypt authentication
│
├── dashboard/                     # Next.js 14 app
│   ├── Dockerfile
│   ├── next.config.js
│   ├── app/
│   │   ├── dashboard/             #   Admin UI — groups, join requests, logs, events
│   │   └── client/                #   Client UI — groups, upload delta, trust, notifs
│   └── components/
│       └── AuthContext.tsx
│
├── tests/                         # pytest tests (~220 passing)
│   ├── conftest.py                #   Test fixtures (TestClient, sample_config)
│   ├── test_smoke.py              #   API smoke tests (health, status, groups, clients)
│   ├── test_aggregator.py
│   ├── test_compression.py
│   ├── test_privacy.py
│   ├── test_reproducibility.py
│   ├── test_trust_manager.py
│   ├── test_upload_endpoint.py    #   POST /api/clients/{id}/delta validation
│   ├── test_external_client_flow.py # End-to-end upload cycle
│   └── test_websocket_cleanup.py  #   WebSocket rejects client-training messages
│
├── .github/workflows/ci.yml       # CI — lint + typecheck + test
├── .pre-commit-config.yaml        # pre-commit hooks (ruff + mypy)
├── config.yaml                    # Default server configuration
├── pyproject.toml                 # Package metadata + tool config (ruff, mypy, pytest)
├── Makefile                       # make test / lint / run-server …
├── Dockerfile                     # Root — API server
├── docker-compose.yml             # Multi-service deployment
├── requirements.txt               # Python dependencies
├── conftest.py                    # Adds src/ to sys.path for pytest
├── LICENSE                        # MIT
├── CHANGELOG.md
├── AGENTS.md                      # AI assistant memory (architecture, gotchas, workflows)
└── .claude/                       # Claude memory (architecture.md, progress.md, decisions.md)
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

pip install -e ".[dev]"

# Start the server (port 8000 by default)
uvicorn astra.app.server_api:app --reload --port 8000
```

The API is live at **http://localhost:8000**. Interactive Swagger docs at `/docs`.

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
# → API on :8000, Dashboard on :3000, optional Redis on :6379
```

---

## Submitting a Delta

Clients train on their own hardware and submit pre-computed weight deltas to the server. Three transports are supported depending on file size.

### 1. Inline REST upload (≤ max_inline_bytes, default 100 MB)

Best for small models or when the delta fits comfortably in memory.

```bash
# Download the current global model as raw float32 weight bytes
curl -OJ -H "Authorization: Bearer $TOKEN" \
  'http://localhost:8000/api/models/$GROUP_ID/download?format=raw'

# Train locally, compute the delta
# delta = (new_weights - old_weights).astype('<f4').tobytes()

# Upload the delta
python -c "
import base64, requests, numpy as np
delta = np.load('my_delta.npy').astype('<f4').tobytes()
r = requests.post(
    f'http://localhost:8000/api/clients/$CLIENT_ID/delta',
    headers={'Authorization': f'Bearer $TOKEN'},
    json={
        'client_id': '$CLIENT_ID',
        'client_version': 0,
        'local_updates': base64.b64encode(delta).decode(),
        'update_type': 'delta',
        'local_dataset_size': 1000,
        'meta': {'train_accuracy': 0.7, 'train_loss': 0.4},
    },
)
print(r.json())
"
```

### 2. Presigned-URL chunked upload (> max_inline_bytes)

For large models (3B+ params, ~12 GB deltas), use the staged upload flow. The server allocates a slot on disk, returns a presigned PUT URL, and the client streams bytes directly to disk. The server verifies sha256 + dispatches into the FLServer only on `complete`.

```bash
# 1. Download the current global model
curl -OJ -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/models/$GROUP_ID/download

# 2. Train locally, compute the delta
python train_and_compute_delta.py   # produces delta.bin + sha256

# 3. Initiate the upload
python -c "
import hashlib, requests
delta = open('delta.bin','rb').read()
sha = hashlib.sha256(delta).hexdigest()
r = requests.post(
    'http://localhost:8000/api/uploads/init',
    headers={'Authorization': 'Bearer $TOKEN'},
    json={
        'client_id': '$CLIENT_ID',
        'group_id': '$GROUP_ID',
        'content_length': len(delta),
        'sha256': sha,
    },
)
info = r.json()
print('upload_id', info['upload_id'])
print('upload_url', info['upload_url'])
print('chunk_size', info['chunk_size'])
"

# 4. PUT the bytes (single PUT or chunked with Content-Range)
curl -X PUT --data-binary @delta.bin \
  -H 'Content-Type: application/octet-stream' \
  'http://localhost:8000/api/uploads/$UPLOAD_ID/blob?expires=$E&sig=$SIG'

# 5. Complete — server verifies sha256, applies size/NaN checks, dispatches
python -c "
import requests
r = requests.post(
    'http://localhost:8000/api/uploads/$UPLOAD_ID/complete',
    headers={'Authorization': 'Bearer $TOKEN'},
    json={'sha256': '$SHA'},
)
print(r.json())   # {status: 'completed', global_version: N, size: ..., sha256: ...}
"
```

### 3. Dashboard

1. Sign in as a client.
2. Navigate to **Upload Delta** in the client sidebar.
3. Pick your group, select a `.pt` / `.npy` / `.bin` file.
4. Files ≤ 100 MB are sent inline. Larger files automatically use the chunked upload flow with a live progress bar and cancel button.
5. (Optional) enter training accuracy / loss for nicer dashboard metrics.

The dashboard also has **Download Weights (.bin, raw float32)** and **Download Full Model (.pt checkpoint)** buttons. Both use the chunked download flow — files stream in 8 MB chunks with live progress, can be cancelled mid-transfer, and the browser verifies the sha256 of the assembled file before saving it. No need to extract weights from the `.pt` checkpoint yourself, and works reliably for 3B+ models over slow connections.

### Server behavior

Regardless of transport, the server:
1. Decodes the delta as float32 little-endian (`<f4`).
2. Verifies byte count matches `total_params × 4` (or `× 8` for float64).
3. Rejects NaN / Inf with HTTP 400.
4. Enforces a per-client rate limit (1 upload / 2 s).
5. Applies server-side DP if configured (`dp_mode: server`).
6. Scores trust via cosine similarity against the running global estimate.
7. Appends to the AsyncServer aggregator buffer (window sized to the group's `window_size`).
8. Triggers aggregation when the window fills or the time limit elapses.
9. Broadcasts `client_update` + `aggregation_complete` over WebSocket.
10. Returns `{status: 'accepted', global_version: N}`.

**Limits:**
- Default `max_inline_bytes = 100 MB`. Larger files must use the chunked flow.
- Default `chunk_size = 8 MB`. The server tells the client the chunk size in the init response.
- Default `presign_ttl = 3600 s` for the presigned PUT URL.
- Per-client rate limit: 1 upload / 2 s (any transport).
- Disk-space check: init refuses uploads that wouldn't fit on the uploads partition.

**Configurable in `config.yaml` or env vars:**
```yaml
uploads:
  max_inline_bytes: 104857600   # 100 MB
  chunk_size: 8388608           # 8 MB
  disk_path: ./uploads
  presign_ttl_seconds: 3600
  min_free_bytes: 104857600     # refuse if < 100 MB free
```

---

## API Reference

Full Swagger UI at **http://localhost:8000/docs**.

### Auth
```
POST /api/auth/signup                  Register (role: admin | client | observer)
POST /api/auth/login                   Login → JWT token
GET  /api/auth/me                      Current user info
```

### Groups
```
GET    /api/groups                      List all groups (authenticated)
POST   /api/groups                      Create group (admin)
GET    /api/groups/{id}                 Group detail
POST   /api/groups/{id}/start           Start accepting deltas (admin)
POST   /api/groups/{id}/pause           Pause
POST   /api/groups/{id}/resume          Resume
POST   /api/groups/{id}/stop            Stop
DELETE /api/groups/{id}                 Delete group
```

### Join Requests
```
POST  /api/join/join-request               Request to join group (client)
GET   /api/join/join-requests              List pending (admin)
POST  /api/join/join-requests/approve      Approve request (admin)
POST  /api/join/join-requests/reject       Reject request (admin)
POST  /api/join/activate/{group_id}        Activate membership (client)
GET   /api/join/my-requests/{group_id}     My join status
```

### Clients
```
GET   /api/clients                          List known FL clients across groups
POST  /api/clients/register                 Register a client (admin / REST)
POST  /api/clients/{client_id}/delta         Submit a pre-computed model delta (JWT, ≤ max_inline_bytes)
GET   /api/clients/{client_id}/status        Per-client server view (JWT)
GET   /api/clients/connected                 List currently connected WebSocket clients
```

### Uploads (presigned-URL flow for files > 100 MB)
```
POST   /api/uploads/init                     Allocate an upload slot (returns presigned PUT URL + chunk_size)
PUT    /api/uploads/{upload_id}/blob         PUT delta bytes to the presigned URL (chunked, resumable)
POST   /api/uploads/{upload_id}/complete     Verify sha256 + dispatch into the FLServer
DELETE /api/uploads/{upload_id}              Abort the upload and free disk
GET    /api/uploads/{upload_id}              Inspect upload state
```

### Downloads (chunked flow for large model files)
```
POST   /api/downloads/init                   Allocate a download slot (returns manifest + signed chunk URLs)
GET    /api/downloads/{id}/chunk/{N}         Stream chunk N (HMAC-signed, Content-Range aware)
POST   /api/downloads/{id}/complete          Mark download finished (telemetry only)
DELETE /api/downloads/{id}                   Abort the download and free the slot
GET    /api/downloads/{id}                   Inspect download slot state
```

The single-shot `/api/models/{group_id}/download[?format=raw]` still works for small files (< a few MB). For 3B+ models, use the chunked flow — it streams bytes with progress, supports cancel, and verifies sha256 client-side before saving the file.

### Models
```
GET   /api/models                           List registered models
POST  /api/models/register/hf               Register a HuggingFace model
POST  /api/models/register/architecture     Register via Python path (e.g. SimpleMLP)
GET   /api/models/{id}                       Model info
GET   /api/models/{group_id}/download        Download global model weights (full)
                                           ?format=raw returns flattened float32 .bin (upload-ready)
GET   /api/models/{group_id}/adapter        Download LoRA adapter only (PEFT)
GET   /api/models/{group_id}/base           Download base model (PEFT)
```

### System
```
GET   /health                               Health check
GET   /api/server/status                    Server runtime status
GET   /api/system/metrics                   System-wide metrics
GET   /api/logs                             Event logs (filterable by group, type)
GET   /api/notifications                    User notifications
WS    /ws?token=<jwt>                       Live dashboard updates
```

---

## Configuration

### Environment Variables

```bash
SECRET_KEY=your-secure-key            # JWT signing secret (required in production)
ENV=dev                               # dev | prod
GEMINI_API_KEY=your-key               # Optional — Gemini model recommendations
ASTRA_DEFAULT_ADMIN_PASSWORD=admin    # Initial admin password on first run
ASTRA_SEED=42                         # Global random seed
```

### `config.yaml` — Server-side Toggles

```yaml
server:
  aggregator_window: 10              # Aggregate after N updates…
  poll_timeout: 1.0                  # …or after T seconds (whichever first)
  async_lambda: 0.2                  # Staleness decay (higher = harder penalty)
  adaptive_lr: true                  # Auto-reduce LR on instability
  momentum: 0.9

robust:
  method: hybrid                     # fedavg | trimmed_mean | median | hybrid
  trim_ratio: 0.1                    # For trimmed mean
  trust_power: 1.0                   # Trust exponent in hybrid weighting

privacy:
  dp_enabled: false
  dp_mode: server                    # server (client-side DP removed with FLClient)
  clip_norm: 1.0                     # Gradient clipping threshold
  sigma: 1.2                         # Gaussian noise multiplier

communication:
  compression: topk                  # topk | none
  topk_ratio: 0.1                    # Fraction of gradients to transmit

trust:
  init: 1.0
  update_alpha: 0.3
  quarantine_threshold: 0.35
  soft_decay: 0.8
```

Full reference in [`config.yaml`](./config.yaml).

---

## Testing

```bash
pip install -e ".[dev]"              # install with dev deps (pytest, ruff, mypy)
pytest tests/ -v                     # ~220 tests across aggregation, compression, DP,
                                     # trust, upload, schema, auth, smoke
```

Sample test layout:

```
tests/test_aggregator.py .............   9 passed
tests/test_compression.py ...........   7 passed
tests/test_privacy.py ...............   9 passed
tests/test_reproducibility.py ......   4 passed
tests/test_trust_manager.py .........   7 passed
tests/test_smoke.py .................   7 passed
tests/test_upload_endpoint.py .......   8 passed
tests/test_external_client_flow.py ..   3 passed
tests/test_websocket_cleanup.py .....   9 passed
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
- [ ] Migrate from SQLite to PostgreSQL for concurrent-write workloads (Alembic ready)
- [ ] Run `docker compose --profile production up -d` for Redis cache layer
- [ ] Set `GEMINI_API_KEY` if you want Gemini-powered model recommendations

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10+ · FastAPI · Uvicorn |
| **Frontend** | Next.js 14 · React 18 · TypeScript · Tailwind CSS · Recharts |
| **ML** | PyTorch · HuggingFace Transformers · PEFT (LoRA) |
| **Database** | SQLite with WAL mode (`astra.db`) |
| **Auth** | PyJWT · bcrypt (4.1+) |
| **Real-time** | WebSocket · Socket.IO (`fastapi-socketio`) |
| **Privacy** | Server-side DP-SGD · Gaussian noise · Gradient clipping |
| **Deployment** | Docker · Docker Compose |
| **CI** | GitHub Actions (lint + typecheck + test) |
| **Linting** | Ruff · mypy · pre-commit |

---

## Contributing

1. Fork the repo and create a branch.
2. Install dev dependencies: `pip install -e ".[dev]"` and set up `pre-commit install`.
3. Make changes, write tests, run `make lint test typecheck`.
4. Open a PR against `main`.

See [`AGENTS.md`](AGENTS.md) and [`.claude/`](.claude/) for architecture notes, data flow diagrams, and known gotchas.

---

## License

MIT — see [LICENSE](LICENSE).