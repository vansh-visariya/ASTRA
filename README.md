<h1 align="center">ASTRA</h1>
<p align="center">
  <strong>Async Scalable Training & Research Architecture</strong><br>
  Distributed Federated Learning — clients train on their own hardware, submit weight deltas to a central server. The server never sees raw data.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/next.js-14-black?logo=next.js" alt="Next.js">
  <a href="/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

## How It Works

```
Admin creates training group → clients join → admin approves
  → clients train locally → compute delta = new_weights - old_weights
  → POST delta to server
  → server validates → optional DP noise → trust scoring → aggregation
  → global model updated → dashboard reflects live metrics
```

**Key design:** Hybrid async windowing — aggregation triggers after N updates **or** T seconds. No straggler bottleneck.

---

## Features

| Area | What |
|---|---|
| **Aggregation** | FedAvg, Trimmed Mean, Coordinate Median, Hybrid (staleness-weighted + trust scoring) |
| **Trust** | Cosine similarity between client updates and global estimate; soft quarantine below threshold |
| **Privacy** | Server-side DP-SGD — Gaussian noise injection + gradient clipping |
| **Models** | Any PyTorch `nn.Module`, HuggingFace models, optional LoRA/PEFT adapters, safetensors |
| **Auth** | JWT + bcrypt, roles: `admin` / `client` / `observer` |
| **Dashboard** | Next.js 14 — admin panel + client upload UI, live WebSocket metrics |
| **Uploads** | Inline (≤100 MB) or presigned-URL chunked flow for large weights |

---

## Quick Start

**Backend**

```bash
git clone https://github.com/vansh-visariya/ASTRA.git && cd ASTRA
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
uvicorn astra.app.server_api:app --reload           # → http://localhost:8000/docs
```

**Dashboard**

```bash
cd dashboard && npm install && npm run dev          # → http://localhost:3000
```

**Docker**

```bash
docker compose up -d                                # API :8000, Dashboard :3000
```

---

## Project Structure

```
src/astra/
├── app/                  # FastAPI layer — routes, DB, groups, notifications
│   ├── routes/           #   REST endpoints (clients, groups, models, uploads, downloads)
│   ├── fl_server.py      #   Orchestrates AsyncServer + GroupManager
│   ├── group_manager.py  #   Concurrent TrainingGroups + hybrid async windowing
│   └── database.py       #   SQLite (astra.db), WAL mode
├── core/                 # FL algorithms — pure Python, no web dependencies
│   ├── server.py         #   AsyncServer — staleness-weighted aggregation engine
│   ├── aggregation/      #   FedAvg, trimmed mean, median, hybrid
│   ├── models/           #   Delta flatten/apply, HuggingFace + PEFT loading
│   ├── trust_manager.py  #   Trust scoring via cosine similarity + quarantine
│   └── privacy/          #   Server-side DP (clip + noise)
└── infra/                # Transport, auth, model registry
    ├── registry.py       #   ModelRegistry — HuggingFace / custom / architecture import
    ├── connection_manager.py  # WebSocket broadcast + per-client send
    └── security/auth.py  #   JWT + bcrypt + role-based access

dashboard/                # Next.js 14 — admin + client panels
tests/                    # ~237 tests
```

---

## Configuration

Environment variables:

```bash
SECRET_KEY=<random-string>    # JWT signing (required in production)
GEMINI_API_KEY=<key>          # Optional — Gemini model recommendations
ASTRA_SEED=42                 # Global random seed
```

`config.yaml` (defaults in repo):

```yaml
server:
  aggregator_window: 10       # Aggregate after N updates or T seconds
  async_lambda: 0.2           # Staleness decay factor
  adaptive_lr: true           # Auto-reduce LR on loss instability

robust:
  method: hybrid              # fedavg | trimmed_mean | median | hybrid

privacy:
  dp_enabled: false           # Server-side DP noise
  sigma: 1.2
  clip_norm: 1.0

trust:
  quarantine_threshold: 0.35  # Clients below this get quarantined
```

---

## Development

```bash
make install                  # pip install -e ".[dev]"
make test                     # pytest tests/ -v
make lint                     # ruff check
make fmt                      # ruff format
make typecheck                # mypy src/astra/
```

Interactive API docs at **http://localhost:8000/docs**.

---

## License

MIT — see [LICENSE](LICENSE).
