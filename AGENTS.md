# Memory

## Project Overview
ASTRA (Async Scalable Training & Research Architecture) is a production-ready distributed Federated Learning platform. Clients train locally without sharing data; the server aggregates updates using hybrid async windowing (N updates OR T seconds). Stack: FastAPI + PyTorch backend (`src/astra/`), Next.js 14 frontend (`dashboard/`).

Key FL features: async aggregation with staleness decay, Byzantine-robust aggregation (FedAvg, Trimmed Mean, Median, Hybrid), trust scoring via cosine similarity, DP-SGD (client-side or server-side), top-k sparsification + quantization, IID/Dirichlet/pathological data splits, HuggingFace model support with optional LoRA/PEFT.

## Package Layout
All Python source lives under `src/astra/` and is importable as `astra.*`. `src/` must be on `sys.path` (handled by `conftest.py` and `pyproject.toml pythonpath = ["src"]`).

```
src/astra/
├── app/          # API layer, orchestration, DB, group lifecycle
│   └── routes/   # FastAPI routers (system, groups, clients, models, experiments)
├── core/         # FL algorithms — pure Python, no web dependencies
│   ├── aggregation/   # aggregator.py, robust.py, heterogeneous.py
│   ├── models/        # model_zoo.py (CNN/MLP), hf_models.py (HuggingFace+PEFT)
│   ├── privacy/       # privacy.py (DP-SGD), malicious_simulator.py
│   └── utils/         # metrics.py, seed.py, logging_utils.py
├── infra/        # Transport, DB schemas, auth, WebSocket, model registry
│   └── security/ # auth.py — JWT + bcrypt + role-based access
└── client/       # Standalone FL client CLI (client.py)
```

## Component Map
- `src/astra/app/server_api.py` — FastAPI entry point; assembles app, mounts routers, WebSocket, Socket.IO; lifespan manages FLServer singleton
- `src/astra/app/fl_server.py` — FLServer: orchestrates AsyncServer + GroupManager + ConnectionManager; handles client register/update
- `src/astra/app/group_manager.py` — GroupManager: manages concurrent TrainingGroups; window-based aggregation, DB persistence
- `src/astra/app/database.py` — AstraDB: single SQLite `astra.db`; WAL mode, thread-local connections
- `src/astra/app/integration.py` — FLPlatformIntegration: wires auth + notifications + trust + recommender + inference
- `src/astra/app/extended_endpoints.py` — Auth/join/notification/trust REST endpoints
- `src/astra/app/notifications.py` — NotificationService: in-app notifications, WebSocket delivery
- `src/astra/core/server.py` — AsyncServer: async FL engine; staleness-weighted aggregation, trust scoring, momentum
- `src/astra/core/aggregation/` — FedAvg, TrimmedMean, CoordinateMedian, Hybrid aggregation
- `src/astra/core/trust_manager.py` — Cosine-similarity trust scoring, exponential decay, quarantine at 0.35
- `src/astra/core/privacy/` — DP-SGD (clip_and_noise, MomentsAccountant), malicious simulator
- `src/astra/infra/connection_manager.py` — WebSocket registry; broadcast + per-client send
- `src/astra/infra/registry.py` — ModelRegistry: register HF / custom / local .pt models
- `src/astra/infra/security/auth.py` — JWT + bcrypt auth, role-based access (admin/client/observer)

## Data / State Flow
```
FederatedClient → POST /api/clients/register → trains locally → POST /api/clients/update (base64 float32 delta)
  → FLServer.handle_client_update() → AsyncServer.handle_update()
    → TrustManager.update_trust() [cosine sim vs. running global estimate]
    → buffer.append(delta, staleness_weight, trust, dataset_size)
    → _maybe_aggregate() if len(buffer) >= window_size:
        → Aggregator.aggregate() [FedAvg or Robust/Hybrid]
        → momentum smoothing → _apply_update() to model params
  → ConnectionManager.broadcast() → Dashboard WebSocket
```

## Key Decisions
- **Hybrid async windowing**: Aggregate when N updates received OR T seconds elapsed — prevents straggler clients from blocking
- **Staleness-weighted aggregation**: Weight each update by `exp(-lambda * staleness)`, default lambda=0.2
- **Trust via cosine similarity**: Client update vs. running global estimate; quarantine below 0.35
- **Single SQLite (astra.db)**: WAL mode, thread-local connections, fine for research/dev scale
- **All imports use `astra.*`**: Zero shim packages, `src/` on sys.path, 37-rule migration completed
- **Extended endpoints guarded**: `_extended_api_registered` flag prevents double-registration on FastAPI dev-reload

## Active Gotchas
- `sys.path` must point to `src/` (not `src/astra/`) — `import astra` needs `src/` as root
- Import migration rule ORDER matters: more-specific strings first before generic prefixes
- Stale imports can hide inside method bodies, not just module level
- `_extended_api_registered` global flag exists to prevent double-registration on dev-reload
- `aggregator_buffer` is `deque(maxlen=window_size)` — oldest entry silently dropped when full (by design)

## Common Workflows
- **Start server**: `uvicorn astra.app.server_api:app --reload` from repo root
- **Start dashboard**: `cd dashboard && npm run dev` → http://localhost:3000
- **Run tests**: `pytest tests/ -v` (36 tests expected)
- **Run client**: `python src/astra/client/client.py --server http://localhost:8000 --client-id client_1`
- **Config**: `config.yaml` for training params; env vars: `SECRET_KEY`, `ENV`, `GEMINI_API_KEY`

## Metrics / Benchmarks
- Tests: 36/36 passing (3.46s)
- Config defaults: 20 clients, window_size=10, MNIST/Dirichlet(0.3), hybrid robust aggregation, DP client-side (sigma=1.2, clip_norm=1.0), top-k compression (ratio=0.1)
