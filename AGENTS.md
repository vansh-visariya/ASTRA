# Memory

## Project Overview
ASTRA (Async Scalable Training & Research Architecture) is a production-ready distributed Federated Learning platform. Clients train locally without sharing data; the server aggregates updates using hybrid async windowing (N updates OR T seconds). Stack: FastAPI + PyTorch backend (`src/astra/`), Next.js 14 frontend (`dashboard/`).

Key FL features: async aggregation with staleness decay, Byzantine-robust aggregation (FedAvg, Trimmed Mean, Median, Hybrid), trust scoring via cosine similarity, DP-SGD (client-side or server-side), top-k sparsification + quantization, IID/Dirichlet/pathological data splits, HuggingFace model support with optional LoRA/PEFT, registry-driven model loading with dynamic import.

## Package Layout
All Python source lives under `src/astra/` and is importable as `astra.*`. `src/` must be on `sys.path` (handled by `conftest.py` and `pyproject.toml pythonpath = ["src"]`).

```
src/astra/
├── app/          # API layer, orchestration, DB, group lifecycle
│   └── routes/   # FastAPI routers (system, groups, clients, models, experiments)
├── core/         # FL algorithms — pure Python, no web dependencies
│   ├── aggregation/   # aggregator.py, robust.py — FedAvg, TrimmedMean, Median, Hybrid
│   ├── models/        # model_zoo.py (flatten_all_params, apply_flat_delta, PEFT utils), hf_models.py (HuggingFace+PEFT)
│   ├── privacy/       # privacy.py (DP-SGD), malicious_simulator.py
│   └── utils/         # metrics.py, seed.py, logging_utils.py
├── infra/        # Transport, DB schemas, auth, WebSocket, model registry
│   └── security/ # auth.py — JWT + bcrypt + role-based access
└── client/       # Standalone FL client CLI (cli.py)
```

## Model System (Registry-Driven)
No hardcoded model classes. All models are registered via the ModelRegistry:

| Registration method | Dashboard tab | What it does |
|---|---|---|
| `register_factory(model_id, factory)` | Programmatic | Register any `lambda: nn.Module()` |
| `POST /api/models/register/hf` | HuggingFace tab | Register from HF Hub with optional PEFT |
| `POST /api/models/register/architecture` | External tab | Dynamic import from Python path (e.g. `torchvision.models.resnet18`) |

Models registered via External/Architecture tab are persisted to `model_registry` table in SQLite and auto-reloaded on server restart. The registry is the single source of truth — client and server both call `registry.build_model(model_id)` to get identical architectures.

## Component Map
- `src/astra/app/server_api.py` — FastAPI entry point; assembles app, mounts routers, WebSocket, Socket.IO; lifespan manages FLServer singleton
- `src/astra/app/fl_server.py` — FLServer: orchestrates AsyncServer + GroupManager + ConnectionManager; loads persisted models from DB on startup
- `src/astra/app/group_manager.py` — GroupManager: manages concurrent TrainingGroups; window-based aggregation, WebSocket-triggered + watchdog-triggered
- `src/astra/app/training_group.py` — TrainingGroup + AsyncWindowConfig dataclass; `add_update()`, `to_dict()` with full clients/accuracy/loss fields
- `src/astra/app/database.py` — AstraDB: single SQLite `astra.db`; WAL mode, thread-local connections; `model_registry` table for persisted models
- `src/astra/app/integration.py` — FLPlatformIntegration: wires auth + notifications + trust + recommender + inference
- `src/astra/app/extended_endpoints.py` — Auth/join/notification/trust REST endpoints + `/api/recommendations/unified`
- `src/astra/app/notifications.py` — NotificationService: in-app notifications, WebSocket delivery
- `src/astra/app/model_recommender.py` — Gemini-powered model suggestions
- `src/astra/core/server.py` — AsyncServer: async FL engine; staleness-weighted aggregation, trust scoring, momentum
- `src/astra/core/aggregation/` — FedAvg, TrimmedMean, CoordinateMedian, Hybrid aggregation
- `src/astra/core/models/model_zoo.py` — `flatten_all_params()`, `apply_flat_delta()`, PEFT utilities, `SimpleMLP`
- `src/astra/core/models/hf_models.py` — HuggingFace model loading with optional LoRA/PEFT
- `src/astra/core/trust_manager.py` — Cosine-similarity trust scoring, exponential decay, quarantine at 0.35
- `src/astra/core/privacy/` — DP-SGD (clip_and_noise, MomentsAccountant), malicious simulator
- `src/astra/core/data_splitter.py` — IID/Dirichlet/pathological data splits (MNIST, CIFAR-10 datasets)
- `src/astra/core/compression.py` — Top-k sparsification and quantization
- `src/astra/infra/connection_manager.py` — WebSocket registry; broadcast + per-client send
- `src/astra/infra/registry.py` — ModelRegistry: `register_factory()`, `build_model()`, HF registration, architecture registration
- `src/astra/infra/security/auth.py` — JWT + bcrypt auth, role-based access (admin/client/observer), join tokens

## Data / State Flow

### Client continuous training cycle
```
CLI startup → connect WebSocket → sync group config → build_model(model_id) → first training round
  → send update via WebSocket → enter _listen_loop():
    → on model_update: _download_model() → REST GET /api/models/{group_id}/download (full) or /adapter (PEFT)
    → on train_command: _run_training() → train → push delta → repeat
  → on disconnect: exponential backoff reconnect (2s → max 60s)
```

### Model update flow
```
FLClient._run_training():
  → capture_initial_weights() → train local_epochs → compute_delta()
  → apply DP (clip + noise) → apply top-k compression → base64 encode
  → send via WebSocket {"type": "update", "update": {...}}

GroupManager.add_client_update():
  → normalize → TrustManager.update_trust() → group.pending_updates.append(delta)
  → if len(pending_updates) >= window_size → TRIGGER aggregate_group()

GroupManager._training_watchdog():
  → asyncio.sleep(1s) loop → if elapsed >= time_limit → TRIGGER aggregate_group()

GroupManager.aggregate_group():
  → is_peft? apply_peft_delta() : apply_flat_delta() → update server model
  → save checkpoint → broadcast model_update → clear buffer
```

## Key Decisions
- **Registry-driven models**: No hardcoded model classes. All models registered via ModelRegistry with factory lambdas. Client and server both call `build_model(model_id)`.
- **Hybrid async windowing**: Aggregate when N updates received OR T seconds elapsed — prevents straggler clients from blocking
- **Staleness-weighted aggregation**: Weight each update by `exp(-lambda * staleness)`, default lambda=0.2
- **Trust via cosine similarity**: Client update vs. running global estimate; quarantine below 0.35
- **Join approval with trust gate**: Admin approves join requests; server checks trust score ≥ 0.10 before activating client
- **Single SQLite (astra.db)**: WAL mode, thread-local connections, fine for research/dev scale
- **All imports use `astra.*`**: Zero shim packages, `src/` on sys.path
- **Extended endpoints guarded**: `_extended_api_registered` flag prevents double-registration on FastAPI dev-reload

## Active Gotchas
- `sys.path` must point to `src/` (not `src/astra/`) — `import astra` needs `src/` as root
- Import migration rule ORDER matters: more-specific strings first before generic prefixes
- `aggregator_buffer` is `deque(maxlen=window_size)` — oldest entry silently dropped when full (by design)
- Models registered via External/Architecture tab are persisted to DB and reloaded on server restart
- Join requests require the group to exist (checked before DB insert) and won't duplicate pending requests
- Client requires authentication (JWT) for both WebSocket and REST calls
- `GET /api/groups` requires authentication (any role)

## Common Workflows
- **Start server**: `uvicorn astra.app.server_api:app --reload` from repo root
- **Start dashboard**: `cd dashboard && npm run dev` → http://localhost:3000
- **Run tests**: `pytest tests/ -v` (223 tests expected)
- **Run client**: `python -m astra.client.cli --server http://localhost:8000 --client-id client_1 --group-id test --username user --password pass`
- **Register model**: Via dashboard (Registry / HuggingFace / External tabs) or programmatically via registry
- **Config**: `config.yaml` for training params; env vars: `SECRET_KEY`, `ENV`, `GEMINI_API_KEY`, `ASTRA_DEFAULT_ADMIN_PASSWORD`, `ASTRA_SEED`

## Metrics / Benchmarks
- Tests: 223/223 passing
- Config defaults: 20 clients, window_size=10, MNIST/Dirichlet(0.3), hybrid robust aggregation, DP client-side (sigma=1.2, clip_norm=1.0), top-k compression (ratio=0.1)
