# Memory

## Project Overview
ASTRA (Async Scalable Training & Research Architecture) is a production-ready distributed Federated Learning platform. **Clients train externally and submit pre-computed model deltas** to the server; the server aggregates using hybrid async windowing (N updates OR T seconds) and broadcasts the new global model. Stack: FastAPI + PyTorch backend (`src/astra/`), Next.js 14 frontend (`dashboard/`).

Key server features: async aggregation with staleness decay, Byzantine-robust aggregation (FedAvg, Trimmed Mean, Median, Hybrid), trust scoring via cosine similarity, server-side DP-SGD, top-k sparsification, HuggingFace model support with optional LoRA/PEFT, registry-driven model loading with dynamic import.

## Package Layout
All Python source lives under `src/astra/` and is importable as `astra.*`. `src/` must be on `sys.path` (handled by `conftest.py` and `pyproject.toml pythonpath = ["src"]`).

```
src/astra/
├── app/          # API layer, orchestration, DB, group lifecycle
│   └── routes/   # FastAPI routers (system, groups, clients, models, experiments)
├── core/         # Server-side FL algorithms — pure Python, no web dependencies
│   ├── aggregation/   # aggregator.py, robust.py — FedAvg, TrimmedMean, Median, Hybrid
│   ├── models/        # model_zoo.py (flatten_all_params, apply_flat_delta, PEFT utils), hf_models.py (HuggingFace+PEFT)
│   ├── privacy/       # privacy.py (server-side DP-SGD)
│   └── utils/         # metrics.py, seed.py, logging_utils.py
├── infra/        # Transport, DB schemas, auth, WebSocket, model registry
│   └── security/ # auth.py — JWT + bcrypt + role-based access
```

No `client/` directory and no `FLClient` class — those were removed. Clients upload deltas via REST or the dashboard.

## Model System (Registry-Driven)
No hardcoded model classes. All models are registered via the ModelRegistry:

| Registration method | Dashboard tab | What it does |
|---|---|---|
| `register_factory(model_id, factory)` | Programmatic | Register any `lambda: nn.Module()` |
| `POST /api/models/register/hf` | HuggingFace tab | Register from HF Hub with optional PEFT |
| `POST /api/models/register/architecture` | External tab | Dynamic import from Python path (e.g. `torchvision.models.resnet18`) |

Models registered via External/Architecture tab are persisted to `model_registry` table in SQLite and auto-reloaded on server restart. The registry is the single source of truth — server calls `registry.build_model(model_id)` to get the architecture.

## Component Map
- `src/astra/app/server_api.py` — FastAPI entry point; assembles app, mounts routers, WebSocket, Socket.IO; lifespan manages FLServer singleton
- `src/astra/app/fl_server.py` — FLServer: orchestrates AsyncServer + GroupManager + ConnectionManager; lazy-builds the AsyncServer for a group's model on first delta upload if the server started with no model_id
- `src/astra/app/group_manager.py` — GroupManager: manages concurrent TrainingGroups; window-based aggregation, watchdog-triggered time-based aggregation
- `src/astra/app/training_group.py` — TrainingGroup + AsyncWindowConfig dataclass; `add_update()`, `to_dict()` with full clients/accuracy/loss fields
- `src/astra/app/database.py` — AstraDB: single SQLite `astra.db`; WAL mode, thread-local connections; `model_registry` table for persisted models
- `src/astra/app/integration.py` — FLPlatformIntegration: wires auth + notifications + trust + recommender + inference
- `src/astra/app/extended_endpoints.py` — Auth/join/notification/trust REST endpoints + `/api/recommendations/unified`
- `src/astra/app/notifications.py` — NotificationService: in-app notifications, WebSocket delivery
- `src/astra/app/model_recommender.py` — Gemini-powered model suggestions
- `src/astra/core/server.py` — AsyncServer: async FL engine; staleness-weighted aggregation, trust scoring, momentum
- `src/astra/core/aggregation/` — FedAvg, TrimmedMean, CoordinateMedian, Hybrid aggregation
- `src/astra/core/models/model_zoo.py` — `flatten_all_params()`, `apply_flat_delta()`, PEFT utilities, `SimpleMLP`
- `src/astra/core/models/hf_models.py` — HuggingFace model loading with optional LoRA/PEFT; `save_base_model_to_disk()` and `load_base_model_from_disk()` for persisting base models in .pt and safetensors formats; `get_download_info()` for metadata about available files
- `src/astra/core/trust_manager.py` — Cosine-similarity trust scoring, exponential decay, quarantine at 0.35
- `src/astra/core/privacy/privacy.py` — Server-side DP (`clip_and_noise`); the `MaliciousSimulator` was removed
- `src/astra/core/compression.py` — Top-k sparsification and quantization
- `src/astra/infra/connection_manager.py` — WebSocket registry; broadcast + per-client send
- `src/astra/infra/registry.py` — ModelRegistry: `register_factory()`, `build_model()`, HF registration, architecture registration
- `src/astra/infra/websocket_handler.py` — WebSocket endpoint (event channel only); rejects `train_command`, `update`, `metrics`, `training_*` messages
- `src/astra/app/routes/clients.py` — **`POST /api/clients/{client_id}/delta`** — the new entry point for external clients to submit deltas (auth required, base64-decodes float32, validates NaN/Inf, enforces ≤100 MB and 2 s rate limit)

## Data / State Flow

### External client upload flow
```
Client signs up → requests to join group → admin approves → client activates
  → client trains externally on their own data
  → computes delta = new_weights - old_weights (float32, little-endian)
  → POST /api/clients/{client_id}/delta with base64(local_updates)
    → JWT verified
    → size ≤ 100 MB, length % 4 == 0, no NaN/Inf
    → rate-limited (1 upload / 2 s per client)
    → if PEFT group: validates adapter-only upload (rejects full model weights)
    → AsyncServer.handle_update(): apply server-side DP, score trust, append to buffer
    → if window full → AsyncServer._perform_aggregation() → apply delta to model → bump global_version
    → also GroupManager.process_client_update + aggregate_group (keeps group model_version in sync)
    → broadcast "client_update" / "aggregation_complete" over WebSocket
  → GET /api/models/{group_id}/download to pull the new global model
```

### HuggingFace model download flow (PEFT groups)
```
Admin creates group with HF model + PEFT enabled:
  → GroupManager.create_group() saves base model to models/hf/{model_id}/
    → base_model.pt (PyTorch format, fast loading)
    → base_model.safetensors (HuggingFace-native format, if safetensors installed)
    → adapter_config.json (metadata)

Client download workflow:
  → GET /api/models/{group_id}/download-info (returns available files, formats, sizes)
  → First time: GET /api/models/{group_id}/base (downloads frozen backbone, ~large)
  → Each round: GET /api/models/{group_id}/adapter (downloads LoRA adapter, ~small)
  → Upload: POST /api/clients/{client_id}/delta with adapter-only weights
    → Server validates: rejects if payload > 50% of full model size (likely full model uploaded by mistake)
```

### Aggregation trigger flow
```
GroupManager._training_watchdog():
  → asyncio.sleep(1s) loop → if elapsed >= time_limit → aggregate_group()

GroupManager.aggregate_group():
  → aggregate deltas (FedAvg / Trimmed Mean / Median / Hybrid)
  → apply_peft_delta() or apply_flat_delta() to live server model
  → save checkpoint → broadcast "model_update" → clear buffer
```

## Key Decisions
- **External training**: Clients train on their own hardware/data. This project only handles aggregation + trust + DP + delivery.
- **Registry-driven models**: No hardcoded model classes. All models registered via ModelRegistry with factory lambdas.
- **Hybrid async windowing**: Aggregate when N updates received OR T seconds elapsed — prevents straggler clients from blocking
- **Staleness-weighted aggregation**: Weight each update by `exp(-lambda * staleness)`, default lambda=0.2
- **Trust via cosine similarity**: Client update vs. running global estimate; quarantine below 0.35
- **Lazy AsyncServer init**: If the FL server starts with no `model_id` in config, it builds the AsyncServer on the first delta upload using the group's `model_id` and `window_size`.
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
- Delta bytes must be float32 little-endian (`np.frombuffer(..., dtype='<f4')`); a wrong-endian numpy build (some Windows configs) will read garbage. The upload endpoint uses `'<f4'` explicitly to avoid this.
- The route's `submit_client_delta` calls BOTH `AsyncServer.handle_update` (for DP + trust + aggregator) AND `GroupManager.process_client_update` + `aggregate_group` (so the group's `model_version` and metrics_history stay in sync with the AsyncServer's `global_version`). Both must stay in step.
- `client_to_group` map is updated by `GroupManager.register_client`, NOT by `group.add_client`. The activate endpoint uses `register_client` to wire this correctly.

## Common Workflows
- **Start server**: `uvicorn astra.app.server_api:app --reload` from repo root
- **Start dashboard**: `cd dashboard && npm run dev` → http://localhost:3000
- **Run tests**: `pytest tests/ -v` (~210 tests)
- **Submit a delta (REST)**:
  ```bash
  curl -X POST http://localhost:8000/api/clients/$CLIENT_ID/delta \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"client_id":"...","client_version":0,"local_updates":"<base64 float32 bytes>","update_type":"delta","local_dataset_size":1000,"meta":{}}'
  ```
- **Submit a delta (dashboard)**: log in as client → Upload Delta → select group, file, click Upload
- **Register model**: Via dashboard (Registry / HuggingFace / External tabs) or programmatically via `registry.register_factory(model_id, factory, info)`
- **Config**: `config.yaml` for training params; env vars: `SECRET_KEY`, `ENV`, `GEMINI_API_KEY`, `ASTRA_DEFAULT_ADMIN_PASSWORD`, `ASTRA_SEED`

## Metrics / Benchmarks
- Tests: ~283 passing (3 new test files for the upload pipeline + WebSocket cleanup + HF download; 3 old test files deleted)
- Config defaults: 20 clients, window_size=10, MNIST/Dirichlet(0.3), hybrid robust aggregation, DP client-side (sigma=1.2, clip_norm=1.0), top-k compression (ratio=0.1)
