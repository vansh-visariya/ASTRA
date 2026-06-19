# ASTRA Architecture

This document describes the internal design of ASTRA — how the federated learning pipeline works end-to-end.

## High-Level Architecture

```
┌─────────────┐     ┌──────────────────────────────────────┐     ┌─────────────┐
│  Dashboard   │────▶│           FastAPI Server              │◀────│  FL Client  │
│  (Next.js)   │     │                                      │     │  (CLI/WS)   │
│              │◀────│  ┌────────┐  ┌──────────┐  ┌──────┐ │────▶│              │
└─────────────┘     │  │FLServer│  │ GroupMgr │  │  DB  │ │     └─────────────┘
      │             │  └───┬────┘  └────┬─────┘  └──────┘ │
      │             │      │            │                   │
      ▼             │  ┌───▼────────────▼─────┐            │
  WebSocket ◀──────▶│  │    AsyncServer        │            │
                    │  │  (aggregation engine) │            │
                    │  └───────────────────────┘            │
                    └──────────────────────────────────────┘
```

## Data Flow

### Client Registration → First Global Model

1. Client starts with `python -m astra.client.cli --server http://host:8000 --client-id my_client`
2. CLI sends `POST /api/clients/register` with client metadata
3. Server registers the client in AstraDB and creates a WebSocket entry
4. Client requests the global model weights for its assigned group via WebSocket
5. Client downloads weights and begins local training

### Training Loop

```
┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐
│  Client  │     │ WebSocket│     │ AsyncServer  │     │ Dashboard│
│ trains   │     │ handler  │     │ / GroupMgr   │     │          │
└────┬─────┘     └────┬─────┘     └──────┬───────┘     └────┬─────┘
     │                │                 │                    │
     │ train 1 epoch  │                 │                    │
     │───────────────▶│                 │                    │
     │                │ push_delta()    │                    │
     │                │────────────────▶│                    │
     │                │                 │ buffer.append(d)   │
     │                │                 │ update_trust()     │
     │                │                 │                    │
     │                │                 │ _maybe_aggregate() │
     │                │                 │─── if len>=N or T  │
     │                │                 │    aggregate()     │
     │                │                 │    momentum()      │
     │                │                 │    _apply_update() │
     │                │                 │                    │
     │                │                 │ broadcast metrics  │
     │                │                 │───────────────────▶│
     │                │   new_weights   │                    │
     │◀───────────────│◀────────────────│                    │
     │                │                 │                    │
```

### Aggregation Strategies

| Strategy | How it works | When to use |
|----------|-------------|-------------|
| **FedAvg** | Weighted average of all updates (by dataset size) | Default, honest clients |
| **Trimmed Mean** | Discards top/bottom `trim_ratio` fraction per coordinate | Suspected outliers |
| **Coordinate Median** | Per-coordinate median across updates | Byzantine attacks |
| **Hybrid** | Trust-weighted FedAvg with staleness decay | Mixed trust environments |

### Trust Scoring

Each client has a trust score (0.0 to 1.0) computed via cosine similarity:

```
trust = cosine_similarity(client_delta, running_global_estimate)
trust = EMWA(trust, previous_trust, beta=0.9)  # exponential decay

if trust < 0.35: quarantine client
if trust > 0.70: fully trusted
else: partial weight scaling
```

### Window-Based Aggregation

ASTRA uses **hybrid async windowing** — aggregation fires when:

- **N updates** have been received (`config.aggregator_window`), OR
- **T seconds** have elapsed since last aggregation (`config.poll_timeout`)

This prevents straggler clients from blocking the entire pipeline.

## Core Module Map

| Module | Location | Responsibility |
|--------|----------|----------------|
| `AsyncServer` | `core/server.py` | Aggregation engine, buffer management, trust coordination, momentum |
| `FLClient` | `core/fl_client.py` | Local training, delta computation, compression, DP |
| `TrainingGroup` | `app/training_group.py` | Group state, async window config |
| `GroupManager` | `app/group_manager.py` | Group lifecycle (create/start/pause/stop/delete), watchdog timer |
| `FLServer` | `app/fl_server.py` | Orchestrator — wires AsyncServer + GroupManager + ConnectionManager |
| `AstraDB` | `app/database.py` | SQLite persistence, WAL mode, thread-local connections |
| `ConnectionManager` | `infra/connection_manager.py` | WebSocket registry, broadcast, per-client send |
| `ModelRegistry` | `infra/registry.py` | Model factory registry, HF integration, dynamic architecture import + DB persistence |
| `hf_models` | `core/models/hf_models.py` | HuggingFace model loading, PEFT application, save/load to disk (.pt + safetensors) |
| `TrustManager` | `core/trust_manager.py` | Cosine similarity trust, quarantine logic |
| `Config` | `core/config.py` | YAML + env config loading |

## WebSocket Protocol

The WebSocket connection uses Socket.IO with the following events:

| Direction | Event | Payload |
|-----------|-------|---------|
| Client→Server | `push_delta` | `PushDeltaPayload` — serialized float32 delta, metadata |
| Client→Server | `request_model` | `RequestModelPayload` — client ID |
| Server→Client | `new_model` | Serialized model weights (bytes) |
| Server→Client | `metrics_update` | Aggregated metrics (accuracy, loss, trust scores) |
| Server→Client | `notification` | Notification object (type, message, timestamp) |

## Client Lifecycle

```
register → join_request → approved → activate → training ↔ aggregation → complete
                                           │
                                     rejected (end)
```

## HuggingFace Model Download (PEFT Groups)

When an admin creates a group with a HuggingFace model and PEFT enabled, the base model is saved to disk so clients can download it efficiently:

```
Admin creates group (PEFT enabled):
  → GroupManager._save_hf_model_to_disk()
    → saves base_model.pt (PyTorch format)
    → saves base_model.safetensors (if safetensors installed)
    → saves adapter_config.json (LoRA config metadata)
    → stored in models/hf/{model_id}/

Client workflow:
  1. GET /api/models/{group_id}/download-info → check available files
  2. GET /api/models/{group_id}/base → download frozen backbone (one-time, large)
  3. Fine-tune locally with LoRA adapters
  4. GET /api/models/{group_id}/adapter → download latest adapter (per-round, small)
  5. POST /api/clients/{client_id}/delta → upload adapter-only weights
     → Server validates: rejects if > 50% of full model (likely full model uploaded by mistake)
```

**Upload validation**: When a PEFT group receives an upload, the server checks the delta size against the expected adapter size. If the payload is > 50% of the full model size, it's rejected with an actionable error message telling the client to upload only adapter weights.

**Safetensors support**: The base model is saved in both `.pt` (PyTorch native, fast loading) and `.safetensors` (HuggingFace-native format) when the `safetensors` package is installed. Clients can choose their preferred format via the `format` query parameter.

## Database Schema

Single SQLite database (`astra.db`) with these tables:

- `users` — id, username, password_hash, role, created_at
- `groups` — id, name, model_id, config JSON, status, created_at
- `group_members` — group_id, client_id, status, joined_at, trust_score
- `clients` — id, name, status, last_seen, metadata JSON
- `join_requests` — id, group_id, client_id, status, created_at
- `notifications` — id, user_id, type, message, read, created_at
- `model_registry` — model_id, model_type, architecture, config JSON, source

All timestamps are Unix floats.

## Adding a Custom Model

See the [Quick Start](../README.md#quick-start) and use the External tab in the Create Group page, or call the API directly:

```
POST /api/models/register/architecture
{
  "model_id": "my_resnet",
  "architecture_path": "torchvision.models.resnet50",
  "model_type": "vision",
  "config": {"pretrained": true}
}
```
