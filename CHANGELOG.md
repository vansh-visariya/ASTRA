# Changelog

All notable changes to the ASTRA project.

## [0.1.0] — Unreleased

### Added
- Async Federated Learning server with staleness-weighted aggregation
- Byzantine-robust aggregation (FedAvg, Trimmed Mean, Coordinate Median, Hybrid)
- Trust scoring via cosine similarity with exponential decay
- Differential Privacy (DP-SGD) — client-side and server-side modes
- Top-k sparsification + quantization for communication compression
- IID, Dirichlet (non-IID), and pathological data splits
- HuggingFace model support with optional LoRA/PEFT
- HuggingFace model download-to-disk: base models saved in .pt and safetensors formats on group creation
- Safetensors format support for model downloads (.pt and .safetensors)
- Download-info endpoint (`GET /api/models/{group_id}/download-info`) for clients to discover available files
- PEFT upload validation: server rejects full model weights for PEFT groups (adapter-only enforcement)
- FastAPI REST API + WebSocket + Socket.IO real-time transport
- Next.js 14 dashboard with admin and client panels
- Group-based training with hybrid async windowing (N updates OR T seconds)
- SQLite persistence (astra.db) with WAL mode
- JWT + bcrypt authentication with role-based access (admin/client/observer)
- Malicious client simulator for robustness testing
- Heterogeneous model aggregation support
- Gemini API-powered model recommendations

### Changed
- Restructured codebase to `src/astra/` package layout
- Migrated all imports to `astra.*` namespace
- Centralized config loading via `astra.core.config.load_config()`
- Renamed `core/client.py` → `core/fl_client.py` and `client/client.py` → `client/cli.py`
- Populated `__init__.py` files with re-exports for clean public API
- Added `[project.scripts]` CLI entry points (`astra-server`, `astra-client`)
- Fixed Dockerfiles to use correct `astra.*` module paths
- Rewritten Makefile with correct paths for tests, lint, and typecheck

### Fixed
- Removed dead `networking.state` import in `/health` endpoint
- Removed unused `aiohttp.web.Application` from `server_api.py`
