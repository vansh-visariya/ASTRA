# SKILL: FL Platform Master Index

## HOW TO USE THESE SKILLS

Before writing any code for this federated learning platform, read the relevant SKILL.md file(s) listed below. They contain battle-tested patterns, known bug fixes, and anti-patterns to avoid.

---

## SKILL FILES MAP

| Component | SKILL File | When to Read |
|-----------|-----------|--------------|
| Federated Learning Algorithms | `FL_CORE_SKILL.md` | Implementing FedAvg, DP, aggregation, round management |
| gRPC + WebSocket Networking | `NETWORKING_SKILL.md` | Client-server communication, streaming, TLS, auth |
| Web Dashboard | `DASHBOARD_SKILL.md` | FastAPI backend, React frontend, real-time updates |
| Model Registry & Storage | `MODEL_MANAGEMENT_SKILL.md` | Versioning, checkpointing, rollback, validation |
| Security & Privacy | `SECURITY_SKILL.md` | JWT, DP budget, Byzantine detection, input validation |
| DevOps & Infrastructure | `DEVOPS_SKILL.md` | Docker, monitoring, CI/CD, logging, certificates |

---

## TECHNOLOGY STACK

```
Backend (Python 3.11+):
  FL Core:        PyTorch, numpy
  Networking:     grpcio, grpcio-tools, websockets
  API:            FastAPI, uvicorn, SQLAlchemy (async), asyncpg
  Auth:           python-jose[cryptography], passlib[bcrypt]
  Privacy:        google-dp-accounting (or opacus for DP-SGD)
  Storage:        aioboto3 (S3/MinIO), aiofiles
  Monitoring:     prometheus-client, structlog
  Validation:     pydantic v2

Frontend (TypeScript):
  Framework:      React 18 + Vite
  State:          Zustand + immer
  Charts:         Recharts
  Data fetching:  SWR or TanStack Query
  Styling:        Tailwind CSS
  WebSocket:      Native WebSocket (see useWebSocket hook)

Infrastructure:
  Database:       PostgreSQL 15
  Cache/Revocation: Redis 7
  Model Storage:  MinIO (self-hosted S3-compatible)
  Container:      Docker + Docker Compose (dev), Kubernetes (prod)
  Monitoring:     Prometheus + Grafana
  CI/CD:          GitHub Actions
```

---

## PROJECT STRUCTURE

```
fl-platform/
├── fl_core/                 # Federated learning algorithms
│   ├── server/
│   ├── client/
│   └── common/
├── networking/              # gRPC + WebSocket
│   ├── grpc/
│   ├── websocket/
│   └── security/
├── dashboard/               # Web dashboard
│   ├── backend/             # FastAPI
│   └── frontend/            # React + Vite
├── model_management/        # Model registry + storage
├── monitoring/              # Prometheus + Grafana configs
├── docker/                  # Dockerfiles
├── k8s/                     # Kubernetes manifests
├── tests/
│   ├── unit/
│   └── integration/
├── proto/                   # gRPC proto definitions
├── generated/               # Auto-generated proto stubs
├── certs/                   # TLS certificates (gitignored)
├── .env.template            # Environment variable template
├── docker-compose.yml
└── requirements.txt
```

---

## UNIVERSAL RULES (apply across ALL files)

### Error Handling
```python
# ✅ ALWAYS use specific custom exceptions
class FLBaseError(Exception): pass
class RoundStateError(FLBaseError): pass
class InsufficientClientsError(FLBaseError): pass
class IntegrityError(FLBaseError): pass
class PrivacyBudgetExhausted(FLBaseError): pass
class ByzantineClientDetected(FLBaseError): pass
class TokenExpiredError(FLBaseError): pass

# ✅ ALWAYS log exceptions with context before re-raising
try:
    await run_round(round_id)
except FLBaseError as e:
    logger.exception("fl_error", round_id=round_id, error=str(e), exc_info=True)
    raise
```

### Configuration
```python
# ✅ ALWAYS load sensitive config from environment — NEVER hardcode
import os

class Settings:
    jwt_secret: str = os.environ["JWT_SECRET"]          # Crash fast if missing
    database_url: str = os.environ["DATABASE_URL"]
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    def __post_init__(self):
        if len(self.jwt_secret) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters")

# ❌ NEVER do this
JWT_SECRET = "mysecret123"  # Hardcoded secret
```

### Async Best Practices
```python
# ✅ Always await async operations — never mix sync I/O in async
async def get_clients():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Client))  # async query
        return result.scalars().all()

# ❌ NEVER call blocking I/O in async context
async def bad_get_clients():
    return db.query(Client).all()  # Blocks event loop!
```

### Testing Pattern
```python
# Every module needs: unit tests + integration tests
# tests/unit/test_aggregator.py
import pytest
import torch

def test_fedavg_weighted_correctly():
    updates = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([3.0])}]
    sizes = [1, 3]  # 25% weight and 75% weight
    result = federated_average(updates, sizes)
    expected = 0.25 * 1.0 + 0.75 * 3.0  # = 2.5
    assert torch.allclose(result["w"], torch.tensor([expected]))

def test_fedavg_rejects_shape_mismatch():
    updates = [{"w": torch.tensor([1.0, 2.0])}, {"w": torch.tensor([1.0])}]
    with pytest.raises(ValueError, match="Shape mismatch"):
        federated_average(updates, [1, 1])
```

---

## GOLDEN PATH: STARTING A NEW FEATURE

1. **Read** the relevant SKILL.md file(s) above  
2. **Define** types/schemas with Pydantic (schema-first)  
3. **Write** the DB model if needed (SQLAlchemy)  
4. **Implement** business logic with proper error handling  
5. **Add** Prometheus metrics for observability  
6. **Write** unit tests → integration tests  
7. **Update** docker-compose.yml if new service needed  
8. **Run** `mypy` + `ruff` before committing  

---

## QUICK REFERENCE: COMMON PITFALLS

| Mistake | Where It Bites | Prevention |
|---------|---------------|------------|
| `model.state_dict()` without `deepcopy` | FL Core | Always `copy.deepcopy()` |
| Sync DB in async FastAPI | Dashboard | Use `AsyncSession` only |
| Missing WS disconnect cleanup | Dashboard | `finally: await event_bus.disconnect()` |
| No round_id check on update | Networking | Validate round_id matches current |
| No NaN check before aggregation | FL Core | `validate_model_update()` |
| Checkpoint non-atomic write | Model Mgmt | `.tmp` file + rename |
| JWT secret hardcoded | Security | `os.environ["JWT_SECRET"]` |
| DP budget not tracked | Security | `PrivacyAccountant.charge()` each round |
| gRPC message > 4MB | Networking | Use streaming for models |
| No connection pool pre_ping | DevOps | `pool_pre_ping=True` in SQLAlchemy |