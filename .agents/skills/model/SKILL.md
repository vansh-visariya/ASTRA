# SKILL: Model Management & Registry

## PURPOSE
Production-grade model versioning, checkpoint storage, lineage tracking, rollback, and serving for the FL platform.

---

## ARCHITECTURE

```
model_management/
├── registry/
│   ├── model_registry.py        # Central registry CRUD
│   ├── version_manager.py       # Semantic versioning, lineage
│   └── registry_db.py           # SQLAlchemy models for registry
├── storage/
│   ├── storage_backend.py       # Abstract storage interface
│   ├── local_storage.py         # Local filesystem backend
│   └── s3_storage.py            # S3/MinIO backend
├── serving/
│   ├── model_server.py          # Serve models for inference
│   └── predictor.py             # Batch/single inference
├── checkpointing/
│   ├── checkpoint_manager.py    # Save/load training checkpoints
│   └── checkpoint_scheduler.py  # Auto-checkpoint every N rounds
└── validation/
    ├── model_validator.py       # Pre-registration quality checks
    └── drift_detector.py        # Performance drift detection
```

---

## DATABASE SCHEMA

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id              = Column(Integer, primary_key=True)
    version         = Column(String(32), unique=True, nullable=False, index=True)  # "1.0.0", "1.1.0"
    round_id        = Column(Integer, ForeignKey("rounds.id"), nullable=False)
    storage_path    = Column(String(512), nullable=False)    # URI: "s3://bucket/models/v1.0.0.pt"
    checksum        = Column(String(64), nullable=False)     # SHA-256
    file_size_bytes = Column(Integer, nullable=False)
    
    # Performance
    val_loss        = Column(Float, nullable=True)
    val_accuracy    = Column(Float, nullable=True)
    
    # Metadata
    architecture    = Column(String(128), nullable=False)    # "ResNet50", "BERT-base"
    framework       = Column(String(32), nullable=False)     # "pytorch", "tensorflow"
    model_config    = Column(JSON, default={})               # Hyperparams, layer dims
    tags            = Column(JSON, default=[])               # ["production", "baseline"]
    
    # Lineage
    parent_version  = Column(String(32), nullable=True)      # Which version this was trained from
    num_clients     = Column(Integer, nullable=False)
    total_rounds    = Column(Integer, nullable=False)
    
    # State
    status          = Column(String(32), default="candidate")  # candidate|active|archived|failed
    is_production   = Column(Boolean, default=False)
    
    created_at      = Column(DateTime, default=datetime.utcnow)
    promoted_at     = Column(DateTime, nullable=True)
    archived_at     = Column(DateTime, nullable=True)
    
    # Notes
    description     = Column(String(1024), default="")
    created_by      = Column(String(64), default="fl-server")
```

---

## CRITICAL RULES — ALWAYS FOLLOW

### 1. Semantic Versioning — Deterministic
```python
from packaging.version import Version

class VersionManager:
    """
    Major.Minor.Patch
    - Major: Architecture change (incompatible weights)
    - Minor: New round of FL training from same architecture
    - Patch: Hotfix / fine-tune on existing model
    """
    
    @staticmethod
    def next_version(current: str, bump: str = "minor") -> str:
        v = Version(current)
        if bump == "major":
            return f"{v.major + 1}.0.0"
        elif bump == "minor":
            return f"{v.major}.{v.minor + 1}.0"
        elif bump == "patch":
            return f"{v.major}.{v.minor}.{v.micro + 1}"
        else:
            raise ValueError(f"Invalid bump type: {bump}. Use 'major', 'minor', or 'patch'")

    @staticmethod
    def is_compatible(base: str, candidate: str) -> bool:
        """Same major version = compatible weights (can continue training)."""
        return Version(base).major == Version(candidate).major
```

### 2. Storage Backend — Abstract Interface
```python
from abc import ABC, abstractmethod
import hashlib

class StorageBackend(ABC):
    @abstractmethod
    async def save(self, key: str, data: bytes) -> str:
        """Save bytes, return full URI."""
        ...
    
    @abstractmethod
    async def load(self, uri: str) -> bytes:
        """Load bytes by URI."""
        ...
    
    @abstractmethod
    async def delete(self, uri: str) -> None: ...
    
    @abstractmethod
    async def exists(self, uri: str) -> bool: ...
    
    @staticmethod
    def compute_checksum(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()


class LocalStorageBackend(StorageBackend):
    def __init__(self, base_path: str):
        import os
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    async def save(self, key: str, data: bytes) -> str:
        import aiofiles
        path = f"{self.base_path}/{key}"
        import os; os.makedirs(os.path.dirname(path), exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        return f"file://{path}"

    async def load(self, uri: str) -> bytes:
        import aiofiles
        path = uri.removeprefix("file://")
        async with aiofiles.open(path, "rb") as f:
            return await f.read()

    async def delete(self, uri: str) -> None:
        import os
        os.remove(uri.removeprefix("file://"))

    async def exists(self, uri: str) -> bool:
        import os
        return os.path.exists(uri.removeprefix("file://"))


class S3StorageBackend(StorageBackend):
    def __init__(self, bucket: str, endpoint_url: str = None):
        import aioboto3
        self.bucket = bucket
        self.session = aioboto3.Session()
        self.endpoint_url = endpoint_url   # For MinIO

    async def save(self, key: str, data: bytes) -> str:
        async with self.session.client("s3", endpoint_url=self.endpoint_url) as s3:
            await s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    async def load(self, uri: str) -> bytes:
        key = uri.removeprefix(f"s3://{self.bucket}/")
        async with self.session.client("s3", endpoint_url=self.endpoint_url) as s3:
            response = await s3.get_object(Bucket=self.bucket, Key=key)
            return await response["Body"].read()
```

### 3. Model Registry — Core Operations
```python
class ModelRegistry:
    def __init__(self, storage: StorageBackend, db: AsyncSession):
        self.storage = storage
        self.db = db

    async def register(
        self,
        model_bytes: bytes,
        version: str,
        round_id: int,
        metadata: dict,
        parent_version: str = None,
    ) -> ModelVersion:
        """Register a new model version — atomic operation."""
        
        # 1. Validate before anything
        await self._validate_model(model_bytes, metadata)
        
        # 2. Check version doesn't already exist
        existing = await self.db.execute(
            select(ModelVersion).where(ModelVersion.version == version)
        )
        if existing.scalar_one_or_none():
            raise DuplicateVersionError(f"Version {version} already registered")
        
        # 3. Compute checksum before storage
        checksum = self.storage.compute_checksum(model_bytes)
        
        # 4. Store model (idempotent key = checksum)
        storage_key = f"models/{version}/{checksum[:12]}.pt"
        uri = await self.storage.save(storage_key, model_bytes)
        
        # 5. Create DB record
        record = ModelVersion(
            version=version,
            round_id=round_id,
            storage_path=uri,
            checksum=checksum,
            file_size_bytes=len(model_bytes),
            parent_version=parent_version,
            **{k: v for k, v in metadata.items() if k in ModelVersion.__table__.columns.keys()}
        )
        self.db.add(record)
        await self.db.flush()   # Get ID before commit
        
        return record

    async def get_production_model(self) -> tuple[bytes, ModelVersion]:
        """Get current production model bytes + metadata."""
        result = await self.db.execute(
            select(ModelVersion)
            .where(ModelVersion.is_production == True)
            .order_by(ModelVersion.promoted_at.desc())
            .limit(1)
        )
        record = result.scalar_one_or_none()
        if not record:
            raise NoProductionModelError("No production model set")
        
        model_bytes = await self.storage.load(record.storage_path)
        
        # Verify integrity on every load
        actual = self.storage.compute_checksum(model_bytes)
        if actual != record.checksum:
            raise IntegrityError(
                f"Model {record.version} checksum mismatch — storage may be corrupted"
            )
        
        return model_bytes, record

    async def promote_to_production(
        self, version: str, promoted_by: str = "system"
    ) -> None:
        """Atomically swap production model. Rollback-safe."""
        async with self.db.begin():
            # 1. Verify candidate exists and is validated
            candidate = await self.db.execute(
                select(ModelVersion).where(ModelVersion.version == version)
            )
            candidate = candidate.scalar_one_or_none()
            if not candidate:
                raise ModelNotFoundError(f"Version {version} not found")
            if candidate.status not in ("candidate", "active"):
                raise InvalidStateError(f"Cannot promote model in state: {candidate.status}")
            
            # 2. Demote all current production models
            await self.db.execute(
                update(ModelVersion)
                .where(ModelVersion.is_production == True)
                .values(is_production=False, status="archived", archived_at=datetime.utcnow())
            )
            
            # 3. Promote new version
            candidate.is_production = True
            candidate.status = "active"
            candidate.promoted_at = datetime.utcnow()
            
            # 4. Audit log
            audit = ModelAuditLog(
                version=version, action="promoted", actor=promoted_by,
                timestamp=datetime.utcnow()
            )
            self.db.add(audit)

    async def rollback(self, target_version: str = None) -> ModelVersion:
        """Roll back to previous production model or specific version."""
        if target_version:
            return await self.promote_to_production(target_version, promoted_by="rollback")
        
        # Find last archived production model
        result = await self.db.execute(
            select(ModelVersion)
            .where(ModelVersion.status == "archived")
            .order_by(ModelVersion.archived_at.desc())
            .limit(1)
        )
        prev = result.scalar_one_or_none()
        if not prev:
            raise RollbackError("No previous version to roll back to")
        
        return await self.promote_to_production(prev.version, promoted_by="rollback")
```

### 4. Checkpoint Manager
```python
import torch
from pathlib import Path
from datetime import datetime

class CheckpointManager:
    """
    Save training state (not just weights) for resumable FL training.
    Checkpoints include: model, optimizer, round state, metrics history.
    """
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    def save(
        self,
        model: torch.nn.Module,
        round_id: int,
        metrics: dict,
        extra: dict = None,
    ) -> Path:
        checkpoint = {
            "round_id": round_id,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "extra": extra or {},
        }
        path = self.dir / f"checkpoint_round_{round_id:06d}.pt"
        # Atomic write: write to temp, then rename
        tmp = path.with_suffix(".tmp")
        torch.save(checkpoint, tmp)
        tmp.rename(path)   # Atomic on POSIX
        
        self._prune_old_checkpoints()
        return path

    def load_latest(self) -> dict | None:
        checkpoints = sorted(self.dir.glob("checkpoint_round_*.pt"))
        if not checkpoints:
            return None
        return torch.load(checkpoints[-1], map_location="cpu", weights_only=False)

    def load_round(self, round_id: int) -> dict | None:
        path = self.dir / f"checkpoint_round_{round_id:06d}.pt"
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=False)

    def _prune_old_checkpoints(self):
        """Keep only the last N checkpoints."""
        checkpoints = sorted(self.dir.glob("checkpoint_round_*.pt"))
        for old in checkpoints[:-self.keep_last_n]:
            old.unlink(missing_ok=True)
```

### 5. Model Validation Before Registration
```python
import torch.nn as nn

class ModelValidator:
    def __init__(self, expected_architecture: str, reference_state_dict: dict):
        self.expected_architecture = expected_architecture
        self.reference_keys = set(reference_state_dict.keys())
        self.reference_shapes = {k: v.shape for k, v in reference_state_dict.items()}

    def validate(self, state_dict: dict, metrics: dict) -> list[str]:
        """Returns list of validation errors (empty = valid)."""
        errors = []
        
        # 1. Key matching
        candidate_keys = set(state_dict.keys())
        missing = self.reference_keys - candidate_keys
        extra = candidate_keys - self.reference_keys
        if missing:
            errors.append(f"Missing layers: {missing}")
        if extra:
            errors.append(f"Unexpected layers: {extra}")
        
        # 2. Shape matching
        for key in self.reference_keys & candidate_keys:
            if state_dict[key].shape != self.reference_shapes[key]:
                errors.append(
                    f"Shape mismatch '{key}': "
                    f"expected {self.reference_shapes[key]}, got {state_dict[key].shape}"
                )
        
        # 3. NaN/Inf check
        for key, tensor in state_dict.items():
            if torch.isnan(tensor).any():
                errors.append(f"NaN values in layer '{key}'")
            if torch.isinf(tensor).any():
                errors.append(f"Inf values in layer '{key}'")
        
        # 4. Metrics sanity
        if "val_loss" in metrics and metrics["val_loss"] < 0:
            errors.append(f"Negative validation loss: {metrics['val_loss']}")
        if "val_accuracy" in metrics:
            acc = metrics["val_accuracy"]
            if not (0.0 <= acc <= 1.0):
                errors.append(f"Accuracy out of range [0,1]: {acc}")
        
        return errors
```

---

## COMMON MODEL MANAGEMENT BUGS & FIXES

| Bug | Symptom | Fix |
|-----|---------|-----|
| No checksum verification on load | Corrupt model silently used | Always verify SHA-256 on `load()` |
| Non-atomic checkpoint write | Corrupt checkpoint if crash | Write `.tmp` then rename (POSIX atomic) |
| Non-atomic promotion swap | Two production models at once | Use DB transaction with `begin()` |
| Missing NaN check before register | NaN model enters production | Run `ModelValidator` before `registry.register()` |
| Checkpoint keeps growing | Disk full | `_prune_old_checkpoints()` with `keep_last_n` |
| Version collision | Duplicate key DB error | Check existence before save |
| S3 large file timeout | Upload fails on big models | Use multipart upload for files >100MB |
| Load on CPU, train on GPU | Shape OK but CUDA error | Always specify `map_location="cpu"` then `.to(device)` |

---

## TESTING CHECKLIST
- [ ] Register model → verify checksum in DB matches computed
- [ ] Load corrupted model bytes → IntegrityError raised
- [ ] Promote version → old version archived atomically
- [ ] Rollback → previous archived version becomes production
- [ ] Checkpoint survive process crash (atomic write)
- [ ] Validator catches NaN, shape mismatch, missing layers
- [ ] Prune keeps exactly N checkpoints
- [ ] S3 backend: verify storage and retrieval are byte-identical
- [ ] Concurrent promotions don't create two production models