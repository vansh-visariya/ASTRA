# SKILL: Federated Learning Core Engine

## PURPOSE
Guide for implementing production-ready federated learning algorithms, aggregation strategies, and client/server coordination logic — bug-free and optimized.

---

## ARCHITECTURE OVERVIEW

```
fl_core/
├── server/
│   ├── aggregator.py          # FedAvg, FedProx, FedNova, SCAFFOLD
│   ├── round_manager.py       # Round orchestration, timeout, retry
│   ├── client_registry.py     # Client tracking, health, capabilities
│   └── strategy.py            # Pluggable strategy interface
├── client/
│   ├── trainer.py             # Local training loop
│   ├── model_diff.py          # Delta compression, gradient clipping
│   └── privacy.py             # DP noise injection, clipping
├── common/
│   ├── model_utils.py         # Serialization, versioning, checksums
│   ├── metrics.py             # Loss, accuracy, convergence tracking
│   └── exceptions.py          # Custom FL exceptions
└── config/
    └── fl_config.py           # Dataclass-based configuration
```

---

## CRITICAL RULES — ALWAYS FOLLOW

### 1. Model Parameter Handling
```python
# ✅ CORRECT — Deep copy always, never reference shared state
import copy
global_weights = copy.deepcopy(model.state_dict())

# ❌ WRONG — Mutations will corrupt the global model
global_weights = model.state_dict()
```

### 2. FedAvg Implementation (Reference Standard)
```python
def federated_average(client_updates: list[dict], client_sizes: list[int]) -> dict:
    """
    Weighted FedAvg: w_global = Σ (n_k / n_total) * w_k
    
    Args:
        client_updates: List of state_dicts from each client
        client_sizes: Number of local training samples per client
    Returns:
        Aggregated global model state_dict
    """
    total_samples = sum(client_sizes)
    if total_samples == 0:
        raise ValueError("Total sample count cannot be zero")
    
    aggregated = {}
    for key in client_updates[0].keys():
        # Validate all clients have same architecture
        shapes = [u[key].shape for u in client_updates]
        if len(set(str(s) for s in shapes)) > 1:
            raise ValueError(f"Shape mismatch for layer '{key}': {shapes}")
        
        weighted_sum = sum(
            (n / total_samples) * update[key].float()
            for update, n in zip(client_updates, client_sizes)
        )
        aggregated[key] = weighted_sum
    
    return aggregated
```

### 3. Round Manager — Async, with Timeouts
```python
import asyncio
from dataclasses import dataclass, field
from enum import Enum

class RoundStatus(Enum):
    WAITING    = "waiting"
    COLLECTING = "collecting"
    AGGREGATING = "aggregating"
    COMPLETE   = "complete"
    FAILED     = "failed"

@dataclass
class RoundConfig:
    round_id: int
    min_clients: int = 2           # Minimum for aggregation
    target_clients: int = 10
    client_timeout_sec: float = 300.0
    stragglers_fraction: float = 0.2  # Accept if 80%+ respond

class RoundManager:
    def __init__(self, config: RoundConfig):
        self.config = config
        self.status = RoundStatus.WAITING
        self._updates: dict[str, dict] = {}   # client_id -> update
        self._lock = asyncio.Lock()

    async def collect_update(self, client_id: str, update: dict, sample_count: int):
        async with self._lock:
            if self.status != RoundStatus.COLLECTING:
                raise RoundStateError(f"Round not accepting updates: {self.status}")
            self._updates[client_id] = {"update": update, "samples": sample_count}

    async def wait_for_round(self) -> dict:
        """Wait with timeout; accept stragglers cutoff."""
        deadline = asyncio.get_event_loop().time() + self.config.client_timeout_sec
        target = self.config.target_clients * (1 - self.config.stragglers_fraction)
        
        while True:
            async with self._lock:
                n = len(self._updates)
                if n >= self.config.target_clients:
                    break
                if n >= max(self.config.min_clients, target):
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
            await asyncio.sleep(0.5)
        
        if len(self._updates) < self.config.min_clients:
            self.status = RoundStatus.FAILED
            raise InsufficientClientsError(
                f"Got {len(self._updates)}/{self.config.min_clients} clients"
            )
        return self._updates
```

### 4. Privacy — Differential Privacy (DP-SGD)
```python
from torch import nn
import torch

class DPTrainer:
    """
    Gaussian mechanism DP. Always clip then noise.
    Never add noise before clipping — order matters!
    """
    def __init__(self, model: nn.Module, max_grad_norm: float, noise_multiplier: float):
        self.model = model
        self.max_grad_norm = max_grad_norm       # L2 sensitivity
        self.noise_multiplier = noise_multiplier  # σ

    def clip_gradients(self):
        """Per-sample gradient clipping."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        clip_coeff = min(1.0, self.max_grad_norm / (total_norm + 1e-8))
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coeff)

    def add_noise(self):
        """Add calibrated Gaussian noise to gradients."""
        std = self.noise_multiplier * self.max_grad_norm
        for p in self.model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * std
                p.grad.data.add_(noise)

    def step(self, optimizer):
        self.clip_gradients()  # 1. Clip FIRST
        self.add_noise()       # 2. Then noise
        optimizer.step()       # 3. Then update
        optimizer.zero_grad()
```

### 5. Model Serialization — Safe & Versioned
```python
import hashlib, io, torch

def serialize_model(state_dict: dict, metadata: dict = None) -> bytes:
    """Serialize with checksum for integrity verification."""
    buffer = io.BytesIO()
    payload = {
        "state_dict": state_dict,
        "metadata": metadata or {},
        "version": "1.0"
    }
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    checksum = hashlib.sha256(raw).hexdigest()
    return checksum.encode() + b"|" + raw   # Prepend checksum

def deserialize_model(data: bytes) -> tuple[dict, dict]:
    """Deserialize with integrity check. Raises on tamper."""
    checksum_str, raw = data.split(b"|", 1)
    expected = hashlib.sha256(raw).hexdigest()
    if checksum_str.decode() != expected:
        raise IntegrityError("Model checksum mismatch — possible corruption or tampering")
    buffer = io.BytesIO(raw)
    payload = torch.load(buffer, map_location="cpu", weights_only=True)
    return payload["state_dict"], payload["metadata"]
```

---

## COMMON BUGS & FIXES

| Bug | Symptom | Fix |
|-----|---------|-----|
| Shared reference to global model | All clients train same (modified) weights | `copy.deepcopy()` before distributing |
| float16 overflow in aggregation | NaN weights after round | Cast to float32 before aggregation |
| Gradient explosion | Loss → NaN | Clip grads; check LR |
| Stale client update | Old-round update accepted | Tag updates with `round_id`, reject mismatches |
| Missing min_clients check | Aggregate with 1 client = no federation | Enforce `min_clients >= 2` |
| No timeout on client collection | Server hangs forever | Always use `asyncio.wait_for()` |
| Integer overflow in sample weighting | Wrong aggregation on large datasets | Use `float(n)` in weighting math |

---

## AGGREGATION STRATEGY INTERFACE
```python
from abc import ABC, abstractmethod

class AggregationStrategy(ABC):
    """All strategies implement this interface — plug-and-play."""
    
    @abstractmethod
    def aggregate(
        self,
        updates: list[dict],        # client state_dicts
        weights: list[float],       # relative weights (sum to 1.0)
        global_model: dict,         # current global state_dict
        round_num: int
    ) -> dict:
        """Return new global state_dict."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str: ...

# Register strategies
STRATEGIES: dict[str, type[AggregationStrategy]] = {}

def register_strategy(cls):
    STRATEGIES[cls.name] = cls
    return cls

@register_strategy
class FedAvgStrategy(AggregationStrategy):
    name = "fedavg"
    def aggregate(self, updates, weights, global_model, round_num):
        result = {}
        for key in updates[0]:
            result[key] = sum(w * u[key].float() for u, w in zip(updates, weights))
        return result
```

---

## CONVERGENCE MONITORING

```python
@dataclass
class RoundMetrics:
    round_id: int
    num_clients: int
    avg_train_loss: float
    avg_train_accuracy: float
    global_val_loss: float | None
    global_val_accuracy: float | None
    round_duration_sec: float
    timestamp: str  # ISO 8601

class ConvergenceDetector:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._history: list[float] = []
        self._best: float = float("inf")
        self._no_improve_count: int = 0

    def update(self, val_loss: float) -> bool:
        """Returns True if training should stop (converged)."""
        self._history.append(val_loss)
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
        return self._no_improve_count >= self.patience
```

---

## CONFIGURATION TEMPLATE
```python
from dataclasses import dataclass

@dataclass
class FLConfig:
    # Rounds
    num_rounds: int = 100
    min_clients_per_round: int = 2
    target_clients_per_round: int = 10
    client_fraction: float = 0.1          # C in FedAvg paper
    
    # Local Training
    local_epochs: int = 5
    local_batch_size: int = 32
    local_learning_rate: float = 0.01
    
    # Aggregation
    strategy: str = "fedavg"             # "fedavg" | "fedprox" | "scaffold"
    
    # Privacy
    dp_enabled: bool = False
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 1.1
    dp_delta: float = 1e-5
    
    # Timeouts
    client_timeout_sec: float = 300.0
    round_timeout_sec: float = 600.0
    
    # Model
    model_version: str = "1.0.0"
    checkpoint_every_n_rounds: int = 10
```

---

## TESTING CHECKLIST
- [ ] FedAvg with 2, 10, 100 clients — verify weight correctness
- [ ] Straggler tolerance — kill clients mid-round
- [ ] DP: verify privacy budget (ε, δ) accounting
- [ ] Serialization round-trip: serialize → deserialize → identical weights
- [ ] Checksum tamper detection
- [ ] Round timeout fires and fails gracefully
- [ ] min_clients=1 is rejected
- [ ] float16 model trained correctly via float32 aggregation cast
- [ ] Convergence detector fires at right patience