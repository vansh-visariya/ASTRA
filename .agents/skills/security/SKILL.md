# SKILL: Security & Privacy for Federated Learning

## PURPOSE
Implement production-grade security: authentication, authorization, mTLS, differential privacy budget tracking, Byzantine fault detection, and secure aggregation.

---

## THREAT MODEL

```
Threats to defend against:
1. Eavesdropping            → mTLS encrypts all traffic
2. Unauthorized clients     → JWT + client certificate auth
3. Byzantine clients        → Anomaly detection on updates
4. Model inversion attacks  → Differential Privacy (DP)
5. Membership inference     → DP + secure aggregation
6. Gradient poisoning       → Norm clipping + detection
7. Model theft              → Signed model distribution
8. MITM on model download   → Checksum verification
9. Replay attacks           → Round-bound tokens + nonces
```

---

## AUTHENTICATION SYSTEM

### JWT Token Management
```python
from datetime import datetime, timedelta
from typing import Optional
import jwt
import secrets
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self._revoked_tokens: set[str] = set()  # In prod: use Redis

    def create_client_token(
        self,
        client_id: str,
        expires_in_hours: float = 24.0,
        scopes: list[str] = None,
    ) -> str:
        now = datetime.utcnow()
        payload = {
            "sub": client_id,
            "iat": now,
            "exp": now + timedelta(hours=expires_in_hours),
            "jti": secrets.token_hex(16),   # JWT ID — unique per token (for revocation)
            "scopes": scopes or ["train"],  # "train" | "admin" | "readonly"
            "type": "client",
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict:
        """Raises on invalid/expired token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"require": ["exp", "iat", "sub", "jti"]},
            )
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired — re-register")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid token: {e}")
        
        if payload["jti"] in self._revoked_tokens:
            raise TokenRevokedError("Token has been revoked")
        
        return payload

    def revoke_token(self, jti: str) -> None:
        """Revoke a token (e.g., on client ban or logout)."""
        self._revoked_tokens.add(jti)

    def require_scope(self, payload: dict, required_scope: str) -> None:
        if required_scope not in payload.get("scopes", []):
            raise InsufficientScopeError(
                f"Scope '{required_scope}' required, have: {payload.get('scopes')}"
            )
```

### Client Registration with Rate Limiting
```python
import time
from collections import defaultdict

class RegistrationRateLimiter:
    """Prevent registration floods (Sybil attack mitigation)."""
    
    def __init__(self, max_per_minute: int = 10, max_per_ip: int = 50):
        self.max_per_minute = max_per_minute
        self.max_per_ip = max_per_ip
        self._ip_timestamps: dict[str, list[float]] = defaultdict(list)
        self._ip_total: dict[str, int] = defaultdict(int)

    def check(self, client_ip: str) -> None:
        """Raises RateLimitError if exceeded."""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Clean old timestamps
        self._ip_timestamps[client_ip] = [
            t for t in self._ip_timestamps[client_ip] if t > window_start
        ]
        
        if len(self._ip_timestamps[client_ip]) >= self.max_per_minute:
            raise RateLimitError(f"Too many registrations from {client_ip}")
        
        self._ip_total[client_ip] += 1
        if self._ip_total[client_ip] > self.max_per_ip:
            raise RateLimitError(f"IP {client_ip} exceeded total registration limit")
        
        self._ip_timestamps[client_ip].append(now)
```

---

## DIFFERENTIAL PRIVACY BUDGET TRACKING

### Privacy Accountant (Rényi DP)
```python
from dataclasses import dataclass

@dataclass
class PrivacyBudget:
    epsilon: float   # Privacy loss ε
    delta: float     # Failure probability δ
    spent: float = 0.0  # ε spent so far

class PrivacyAccountant:
    """
    Track (ε, δ)-DP budget across FL rounds using Rényi DP moments accountant.
    Refuse training when budget exhausted.
    """
    
    def __init__(self, total_epsilon: float, delta: float):
        self.total_epsilon = total_epsilon
        self.delta = delta
        self._spent_epsilon = 0.0
        self._round_history: list[dict] = []

    def compute_epsilon_per_round(
        self,
        noise_multiplier: float,
        sample_rate: float,           # q = batch_size / dataset_size
        num_steps: int,               # local_epochs * steps_per_epoch
    ) -> float:
        """
        Gaussian mechanism epsilon estimate.
        For exact accounting use Google's dp_accounting library.
        Approximation: ε ≈ q * sqrt(2 * num_steps * log(1/δ)) / noise_multiplier
        """
        import math
        eps = sample_rate * math.sqrt(2 * num_steps * math.log(1 / self.delta)) / noise_multiplier
        return eps

    def charge(self, epsilon: float, round_id: int) -> PrivacyBudget:
        """Charge epsilon to budget. Raises if budget exceeded."""
        if self._spent_epsilon + epsilon > self.total_epsilon:
            raise PrivacyBudgetExhausted(
                f"Charging ε={epsilon:.4f} would exceed budget "
                f"(spent={self._spent_epsilon:.4f}, total={self.total_epsilon:.4f})"
            )
        self._spent_epsilon += epsilon
        self._round_history.append({
            "round_id": round_id,
            "epsilon_charged": epsilon,
            "total_spent": self._spent_epsilon,
            "remaining": self.total_epsilon - self._spent_epsilon,
        })
        return PrivacyBudget(
            epsilon=self.total_epsilon,
            delta=self.delta,
            spent=self._spent_epsilon,
        )

    @property
    def remaining_epsilon(self) -> float:
        return self.total_epsilon - self._spent_epsilon

    @property
    def is_exhausted(self) -> bool:
        return self._spent_epsilon >= self.total_epsilon * 0.95  # 95% threshold warning
```

---

## BYZANTINE FAULT DETECTION

### Anomaly Detection on Client Updates
```python
import torch
import numpy as np

class ByzantineDetector:
    """
    Detect and exclude potentially malicious/faulty client updates.
    Uses multiple detection methods — run all, flag if any trigger.
    """
    
    def __init__(
        self,
        norm_threshold_multiplier: float = 3.0,  # Flag if norm > mean + 3*std
        cosine_threshold: float = -0.5,           # Flag if cosine sim to median < -0.5
    ):
        self.norm_threshold_multiplier = norm_threshold_multiplier
        self.cosine_threshold = cosine_threshold

    def detect(
        self,
        updates: list[dict],      # client state_dicts (deltas from global)
        client_ids: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Returns (clean_client_ids, flagged_client_ids).
        """
        # Flatten updates to vectors for analysis
        vectors = [self._flatten(u) for u in updates]
        
        flagged = set()
        
        # Method 1: L2 norm outlier detection
        norms = np.array([v.norm().item() for v in vectors])
        mean_norm = norms.mean()
        std_norm  = norms.std() + 1e-8
        threshold = mean_norm + self.norm_threshold_multiplier * std_norm
        for i, (cid, norm) in enumerate(zip(client_ids, norms)):
            if norm > threshold:
                flagged.add(cid)
                print(f"[BYZANTINE] Client {cid} flagged: L2 norm {norm:.4f} > threshold {threshold:.4f}")

        # Method 2: Cosine similarity to geometric median
        median_vec = self._geometric_median(vectors)
        for cid, vec in zip(client_ids, vectors):
            cos_sim = torch.nn.functional.cosine_similarity(
                vec.unsqueeze(0), median_vec.unsqueeze(0)
            ).item()
            if cos_sim < self.cosine_threshold:
                flagged.add(cid)
                print(f"[BYZANTINE] Client {cid} flagged: cosine_sim={cos_sim:.4f} < {self.cosine_threshold}")

        clean = [cid for cid in client_ids if cid not in flagged]
        return clean, list(flagged)

    def _flatten(self, state_dict: dict) -> torch.Tensor:
        return torch.cat([v.float().flatten() for v in state_dict.values()])

    def _geometric_median(self, vectors: list[torch.Tensor], max_iter: int = 100) -> torch.Tensor:
        """Weiszfeld algorithm — robust mean under Byzantine attacks."""
        estimate = torch.stack(vectors).mean(dim=0)
        for _ in range(max_iter):
            dists = torch.stack([torch.dist(estimate, v) for v in vectors])
            dists = torch.clamp(dists, min=1e-8)
            weights = 1.0 / dists
            weights /= weights.sum()
            estimate = sum(w * v for w, v in zip(weights, vectors))
        return estimate
```

---

## SECURE AGGREGATION (MASKING PROTOCOL)

```python
"""
Secure Aggregation: Clients add pairwise random masks before sending.
Server aggregates masked updates — individual updates are never revealed.

Protocol (simplified Bonawitz et al. 2017):
1. Each pair (i,j) agrees on random seed s_ij
2. Client i adds  Σ_j PRG(s_ij) if j > i, else -PRG(s_ij)
3. Masks cancel out in the sum → server gets unmasked aggregate
"""

import torch
import hashlib

def generate_pairwise_mask(seed: bytes, shape_total: int) -> torch.Tensor:
    """Deterministic PRG from seed."""
    g = torch.Generator()
    # Use SHA-256 hash of seed as generator seed (fits in 64-bit int)
    seed_int = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")
    g.manual_seed(seed_int)
    return torch.randn(shape_total, generator=g)

def apply_mask(flat_update: torch.Tensor, client_id: str, peer_ids: list[str], round_id: int) -> torch.Tensor:
    """Add pairwise masks to update."""
    masked = flat_update.clone()
    for peer_id in peer_ids:
        # Deterministic seed from the pair (order-independent)
        pair = tuple(sorted([client_id, peer_id]))
        seed = f"{pair[0]}:{pair[1]}:{round_id}".encode()
        mask = generate_pairwise_mask(seed, flat_update.numel())
        # Canonical direction: add if my ID is "greater"
        if client_id > peer_id:
            masked += mask
        else:
            masked -= mask
    return masked

# Server: masks cancel in sum — no individual update revealed
# Requires ALL clients to complete (drop-resistant version uses secret sharing)
```

---

## INPUT VALIDATION — ALWAYS VALIDATE BEFORE USE

```python
from pydantic import BaseModel, Field, validator
import re

class ClientRegistrationRequest(BaseModel):
    client_id: str = Field(min_length=8, max_length=64)
    dataset_size: int = Field(gt=0, le=10_000_000)  # Sanity: max 10M samples
    device_info: str = Field(max_length=256)

    @validator("client_id")
    def validate_client_id(cls, v):
        # Only alphanumeric + hyphens + underscores
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("client_id may only contain alphanumeric, hyphen, underscore")
        # Prevent SQL injection / path traversal (belt-and-suspenders)
        if any(bad in v for bad in ["../", ";", "'", '"', "<", ">"]):
            raise ValueError("Invalid characters in client_id")
        return v

def validate_model_update(update: dict, max_norm: float = 100.0) -> None:
    """Validate incoming client update before aggregation."""
    # 1. Must be a dict
    if not isinstance(update, dict):
        raise ValidationError("Update must be a dict of tensors")
    
    # 2. Check for NaN/Inf in each layer
    for key, tensor in update.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"Layer '{key}' is not a tensor")
        if torch.isnan(tensor).any():
            raise ValidationError(f"NaN in layer '{key}' — client may be unstable")
        if torch.isinf(tensor).any():
            raise ValidationError(f"Inf in layer '{key}' — gradient explosion")
    
    # 3. Norm sanity check
    flat = torch.cat([v.float().flatten() for v in update.values()])
    norm = flat.norm().item()
    if norm > max_norm:
        raise ValidationError(f"Update norm {norm:.2f} exceeds max {max_norm} — clipping or rejecting")
```

---

## SECURITY CONFIGURATION CHECKLIST

```python
@dataclass
class SecurityConfig:
    # TLS
    tls_enabled: bool = True          # NEVER disable in production
    mtls_enabled: bool = False         # Enable for high-security deployments
    tls_cert_path: str = "certs/server.crt"
    tls_key_path: str = "certs/server.key"
    tls_ca_path: str = "certs/ca.crt"
    
    # Auth
    jwt_secret: str = ""               # MUST be set via env var, never hardcoded
    jwt_expiry_hours: float = 24.0
    max_clients_per_ip: int = 50
    
    # DP
    dp_enabled: bool = True
    dp_total_epsilon: float = 10.0     # Total ε budget
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 1.1
    
    # Byzantine detection
    byzantine_detection: bool = True
    norm_outlier_multiplier: float = 3.0
    cosine_similarity_threshold: float = -0.5
    
    # Rate limiting
    registration_rate_limit: int = 10  # per minute per IP
    update_rate_limit: int = 1         # per round per client (enforce 1 update/round)
```

---

## COMMON SECURITY BUGS & FIXES

| Bug | Vulnerability | Fix |
|-----|--------------|-----|
| JWT secret in source code | Key exposure → token forgery | Load from `os.environ["JWT_SECRET"]` |
| No token revocation | Banned client reuses token | Maintain `revoked_jti` set in Redis |
| Accepting updates from wrong round | Replay / stale update attack | Check `round_id` matches current round |
| No client update rate limit | DoS on aggregation | Max 1 update per client per round |
| TLS disabled in dev, forgotten in prod | Plaintext traffic | Default `tls_enabled=True`; require env override |
| No DP budget exhaustion check | Unlimited privacy leakage | `PrivacyAccountant.charge()` before every round |
| NaN updates silently included | Model poisoning | `validate_model_update()` before aggregation |
| User-controlled client_id | Path traversal / injection | Regex whitelist validation |

---

## TESTING CHECKLIST
- [ ] Expired JWT returns 401 (not 500)
- [ ] Revoked JWT token is rejected
- [ ] Byzantine client with inverted gradients is detected and excluded
- [ ] DP budget exhaustion raises `PrivacyBudgetExhausted`
- [ ] Client submitting update for wrong round is rejected
- [ ] Registration rate limit triggers at 11th request/minute
- [ ] Model with NaN weights is rejected at validation
- [ ] TLS: plaintext connection refused on secure server
- [ ] mTLS: client without cert gets UNAVAILABLE
- [ ] Norm outlier (10x) detected by Byzantine filter