"""Privacy — DP-SGD and malicious client simulator."""

from astra.core.privacy.malicious_simulator import MaliciousSimulator
from astra.core.privacy.privacy import clip_and_noise

__all__ = [
    "clip_and_noise",
    "MaliciousSimulator",
]
