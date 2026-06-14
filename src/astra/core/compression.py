"""
Communication Compression for Federated Learning.

Top-k sparsification for gradient compression.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["topk_sparsify"]


def topk_sparsify(
    vector: np.ndarray,
    k_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Top-k sparsification: keep only k% largest magnitude elements.

    Args:
        vector: Input vector.
        k_ratio: Fraction of elements to keep (0 to 1).

    Returns:
        Tuple of (sparse_vector, metadata).

    Raises:
        ValueError: If k_ratio is invalid.
    """
    if not 0 < k_ratio <= 1:
        raise ValueError("k_ratio must be between 0 and 1")

    k = max(1, int(len(vector) * k_ratio))

    magnitudes = np.abs(vector)
    threshold = np.sort(magnitudes)[-k] if k < len(vector) else 0

    mask = magnitudes >= threshold

    sparse_vector = np.where(mask, vector, 0.0)

    metadata = {
        "k_ratio": k_ratio,
        "k": k,
        "original_size": len(vector),
        "compressed_size": int(np.sum(mask)),
        "compression_ratio": len(vector) / max(np.sum(mask), 1),
        "threshold": float(threshold),
    }

    return sparse_vector, metadata
