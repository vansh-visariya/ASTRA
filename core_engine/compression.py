"""
Communication Compression for Federated Learning.

Implements gradient compression techniques:
- Top-k sparsification
- Quantization (optional)

References:
- Sattler et al., "Sparse and Ternary Communication for Efficient Distributed Learning"
- Aji & Heafield, "Sparse Communication for Distributed Gradient Descent"
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "topk_sparsify",
    "decompress_topk",
    "quantize_vector",
    "decompress_quantize",
    "CompressionStats",
    "apply_compression",
]


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


def decompress_topk(
    sparse_vector: np.ndarray,
    metadata: dict[str, Any],
    full_size: int,
) -> np.ndarray:
    """
    Decompress top-k sparse vector.

    Args:
        sparse_vector: Sparse vector.
        metadata: Compression metadata.
        full_size: Original full size.

    Returns:
        Decompressed full-size vector.
    """
    if len(sparse_vector) == full_size:
        return sparse_vector

    full_vector = np.zeros(full_size, dtype=sparse_vector.dtype)

    k = metadata.get("k", int(full_size * metadata.get("k_ratio", 0.1)))

    if k > 0 and len(sparse_vector) > 0:
        topk_indices = np.argpartition(np.abs(sparse_vector), -k)[-k:]
        full_vector[topk_indices] = sparse_vector[topk_indices]

    return full_vector


def quantize_vector(
    vector: np.ndarray,
    bits: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Quantize vector to specified bit depth.

    Args:
        vector: Input vector.
        bits: Number of bits for quantization.

    Returns:
        Tuple of (quantized_vector, metadata).

    Raises:
        ValueError: If bits is invalid.
    """
    if bits < 1 or bits > 32:
        raise ValueError("bits must be between 1 and 32")

    vmin = np.min(vector)
    vmax = np.max(vector)

    levels = 2**bits

    normalized = (vector - vmin) / (vmax - vmin + 1e-8)

    quantized = np.round(normalized * (levels - 1)).astype(np.int32)

    dequantized = quantized / (levels - 1) * (vmax - vmin) + vmin

    metadata = {
        "bits": bits,
        "levels": levels,
        "vmin": float(vmin),
        "vmax": float(vmax),
    }

    return dequantized, metadata


def decompress_quantize(
    quantized: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    """
    Decompress quantized vector.

    Args:
        quantized: Quantized vector.
        metadata: Compression metadata.

    Returns:
        Decompressed vector.
    """
    levels = metadata["levels"]
    vmin = metadata["vmin"]
    vmax = metadata["vmax"]

    normalized = quantized / (levels - 1)

    return normalized * (vmax - vmin) + vmin


class CompressionStats:
    """Track compression statistics."""

    def __init__(self) -> None:
        """Initialize compression stats."""
        self.total_original_bytes = 0
        self.total_compressed_bytes = 0
        self.num_compressions = 0

    def record(self, original_size: int, compressed_size: int) -> None:
        """
        Record compression statistics.

        Args:
            original_size: Original vector size.
            compressed_size: Compressed vector size.
        """
        self.total_original_bytes += original_size * 4
        self.total_compressed_bytes += compressed_size * 4
        self.num_compressions += 1

    def get_average_ratio(self) -> float:
        """Get average compression ratio."""
        if self.total_original_bytes == 0:
            return 1.0
        return self.total_original_bytes / max(self.total_compressed_bytes, 1)

    def reset(self) -> None:
        """Reset statistics."""
        self.total_original_bytes = 0
        self.total_compressed_bytes = 0
        self.num_compressions = 0


def apply_compression(
    vector: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply configured compression method.

    Args:
        vector: Input vector.
        config: Configuration dictionary.

    Returns:
        Tuple of (compressed_vector, metadata).
    """
    method = config.get("communication", {}).get("compression", "none")

    if method == "topk":
        k_ratio = config.get("communication", {}).get("topk_ratio", 0.1)
        return topk_sparsify(vector, k_ratio)
    if method == "quantize":
        bits = config.get("communication", {}).get("quantize_bits", 8)
        return quantize_vector(vector, bits)
    return vector, {"method": "none"}
