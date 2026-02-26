"""
Communication Compression for Federated Learning.

Implements gradient compression techniques:
- Top-k sparsification
- Quantization (optional)

References:
- Sattler et al., "Sparse and Ternary Communication for Efficient Distributed Learning"
- Aji & Heafield, "Sparse Communication for Distributed Gradient Descent"
"""

from typing import Any, Dict, List, Tuple

import numpy as np


def topk_sparsify(
    vector: np.ndarray,
    k_ratio: float
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Top-k sparsification: keep only k% largest magnitude elements.
    
    Args:
        vector: Input vector.
        k_ratio: Fraction of elements to keep (0 to 1).
    
    Returns:
        Tuple of (sparse_vector, metadata).
    """
    k = max(1, int(len(vector) * k_ratio))
    
    magnitudes = np.abs(vector)
    threshold = np.sort(magnitudes)[-k] if k < len(vector) else 0
    
    mask = magnitudes >= threshold
    
    sparse_vector = np.where(mask, vector, 0.0)
    
    metadata = {
        'k_ratio': k_ratio,
        'k': k,
        'original_size': len(vector),
        'compressed_size': np.sum(mask),
        'compression_ratio': len(vector) / max(np.sum(mask), 1),
        'threshold': float(threshold)
    }
    
    return sparse_vector, metadata


def decompress_topk(
    sparse_vector: np.ndarray,
    metadata: Dict[str, Any],
    full_size: int
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
    return sparse_vector


def quantize_vector(
    vector: np.ndarray,
    bits: int = 8
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Quantize vector to specified bit depth.
    
    Args:
        vector: Input vector.
        bits: Number of bits for quantization.
    
    Returns:
        Tuple of (quantized_vector, metadata).
    """
    vmin = np.min(vector)
    vmax = np.max(vector)
    
    levels = 2 ** bits
    
    normalized = (vector - vmin) / (vmax - vmin + 1e-8)
    
    quantized = np.round(normalized * (levels - 1)).astype(np.int32)
    
    dequantized = quantized / (levels - 1) * (vmax - vmin) + vmin
    
    metadata = {
        'bits': bits,
        'levels': levels,
        'vmin': float(vmin),
        'vmax': float(vmax)
    }
    
    return dequantized, metadata


def decompress_quantize(
    quantized: np.ndarray,
    metadata: Dict[str, Any]
) -> np.ndarray:
    """Decompress quantized vector."""
    levels = metadata['levels']
    vmin = metadata['vmin']
    vmax = metadata['vmax']
    
    normalized = quantized / (levels - 1)
    
    return normalized * (vmax - vmin) + vmin


class CompressionStats:
    """Track compression statistics."""
    
    def __init__(self):
        self.total_original_bytes = 0
        self.total_compressed_bytes = 0
        self.num_compressions = 0
    
    def record(self, original_size: int, compressed_size: int) -> None:
        """Record compression statistics."""
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
    config: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply configured compression method.
    
    Args:
        vector: Input vector.
        config: Configuration dictionary.
    
    Returns:
        Tuple of (compressed_vector, metadata).
    """
    method = config.get('communication', {}).get('compression', 'none')
    
    if method == 'topk':
        k_ratio = config.get('communication', {}).get('topk_ratio', 0.1)
        return topk_sparsify(vector, k_ratio)
    elif method == 'quantize':
        bits = config.get('communication', {}).get('quantize_bits', 8)
        return quantize_vector(vector, bits)
    else:
        return vector, {'method': 'none'}
