"""
Unit tests for compression methods.
"""

import numpy as np
import pytest

from astra.core.compression import topk_sparsify


class TestTopKSparsification:
    """Tests for top-k sparsification."""

    def test_preserves_k_ratio(self):
        """Test that sparse vector keeps roughly k_ratio non-zero."""
        n = 1000
        vector = np.random.randn(n)
        k_ratio = 0.2

        sparse, metadata = topk_sparsify(vector, k_ratio)

        non_zero = int(np.sum(np.abs(sparse) > 1e-8))
        expected = int(n * k_ratio)

        assert non_zero >= expected * 0.8

    def test_keeps_largest_magnitudes(self):
        """Test that largest elements are preserved."""
        vector = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 100.0])
        sparse, _ = topk_sparsify(vector, 0.4)

        assert sparse[5] == pytest.approx(100.0)
        assert sparse[2] == pytest.approx(3.0)

    def test_compression_ratio(self):
        """Test compression ratio metadata."""
        n = 1000
        vector = np.random.randn(n)
        k_ratio = 0.3

        sparse, metadata = topk_sparsify(vector, k_ratio)

        assert metadata["compression_ratio"] >= (1.0 / k_ratio) * 0.8

    def test_empty_vector(self):
        """Test with nearly zero vector."""
        vector = np.zeros(100)
        sparse, metadata = topk_sparsify(vector, 0.1)

        assert sparse.shape == vector.shape
        assert metadata is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
