"""
Unit tests for compression methods.
"""

import numpy as np
import pytest

from federated.compression import (
    topk_sparsify,
    decompress_topk,
    quantize_vector,
    decompress_quantize
)


class TestTopKSparsification:
    """Tests for top-k sparsification."""
    
    def test_preserves_k_ratio(self):
        """Test that k ratio is approximately preserved."""
        vector = np.random.randn(1000)
        
        k_ratio = 0.1
        sparse, metadata = topk_sparsify(vector, k_ratio)
        
        expected_k = int(1000 * k_ratio)
        
        actual_sparse = np.sum(sparse != 0)
        
        assert actual_sparse <= expected_k + 1
    
    def test_keeps_largest_magnitudes(self):
        """Test that largest magnitude elements are kept."""
        vector = np.array([1.0, 100.0, 2.0, 99.0, 3.0])
        
        sparse, _ = topk_sparsify(vector, 0.4)
        
        non_zero_indices = np.where(sparse != 0)[0]
        
        assert 1 in non_zero_indices
        assert 3 in non_zero_indices
    
    def test_compression_ratio(self):
        """Test compression ratio in metadata."""
        vector = np.random.randn(1000)
        
        k_ratio = 0.1
        sparse, metadata = topk_sparsify(vector, k_ratio)
        
        assert 'compression_ratio' in metadata
        assert metadata['compression_ratio'] > 5.0
    
    def test_empty_vector(self):
        """Test with empty vector."""
        vector = np.array([])
        
        sparse, metadata = topk_sparsify(vector, 0.1)
        
        assert len(sparse) == 0


class TestQuantization:
    """Tests for quantization."""
    
    def test_reduces_unique_values(self):
        """Test that quantization reduces unique values."""
        vector = np.random.randn(1000)
        
        bits = 4
        quantized, metadata = quantize_vector(vector, bits)
        
        assert len(np.unique(quantized)) <= 2 ** bits
    
    def test_preserves_shape(self):
        """Test that quantization preserves shape."""
        vector = np.random.randn(100, 50)
        
        quantized, _ = quantize_vector(vector, 8)
        
        assert quantized.shape == vector.shape
    
    def test_roundtrip(self):
        """Test quantization and dequantization."""
        vector = np.random.randn(100)
        
        bits = 8
        quantized, metadata = quantize_vector(vector, bits)
        decompressed = decompress_quantize(quantized, metadata)
        
        assert decompressed.shape == vector.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
