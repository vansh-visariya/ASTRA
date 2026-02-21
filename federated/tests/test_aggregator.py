"""
Unit tests for robust aggregation methods.
"""

import numpy as np
import pytest

from federated.robust_aggregation import (
    trimmed_mean,
    coordinate_median,
    hybrid_aggregator
)


class TestTrimmedMean:
    """Tests for trimmed mean aggregator."""
    
    def test_single_update(self):
        """Test trimmed mean with single update."""
        updates = [np.array([1.0, 2.0, 3.0])]
        result = trimmed_mean(updates, 0.1)
        
        np.testing.assert_array_almost_equal(result, updates[0])
    
    def test_identical_updates(self):
        """Test trimmed mean with identical updates."""
        updates = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0])
        ]
        result = trimmed_mean(updates, 0.1)
        
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])
    
    def test_trim_ratio_near_half(self):
        """Test trimmed mean with trim ratio near 0.5."""
        updates = [
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]),
            np.array([100.0, 200.0, 300.0])
        ]
        
        result = trimmed_mean(updates, 0.4)
        
        assert result is not None
        assert len(result) == 3
    
    def test_empty_updates(self):
        """Test trimmed mean with empty updates."""
        result = trimmed_mean([], 0.1)
        
        assert len(result) == 0


class TestCoordinateMedian:
    """Tests for coordinate-wise median aggregator."""
    
    def test_single_update(self):
        """Test median with single update."""
        updates = [np.array([1.0, 2.0, 3.0])]
        result = coordinate_median(updates)
        
        np.testing.assert_array_almost_equal(result, updates[0])
    
    def test_multiple_updates(self):
        """Test median with multiple updates."""
        updates = [
            np.array([1.0, 4.0]),
            np.array([2.0, 5.0]),
            np.array([3.0, 6.0])
        ]
        result = coordinate_median(updates)
        
        np.testing.assert_array_almost_equal(result, [2.0, 5.0])
    
    def test_consistency_with_numpy(self):
        """Test consistency with numpy median."""
        updates = [
            np.random.randn(100),
            np.random.randn(100),
            np.random.randn(100),
            np.random.randn(100),
            np.random.randn(100)
        ]
        
        result = coordinate_median(updates)
        
        expected = np.median(np.array(updates), axis=0)
        
        np.testing.assert_array_almost_equal(result, expected)


class TestHybridAggregator:
    """Tests for hybrid robust aggregator."""
    
    def test_basic_functionality(self):
        """Test basic hybrid aggregation."""
        updates = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),
            np.array([0.9, 1.9, 2.9])
        ]
        trust_scores = [1.0, 1.0, 1.0]
        staleness_weights = [1.0, 1.0, 1.0]
        
        config = {
            'robust': {
                'method': 'hybrid',
                'trim_ratio': 0.1,
                'norm_clip': 10.0,
                'anomaly_k': 3.0,
                'sim_threshold': 0.2,
                'trust_power': 1.0
            },
            'trust': {
                'init': 1.0,
                'update_alpha': 0.3,
                'quarantine_threshold': 0.35,
                'soft_decay': 0.8
            }
        }
        
        result = hybrid_aggregator(updates, trust_scores, staleness_weights, config)
        
        assert result is not None
        assert len(result) == 3
    
    def test_staleness_weighting(self):
        """Test that staleness weighting affects results."""
        updates = [
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0])
        ]
        trust_scores = [1.0, 1.0]
        
        staleness_high = [1.0, 1.0]
        staleness_low = [0.1, 1.0]
        
        config = {
            'robust': {
                'method': 'hybrid',
                'trim_ratio': 0.1,
                'norm_clip': 100.0,
                'anomaly_k': 5.0,
                'sim_threshold': 0.0,
                'trust_power': 1.0
            },
            'trust': {
                'init': 1.0,
                'update_alpha': 0.3,
                'quarantine_threshold': 0.35,
                'soft_decay': 0.8
            }
        }
        
        result_high = hybrid_aggregator(updates, trust_scores, staleness_high, config)
        result_low = hybrid_aggregator(updates, trust_scores, staleness_low, config)
        
        assert not np.array_equal(result_high, result_low)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
