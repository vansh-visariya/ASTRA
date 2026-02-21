"""
Unit tests for privacy mechanisms.
"""

import numpy as np
import pytest

from federated.privacy import (
    clip_and_noise,
    secure_aggregate_masking,
    estimate_epsilon
)


class TestDifferentialPrivacy:
    """Tests for DP mechanisms."""
    
    def test_clip_and_noise_output_norm(self):
        """Test that output norm is bounded by clipping."""
        gradient = np.random.randn(1000) * 10
        
        clip_norm = 1.0
        sigma = 0.0  # No noise for this test
        
        result = clip_and_noise(gradient, clip_norm, sigma)
        
        result_norm = np.linalg.norm(result)
        
        assert result_norm <= clip_norm * 1.01  # Allow small tolerance
    
    def test_clip_and_noise_noise_added(self):
        """Test that noise is actually added."""
        gradient = np.zeros(1000)
        
        sigma = 1.0
        result = clip_and_noise(gradient, clip_norm=10.0, sigma=sigma)
        
        assert not np.array_equal(result, gradient)
    
    def test_clip_and_noise_deterministic_given_seed(self):
        """Test noise is random (not deterministic)."""
        gradient = np.ones(100)
        
        result1 = clip_and_noise(gradient, clip_norm=10.0, sigma=1.0)
        
        result2 = clip_and_noise(gradient, clip_norm=10.0, sigma=1.0)
        
        assert not np.array_equal(result1, result2)
    
    def test_no_clipping_when_below_threshold(self):
        """Test clipping is applied (gradient is close to original when small)."""
        gradient = np.random.randn(100) * 0.1
        
        clip_norm = 1.0
        sigma = 0.0  # No noise for comparison
        
        result = clip_and_noise(gradient, clip_norm, sigma)
        
        # With sigma=0, result should be close to gradient (already below threshold)
        np.testing.assert_array_almost_equal(result, gradient, decimal=1)


class TestSecureAggregation:
    """Tests for secure aggregation simulation."""
    
    def test_masking_protocol(self):
        """Test secure aggregation masking."""
        updates = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])
        ]
        
        seed = 42
        
        result = secure_aggregate_masking(updates, seed)
        
        assert result is not None
        assert len(result) == 3
    
    def test_masking_hides_individual_updates(self):
        """Test that masking obscures individual updates."""
        update1 = np.array([100.0, 200.0])
        update2 = np.array([1.0, 2.0])
        
        updates = [update1, update2]
        
        result = secure_aggregate_masking(updates, 12345)
        
        assert not np.array_equal(result, update1)
        assert not np.array_equal(result, update2)


class TestEpsilonEstimation:
    """Tests for epsilon estimation."""
    
    def test_epsilon_increases_with_steps(self):
        """Test that epsilon increases with more steps."""
        sigma = 1.0
        clip_norm = 1.0
        
        epsilon_10 = estimate_epsilon(10, sigma, clip_norm)
        epsilon_100 = estimate_epsilon(100, sigma, clip_norm)
        
        assert epsilon_100 > epsilon_10
    
    def test_epsilon_increases_with_noise(self):
        """Test that epsilon increases with less noise."""
        steps = 100
        clip_norm = 1.0
        
        epsilon_low_noise = estimate_epsilon(steps, sigma=0.5, clip_norm=clip_norm)
        epsilon_high_noise = estimate_epsilon(steps, sigma=2.0, clip_norm=clip_norm)
        
        assert epsilon_low_noise > epsilon_high_noise
    
    def test_epsilon_format(self):
        """Test epsilon is positive float."""
        epsilon = estimate_epsilon(100, 1.0, 1.0)
        
        assert isinstance(epsilon, float)
        assert epsilon > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
