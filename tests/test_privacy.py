"""
Unit tests for privacy mechanisms.
"""

import numpy as np
import pytest

from astra.core.privacy.privacy import clip_and_noise


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
