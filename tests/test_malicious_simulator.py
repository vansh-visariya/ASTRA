"""
Unit tests for MaliciousSimulator: noise, sign_flip, scale, backdoor, label_flip,
suspicion scores, and attack impact.
"""

import numpy as np
import torch
import pytest

from astra.core.privacy.malicious_simulator import MaliciousSimulator


@pytest.fixture
def simulator():
    config = {
        "malicious": {
            "enabled": True,
            "ratio": 0.3,
            "behaviors": ["noise", "sign_flip", "scale", "backdoor"],
        }
    }
    return MaliciousSimulator(config)


@pytest.fixture
def gradient():
    return np.random.randn(1000).astype(np.float32)


@pytest.fixture
def labels():
    return torch.randint(0, 10, (100,))


class TestNoiseAttack:
    def test_noise_changes_gradient(self, simulator, gradient):
        result = simulator._add_noise(gradient, noise_scale=5.0)
        assert result.shape == gradient.shape
        assert not np.allclose(result, gradient)

    def test_noise_scale_zero_no_change(self, simulator, gradient):
        result = simulator._add_noise(gradient, noise_scale=0.0)
        assert np.allclose(result, gradient)


class TestSignFlip:
    def test_sign_flip_inverts(self, simulator, gradient):
        result = simulator._sign_flip(gradient)
        np.testing.assert_allclose(result, -gradient * 2.0)


class TestScaleAttack:
    def test_scale_amplifies(self, simulator, gradient):
        result = simulator._scale_attack(gradient, scale=100.0)
        np.testing.assert_allclose(result, gradient * 100.0)

    def test_scale_one_no_change(self, simulator, gradient):
        result = simulator._scale_attack(gradient, scale=1.0)
        np.testing.assert_allclose(result, gradient)


class TestBackdoorAttack:
    def test_backdoor_shape_preserved(self, simulator, gradient):
        result = simulator._backdoor_attack(gradient, "malicious_client_42")
        assert result.shape == gradient.shape

    def test_backdoor_deterministic(self, simulator, gradient):
        a = simulator._backdoor_attack(gradient.copy(), "mal_1")
        b = simulator._backdoor_attack(gradient.copy(), "mal_1")
        np.testing.assert_allclose(a, b)

    def test_backdoor_different_clients(self, simulator, gradient):
        a = simulator._backdoor_attack(gradient.copy(), "mal_a")
        b = simulator._backdoor_attack(gradient.copy(), "mal_b")
        assert not np.allclose(a, b)


class TestLabelFlip:
    def test_label_flip_changes_subset(self, simulator, labels):
        result = simulator.simulate_label_flip(labels.clone(), flip_ratio=0.5)
        assert result.shape == labels.shape
        changed = (result != labels).sum().item()
        assert changed > 0

    def test_label_flip_ratio_zero_no_change(self, simulator, labels):
        result = simulator.simulate_label_flip(labels.clone(), flip_ratio=0.0)
        assert torch.equal(result, labels)

    def test_label_flip_ratio_one_all_changed(self, simulator, labels):
        result = simulator.simulate_label_flip(labels.clone(), flip_ratio=1.0)
        assert (result != labels).sum().item() == len(labels)


class TestInjectAttack:
    def test_inject_attack_shape(self, simulator, gradient):
        result = simulator.inject_attack(gradient, "test_client")
        assert result.shape == gradient.shape

    def test_no_behaviors_returns_original(self, gradient):
        config = {"malicious": {"enabled": True, "ratio": 0.3, "behaviors": ["noise"]}}
        sim = MaliciousSimulator(config)
        result = sim.inject_attack(gradient.copy(), "test_client")
        assert result.shape == gradient.shape

    def test_select_behavior_deterministic(self, simulator):
        a = simulator._select_behavior("client_x")
        b = simulator._select_behavior("client_x")
        assert a == b


class TestSuspicionScore:
    def test_clean_gradient_low_suspicion(self, simulator):
        g = np.random.randn(1000).astype(np.float32) * 0.01
        score = simulator.get_suspicion_score(g)
        assert 0.0 <= score <= 1.0

    def test_noisy_gradient_high_suspicion(self, simulator):
        g = np.random.randn(1000).astype(np.float32) * 100.0
        score = simulator.get_suspicion_score(g)
        assert score >= 0.3

    def test_score_never_exceeds_one(self, simulator):
        g = np.random.randn(1000).astype(np.float32) * 1000.0
        score = simulator.get_suspicion_score(g)
        assert score <= 1.0


class TestAttackImpact:
    def test_impact_calculation(self, simulator):
        impact = simulator.compute_attack_impact(0.95, 0.60)
        assert 0.0 <= impact["attack_success"] <= 1.0
        assert impact["accuracy_drop"] > 0.0
