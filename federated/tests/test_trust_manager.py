"""
Unit tests for trust manager.
"""

import numpy as np
import pytest

from federated.trust_manager import TrustManager


class TestTrustManager:
    """Tests for TrustManager."""
    
    @pytest.fixture
    def config(self):
        return {
            'trust': {
                'init': 1.0,
                'update_alpha': 0.3,
                'quarantine_threshold': 0.35,
                'soft_decay': 0.8
            }
        }
    
    @pytest.fixture
    def trust_manager(self, config):
        return TrustManager(config)
    
    def test_initial_trust(self, trust_manager):
        """Test initial trust score."""
        trust = trust_manager.get_trust('client_001')
        assert trust == 1.0
    
    def test_trust_bounds(self, trust_manager):
        """Test trust is bounded in [0, 1]."""
        global_vec = np.random.randn(100)
        
        for i in range(20):
            update = np.random.randn(100) + (i * 0.5)
            trust = trust_manager.update_trust(f'client_{i:03d}', update, global_vec)
            
            assert 0.0 <= trust <= 1.0
    
    def test_trust_update_formula(self, trust_manager):
        """Test trust update formula application."""
        trust_manager.trust_scores['test_client'] = 0.5
        
        update = np.ones(100)
        global_vec = np.ones(100)
        
        trust = trust_manager.update_trust('test_client', update, global_vec)
        
        expected = 0.5 + 0.3 * 1.0
        
        assert abs(trust - min(expected, 1.0)) < 0.01
    
    def test_quarantine_threshold(self, trust_manager):
        """Test quarantine threshold configuration."""
        # Test that quarantine_threshold is correctly set
        assert trust_manager.quarantine_threshold == 0.35
        
        # Test that trust bounds are enforced
        trust_manager.trust_scores['test'] = 1.5
        trust = trust_manager.update_trust('test', np.ones(100), np.ones(100))
        
        # Trust should be bounded to [0, 1]
        assert trust <= 1.0
    
    def test_stats(self, trust_manager):
        """Test trust statistics."""
        for i in range(5):
            trust_manager.trust_scores[f'client_{i}'] = 0.5 + i * 0.1
        
        stats = trust_manager.get_stats()
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
    
    def test_history_tracking(self, trust_manager):
        """Test trust history tracking."""
        global_vec = np.ones(10)
        
        for i in range(5):
            trust_manager.update_trust('client_001', np.ones(10), global_vec)
        
        history = trust_manager.get_history('client_001')
        
        assert len(history) == 5
    
    def test_reset(self, trust_manager):
        """Test trust manager reset."""
        trust_manager.trust_scores['client_001'] = 0.5
        trust_manager.quarantined['client_001'] = True
        
        trust_manager.reset()
        
        assert trust_manager.get_trust('client_001') == 1.0
        assert not trust_manager.is_quarantined('client_001')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
