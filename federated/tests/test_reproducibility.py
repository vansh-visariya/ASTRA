"""
Test reproducibility with fixed seeds.
"""

import numpy as np
import pytest
import torch

from federated.utils.seed import set_seed


class TestReproducibility:
    """Tests for reproducibility."""
    
    def test_numpy_reproducibility(self):
        """Test numpy random is reproducible."""
        seed = 42
        
        set_seed(seed)
        result1 = np.random.randn(100)
        
        set_seed(seed)
        result2 = np.random.randn(100)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_torch_reproducibility(self):
        """Test torch random is reproducible."""
        seed = 42
        
        set_seed(seed)
        result1 = torch.randn(100)
        
        set_seed(seed)
        result2 = torch.randn(100)
        
        assert torch.equal(result1, result2)
    
    def test_model_weights_reproducibility(self):
        """Test model initialization is reproducible."""
        seed = 42
        
        set_seed(seed)
        model1 = torch.nn.Linear(10, 10)
        
        set_seed(seed)
        model2 = torch.nn.Linear(10, 10)
        
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
    
    def test_training_reproducibility(self):
        """Test training with same seed produces similar results."""
        seed = 42
        
        set_seed(seed)
        model1 = torch.nn.Linear(784, 10)
        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        data1 = torch.randn(32, 784)
        target1 = torch.randint(0, 10, (32,))
        
        for _ in range(10):
            optimizer1.zero_grad()
            output = model1(data1)
            loss = torch.nn.functional.cross_entropy(output, target1)
            loss.backward()
            optimizer1.step()
        
        set_seed(seed)
        model2 = torch.nn.Linear(784, 10)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        
        for _ in range(10):
            optimizer2.zero_grad()
            output = model2(data1)
            loss = torch.nn.functional.cross_entropy(output, target1)
            loss.backward()
            optimizer2.step()
        
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
