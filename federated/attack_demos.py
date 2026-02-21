"""
Attack Demonstrations for Privacy Evaluation.

Implements gradient inversion attacks to demonstrate privacy risks.
Used for educational purposes and privacy evaluation.

References:
- Zhu & Han, "Deep Learning with Gradient Leakage"
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from PIL import Image


class GradientInversionAttack:
    """Gradient inversion attack for privacy evaluation."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def reconstruct(
        self,
        target_gradient: np.ndarray,
        target_label: int,
        num_iterations: int = 100,
        learning_rate: float = 0.1
    ) -> Tuple[np.ndarray, float]:
        """
        Attempt to reconstruct input from gradient.
        
        Args:
            target_gradient: Observed gradient vector.
            target_label: Target class label.
            num_iterations: Number of optimization iterations.
            learning_rate: Learning rate for optimization.
        
        Returns:
            Tuple of (reconstructed image, final loss).
        """
        dummy_input = torch.randn(1, 1, 28, 28, requires_grad=True)
        
        optimizer = optim.Adam([dummy_input], lr=learning_rate)
        
        criterion = nn.CrossEntropyLoss()
        
        final_loss = float('inf')
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            output = self.model(dummy_input)
            loss = criterion(output, torch.tensor([target_label]))
            
            gradient_placeholder = torch.tensor(
                target_gradient[:dummy_input.numel()],
                dtype=torch.float32
            ).view_as(dummy_input)
            
            grad_loss = torch.nn.functional.mse_loss(
                dummy_input.grad if dummy_input.grad is not None else torch.zeros_like(dummy_input),
                gradient_placeholder
            )
            
            total_loss = loss + 0.1 * grad_loss
            
            total_loss.backward()
            optimizer.step()
            
            if iteration % 20 == 0:
                final_loss = total_loss.item()
        
        return dummy_input.detach().numpy(), final_loss


def run_inversion_demo(
    model: nn.Module,
    data_loader: DataLoader,
    config: Dict[str, Any],
    output_dir: Path,
    with_dp: bool = True
) -> Dict[str, Any]:
    """
    Run gradient inversion demo.
    
    Args:
        model: Model to attack.
        data_loader: Data loader.
        config: Configuration.
        output_dir: Output directory.
        with_dp: Whether DP is applied.
    
    Returns:
        Results dictionary.
    """
    from federated.privacy import clip_and_noise
    
    model.eval()
    
    data, target = next(iter(data_loader))
    
    data.requires_grad = True
    
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)
    
    loss.backward()
    
    gradient = data.grad.data.cpu().numpy()
    
    if with_dp and config.get('privacy', {}).get('dp_enabled', False):
        clip_norm = config['privacy'].get('clip_norm', 1.0)
        sigma = config['privacy'].get('sigma', 1.0)
        
        gradient = clip_and_noise(gradient.flatten(), clip_norm, sigma)
        gradient = gradient.reshape(data.shape)
    
    attack = GradientInversionAttack(model, config)
    
    reconstructed, final_loss = attack.reconstruct(
        gradient.flatten(),
        target.item(),
        num_iterations=100
    )
    
    original_np = data.detach().cpu().numpy()
    
    mse = np.mean((reconstructed - original_np) ** 2)
    
    results = {
        'with_dp': with_dp,
        'mse': float(mse),
        'final_loss': final_loss
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / f'reconstructed_{"dp" if with_dp else "no_dp"}.npy', reconstructed)
    np.save(output_dir / f'original_{"dp" if with_dp else "no_dp"}.npy', original_np)
    
    return results


def compare_dp_vs_no_dp(
    model: nn.Module,
    data_loader: DataLoader,
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Compare gradient inversion with and without DP.
    
    Args:
        model: Model to attack.
        data_loader: Data loader.
        config: Configuration.
        output_dir: Output directory.
    
    Returns:
        Comparison results.
    """
    results_no_dp = run_inversion_demo(
        model, data_loader, config, output_dir / 'no_dp', with_dp=False
    )
    
    config['privacy']['dp_enabled'] = True
    
    results_with_dp = run_inversion_demo(
        model, data_loader, config, output_dir / 'with_dp', with_dp=True
    )
    
    comparison = {
        'no_dp_mse': results_no_dp['mse'],
        'with_dp_mse': results_with_dp['mse'],
        'improvement': results_no_dp['mse'] - results_with_dp['mse']
    }
    
    return comparison
