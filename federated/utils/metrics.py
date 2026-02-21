"""
Metrics computation for federated learning evaluation.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_accuracy(model: nn.Module, data_loader: DataLoader) -> float:
    """
    Compute model accuracy on data loader.
    
    Args:
        model: PyTorch model.
        data_loader: Data loader.
    
    Returns:
        Accuracy as float between 0 and 1.
    """
    model.eval()
    
    device = next(model.parameters()).device
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / total if total > 0 else 0.0


def compute_loss(model: nn.Module, data_loader: DataLoader) -> float:
    """
    Compute model loss on data loader.
    
    Args:
        model: PyTorch model.
        data_loader: Data loader.
    
    Returns:
        Average loss.
    """
    model.eval()
    
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * len(target)
            total_samples += len(target)
    
    return total_loss / total_samples if total_samples > 0 else 0.0


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute L2 norm of gradients.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Gradient L2 norm.
    """
    total_norm = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    return total_norm ** 0.5


def compute_weight_norm(model: nn.Module) -> float:
    """
    Compute L2 norm of model weights.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Weight L2 norm.
    """
    total_norm = 0.0
    
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    
    return total_norm ** 0.5


def compute_weight_stats(weights: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics of weight vector.
    
    Args:
        weights: Flattened weight vector.
    
    Returns:
        Dictionary of statistics.
    """
    return {
        'mean': float(np.mean(weights)),
        'std': float(np.std(weights)),
        'min': float(np.min(weights)),
        'max': float(np.max(weights)),
        'norm': float(np.linalg.norm(weights)),
        'abs_mean': float(np.mean(np.abs(weights)))
    }


def compute_similarity(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Compute similarity metrics between two vectors.
    
    Args:
        a: First vector.
        b: Second vector.
    
    Returns:
        Dictionary of similarity metrics.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a < 1e-8 or norm_b < 1e-8:
        return {'cosine': 0.0, 'euclidean': 0.0, 'mse': 0.0}
    
    cosine = np.dot(a, b) / (norm_a * norm_b)
    euclidean = np.linalg.norm(a - b)
    mse = np.mean((a - b) ** 2)
    
    return {
        'cosine': float(cosine),
        'euclidean': float(euclidean),
        'mse': float(mse)
    }


class MetricsTracker:
    """Track and log metrics during training."""
    
    def __init__(
        self,
        experiment_id: str,
        log_dir: Any,
        config: Dict[str, Any]
    ):
        self.experiment_id = experiment_id
        self.log_dir = log_dir
        self.config = config
        
        self.metrics_history: List[Dict[str, Any]] = []
        
        self._init_loggers()
    
    def _init_loggers(self) -> None:
        """Initialize loggers."""
        from federated.utils.logging_utils import JSONLLogger
        
        self.jsonl_logger = JSONLLogger(
            self.log_dir / 'metrics.jsonl'
        )
    
    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        """
        Log metrics at a step.
        
        Args:
            step: Current training step.
            metrics: Metrics dictionary.
        """
        entry = {
            'experiment_id': self.experiment_id,
            'step': step,
            **metrics
        }
        
        self.metrics_history.append(entry)
        self.jsonl_logger.log(entry)
    
    def save_all(self) -> None:
        """Save all tracked metrics."""
        self.jsonl_logger.close()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get metrics history."""
        return self.metrics_history


def compute_trust_metrics(trust_scores: Dict[str, float]) -> Dict[str, float]:
    """Compute statistics on trust scores."""
    if not trust_scores:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    values = list(trust_scores.values())
    
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }


def compute_attack_metrics(
    pre_attack_accuracy: float,
    post_attack_accuracy: float,
    recovery_step: int
) -> Dict[str, Any]:
    """Compute attack impact metrics."""
    accuracy_drop = pre_attack_accuracy - post_attack_accuracy
    relative_drop = accuracy_drop / max(pre_attack_accuracy, 1e-8)
    
    return {
        'accuracy_drop': accuracy_drop,
        'relative_drop': relative_drop,
        'recovery_step': recovery_step,
        'attack_success': 1.0 if relative_drop > 0.1 else 0.0
    }
