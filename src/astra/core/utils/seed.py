"""
Seed utilities for reproducibility.
"""

import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seed_state() -> Dict[str, Any]:
    """Get current random state."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_seed_state(state: Dict[str, Any]) -> None:
    """Restore random state."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def seed_worker(worker_id: int) -> None:
    """
    Seed worker for DataLoader multiprocessing.
    
    Args:
        worker_id: Worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_deterministic_execution() -> None:
    """Configure PyTorch for deterministic execution."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            logging.warning("Some operations don't have deterministic implementations")
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
