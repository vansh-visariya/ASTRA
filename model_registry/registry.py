"""
Model Registry for managing baseline models.

Supports:
- HuggingFace models (with PEFT)
- Custom CNN/MLP models
- Local model files

References:
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
"""

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class ModelInfo:
    """Model metadata container."""
    
    def __init__(
        self,
        model_id: str,
        model_type: str,
        architecture: str,
        total_params: int,
        trainable_params: int,
        is_peft: bool = False,
        peft_method: Optional[str] = None,
        source: str = "local",
        model_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        self.model_id = model_id
        self.model_type = model_type  # vision, text, multimodal
        self.architecture = architecture
        self.total_params = total_params
        self.trainable_params = trainable_params
        self.is_peft = is_peft
        self.peft_method = peft_method
        self.source = source
        self.model_path = model_path
        self.config = config or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'architecture': self.architecture,
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            'is_peft': self.is_peft,
            'peft_method': self.peft_method,
            'source': self.source,
            'model_path': self.model_path,
            'config': self.config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        return cls(**data)


class ModelRegistry:
    """
    Central registry for managing baseline models.
    
    Features:
    - Register HF models with optional PEFT
    - Register custom CNN/MLP
    - Load local .pt files
    - Validate model compatibility
    """
    
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelInfo] = {}
        self.model_instances: Dict[str, nn.Module] = {}
        
        self.logger = logging.getLogger(__name__)
        
        self._register_builtin_models()
    
    def _register_builtin_models(self) -> None:
        """Register built-in CNN/MLP models."""
        from core_engine.model_zoo import SimpleCNN, CIFAR10CNN, SimpleMLP
        
        # SimpleCNN for MNIST
        model = SimpleCNN(num_classes=10)
        param_count = sum(p.numel() for p in model.parameters())
        
        self.models['simple_cnn_mnist'] = ModelInfo(
            model_id='simple_cnn_mnist',
            model_type='vision',
            architecture='SimpleCNN',
            total_params=param_count,
            trainable_params=param_count,
            is_peft=False,
            source='builtin',
            config={'dataset': 'MNIST', 'num_classes': 10}
        )
        
        # CIFAR10 CNN
        model = CIFAR10CNN(num_classes=10)
        param_count = sum(p.numel() for p in model.parameters())
        
        self.models['simple_cnn_cifar10'] = ModelInfo(
            model_id='simple_cnn_cifar10',
            model_type='vision',
            architecture='CIFAR10CNN',
            total_params=param_count,
            trainable_params=param_count,
            is_peft=False,
            source='builtin',
            config={'dataset': 'CIFAR10', 'num_classes': 10}
        )
        
        self.logger.info(f"Registered {len(self.models)} builtin models")
    
    def register_hf_model(
        self,
        model_name: str,
        use_peft: bool = False,
        peft_config: Optional[Dict] = None
    ) -> ModelInfo:
        """
        Register a HuggingFace model.
        
        Args:
            model_name: HF model name (e.g., "openai/clip-vit-base-patch32")
            use_peft: Whether to apply PEFT (LoRA)
            peft_config: PEFT configuration if use_peft=True
        
        Returns:
            ModelInfo object
        """
        from core_engine.hf_models import load_hf_peft_model
        
        model_id = f"hf_{model_name.replace('/', '_')}"
        
        if use_peft:
            model_id += "_peft"
        
        if model_id in self.models:
            self.logger.info(f"Model {model_id} already registered")
            return self.models[model_id]
        
        self.logger.info(f"Loading HF model: {model_name}")
        
        try:
            peft_cfg = peft_config or {
                'enabled': use_peft,
                'method': 'lora',
                'lora_rank': 8,
                'lora_alpha': 16,
                'target_modules': ['q_proj', 'v_proj']
            }
            
            model, processor = load_hf_peft_model(
                model_name,
                peft_cfg,
                device='cpu'  # Load on CPU first
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            
            if use_peft:
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            else:
                trainable = total_params
            
            model_type = 'vision'
            if 'text' in model_name.lower() or 'bert' in model_name.lower() or 'gpt' in model_name.lower():
                model_type = 'text'
            if 'clip' in model_name.lower() or 'blip' in model_name.lower():
                model_type = 'multimodal'
            
            model_info = ModelInfo(
                model_id=model_id,
                model_type=model_type,
                architecture=model_name,
                total_params=total_params,
                trainable_params=trainable,
                is_peft=use_peft,
                peft_method=peft_cfg.get('method') if use_peft else None,
                source='huggingface',
                model_path=model_name,
                config=peft_cfg
            )
            
            self.models[model_id] = model_info
            self.model_instances[model_id] = model
            
            self.logger.info(f"Registered HF model: {model_id} ({total_params:,} params)")
            
            return model_info
        
        except Exception as e:
            self.logger.error(f"Failed to load HF model {model_name}: {e}")
            raise
    
    def register_local_model(
        self,
        model_id: str,
        model_path: str,
        architecture: str = "Custom"
    ) -> ModelInfo:
        """
        Register a local .pt model file.
        
        Args:
            model_id: Unique identifier
            model_path: Path to .pt file
            architecture: Model architecture name
        
        Returns:
            ModelInfo object
        """
        if model_id in self.models:
            return self.models[model_id]
        
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Create dummy model to count params
            model = nn.Linear(10, 10)  # Placeholder
            model.load_state_dict(state_dict)
            
            total_params = sum(p.numel() for p in model.parameters())
            
            model_info = ModelInfo(
                model_id=model_id,
                model_type='custom',
                architecture=architecture,
                total_params=total_params,
                trainable_params=total_params,
                is_peft=False,
                source='local',
                model_path=model_path
            )
            
            self.models[model_id] = model_info
            
            self.logger.info(f"Registered local model: {model_id}")
            
            return model_info
        
        except Exception as e:
            self.logger.error(f"Failed to load local model: {e}")
            raise
    
    def register_custom_architecture(
        self,
        model_id: str,
        architecture: str,
        model_type: str,
        config: Dict[str, Any]
    ) -> ModelInfo:
        """
        Register a custom CNN/MLP architecture.
        
        Args:
            model_id: Unique identifier
            architecture: Architecture type (cnn, mlp, vit)
            model_type: vision, text, multimodal
            config: Model configuration
        
        Returns:
            ModelInfo object
        """
        from core_engine.model_zoo import create_model
        
        config['model']['type'] = architecture
        
        try:
            model = create_model(config)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = ModelInfo(
                model_id=model_id,
                model_type=model_type,
                architecture=architecture,
                total_params=total_params,
                trainable_params=trainable_params,
                is_peft=False,
                source='custom',
                config=config
            )
            
            self.models[model_id] = model_info
            self.model_instances[model_id] = model
            
            return model_info
        
        except Exception as e:
            self.logger.error(f"Failed to create custom model: {e}")
            raise
    
    def load_model(self, model_id: str, device: str = 'cpu') -> nn.Module:
        """
        Load model instance.
        
        Args:
            model_id: Model identifier
            device: Target device (cpu/cuda)
        
        Returns:
            Model instance
        """
        if model_id in self.model_instances:
            model = self.model_instances[model_id]
            return model.to(device)
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_info = self.models[model_id]
        
        if model_info.source == 'huggingface':
            from core_engine.hf_models import load_hf_peft_model
            model, _ = load_hf_peft_model(
                model_info.model_path,
                model_info.config,
                device=device
            )
        elif model_info.source == 'local':
            model = torch.load(model_info.model_path, map_location=device)
        else:
            from core_engine.model_zoo import create_model
            config = {'model': {'type': model_info.architecture}, **model_info.config}
            model = create_model(config)
            model = model.to(device)
        
        self.model_instances[model_id] = model
        return model
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Args:
            model_type: Optional filter by type
        
        Returns:
            List of model info dictionaries
        """
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        return [m.to_dict() for m in models]
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model info by ID."""
        if model_id in self.models:
            return self.models[model_id].to_dict()
        return None
    
    def validate_model(self, model_id: str) -> Tuple[bool, str]:
        """
        Validate model compatibility.
        
        Args:
            model_id: Model to validate
        
        Returns:
            (is_valid, message)
        """
        if model_id not in self.models:
            return False, f"Model {model_id} not found"
        
        model_info = self.models[model_id]
        
        if model_info.total_params > 1_000_000_000:
            return False, "Model exceeds 1B parameters"
        
        return True, "Valid"
    
    def save_registry(self, path: str) -> None:
        """Save registry to JSON file."""
        data = {
            model_id: info.to_dict() 
            for model_id, info in self.models.items()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_registry(self, path: str) -> None:
        """Load registry from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for model_id, info_dict in data.items():
            self.models[model_id] = ModelInfo.from_dict(info_dict)


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get global model registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry
