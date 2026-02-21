"""
HuggingFace Models and PEFT Integration.

Supports:
- Loading HuggingFace models
- PEFT (LoRA, adapters) configuration
- Parameter extraction and aggregation for PEFT-only federation

References:
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
- https://github.com/huggingface/peft
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, CLIPVisionModel, CLIPProcessor


def load_hf_peft_model(
    model_name: str,
    peft_config: Dict[str, Any],
    device: str = 'cuda'
) -> Tuple[nn.Module, Any]:
    """
    Load HuggingFace model with PEFT.
    
    Args:
        model_name: HuggingFace model name.
        peft_config: PEFT configuration.
        device: Device to load model on.
    
    Returns:
        Tuple of (model, processor/tokenizer).
    """
    logger = logging.getLogger(__name__)
    
    peft_enabled = peft_config.get('enabled', False)
    
    if 'clip' in model_name.lower():
        model = CLIPVisionModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        if peft_enabled:
            model = apply_peft(model, peft_config)
            logger.info(f"PEFT enabled with method: {peft_config.get('method', 'lora')}")
        
        model = model.to(device)
        return model, processor
    
    try:
        model = AutoModel.from_pretrained(model_name)
        
        if peft_enabled:
            model = apply_peft(model, peft_config)
            logger.info(f"PEFT enabled with method: {peft_config.get('method', 'lora')}")
        
        model = model.to(device)
        
        return model, None
    
    except Exception as e:
        logger.warning(f"Could not load model {model_name}: {e}")
        logger.info("Falling back to CLIP vision model")
        
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if peft_enabled:
            model = apply_peft(model, peft_config)
        
        model = model.to(device)
        return model, processor


def apply_peft(
    model: nn.Module,
    peft_config: Dict[str, Any]
) -> nn.Module:
    """
    Apply PEFT to model.
    
    Args:
        model: Base model.
        peft_config: PEFT configuration.
    
    Returns:
        Model with PEFT applied.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        method = peft_config.get('method', 'lora').lower()
        
        if method == 'lora':
            lora_rank = peft_config.get('lora_rank', 8)
            lora_alpha = peft_config.get('lora_alpha', 16)
            target_modules = peft_config.get('target_modules', ['q_proj', 'v_proj'])
            
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                task_type=TaskType.FEATURE_EXTRACTION
            )
            
            model = get_peft_model(model, lora_config)
            
        return model
    
    except ImportError:
        logging.warning("PEFT not installed, returning base model")
        return model


def extract_peft_params(model: nn.Module) -> Dict[str, np.ndarray]:
    """
    Extract only PEFT (LoRA) parameters.
    
    Args:
        model: Model with PEFT.
    
    Returns:
        Dictionary of parameter name -> numpy array.
    """
    peft_params = {}
    
    for name, param in model.named_parameters():
        if 'lora' in name.lower() or 'adapter' in name.lower():
            peft_params[name] = param.data.cpu().numpy().copy()
    
    if not peft_params:
        for param in model.parameters():
            peft_params[f"param_{len(peft_params)}"] = param.data.cpu().numpy().copy()
    
    return peft_params


def apply_peft_update(
    model: nn.Module,
    param_dict: Dict[str, np.ndarray]
) -> None:
    """
    Apply PEFT parameter update to model.
    
    Args:
        model: Model to update.
        param_dict: Parameter dictionary.
    """
    state_dict = model.state_dict()
    
    for name, values in param_dict.items():
        if name in state_dict:
            state_dict[name] = torch.from_numpy(values)
    
    model.load_state_dict(state_dict)


def get_peft_param_size(model: nn.Module) -> int:
    """Get total size of PEFT parameters in bytes."""
    total = 0
    
    for name, param in model.named_parameters():
        if 'lora' in name.lower() or 'adapter' in name.lower():
            total += param.numel() * param.element_size()
    
    return total


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone, keep only PEFT parameters trainable."""
    for name, param in model.named_parameters():
        if 'lora' not in name.lower() and 'adapter' not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True


def get_trainable_params(model: nn.Module) -> int:
    """Get number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_communication_bytes(model: nn.Module) -> int:
    """
    Compute communication cost for sending PEFT parameters.
    
    Returns:
        Number of bytes to transmit.
    """
    return get_peft_param_size(model)
