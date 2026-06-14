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
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, CLIPProcessor, CLIPVisionModel


def load_hf_peft_model(
    model_name: str,
    peft_config: dict[str, Any],
    device: str = 'cuda'
) -> tuple[nn.Module, Any]:
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
        model: nn.Module = CLIPVisionModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        if peft_enabled:
            model = apply_peft(model, peft_config)
            logger.info(f"PEFT enabled with method: {peft_config.get('method', 'lora')}")

        model = model.to(device)
        return model, processor

    try:
        model_auto: nn.Module = AutoModel.from_pretrained(model_name)

        if peft_enabled:
            model_auto = apply_peft(model_auto, peft_config)
            logger.info(f"PEFT enabled with method: {peft_config.get('method', 'lora')}")

        model_auto = model_auto.to(device)

        return model_auto, None

    except Exception as e:
        logger.warning(f"Could not load model {model_name}: {e}")
        logger.info("Falling back to CLIP vision model")

        model_fallback: nn.Module = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        if peft_enabled:
            model_fallback = apply_peft(model_fallback, peft_config)

        model_fallback = model_fallback.to(device)
        return model_fallback, processor


def apply_peft(
    model: nn.Module,
    peft_config: dict[str, Any]
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
        from peft import LoraConfig, TaskType, get_peft_model

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

            model = get_peft_model(model, lora_config)  # type: ignore[arg-type]

        return model

    except ImportError:
        logging.warning("PEFT not installed, returning base model")
        return model


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA adapter weights as a state_dict."""
    return {
        name: param.data.cpu().clone()
        for name, param in model.named_parameters()
        if 'lora' in name.lower() or 'adapter' in name.lower()
    }


def load_lora_state_dict(model: nn.Module, lora_state: dict[str, torch.Tensor]) -> None:
    """Load LoRA adapter weights into model (in-place)."""
    for name, tensor in lora_state.items():
        for target_name, param in model.named_parameters():
            if target_name == name:
                param.data.copy_(tensor.to(param.device))
                break


def get_base_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract non-LoRA (backbone) weights as a state_dict."""
    return {
        name: param.data.cpu().clone()
        for name, param in model.named_parameters()
        if 'lora' not in name.lower() and 'adapter' not in name.lower()
    }


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone, keep only PEFT parameters trainable."""
    for name, param in model.named_parameters():
        if 'lora' not in name.lower() and 'adapter' not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True
