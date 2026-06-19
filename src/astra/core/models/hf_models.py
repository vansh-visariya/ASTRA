"""
HuggingFace Models and PEFT Integration.

Supports:
- Loading HuggingFace models
- PEFT (LoRA, adapters) configuration
- Parameter extraction and aggregation for PEFT-only federation
- Saving/loading base models in .pt and safetensors formats

References:
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
- https://github.com/huggingface/peft
"""

import logging
import os
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


def save_base_model_to_disk(
    model: nn.Module,
    save_dir: str,
    model_name: str,
    peft_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save base model (backbone) and adapter config to disk in both .pt and safetensors.

    Args:
        model: The PEFT-wrapped model (or base model if no PEFT).
        save_dir: Directory to save files into (e.g. models/hf/{model_name}).
        model_name: HuggingFace model name (for metadata).
        peft_config: PEFT configuration dict (if PEFT is enabled).

    Returns:
        Dict mapping format to file path (e.g. {"pt": "...", "safetensors": "..."}).
    """
    logger = logging.getLogger(__name__)
    os.makedirs(save_dir, exist_ok=True)

    base_state = get_base_model_state_dict(model)
    saved: dict[str, str] = {}

    # Save as .pt (PyTorch native)
    pt_path = os.path.join(save_dir, "base_model.pt")
    torch.save({
        "base_state_dict": base_state,
        "model_name": model_name,
        "peft_config": peft_config or {},
    }, pt_path)
    saved["pt"] = pt_path
    logger.info("Saved base model .pt -> %s (%d params)", pt_path, len(base_state))

    # Save as safetensors (if safetensors is available)
    try:
        from safetensors.torch import save_file as _safetensors_save

        sf_path = os.path.join(save_dir, "base_model.safetensors")
        # safetensors requires flat dict of tensors — filter to only tensors
        tensor_dict = {
            name: param.data.cpu()
            for name, param in model.named_parameters()
            if 'lora' not in name.lower() and 'adapter' not in name.lower()
        }
        if tensor_dict:
            _safetensors_save(tensor_dict, sf_path)
            saved["safetensors"] = sf_path
            logger.info("Saved base model safetensors -> %s", sf_path)
    except ImportError:
        logger.debug("safetensors not installed, skipping safetensors save")

    # Save adapter config metadata
    import json as _json

    config_path = os.path.join(save_dir, "adapter_config.json")
    config_data = {
        "model_name": model_name,
        "has_peft": peft_config is not None and peft_config.get("enabled", False),
        "peft_config": peft_config or {},
        "base_model_files": {fmt: os.path.basename(p) for fmt, p in saved.items()},
    }
    with open(config_path, "w") as f:
        _json.dump(config_data, f, indent=2)

    return saved


def get_download_info(save_dir: str) -> dict[str, Any]:
    """Get metadata about available model files on disk.

    Returns:
        Dict with file info (sizes, formats, existence flags).
    """
    import json as _json

    info: dict[str, Any] = {
        "has_base_model": False,
        "has_adapter": False,
        "formats": {},
        "adapter_versions": [],
    }

    # Check base model files
    pt_path = os.path.join(save_dir, "base_model.pt")
    sf_path = os.path.join(save_dir, "base_model.safetensors")

    if os.path.exists(pt_path):
        info["has_base_model"] = True
        info["formats"]["pt"] = {
            "path": pt_path,
            "size_bytes": os.path.getsize(pt_path),
        }
    if os.path.exists(sf_path):
        info["has_base_model"] = True
        info["formats"]["safetensors"] = {
            "path": sf_path,
            "size_bytes": os.path.getsize(sf_path),
        }

    # Check adapter files
    adapter_latest = os.path.join(save_dir, "adapter_latest.pt")
    if os.path.exists(adapter_latest):
        info["has_adapter"] = True
        info["adapter_latest_size"] = os.path.getsize(adapter_latest)

    # Scan for versioned adapters
    if os.path.exists(save_dir):
        for fname in os.listdir(save_dir):
            if fname.startswith("adapter_v") and fname.endswith(".pt"):
                try:
                    ver = int(fname.replace("adapter_v", "").replace(".pt", ""))
                    info["adapter_versions"].append(ver)
                except ValueError:
                    pass
    info["adapter_versions"].sort()

    # Load adapter config if available
    config_path = os.path.join(save_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            info["adapter_config"] = _json.load(f)

    return info
