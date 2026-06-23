"""
Quick LoRA fine-tune + export adapter weights for ASTRA upload.

Usage:
    pip install torch peft transformers datasets
    python scripts/train_adapter.py --group-id <GROUP_ID> --token <JWT>
    python scripts/train_adapter.py  # uses hardcoded defaults (no server)

Outputs:
    adapter_delta.pt    — raw float32 adapter weight bytes (upload this)
    adapter_weights.pt  — full adapter state dict (backup)
"""

import argparse
import json
import sys
import struct
import urllib.request
import urllib.error
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DELTA = "adapter_delta.pt"
OUTPUT_STATE = "adapter_weights.pt"


def fetch_manifest(base_url: str, group_id: str, token: str) -> dict | None:
    """Fetch training manifest from the ASTRA server. Returns None if unavailable."""
    url = f"{base_url}/api/groups/{group_id}/manifest"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("manifest")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  No training manifest found for group '{group_id}'. Using defaults.")
        else:
            print(f"  Warning: could not fetch manifest (HTTP {e.code}). Using defaults.")
        return None
    except Exception as e:
        print(f"  Warning: could not reach server ({e}). Using defaults.")
        return None


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune + export for ASTRA upload")
    parser.add_argument("--group-id", help="ASTRA group ID (fetches manifest from server)")
    parser.add_argument("--token", help="JWT auth token (required with --group-id)")
    parser.add_argument("--base-url", default="http://localhost:8000", help="ASTRA server URL")
    parser.add_argument("--model-id", default="SupraLabs/Supra-1.5-50M-Instruct-exp",
                        help="HuggingFace model ID (ignored if manifest provides model_id)")
    parser.add_argument("--steps", type=int, default=3, help="Training steps")
    args = parser.parse_args()

    # --- 1. Fetch manifest (if group-id provided) ---
    manifest = None
    if args.group_id:
        if not args.token:
            print("Error: --token is required when --group-id is provided.")
            sys.exit(1)
        print(f"Fetching manifest for group '{args.group_id}' ...")
        manifest = fetch_manifest(args.base_url, args.group_id, args.token)

    # --- 2. Resolve config from manifest or defaults ---
    model_id = args.model_id
    target_modules = "all-linear"
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lr = 1e-4
    expected_delta_bytes = None

    if manifest:
        model_id = manifest.get("model_id", model_id)
        target_modules = manifest.get("target_modules", target_modules)
        lora_r = manifest.get("lora_rank", lora_r)
        lora_alpha = manifest.get("lora_alpha", lora_alpha)
        lora_dropout = manifest.get("lora_dropout", lora_dropout)
        lr = manifest.get("lr", lr)
        expected_delta_bytes = manifest.get("expected_delta_bytes")
        print(f"  model_id:       {model_id}")
        print(f"  target_modules: {target_modules}")
        print(f"  lora_rank:      {lora_r}")
        print(f"  lora_alpha:     {lora_alpha}")
        print(f"  lr:             {lr}")
        if expected_delta_bytes:
            print(f"  expected_delta: {expected_delta_bytes} bytes ({expected_delta_bytes / 1024:.1f} KB)")
    else:
        print(f"  Using defaults: model={model_id}, target_modules={target_modules}, lr={lr}")

    # --- 3. Load base model ---
    print(f"\nLoading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    # --- 4. Apply LoRA ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 5. Dummy training ---
    dummy_texts = [
        "The capital of France is Paris.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for step in range(args.steps):
        inputs = tokenizer(
            dummy_texts[step % len(dummy_texts)],
            return_tensors="pt", padding=True, truncation=True, max_length=128,
        )
        labels = inputs["input_ids"].clone()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  step {step+1}: loss={loss.item():.4f}")

    model.eval()

    # --- 6. Extract adapter weights ---
    adapter_state = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            adapter_state[name] = param.data.detach().cpu().contiguous()

    print(f"\nExtracted {len(adapter_state)} adapter tensors:")
    for name, tensor in adapter_state.items():
        print(f"  {name}: {tensor.shape} ({tensor.numel()} params)")

    # Save state dict
    torch.save(adapter_state, OUTPUT_STATE)
    print(f"\nSaved adapter state dict -> {OUTPUT_STATE}")

    # --- 7. Export as raw float32 bytes ---
    all_bytes = b""
    for tensor in adapter_state.values():
        all_bytes += tensor.numpy().astype("<f4").tobytes()

    with open(OUTPUT_DELTA, "wb") as f:
        f.write(all_bytes)

    total_params = sum(t.numel() for t in adapter_state.values())
    print(f"Saved raw float32 delta ({len(all_bytes)} bytes, {total_params} params) -> {OUTPUT_DELTA}")

    # --- 8. Validate against manifest ---
    if expected_delta_bytes and len(all_bytes) != expected_delta_bytes:
        print(f"\n  WARNING: Delta size mismatch!")
        print(f"  Expected: {expected_delta_bytes} bytes (manifest)")
        print(f"  Got:      {len(all_bytes)} bytes")
        print(f"  Upload will be REJECTED by the server.")
        print(f"  Fix: ensure target_modules={target_modules} and lora_rank={lora_r} match the group config.")
        sys.exit(1)
    elif expected_delta_bytes:
        print(f"\n  Delta size matches manifest ({len(all_bytes)} bytes).")

    print(f"\nUpload {OUTPUT_DELTA} to ASTRA via the Upload Delta page.")


if __name__ == "__main__":
    main()
