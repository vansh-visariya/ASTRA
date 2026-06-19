"""
Quick LoRA fine-tune + export adapter weights for ASTRA upload.

Usage:
    pip install torch peft transformers datasets
    python scripts/train_adapter.py

Outputs:
    adapter_delta.pt    — raw float32 adapter weight bytes (upload this)
    adapter_weights.pt  — full adapter state dict (backup)
"""

import struct
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "SupraLabs/Supra-1.5-50M-Instruct-exp"
OUTPUT_DELTA = "adapter_delta.pt"
OUTPUT_STATE = "adapter_weights.pt"

# --- 1. Load base model ---
print(f"Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()

# --- 2. Apply LoRA ---
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 3. Dummy training (3 steps) ---
dummy_texts = [
    "The capital of France is Paris.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language.",
]
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

for step in range(3):
    inputs = tokenizer(dummy_texts[step % len(dummy_texts)], return_tensors="pt", padding=True, truncation=True, max_length=128)
    labels = inputs["input_ids"].clone()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"  step {step+1}: loss={loss.item():.4f}")

model.eval()

# --- 4. Extract adapter weights ---
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

# --- 5. Export as raw float32 bytes ---
all_bytes = b""
for tensor in adapter_state.values():
    all_bytes += tensor.numpy().astype("<f4").tobytes()

with open(OUTPUT_DELTA, "wb") as f:
    f.write(all_bytes)

total_params = sum(t.numel() for t in adapter_state.values())
print(f"Saved raw float32 delta ({len(all_bytes)} bytes, {total_params} params) -> {OUTPUT_DELTA}")
print(f"\nUpload {OUTPUT_DELTA} to ASTRA via the Upload Delta page.")
