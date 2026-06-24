"""
Client-side training and delta upload for ASTRA E2E testing.

Downloads the global model, trains on local synthetic data,
computes a weight delta, and uploads it via REST API.

Usage:
    python scripts/client_trainer.py \
        --base-url http://localhost:8000 \
        --token <JWT> \
        --client-id client1_group1 \
        --group-id group1 \
        --data-file scripts/data/client_0_data.pt \
        --local-epochs 2 \
        --lr 0.01

    Or use as a library:
        from scripts.client_trainer import train_and_upload
        result = train_and_upload(base_url, token, client_id, group_id, X, y)
"""

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow importing from project root
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
from astra.core.models.model_zoo import SimpleMLP, flatten_all_params


# ---------------------------------------------------------------------------
# Download global model weights
# ---------------------------------------------------------------------------

def download_model_weights(base_url: str, group_id: str, token: str) -> np.ndarray:
    """Download the global model as flat float32 bytes via format=raw."""
    url = f"{base_url}/api/models/{group_id}/download?format=raw"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        num_params = int(resp.headers.get("X-Num-Parameters", len(raw) // 4))
        return np.frombuffer(raw, dtype="<f4").copy()


def download_model_weights_as_tensor(
    base_url: str, group_id: str, token: str, device: str = "cpu"
) -> torch.Tensor:
    """Download global model weights as a flat torch tensor."""
    arr = download_model_weights(base_url, group_id, token)
    return torch.from_numpy(arr).float().to(device)


# ---------------------------------------------------------------------------
# Load model architecture and apply weights
# ---------------------------------------------------------------------------

def load_model_with_weights(
    base_url: str, group_id: str, token: str, device: str = "cpu"
) -> SimpleMLP:
    """Download weights and load them into a fresh SimpleMLP."""
    model = SimpleMLP(input_dim=784, num_classes=10, hidden_dim=256)
    flat_weights = download_model_weights(base_url, group_id, token, device)

    # Apply weights using the same flatten order as flatten_all_params
    params = sorted(
        [(name, param) for name, param in model.named_parameters()],
        key=lambda x: x[0],
    )
    offset = 0
    for name, param in params:
        size = param.numel()
        param.data = flat_weights[offset : offset + size].reshape(param.shape).to(device)
        offset += size

    return model


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def train_local(
    model: SimpleMLP,
    X: torch.Tensor,
    y: torch.Tensor,
    local_epochs: int = 2,
    lr: float = 0.01,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    """Train the model on local data. Returns metrics dict."""
    model.to(device)
    model.train()

    dataset = TensorDataset(X.to(device), y.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(local_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += len(y_batch)
        avg_loss = epoch_loss / total
        acc = correct / total
        history.append({"loss": avg_loss, "accuracy": acc})

    return {
        "train_loss": history[-1]["loss"],
        "train_accuracy": history[-1]["accuracy"],
        "epochs": local_epochs,
    }


# ---------------------------------------------------------------------------
# Compute and upload delta
# ---------------------------------------------------------------------------

def compute_delta(model_before: SimpleMLP, model_after: SimpleMLP) -> np.ndarray:
    """Compute delta = new_weights - old_weights (flat float32)."""
    old = flatten_all_params(model_before)
    new = flatten_all_params(model_after)
    return (new - old).astype(np.float32)


def upload_delta(
    base_url: str,
    token: str,
    client_id: str,
    client_version: int,
    delta: np.ndarray,
    dataset_size: int,
    meta: dict | None = None,
) -> dict:
    """Upload a delta to the ASTRA server via POST /api/clients/{client_id}/delta."""
    delta_bytes = delta.astype("<f4").tobytes()
    b64 = base64.b64encode(delta_bytes).decode("ascii")

    payload = {
        "client_id": client_id,
        "client_version": client_version,
        "local_updates": b64,
        "update_type": "delta",
        "local_dataset_size": dataset_size,
        "meta": meta or {},
    }

    data = json.dumps(payload).encode("utf-8")
    url = f"{base_url}/api/clients/{client_id}/delta"
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return {"status": "error", "code": e.code, "detail": body}


# ---------------------------------------------------------------------------
# High-level: train and upload in one call
# ---------------------------------------------------------------------------

def train_and_upload(
    base_url: str,
    token: str,
    client_id: str,
    group_id: str,
    X: torch.Tensor,
    y: torch.Tensor,
    client_version: int = 0,
    local_epochs: int = 2,
    lr: float = 0.01,
    batch_size: int = 32,
) -> dict:
    """End-to-end: download → train → compute delta → upload.

    Returns a dict with training metrics and upload response.
    """
    device = "cpu"

    # 1. Snapshot initial weights
    model_before = load_model_with_weights(base_url, group_id, token, device)

    # 2. Clone for training
    import copy
    model_after = copy.deepcopy(model_before)

    # 3. Train
    metrics = train_local(model_after, X, y, local_epochs, lr, batch_size, device)

    # 4. Compute delta
    delta = compute_delta(model_before, model_after)

    # 5. Upload
    response = upload_delta(
        base_url, token, client_id, client_version,
        delta, len(X), meta={"train_accuracy": metrics["train_accuracy"],
                               "train_loss": metrics["train_loss"]},
    )

    return {
        **metrics,
        "delta_norm": float(np.linalg.norm(delta)),
        "delta_size_bytes": len(delta) * 4,
        "num_params": len(delta),
        "upload_response": response,
    }


# ---------------------------------------------------------------------------
# Evaluate global model on test data
# ---------------------------------------------------------------------------

def evaluate_model(
    base_url: str, group_id: str, token: str,
    X_test: torch.Tensor, y_test: torch.Tensor,
) -> dict:
    """Download the global model and evaluate on test data."""
    model = load_model_with_weights(base_url, group_id, token)
    model.eval()

    with torch.no_grad():
        output = model(X_test)
        pred = output.argmax(dim=1)
        correct = (pred == y_test).sum().item()
        total = len(y_test)
        loss = nn.CrossEntropyLoss()(output, y_test).item()

    return {"accuracy": correct / total, "loss": loss, "total": total}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train locally and upload delta to ASTRA")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--token", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--data-file", required=True, help="Path to .pt file with X, y tensors")
    parser.add_argument("--client-version", type=int, default=0)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data = torch.load(args.data_file, weights_only=False)
    X, y = data["X"], data["y"]
    print(f"Loaded {len(X)} samples from {args.data_file}")

    result = train_and_upload(
        args.base_url, args.token, args.client_id, args.group_id,
        X, y, args.client_version, args.local_epochs, args.lr, args.batch_size,
    )

    print(f"\nTraining: loss={result['train_loss']:.4f}, acc={result['train_accuracy']:.4f}")
    print(f"Delta: {result['delta_size_bytes']} bytes, L2 norm={result['delta_norm']:.4f}")
    print(f"Upload: {json.dumps(result['upload_response'], indent=2)}")


if __name__ == "__main__":
    main()
