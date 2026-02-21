"""
Standalone federated learning client runner.

Designed to run inside its own Docker container. On each round it:
  1. Fetches the latest global model from the server.
  2. Loads a local data shard (deterministic by CLIENT_ID + seed).
  3. Trains for `local_epochs` on that shard.
  4. Uploads the weight delta to the server over HTTP.
  5. Waits briefly, then repeats.

Environment variables
---------------------
SERVER_URL   : base URL of the FL server  (default: http://server:5000)
CLIENT_ID    : unique identifier           (default: client_000)
NUM_ROUNDS   : how many training rounds    (default: read from server config)
"""

import io
import json
import logging
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from federated.model_zoo import create_model
from federated.data_splitter import DataSplitter
from federated.privacy import clip_and_noise
from federated.compression import topk_sparsify
from federated.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fl_client")

SERVER_URL = os.environ.get("SERVER_URL", "http://server:5000")
CLIENT_ID = os.environ.get("CLIENT_ID", "client_000")
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "0"))  # 0 = use server config


# ── helpers ─────────────────────────────────────────────────────────────

def fetch_config() -> Dict[str, Any]:
    """Download training configuration from the server."""
    url = f"{SERVER_URL}/api/config"
    logger.info(f"Fetching config from {url}")
    for attempt in range(30):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            wait = min(2 ** attempt, 30)
            logger.warning(f"Server not ready, retrying in {wait}s …")
            time.sleep(wait)
    logger.error("Could not reach the server after 30 attempts – exiting")
    sys.exit(1)


def fetch_global_model(model: nn.Module) -> None:
    """Download the current global model weights and load them."""
    url = f"{SERVER_URL}/api/model"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    buf = io.BytesIO(resp.content)
    state_dict = torch.load(buf, map_location="cpu")
    model.load_state_dict(state_dict)
    logger.info("Global model weights loaded")


def upload_update(update: Dict[str, Any]) -> Dict[str, Any]:
    """POST the training update to the server."""
    url = f"{SERVER_URL}/api/update"

    # Convert numpy delta to a plain list for JSON serialisation
    payload = {
        "client_id": update["client_id"],
        "client_version": update["client_version"],
        "local_updates": update["local_updates"].tolist(),
        "update_type": update.get("update_type", "delta"),
        "local_dataset_size": update.get("local_dataset_size", 0),
        "timestamp": update.get("timestamp", time.time()),
        "meta": update.get("meta", {}),
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ── local training ──────────────────────────────────────────────────────

def get_weights(model: nn.Module) -> np.ndarray:
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights)


def local_train(
    model: nn.Module,
    train_loader: DataLoader,
    config: Dict[str, Any],
    client_version: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run local SGD and return the update dict."""
    model.train()
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["client"]["lr"],
        weight_decay=config["client"].get("weight_decay", 0.0),
    )

    local_epochs = config["client"]["local_epochs"]
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    initial_weights = get_weights(model)

    for _ in range(local_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(target)
            total_correct += (output.argmax(dim=1) == target).sum().item()
            total_samples += len(target)

    final_weights = get_weights(model)
    weight_delta = final_weights - initial_weights

    # Optional client-side DP
    if config["privacy"]["dp_enabled"] and config["privacy"]["dp_mode"] == "client":
        weight_delta = clip_and_noise(
            weight_delta,
            config["privacy"]["clip_norm"],
            config["privacy"]["sigma"],
        )

    # Optional compression
    if config["communication"]["compression"] == "topk":
        k_ratio = config["communication"].get("topk_ratio", 0.1)
        weight_delta, _ = topk_sparsify(weight_delta, k_ratio)

    train_loss = total_loss / total_samples if total_samples else 0.0
    train_acc = total_correct / total_samples if total_samples else 0.0

    return {
        "client_id": CLIENT_ID,
        "client_version": client_version,
        "local_updates": weight_delta,
        "update_type": "delta",
        "local_dataset_size": total_samples,
        "timestamp": time.time(),
        "meta": {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "local_steps": local_epochs * len(train_loader),
        },
    }


# ── main loop ───────────────────────────────────────────────────────────

def main():
    logger.info(f"Starting client {CLIENT_ID}  server={SERVER_URL}")

    config = fetch_config()
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Build local model (same architecture as server)
    model = create_model(config)

    # Build local data shard
    # Extract the numeric index from the CLIENT_ID (e.g. "client_002" -> 2)
    try:
        client_index = int(CLIENT_ID.split("_")[-1])
    except ValueError:
        client_index = hash(CLIENT_ID) % config["client"]["num_clients"]

    data_splitter = DataSplitter(config)
    data_splitter.split_data()
    train_data = data_splitter.get_client_data(client_index)
    train_loader = DataLoader(
        train_data,
        batch_size=config["client"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    logger.info(f"Local dataset size: {len(train_data)} samples")

    total_rounds = NUM_ROUNDS if NUM_ROUNDS > 0 else config["training"]["total_steps"]
    client_version = 0

    for rnd in range(1, total_rounds + 1):
        logger.info(f"── Round {rnd}/{total_rounds} ──")

        # 1. Sync with global model
        fetch_global_model(model)

        # 2. Train locally
        client_version += 1
        update = local_train(model, train_loader, config, client_version, device)
        logger.info(
            f"Local train done  loss={update['meta']['train_loss']:.4f}  "
            f"acc={update['meta']['train_accuracy']:.4f}"
        )

        # 3. Upload delta
        resp = upload_update(update)
        logger.info(f"Upload OK  server_version={resp.get('global_version')}")

    logger.info("All rounds completed – client shutting down")


if __name__ == "__main__":
    main()
