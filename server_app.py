"""
FastAPI server for the Async Federated Learning framework.

Exposes HTTP endpoints so that individual client containers can:
  - GET  /api/config   -> fetch training configuration
  - GET  /api/model    -> download the current global model weights
  - POST /api/update   -> submit a local training update
  - GET  /api/status   -> server health and training status
"""

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from federated.server import AsyncServer
from federated.aggregator import create_aggregator
from federated.model_zoo import create_model
from federated.data_splitter import DataSplitter
from federated.utils.seed import set_seed
from federated.utils.logging_utils import setup_logging, get_logger
from federated.utils.metrics import MetricsTracker


# ── globals populated by lifespan ───────────────────────────────────────
server: AsyncServer = None
config: Dict[str, Any] = {}
metrics_tracker: MetricsTracker = None
log_dir: Path = None
start_time: float = 0.0
updates_received: int = 0


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def init_server():
    """Initialise the global model, aggregator and async server."""
    global server, config, metrics_tracker, log_dir, start_time

    config = load_config()
    set_seed(config["seed"])

    config.setdefault("experiment_id", f"docker_exp_{int(time.time())}")

    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "plots").mkdir(exist_ok=True)
    (log_dir / "logs").mkdir(exist_ok=True)
    setup_logging(log_dir, config["logging"]["jsonl_logging"])

    logger = get_logger(__name__)
    logger.info("Initialising federated server …")

    model = create_model(config)
    aggregator = create_aggregator(config)

    data_splitter = DataSplitter(config)
    _, val_loader = data_splitter.create_data_loaders()

    server = AsyncServer(
        model=model,
        aggregator=aggregator,
        config=config,
        val_loader=val_loader,
    )
    server.start()

    metrics_tracker = MetricsTracker(
        experiment_id=config["experiment_id"],
        log_dir=log_dir,
        config=config,
    )

    start_time = time.time()
    logger.info("Server ready – waiting for client updates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_server()
    yield


app = FastAPI(
    title="Async Federated Learning Server",
    description="Aggregation server for distributed federated learning clients",
    lifespan=lifespan,
)


# ── Pydantic models ────────────────────────────────────────────────────

class UpdateMeta(BaseModel):
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    local_steps: int = 0


class ClientUpdate(BaseModel):
    client_id: str
    client_version: int = 0
    local_updates: List[float]
    update_type: str = "delta"
    local_dataset_size: int = 1
    timestamp: Optional[float] = None
    meta: UpdateMeta = UpdateMeta()


class UpdateResponse(BaseModel):
    status: str
    global_version: int


class StatusResponse(BaseModel):
    running: bool
    global_version: int
    updates_received: int
    uptime_seconds: float


# ── API endpoints ───────────────────────────────────────────────────────

@app.get("/api/config")
def get_config():
    """Return the training configuration so clients can self-configure."""
    return config


@app.get("/api/model")
def get_model():
    """Return current global model state_dict as a binary .pt file."""
    buf = io.BytesIO()
    torch.save(server.model.state_dict(), buf)
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=global_model.pt"},
    )


@app.post("/api/update", response_model=UpdateResponse)
def post_update(data: ClientUpdate):
    """Receive a training update from a client."""
    global updates_received

    delta = np.array(data.local_updates, dtype=np.float32)

    update = {
        "client_id": data.client_id,
        "client_version": data.client_version,
        "local_updates": delta,
        "update_type": data.update_type,
        "local_dataset_size": data.local_dataset_size,
        "timestamp": data.timestamp or time.time(),
        "meta": data.meta.model_dump(),
    }

    server.handle_update(update)
    updates_received += 1

    logger = get_logger(__name__)
    logger.info(
        f"Update #{updates_received} from {data.client_id}  "
        f"loss={data.meta.train_loss}"
    )

    eval_interval = config["training"]["eval_interval_steps"]
    if updates_received % eval_interval == 0:
        metrics = server.evaluate()
        metrics_tracker.log_metrics(updates_received, metrics)
        logger.info(
            f"Eval @ update {updates_received}: "
            f"acc={metrics.get('accuracy', 0):.4f}  loss={metrics.get('loss', 0):.4f}"
        )
        server.save_checkpoint(str(log_dir / f"checkpoint_{updates_received}.pt"))

    return UpdateResponse(status="ok", global_version=server.global_version)


@app.get("/api/status", response_model=StatusResponse)
def get_status():
    """Health / status endpoint."""
    return StatusResponse(
        running=server.running if server else False,
        global_version=server.global_version if server else 0,
        updates_received=updates_received,
        uptime_seconds=round(time.time() - start_time, 1),
    )


# ── Entrypoint ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server_app:app", host="0.0.0.0", port=5000)
