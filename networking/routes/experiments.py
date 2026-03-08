"""
Experiment management REST endpoints.
"""

from fastapi import APIRouter, HTTPException

from networking.models import ExperimentConfig, ControlCommand
from networking.state import get_fl_server

router = APIRouter()


@router.post("/api/experiments/start")
async def start_experiment(config: ExperimentConfig):
    """Start a new federated learning experiment."""
    fl_server = get_fl_server()
    fl_server.start_experiment(config.experiment_id, config.config)
    return {"status": "started", "experiment_id": config.experiment_id}


@router.post("/api/experiments/{experiment_id}/control")
async def control_experiment(experiment_id: str, command: ControlCommand):
    """Control experiment (pause, resume, stop)."""
    fl_server = get_fl_server()
    if command.command == "pause":
        fl_server.pause_experiment()
    elif command.command == "resume":
        fl_server.resume_experiment()
    elif command.command == "stop":
        fl_server.stop_experiment()

    return {"status": "ok", "command": command.command}


@router.get("/api/experiments/{experiment_id}/metrics")
async def get_experiment_metrics(experiment_id: str):
    """Get experiment metrics history."""
    fl_server = get_fl_server()
    metrics = fl_server.db.get_experiment_metrics(experiment_id)
    return {"experiment_id": experiment_id, "metrics": metrics}
