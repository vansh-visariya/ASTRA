"""
System-level REST endpoints: root, health, metrics, status, logs.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from networking.state import get_fl_server

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Federated Learning API", "version": "1.0.0"}


@router.get("/health")
async def health():
    from networking import state
    return {"status": "healthy", "server_ready": state.fl_server is not None}


@router.get("/api/system/metrics")
async def get_system_metrics():
    """Get system-wide metrics for dashboard."""
    fl_server = get_fl_server()
    groups = fl_server.group_manager.get_all_groups()
    clients = fl_server.group_manager.get_all_client_status()

    active_groups = [g for g in groups if g.get('status') == 'TRAINING']
    dp_enabled = sum(1 for g in groups if g.get('config', {}).get('dp_enabled', False))

    latest_metric = None
    latest_group_id = None
    for group in groups:
        history = group.get('metrics_history') or []
        if not history:
            continue
        candidate = history[-1]
        if not latest_metric or candidate.get('timestamp', 0) > latest_metric.get('timestamp', 0):
            latest_metric = candidate
            latest_group_id = group.get('group_id')

    return {
        "total_groups": len(groups),
        "active_groups": len(active_groups),
        "total_participants": len(clients),
        "active_participants": len([c for c in clients if c.get('status') == 'active']),
        "dp_enabled_groups": dp_enabled,
        "total_aggregations": sum(g.get('model_version', 0) for g in groups),
        "latest_group_id": latest_group_id,
        "latest_accuracy": (latest_metric or {}).get('accuracy', 0),
        "latest_loss": (latest_metric or {}).get('loss', 0),
        "latest_version": (latest_metric or {}).get('version', 0),
        "latest_timestamp": (latest_metric or {}).get('timestamp', 0)
    }


@router.get("/api/server/status")
async def get_server_status():
    """Get server status."""
    fl_server = get_fl_server()
    return {
        "running": fl_server.is_running,
        "paused": fl_server.is_paused,
        "experiment_id": fl_server.experiment_id,
        "global_version": fl_server.server.global_version if fl_server.server else 0,
        "connected_clients": len(fl_server.connection_manager.client_sockets)
    }


@router.get("/api/logs")
async def get_logs(limit: int = 100, event_type: Optional[str] = None, group_id: Optional[str] = None):
    """Get server event logs."""
    fl_server = get_fl_server()
    logs = fl_server.group_manager.get_logs(limit, event_type, group_id)
    return {"logs": logs, "count": len(logs)}
