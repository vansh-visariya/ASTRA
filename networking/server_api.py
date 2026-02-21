"""
FastAPI Server for Distributed Federated Learning.

Provides:
- REST API for training control
- WebSocket for live updates
- Client registration and management
- Experiment tracking with SQLite
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core_engine.aggregator import create_aggregator
from core_engine.data_splitter import DataSplitter
from core_engine.server import AsyncServer
from core_engine.utils.seed import set_seed
from model_registry.registry import get_registry


# ============================================================================
# Data Models
# ============================================================================

class ClientRegister(BaseModel):
    client_id: str
    capabilities: Dict[str, Any] = {}

class ClientUpdate(BaseModel):
    client_id: str
    client_version: int
    local_updates: str  # Base64 encoded
    update_type: str = "delta"
    local_dataset_size: int
    meta: Dict[str, Any] = {}

class ExperimentConfig(BaseModel):
    experiment_id: str
    config: Dict[str, Any]

class ControlCommand(BaseModel):
    command: str  # start, pause, resume, stop
    params: Dict[str, Any] = {}


# ============================================================================
# Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for live updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_sockets: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass
    
    async def send_to(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client."""
        if client_id in self.client_sockets:
            try:
                await self.client_sockets[client_id].send_json(message)
            except Exception:
                pass
    
    def register_client(self, client_id: str, websocket: WebSocket):
        self.client_sockets[client_id] = websocket
    
    def unregister_client(self, client_id: str):
        if client_id in self.client_sockets:
            del self.client_sockets[client_id]


# ============================================================================
# Database Manager
# ============================================================================

class ExperimentDB:
    """SQLite-based experiment tracking."""
    
    def __init__(self, db_path: str = "./experiments.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY,
                experiment_id TEXT UNIQUE,
                config_json TEXT,
                status TEXT,
                start_time TEXT,
                end_time TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY,
                experiment_id TEXT,
                step INTEGER,
                timestamp TEXT,
                metrics_json TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clients (
                id INTEGER PRIMARY KEY,
                client_id TEXT UNIQUE,
                experiment_id TEXT,
                status TEXT,
                trust_score REAL,
                last_seen TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_experiment(self, experiment_id: str, config: Dict) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO experiments (experiment_id, config_json, status, start_time) VALUES (?, ?, ?, ?)',
            (experiment_id, json.dumps(config), 'pending', datetime.now().isoformat())
        )
        
        conn.commit()
        conn.close()
    
    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        end_time = datetime.now().isoformat() if status in ['completed', 'failed'] else None
        
        cursor.execute(
            'UPDATE experiments SET status = ?, end_time = ? WHERE experiment_id = ?',
            (status, end_time, experiment_id)
        )
        
        conn.commit()
        conn.close()
    
    def log_metrics(self, experiment_id: str, step: int, metrics: Dict) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO metrics (experiment_id, step, timestamp, metrics_json) VALUES (?, ?, ?, ?)',
            (experiment_id, step, datetime.now().isoformat(), json.dumps(metrics))
        )
        
        conn.commit()
        conn.close()
    
    def register_client(self, client_id: str, experiment_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT OR REPLACE INTO clients (client_id, experiment_id, status, trust_score, last_seen) VALUES (?, ?, ?, ?, ?)',
            (client_id, experiment_id, 'active', 1.0, datetime.now().isoformat())
        )
        
        conn.commit()
        conn.close()
    
    def update_client(self, client_id: str, trust_score: float, status: str = 'active') -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE clients SET trust_score = ?, status = ?, last_seen = ? WHERE client_id = ?',
            (trust_score, status, datetime.now().isoformat(), client_id)
        )
        
        conn.commit()
        conn.close()
    
    def get_experiment_metrics(self, experiment_id: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT step, timestamp, metrics_json FROM metrics WHERE experiment_id = ? ORDER BY step',
            (experiment_id,)
        )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'step': row[0],
                'timestamp': row[1],
                **json.loads(row[2])
            })
        
        conn.close()
        return results


# ============================================================================
# FL Server Application
# ============================================================================

class FLServer:
    """Federated Learning Server with API."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_manager = ConnectionManager()
        self.db = ExperimentDB()
        
        self.server: Optional[AsyncServer] = None
        self.model_registry = get_registry()
        
        self.experiment_id: Optional[str] = None
        self.is_running = False
        self.is_paused = False
        
        self.logger = logging.getLogger(__name__)
        
        self._setup_server()
    
    def _setup_server(self):
        """Initialize the FL server."""
        from core_engine.model_zoo import create_model
        
        # Create model
        model = create_model(self.config)
        
        # Create aggregator
        aggregator = create_aggregator(self.config)
        
        # Create data splitter (for validation)
        data_splitter = DataSplitter(self.config)
        _, val_loader = data_splitter.create_data_loaders()
        
        # Create async server
        self.server = AsyncServer(
            model=model,
            aggregator=aggregator,
            config=self.config,
            val_loader=val_loader
        )
        
        self.logger.info("FL Server initialized")
    
    async def handle_client_register(self, client_id: str, capabilities: Dict) -> Dict:
        """Handle client registration."""
        self.connection_manager.register_client(client_id, None)
        self.db.register_client(client_id, self.experiment_id or 'default')
        
        self.logger.info(f"Client registered: {client_id}")
        
        return {
            'status': 'registered',
            'client_id': client_id,
            'config': self.config
        }
    
    async def handle_client_update(self, update: ClientUpdate) -> Dict:
        """Handle incoming client update."""
        if not self.server or not self.is_running or self.is_paused:
            return {'status': 'rejected', 'reason': 'server_not_ready'}
        
        # Decode update (simplified - would normally use base64)
        import base64
        try:
            delta_bytes = base64.b64decode(update.local_updates)
            delta = np.frombuffer(delta_bytes, dtype=np.float32)
        except Exception:
            delta = np.array([])
        
        client_update = {
            'client_id': update.client_id,
            'client_version': update.client_version,
            'local_updates': delta.tobytes(),
            'update_type': update.update_type,
            'local_dataset_size': update.local_dataset_size,
            'timestamp': time.time(),
            'meta': update.meta
        }
        
        # Process update
        self.server.handle_update(client_update)
        
        # Broadcast update to dashboard
        await self.connection_manager.broadcast({
            'type': 'client_update',
            'client_id': update.client_id,
            'step': self.server.global_version
        })
        
        return {'status': 'accepted', 'global_version': self.server.global_version}
    
    async def get_global_model(self) -> Dict:
        """Get global model state (simplified)."""
        if not self.server:
            return {}
        
        return {
            'global_version': self.server.global_version,
            'model_type': 'simple_cnn'
        }
    
    def start_experiment(self, experiment_id: str, config: Dict) -> None:
        """Start a new experiment."""
        self.experiment_id = experiment_id
        self.config = config
        
        set_seed(config.get('seed', 42))
        
        self.db.create_experiment(experiment_id, config)
        self.db.update_experiment_status(experiment_id, 'running')
        
        self.is_running = True
        self.is_paused = False
        
        self._setup_server()
        
        self.logger.info(f"Experiment started: {experiment_id}")
    
    def pause_experiment(self) -> None:
        self.is_paused = True
        self.logger.info("Experiment paused")
    
    def resume_experiment(self) -> None:
        self.is_paused = False
        self.logger.info("Experiment resumed")
    
    def stop_experiment(self) -> None:
        self.is_running = False
        if self.experiment_id:
            self.db.update_experiment_status(self.experiment_id, 'completed')
        self.logger.info("Experiment stopped")


# ============================================================================
# FastAPI App with Socket.IO
# ============================================================================

fl_server: Optional[FLServer] = None
socketio_app: Optional[AsyncServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fl_server, socketio_app
    
    config = {
        'seed': 42,
        'dataset': {'name': 'MNIST', 'split': 'dirichlet', 'dirichlet_alpha': 0.3},
        'model': {'type': 'cnn', 'cnn': {'name': 'simple_cnn'}},
        'client': {'num_clients': 10, 'local_epochs': 2, 'batch_size': 32, 'lr': 0.01},
        'server': {'optimizer': 'sgd', 'server_lr': 0.5, 'momentum': 0.9, 'async_lambda': 0.2, 'aggregator_window': 5},
        'robust': {'method': 'fedavg', 'trim_ratio': 0.1},
        'privacy': {'dp_enabled': False},
        'training': {'total_steps': 1000, 'eval_interval_steps': 10}
    }
    
    fl_server = FLServer(config)
    
    # Setup Socket.IO
    from aiohttp import web
    socketio_app = web.Application()
    
    yield
    
    if fl_server:
        fl_server.stop_experiment()


app = FastAPI(title="Federated Learning API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Socket.IO support
from fastapi_socketio import SocketManager

socket_manager = SocketManager(app, cors_allowed_origins="*")


@socket_manager.on('connect')
async def connect(sid, environ):
    print(f"Client connected: {sid}")


@socket_manager.on('disconnect')
async def disconnect(sid):
    print(f"Client disconnected: {sid}")


@socket_manager.on('register')
async def register(sid, data):
    """Handle client registration via Socket.IO"""
    client_id = data.get('client_id')
    capabilities = data.get('capabilities', {})
    
    if fl_server:
        result = await fl_server.handle_client_register(client_id, capabilities)
        await socket_manager.emit('registered', result, room=sid)


@socket_manager.on('update')
async def handle_update(sid, data):
    """Handle client update via Socket.IO"""
    if not fl_server:
        return
    
    try:
        update = ClientUpdate(**data)
        result = await fl_server.handle_client_update(update)
        await socket_manager.emit('update_ack', result, room=sid)
    except Exception as e:
        await socket_manager.emit('error', {'message': str(e)}, room=sid)


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {"message": "Federated Learning API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "server_ready": fl_server is not None}


@app.post("/api/experiments/start")
async def start_experiment(config: ExperimentConfig):
    """Start a new federated learning experiment."""
    fl_server.start_experiment(config.experiment_id, config.config)
    return {"status": "started", "experiment_id": config.experiment_id}


@app.post("/api/experiments/{experiment_id}/control")
async def control_experiment(experiment_id: str, command: ControlCommand):
    """Control experiment (pause, resume, stop)."""
    if command.command == "pause":
        fl_server.pause_experiment()
    elif command.command == "resume":
        fl_server.resume_experiment()
    elif command.command == "stop":
        fl_server.stop_experiment()
    
    return {"status": "ok", "command": command.command}


@app.get("/api/experiments/{experiment_id}/metrics")
async def get_experiment_metrics(experiment_id: str):
    """Get experiment metrics history."""
    metrics = fl_server.db.get_experiment_metrics(experiment_id)
    return {"experiment_id": experiment_id, "metrics": metrics}


@app.get("/api/models")
async def list_models():
    """List registered models."""
    models = fl_server.model_registry.list_models()
    return {"models": models}


@app.post("/api/models/register")
async def register_model(model_id: str, model_type: str, model_source: str, config: Dict):
    """Register a new model."""
    if model_source == "huggingface":
        model_info = fl_server.model_registry.register_hf_model(
            config['model_name'],
            use_peft=config.get('use_peft', False),
            peft_config=config.get('peft_config')
        )
    elif model_source == "custom":
        model_info = fl_server.model_registry.register_custom_architecture(
            model_id,
            config['architecture'],
            model_type,
            config
        )
    
    return {"status": "registered", "model": model_info.to_dict()}


@app.get("/api/clients")
async def list_clients():
    """List connected clients."""
    clients = list(fl_server.connection_manager.client_sockets.keys())
    return {"clients": clients, "count": len(clients)}


@app.post("/api/clients/register")
async def register_client(client: ClientRegister):
    """Register a client via REST."""
    client_id = client.client_id
    capabilities = client.capabilities
    
    # Register in database
    fl_server.db.register_client(client_id, fl_server.experiment_id or 'default')
    
    # Add to connected clients
    fl_server.connection_manager.register_client(client_id, None)
    
    return {"status": "registered", "client_id": client_id}


@app.get("/api/server/status")
async def get_server_status():
    """Get server status."""
    return {
        "running": fl_server.is_running,
        "paused": fl_server.is_paused,
        "experiment_id": fl_server.experiment_id,
        "global_version": fl_server.server.global_version if fl_server.server else 0,
        "connected_clients": len(fl_server.connection_manager.client_sockets)
    }


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live updates."""
    await fl_server.connection_manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'register':
                await fl_server.handle_client_register(
                    message['client_id'],
                    message.get('capabilities', {})
                )
            
            elif message.get('type') == 'update':
                update = ClientUpdate(**message['update'])
                result = await fl_server.handle_client_update(update)
                await websocket.send_json(result)
    
    except WebSocketDisconnect:
        fl_server.connection_manager.disconnect(websocket)


# ============================================================================
# Main
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
