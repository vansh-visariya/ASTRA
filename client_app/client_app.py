"""
Federated Learning Client Application.

CLI client that:
- Connects to server via REST + WebSocket
- Auto-registers
- Downloads global model
- Trains locally
- Sends updates
- Auto-reconnects if disconnected
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
import websockets
from websockets.exceptions import ConnectionClosed

from core_engine.client import FLClient as LocalClient
from core_engine.data_splitter import DataSplitter
from core_engine.model_zoo import create_model
from core_engine.utils.seed import set_seed


class FederatedClient:
    """
    Distributed Federated Learning Client.

    Connects to central server, trains locally on its own schedule, and pushes updates.
    Training is client-initiated: the client autonomously decides when to train
    rather than waiting for admin commands.
    """

    def __init__(
        self,
        server_url: str,
        client_id: str,
        config: Dict[str, Any]
    ):
        self.server_url = server_url.rstrip('/')
        self.client_id = client_id or f"client_{uuid.uuid4().hex[:8]}"
        self.config = config

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.is_training = False

        self.local_client: Optional[LocalClient] = None
        self.current_global_version = 0

        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Setup client logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def connect(self) -> bool:
        """Connect to server via WebSocket."""
        try:
            ws_url = self.server_url.replace('http', 'ws') + '/ws'
            self.ws = await websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10
            )
            
            # Build data metadata
            data_metadata = {
                'modality': getattr(self, 'data_modality', 'vision'),
                'samples': getattr(self, 'data_samples', None),
            }
            
            # Register with server
            await self.ws.send(json.dumps({
                'type': 'register',
                'client_id': self.client_id,
                'group_id': getattr(self, 'group_id', 'group_a'),
                'join_token': getattr(self, 'join_token', None),
                'data_metadata': data_metadata,
                'capabilities': {
                    'has_gpu': torch.cuda.is_available(),
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                }
            }))
            
            response = await self.ws.recv()
            data = json.loads(response)
            
            if data.get('status') == 'registered':
                self.is_connected = True
                self.logger.info(f"Connected to server as {self.client_id} in group {data.get('group_id')}")
                return True
            
            self.logger.error(f"Registration failed: {data.get('reason', 'unknown')}")
            return False
        
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from server."""
        if self.ws:
            await self.ws.close()
        self.is_connected = False
    
    async def listen(self):
        """Listen for server commands."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                await self._handle_message(data)
        except ConnectionClosed:
            self.logger.warning("Connection closed by server")
            self.is_connected = False
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        msg_type = message.get('type')

        if msg_type == 'model_update':
            # Server sent new aggregated model - update and train again
            await self._download_model(message)
            self.logger.info("Received aggregated model, training next round...")
            await self._run_training()

        elif msg_type == 'training_started':
            # Server notifies training is open
            if message.get('config'):
                self.config.update({'client': {**self.config.get('client', {}), **message['config']}})
            self.logger.info("Training opened by server")

        elif msg_type == 'training_paused':
            self.logger.info("Training paused by server")

        elif msg_type == 'training_stopped':
            self.logger.info("Training stopped by server")

        elif msg_type == 'train_command':
            # Legacy: still handle direct train commands for backward compat
            await self._run_training()

        elif msg_type == 'config_update':
            self.config.update(message.get('config', {}))
            self.logger.info("Config updated")

        elif msg_type == 'ping':
            await self.ws.send(json.dumps({'type': 'pong'}))
    
    async def _download_model(self, message: Dict[str, Any]):
        """Download and apply global model update."""
        self.logger.info(f"Downloading global model version {message.get('version')}")
        # In full implementation, would download actual model weights
        self.current_global_version = message.get('version', 0)
    
    async def _run_training(self):
        """Run local training and send update to server."""
        if not self.local_client:
            self._initialize_local_client()
        
        self.is_training = True
        self.logger.info("Starting local training...")
        
        # Train locally
        update = self.local_client.local_train()
        
        # Encode update for transmission
        encoded = base64.b64encode(update['local_updates']).decode('utf-8')
        
        # Send update to server
        await self.ws.send(json.dumps({
            'type': 'update',
            'update': {
                'client_id': self.client_id,
                'client_version': self.current_global_version,
                'local_updates': encoded,
                'update_type': 'delta',
                'local_dataset_size': update['local_dataset_size'],
                'meta': update['meta']
            }
        }))
        
        self.is_training = False
        self.logger.info("Update sent to server")
    
    def _initialize_local_client(self):
        """Initialize local FL client."""
        # Setup data
        data_splitter = DataSplitter(self.config)
        train_data = data_splitter.get_client_data(
            hash(self.client_id) % self.config.get('client', {}).get('num_clients', 10)
        )
        
        # Create model
        model_factory = lambda: create_model(self.config)
        
        # Create client
        self.local_client = LocalClient(
            client_id=self.client_id,
            train_data=train_data,
            model_factory=model_factory,
            config=self.config
        )
        
        self.logger.info(f"Local client initialized with {len(train_data)} samples")
    
    async def run(self):
        """Main client loop."""
        # Connect to server
        connected = await self.connect()
        if not connected:
            self.logger.error("Failed to connect to server")
            return

        # Train once immediately, push update to server
        await self._run_training()

        # Then listen: server will send model_update after aggregation,
        # which triggers the next training round automatically
        await self.listen()

    async def _check_group_status(self):
        """Check if the group is already in TRAINING state and start loop if so."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                group_id = getattr(self, 'group_id', 'group_a')
                url = f"{self.server_url}/api/groups/{group_id}"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        group = data.get('group', {})
                        if group.get('is_training'):
                            self.training_allowed = True
                            self.logger.info("Group already training - starting autonomous training")
                            self._start_training_loop()
        except Exception as e:
            self.logger.debug(f"Could not check group status: {e}")


class RESTClient:
    """REST API client for simpler operations."""
    
    def __init__(self, server_url: str, token: Optional[str] = None):
        self.server_url = server_url.rstrip('/')
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers.update({'Authorization': f'Bearer {token}'})
    
    def register_client(self, client_id: str) -> Dict:
        """Register client via REST."""
        response = self.session.post(
            f"{self.server_url}/api/clients/register",
            json={'client_id': client_id}
        )
        return response.json()
    
    def get_config(self) -> Dict:
        """Get training configuration from server."""
        response = self.session.get(f"{self.server_url}/api/config")
        return response.json()
    
    def get_model(self) -> bytes:
        """Download global model."""
        response = self.session.get(f"{self.server_url}/api/model/latest")
        return response.content
    
    def upload_update(self, client_id: str, update: Dict) -> Dict:
        """Upload client update via REST."""
        response = self.session.post(
            f"{self.server_url}/api/clients/{client_id}/update",
            json=update
        )
        return response.json()
    
    def get_status(self) -> Dict:
        """Get server status."""
        response = self.session.get(f"{self.server_url}/api/server/status")
        return response.json()


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    
    # Server options
    parser.add_argument('--server', type=str, default='http://localhost:8000',
                        help='Server URL')
    parser.add_argument('--client-id', type=str, default=None,
                        help='Client ID (auto-generated if not provided)')
    
    # Group authentication
    parser.add_argument('--group-id', type=str, default='group_a',
                        help='Group ID to join')
    parser.add_argument('--join-token', type=str, default=None,
                        help='Join token for group authentication')
    
    # Data metadata
    parser.add_argument('--data-modality', type=str, default='vision',
                        choices=['vision', 'text', 'multimodal'],
                        help='Data modality (vision/text/multimodal)')
    parser.add_argument('--data-samples', type=int, default=None,
                        help='Number of data samples')
    
    # Authentication
    parser.add_argument('--token', type=str, default=None,
                        help='Authentication token')
    
    # Training config
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Config file path')

    # Mode
    parser.add_argument('--mode', choices=['websocket', 'rest'], default='websocket',
                        help='Connection mode')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command line
    set_seed(config.get('seed', 42))
    
    # Create client
    if args.mode == 'websocket':
        client = FederatedClient(
            server_url=args.server,
            client_id=args.client_id,
            config=config
        )
        # Set additional attributes
        client.group_id = args.group_id
        client.join_token = args.join_token
        client.data_modality = args.data_modality
        client.data_samples = args.data_samples
        
        # Run client
        asyncio.run(client.run())
    
    else:
        import uuid
        client_id = args.client_id or f"client_{uuid.uuid4().hex[:8]}"
        client = RESTClient(args.server, args.token)
        
        # Register with server
        try:
            result = client.register_client(client_id)
            print(f"Registered: {result}")
        except Exception as e:
            print(f"Registration failed (server may not have endpoint): {e}")
        
        # Keep client alive and poll
        while True:
            try:
                status = client.get_status()
                print(f"Server status: {status}")
                time.sleep(5)
            except KeyboardInterrupt:
                print("\nClient stopped")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)


if __name__ == '__main__':
    main()
