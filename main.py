#!/usr/bin/env python3
"""
Main entry point for the Async Federated Learning framework.

This module initializes the server, clients, and runs the training loop.
Supports both CNN (MNIST/CIFAR) and HuggingFace multimodal models with PEFT.

References:
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from federated.server import AsyncServer
from federated.client import FLClient
from federated.data_splitter import DataSplitter
from federated.model_zoo import create_model
from federated.hf_models import load_hf_peft_model
from federated.utils.seed import set_seed
from federated.utils.logging_utils import setup_logging, get_logger
from federated.utils.metrics import MetricsTracker


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment_dirs(config: Dict[str, Any]) -> Path:
    """Create experiment directories."""
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / 'plots').mkdir(exist_ok=True)
    (log_dir / 'logs').mkdir(exist_ok=True)
    return log_dir


def create_clients(config: Dict[str, Any], data_splitter: DataSplitter, 
                  server: AsyncServer, model_factory: callable) -> Dict[str, FLClient]:
    """Create federated learning clients."""
    clients = {}
    num_clients = config['client']['num_clients']
    
    for i in range(num_clients):
        client_id = f"client_{i:03d}"
        train_data = data_splitter.get_client_data(i)
        
        client = FLClient(
            client_id=client_id,
            train_data=train_data,
            model_factory=model_factory,
            config=config
        )
        clients[client_id] = client
    
    return clients


def run_training_loop(
    server: AsyncServer,
    clients: Dict[str, FLClient],
    config: Dict[str, Any],
    metrics_tracker: MetricsTracker,
    log_dir: Path
) -> None:
    """Run the main asynchronous federated learning training loop."""
    logger = get_logger(__name__)
    
    total_steps = config['training']['total_steps']
    eval_interval = config['training']['eval_interval_steps']
    
    logger.info(f"Starting training for {total_steps} steps")
    print(f"[TRAINING] Starting {total_steps} steps with {len(clients)} clients")
    
    server.start()
    
    step = 0
    while step < total_steps:
        for client_id, client in clients.items():
            print(f"[CLIENT {client_id}] Training...", end=" ", flush=True)
            update = client.local_train()
            print(f"Done. Loss={update['meta']['train_loss']:.4f}")
            server.handle_update(update)
        
        step += 1
        print(f"[STEP {step}/{total_steps}] Completed", flush=True)
        
        if step % eval_interval == 0:
            print(f"[EVAL] Evaluating global model...", flush=True)
            metrics = server.evaluate()
            metrics_tracker.log_metrics(step, metrics)
            logger.info(f"Step {step}: Accuracy={metrics.get('accuracy', 0):.4f}, Loss={metrics.get('loss', 0):.4f}")
            print(f"[EVAL] Step {step}: Accuracy={metrics.get('accuracy', 0):.4f}, Loss={metrics.get('loss', 0):.4f}")
            
            checkpoint_path = log_dir / f"checkpoint_step_{step}.pt"
            server.save_checkpoint(str(checkpoint_path))
    
    print(f"[TRAINING] Completed {total_steps} steps!")
    logger.info("Training completed")
    server.stop()


def run_demo_mode(config: Dict[str, Any]) -> None:
    """Run a quick demo with reduced parameters."""
    logger = get_logger(__name__)
    logger.info("Running in DEMO mode")
    print("="*50)
    print("DEMO MODE: Minimal config for testing")
    print("="*50)
    
    # Check for GPU
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print(f"[GPU] CUDA available: {torch.cuda.get_device_name(0)}")
        print("[GPU] Will use GPU for training")
    else:
        print("[CPU] No GPU detected, using CPU")
    
    # GPU-optimized settings
    if has_gpu:
        config['client']['num_clients'] = 5
        config['training']['total_steps'] = 20
        config['training']['eval_interval_steps'] = 5
        config['server']['aggregator_window'] = 5
        config['client']['local_epochs'] = 2
    else:
        # CPU - much smaller
        config['client']['num_clients'] = 2
        config['training']['total_steps'] = 5
        config['training']['eval_interval_steps'] = 1
        config['server']['aggregator_window'] = 2
        config['client']['local_epochs'] = 1
    
    config['robust']['method'] = 'fedavg'
    config['privacy']['dp_enabled'] = False
    config['malicious']['ratio'] = 0.0
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Async Federated Learning Framework')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo mode with reduced parameters')
    parser.add_argument('--experiment-id', type=str, default=None,
                        help='Experiment ID for logging')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed override')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.demo:
        config = run_demo_mode(config)
    
    if args.seed is not None:
        config['seed'] = args.seed
    
    if args.experiment_id:
        config['experiment_id'] = args.experiment_id
    else:
        config['experiment_id'] = f"exp_{int(time.time())}"
    
    set_seed(config['seed'])
    
    # Set device globally
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}")
    if torch.cuda.is_available():
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
    
    log_dir = setup_experiment_dirs(config)
    setup_logging(log_dir, config['logging']['jsonl_logging'])
    
    logger = get_logger(__name__)
    logger.info(f"Experiment ID: {config['experiment_id']}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    data_splitter = DataSplitter(config)
    train_loaders, val_loader = data_splitter.create_data_loaders()
    
    if config['model']['type'] == 'hf_transformer' or config['peft']['enabled']:
        model, tokenizer = load_hf_peft_model(
            config['model'].get('hf', {}).get('hf_model_name', 'openai/clip-vit-base-patch32'),
            config['peft'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        model_factory = lambda: model
    else:
        model = create_model(config['model'])
        model_factory = lambda: create_model(config['model'])
    
    from federated.aggregator import create_aggregator
    aggregator = create_aggregator(config)
    
    server = AsyncServer(
        model=model_factory(),
        aggregator=aggregator,
        config=config,
        val_loader=val_loader
    )
    
    clients = create_clients(config, data_splitter, server, model_factory)
    
    metrics_tracker = MetricsTracker(
        experiment_id=config['experiment_id'],
        log_dir=log_dir,
        config=config
    )
    
    try:
        run_training_loop(server, clients, config, metrics_tracker, log_dir)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        metrics_tracker.save_all()
        server.save_checkpoint(str(log_dir / 'final_model.pt'))
        
        metadata = {
            'experiment_id': config['experiment_id'],
            'config': config,
            'seed': config['seed'],
            'completed': True
        }
        with open(log_dir / 'experiment_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
