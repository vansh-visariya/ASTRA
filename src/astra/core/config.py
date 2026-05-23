"""
Centralized configuration loader for ASTRA.

Loads from config.yaml or environment variables, with sensible defaults.
All code should import config from here — single source of truth.
"""

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    'seed': 42,
    'dataset': {
        'name': 'MNIST',
        'split': 'dirichlet',
        'dirichlet_alpha': 0.3,
        'imbalance': True,
    },
    'model': {
        'type': 'cnn',
        'cnn': {'name': 'simple_cnn'},
        'hf': {
            'hf_model_name': 'openai/clip-vit-base-patch32',
            'quantization': '8bit',
            'gradient_checkpointing': True,
        },
    },
    'client': {
        'num_clients': 20,
        'local_epochs': 2,
        'batch_size': 32,
        'lr': 0.01,
        'weight_decay': 0.0,
    },
    'server': {
        'optimizer': 'sgd',
        'server_lr': 0.5,
        'momentum': 0.9,
        'async_lambda': 0.2,
        'aggregator_window': 10,
        'adaptive_lr': True,
        'lr_decay_factor': 0.5,
        'instability_threshold': 0.15,
    },
    'robust': {
        'method': 'fedavg',
        'trim_ratio': 0.1,
        'norm_clip': 5.0,
        'anomaly_k': 3,
        'sim_threshold': 0.2,
        'trust_power': 1.0,
    },
    'trust': {
        'init': 1.0,
        'update_alpha': 0.3,
        'quarantine_threshold': 0.35,
        'soft_decay': 0.8,
    },
    'malicious': {
        'enabled': False,
        'ratio': 0.0,
        'behaviors': [],
    },
    'privacy': {
        'dp_enabled': False,
        'dp_mode': 'client',
        'clip_norm': 1.0,
        'sigma': 1.2,
        'epsilon': 8.0,
        'epsilon_target': None,
        'noise_multiplier': 1.2,
        'max_grad_norm': 1.0,
    },
    'communication': {
        'compression': 'none',
        'topk_ratio': 0.1,
    },
    'training': {
        'total_steps': 1000,
        'eval_interval_steps': 10,
    },
    'heterogeneous': {
        'mapping_method': 'average',
        'allow_partial_updates': True,
        'min_param_overlap': 0.5,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _find_config_yaml() -> Path | None:
    candidates = [
        Path.cwd() / 'config.yaml',
        Path(__file__).parent.parent.parent.parent.parent / 'config.yaml',
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()

    if config_path is not None:
        path = Path(config_path)
    else:
        path = _find_config_yaml()

    if path is not None and path.exists():
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config:
            _deep_merge(config, yaml_config)

    env_seed = os.environ.get('ASTRA_SEED')
    if env_seed is not None:
        config['seed'] = int(env_seed)

    env_db = os.environ.get('DB_PATH')
    if env_db is not None:
        config['db_path'] = env_db

    env_secret = os.environ.get('SECRET_KEY')
    if env_secret is not None:
        config['secret_key'] = env_secret

    env_gemini = os.environ.get('GEMINI_API_KEY')
    if env_gemini is not None:
        config['gemini_api_key'] = env_gemini

    return config
