"""
Logging utilities for federated learning experiments.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def setup_logging(
    log_dir: Path,
    jsonl_logging: bool = True,
    level: int = logging.INFO
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files.
        jsonl_logging: Whether to use JSONL format.
        level: Logging level.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    file_handler = logging.FileHandler(log_dir / 'logs' / 'debug.log')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        handlers=handlers
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


class JSONLLogger:
    """JSON Lines logger for machine-readable experiment logs."""
    
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = open(self.log_path, 'a')
    
    def log(self, data: Dict[str, Any]) -> None:
        """Log a JSON entry."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            **data
        }
        self.file_handle.write(json.dumps(entry) + '\n')
        self.file_handle.flush()
    
    def close(self) -> None:
        """Close the file handle."""
        self.file_handle.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ExperimentMetadata:
    """Track experiment metadata."""
    
    def __init__(self, config: Dict[str, Any], log_dir: Path):
        self.config = config
        self.log_dir = Path(log_dir)
        self.metadata: Dict[str, Any] = {}
        
        self._init_metadata()
    
    def _init_metadata(self) -> None:
        """Initialize metadata."""
        self.metadata = {
            'experiment_id': self.config.get('experiment_id', 'unknown'),
            'seed': self.config.get('seed', 42),
            'timestamp_start': datetime.now().isoformat(),
            'config': self.config,
            'git_hash': self._get_git_hash(),
            'pip_freeze': self._get_pip_freeze()
        }
    
    def _get_git_hash(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return None
    
    def _get_pip_freeze(self) -> Optional[str]:
        """Get pip freeze output."""
        try:
            import subprocess
            result = subprocess.run(
                ['pip', 'freeze'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout
        except Exception:
            return None
    
    def add_metric(self, metric_name: str, value: Any) -> None:
        """Add a metric to metadata."""
        if 'metrics' not in self.metadata:
            self.metadata['metrics'] = {}
        self.metadata['metrics'][metric_name] = value
    
    def save(self) -> None:
        """Save metadata to file."""
        self.metadata['timestamp_end'] = datetime.now().isoformat()
        
        metadata_path = self.log_dir / 'experiment_metadata.json'
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_path}")


def log_hyperparameters(config: Dict[str, Any], log_dir: Path) -> None:
    """Log hyperparameters to file."""
    hp_path = Path(log_dir) / 'hyperparameters.json'
    
    with open(hp_path, 'w') as f:
        json.dump(config, f, indent=2)
