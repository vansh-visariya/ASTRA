"""
Experiment Runner for Federated Learning Experiments.

Runs individual experiments from the experiment specification.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_experiment_spec(spec_path: str) -> List[Dict[str, Any]]:
    """Load experiment specification."""
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)
    return spec.get('experiments', [])


def modify_config_for_experiment(
    base_config: Dict[str, Any],
    experiment: Dict[str, Any]
) -> Dict[str, Any]:
    """Modify base config for specific experiment."""
    config = base_config.copy()
    
    exp_id = experiment.get('experiment_id', 'exp')
    
    if 'baseline' in exp_id.lower() and 'fedavg' in exp_id.lower():
        config['server']['aggregator_window'] = 20
        config['robust']['method'] = 'fedavg'
        config['malicious']['ratio'] = 0.0
        config['privacy']['dp_enabled'] = False
    
    elif 'no_robust' in exp_id.lower():
        config['robust']['method'] = 'fedavg'
    
    elif 'hybrid' in exp_id.lower() and 'robust' in exp_id.lower():
        config['robust']['method'] = 'hybrid'
    
    elif 'trust' in exp_id.lower():
        config['robust']['method'] = 'hybrid'
        config['trust']['init'] = 1.0
    
    elif 'dp' in exp_id.lower():
        config['privacy']['dp_enabled'] = True
    
    if 'iid' in exp_id.lower():
        config['dataset']['split'] = 'iid'
    else:
        config['dataset']['split'] = 'dirichlet'
    
    config['seed'] = experiment.get('seed', config.get('seed', 42))
    config['experiment_id'] = exp_id
    
    return config


def save_experiment_config(config: Dict[str, Any], output_dir: Path) -> None:
    """Save experiment configuration."""
    config_path = output_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def run_experiment(
    experiment: Dict[str, Any],
    base_config: Dict[str, Any],
    output_base: Path
) -> bool:
    """
    Run a single experiment.
    
    Args:
        experiment: Experiment specification.
        base_config: Base configuration.
        output_base: Base output directory.
    
    Returns:
        True if experiment succeeded.
    """
    exp_id = experiment['experiment_id']
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running experiment: {exp_id}")
    
    config = modify_config_for_experiment(base_config, experiment)
    
    output_dir = output_base / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config['logging']['log_dir'] = str(output_dir)
    
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    save_experiment_config(config, output_dir)
    
    log_file = output_dir / 'logs' / f'{exp_id}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(log_file, 'w') as log_f:
            result = subprocess.run(
                [sys.executable, 'main.py', '--config', str(config_path)],
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=Path.cwd()
            )
            
            log_f.write(result.stdout)
            log_f.write(result.stderr)
        
        if result.returncode == 0:
            logger.info(f"Experiment {exp_id} completed successfully")
            return True
        else:
            logger.error(f"Experiment {exp_id} failed with return code {result.returncode}")
            return False
    
    except subprocess.TimeoutExpired:
        logger.error(f"Experiment {exp_id} timed out")
        return False
    except Exception as e:
        logger.error(f"Experiment {exp_id} failed with error: {e}")
        return False


def run_all_experiments(
    spec_path: str,
    config_path: str,
    output_base: str
) -> None:
    """Run all experiments from specification."""
    logger = logging.getLogger(__name__)
    
    experiments = load_experiment_spec(spec_path)
    
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for experiment in experiments:
        exp_id = experiment.get('experiment_id', 'unknown')
        
        success = run_experiment(experiment, base_config, output_base)
        results.append({
            'experiment_id': exp_id,
            'success': success
        })
        
        logger.info(f"Results: {results[-1]}")
    
    results_path = output_base / 'experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All experiments completed. Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Run federated learning experiments')
    parser.add_argument('--spec', type=str, default='experiments/experiments_spec.yaml',
                        help='Path to experiments specification')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to base config')
    parser.add_argument('--output', type=str, default='./experiments_results',
                        help='Output directory')
    parser.add_argument('--experiment-id', type=str, default=None,
                        help='Run specific experiment')
    
    args = parser.parse_args()
    
    if args.experiment_id:
        experiments = load_experiment_spec(args.spec)
        experiment = next((e for e in experiments if e['experiment_id'] == args.experiment_id), None)
        
        if experiment is None:
            print(f"Experiment {args.experiment_id} not found")
            sys.exit(1)
        
        with open(args.config, 'r') as f:
            base_config = yaml.safe_load(f)
        
        run_experiment(experiment, base_config, Path(args.output))
    else:
        run_all_experiments(args.spec, args.config, args.output)


if __name__ == '__main__':
    main()
