"""
Plotting utilities for federated learning experiments.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_vs_steps(
    history: List[Dict[str, Any]],
    save_path: Path,
    title: str = "Global Accuracy vs Steps"
) -> None:
    """Plot accuracy over training steps."""
    steps = [h['step'] for h in history if 'accuracy' in h]
    accuracies = [h['accuracy'] for h in history if 'accuracy' in h]
    
    if not steps:
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, accuracies, marker='o', linewidth=2)
    plt.xlabel('Global Steps')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    _save_caption(save_path, title, "Shows global test accuracy progression over federated training rounds.")


def plot_loss_vs_steps(
    history: List[Dict[str, Any]],
    save_path: Path,
    title: str = "Global Loss vs Steps"
) -> None:
    """Plot loss over training steps."""
    steps = [h['step'] for h in history if 'loss' in h]
    losses = [h['loss'] for h in history if 'loss' in h]
    
    if not steps:
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='o', linewidth=2, color='red')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    _save_caption(save_path, title, "Shows global test loss progression over federated training rounds.")


def plot_trust_evolution(
    trust_history: Dict[str, List[float]],
    save_path: Path,
    title: str = "Trust Score Evolution"
) -> None:
    """Plot trust score evolution for suspicious clients."""
    if not trust_history:
        return
    
    plt.figure(figsize=(12, 6))
    
    sorted_clients = sorted(
        trust_history.keys(),
        key=lambda c: max(trust_history[c]) if trust_history[c] else 0,
        reverse=True
    )[:6]
    
    for client_id in sorted_clients:
        history = trust_history[client_id]
        plt.plot(history, label=client_id, alpha=0.7)
    
    plt.xlabel('Update Count')
    plt.ylabel('Trust Score')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    _save_caption(save_path, title, "Trust score evolution for the most suspicious clients.")


def plot_gradient_norm_histogram(
    norms: List[float],
    save_path: Path,
    title: str = "Gradient Norm Distribution"
) -> None:
    """Plot histogram of gradient norms."""
    if not norms:
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(norms, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Gradient L2 Norm')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    _save_caption(save_path, title, "Distribution of gradient L2 norms from client updates.")


def plot_staleness_heatmap(
    staleness_matrix: List[List[float]],
    save_path: Path,
    title: str = "Staleness Heatmap"
) -> None:
    """Plot staleness heatmap (clients x time)."""
    if not staleness_matrix:
        return
    
    plt.figure(figsize=(12, 8))
    
    arr = np.array(staleness_matrix)
    plt.imshow(arr, aspect='auto', cmap='YlOrRd')
    
    plt.xlabel('Time Step')
    plt.ylabel('Client')
    plt.title(title)
    plt.colorbar(label='Staleness')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    _save_caption(save_path, title, "Heatmap showing staleness of client updates over time.")


def plot_client_class_distribution(
    distributions: Dict[int, Dict[int, float]],
    num_classes: int,
    save_path: Path,
    title: str = "Client Class Distribution"
) -> None:
    """Plot class distribution for each client."""
    if not distributions:
        return
    
    num_clients = len(distributions)
    
    matrix = np.zeros((num_clients, num_classes))
    for client_id, dist in distributions.items():
        for class_id, proportion in dist.items():
            matrix[client_id, class_id] = proportion
    
    plt.figure(figsize=(14, 8))
    plt.imshow(matrix, aspect='auto', cmap='Blues')
    
    plt.xlabel('Class')
    plt.ylabel('Client')
    plt.title(title)
    plt.colorbar(label='Proportion')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    _save_caption(save_path, title, "Class distribution across federated clients (non-IID visualization).")


def plot_attack_impact(
    history: List[Dict[str, Any]],
    attack_steps: List[int],
    save_path: Path,
    title: str = "Attack Impact on Accuracy"
) -> None:
    """Plot accuracy with attack injection points."""
    steps = [h['step'] for h in history if 'accuracy' in h]
    accuracies = [h['accuracy'] for h in history if 'accuracy' in h]
    
    if not steps:
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, accuracies, marker='o', linewidth=2)
    
    for attack_step in attack_steps:
        if attack_step <= max(steps):
            idx = steps.index(attack_step) if attack_step in steps else None
            if idx is not None:
                plt.axvline(x=attack_step, color='red', linestyle='--', alpha=0.7)
                plt.annotate('Attack', xy=(attack_step, accuracies[idx]))
    
    plt.xlabel('Global Steps')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    _save_caption(save_path, title, "Accuracy timeline with attack injection points and recovery.")


def plot_epsilon_vs_accuracy(
    epsilon_history: List[float],
    accuracy_history: List[float],
    save_path: Path,
    title: str = "Privacy-Accuracy Tradeoff"
) -> None:
    """Plot epsilon vs accuracy tradeoff."""
    if not epsilon_history or not accuracy_history:
        return
    
    min_len = min(len(epsilon_history), len(accuracy_history))
    epsilon_history = epsilon_history[:min_len]
    accuracy_history = accuracy_history[:min_len]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_history, accuracy_history, marker='o', linewidth=2)
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    _save_caption(save_path, title, "Tradeoff between privacy budget (epsilon) and model accuracy.")


def _save_caption(
    save_path: Path,
    title: str,
    caption: str
) -> None:
    """Save caption as JSON."""
    caption_data = {
        'title': title,
        'caption': caption,
        'generation_timestamp': datetime.now().isoformat(),
        'file': save_path.name
    }
    
    caption_path = save_path.with_suffix('.caption.json')
    with open(caption_path, 'w') as f:
        json.dump(caption_data, f, indent=2)


def generate_all_plots(
    metrics_history: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate all standard plots from metrics history."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_accuracy_vs_steps(
        metrics_history,
        output_dir / 'accuracy_vs_steps.png'
    )
    
    plot_loss_vs_steps(
        metrics_history,
        output_dir / 'loss_vs_steps.png'
    )
