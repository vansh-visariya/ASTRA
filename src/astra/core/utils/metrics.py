"""
Metrics computation for federated learning evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_accuracy(model: nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0.0


def compute_loss(model: nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * len(target)
            total_samples += len(target)
    return total_loss / total_samples if total_samples > 0 else 0.0
