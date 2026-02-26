"""
Personalization Module for Federated Learning.

Implements personalized models with shared backbone + client-specific heads.

References:
- Hanzely et al., "Federated Learning of a Mixture of Global and Local Models"
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonalizedHead(nn.Module):
    """Client-specific prediction head."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class PersonalizedModel(nn.Module):
    """Personalized model with shared backbone and client heads."""
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 10,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.backbone = backbone
        
        self.backbone_output_dim = hidden_dim
        
        self.client_heads = nn.ModuleDict()
        
        self.num_classes = num_classes
    
    def add_client_head(self, client_id: str) -> None:
        """Add a personalized head for a client."""
        if client_id not in self.client_heads:
            self.client_heads[client_id] = PersonalizedHead(
                self.backbone_output_dim,
                self.num_classes
            )
    
    def forward(self, x, client_id: str = 'default'):
        """Forward pass with client-specific head."""
        features = self.backbone(x)
        
        if hasattr(features, 'last_hidden_state'):
            features = features.last_hidden_state[:, 0]
        
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        if features.size(-1) != self.backbone_output_dim:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        if client_id in self.client_heads:
            return self.client_heads[client_id](features)
        else:
            return features


def personalize_model(
    global_model: nn.Module,
    client_ids: List[str],
    num_classes: int = 10
) -> PersonalizedModel:
    """
    Create personalized model for clients.
    
    Args:
        global_model: Global model to use as backbone.
        client_ids: List of client IDs.
        num_classes: Number of output classes.
    
    Returns:
        Personalized model.
    """
    backbone = global_model
    
    personalized = PersonalizedModel(
        backbone=backbone,
        num_classes=num_classes
    )
    
    for client_id in client_ids:
        personalized.add_client_head(client_id)
    
    return personalized


def aggregate_personalized_heads(
    client_heads: Dict[str, nn.Module],
    trust_scores: Dict[str, float]
) -> nn.Module:
    """
    Aggregate client heads using trust scores.
    
    Args:
        client_heads: Dictionary of client head modules.
        trust_scores: Trust scores for each client.
    
    Returns:
        Aggregated head module.
    """
    total_weight = sum(trust_scores.values())
    
    first_head = list(client_heads.values())[0]
    
    aggregated_state = {}
    
    for name, param in first_head.named_parameters():
        aggregated_state[name] = torch.zeros_like(param.data)
    
    for client_id, head in client_heads.items():
        weight = trust_scores.get(client_id, 1.0) / total_weight
        
        for name, param in head.named_parameters():
            aggregated_state[name] += weight * param.data
    
    aggregated_head = PersonalizedHead(
        first_head.fc.in_features,
        first_head.fc.out_features
    )
    
    aggregated_head.load_state_dict({
        'fc.weight': aggregated_state['fc.weight'],
        'fc.bias': aggregated_state['fc.bias']
    })
    
    return aggregated_head
