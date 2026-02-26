"""
Heterogeneous Model Aggregation for Federated Learning.

Handles aggregation of models with different:
- Architectures (CNN, MLP, etc.)
- Parameter counts
- Layer configurations

This module ensures the baseline model (selected by admin) is correctly
updated even when clients use different model variants.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ParameterMapper:
    """Maps parameters between different model architectures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_layer_type(self, param_name: str) -> str:
        """Infer layer type from parameter name."""
        if 'conv' in param_name.lower():
            return 'conv'
        elif 'fc' in param_name.lower() or 'linear' in param_name.lower():
            return 'linear'
        elif 'bn' in param_name.lower() or 'batchnorm' in param_name.lower():
            return 'batchnorm'
        elif 'bias' in param_name.lower():
            return 'bias'
        elif 'embedding' in param_name.lower():
            return 'embedding'
        else:
            return 'other'
    
    def can_map(self, source_param: np.ndarray, target_param: np.ndarray) -> bool:
        """Check if source can be mapped to target."""
        source_size = source_param.size
        target_size = target_param.size
        
        # Same size: direct mapping
        if source_size == target_size:
            return True
        
        # Similar size (within 2x): can use projection
        if target_size <= source_size <= target_size * 2:
            return True
        
        return False
    
    def map_parameters(
        self,
        source_param: np.ndarray,
        target_param: np.ndarray,
        method: str = 'average'
    ) -> np.ndarray:
        """Map source parameters to target parameter space.
        
        Methods:
        - 'average': Average matching indices
        - 'projection': Project to target size
        - 'truncate': Take first N elements
        - 'pad': Pad to target size
        """
        source_size = source_param.size
        target_size = target_param.size
        
        if source_size == target_size:
            return source_param.copy()
        
        if source_size > target_size:
            # Truncate or downsample
            if method == 'truncate':
                return source_param.flatten()[:target_size].reshape(target_param.shape)
            elif method == 'average':
                # Average blocks
                block_size = source_size // target_size
                reshaped = source_param.flatten()[:block_size * target_size].reshape(block_size, target_size)
                return reshaped.mean(axis=0)
            else:
                return source_param.flatten()[:target_size].reshape(target_param.shape)
        else:
            # Pad or upsample
            if method == 'pad':
                result = np.zeros(target_size)
                result[:source_size] = source_param.flatten()
                return result.reshape(target_param.shape)
            elif method == 'repeat':
                # Repeat values
                repeat_factor = (target_size // source_size) + 1
                repeated = np.tile(source_param.flatten(), repeat_factor)[:target_size]
                return repeated.reshape(target_param.shape)
            else:
                result = np.zeros(target_size)
                result[:source_size] = source_param.flatten()
                return result.reshape(target_param.shape)
    
    def align_models(
        self,
        source_params: Dict[str, np.ndarray],
        target_param_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Align source model to target parameter structure."""
        aligned = {}
        
        for target_name in target_param_names:
            # Try exact match first
            if target_name in source_params:
                aligned[target_name] = source_params[target_name].copy()
                continue
            
            # Try to find similar parameter
            target_type = self.get_layer_type(target_name)
            
            matched = False
            for source_name, source_param in source_params.items():
                source_type = self.get_layer_type(source_name)
                
                if source_type == target_type and self.can_map(source_param, np.array([])):
                    # Get target shape from baseline (we'll handle shapes separately)
                    aligned[target_name] = source_param.copy()
                    matched = True
                    break
            
            if not matched:
                aligned[target_name] = None
        
        return aligned


class HeterogeneousAggregator:
    """Handles aggregation of heterogeneous model updates.
    
    Architecture:
    
    ┌─────────────────────────────────────────────────────────────┐
    │              Heterogeneous Aggregation Pipeline            │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Client Updates ──▶ Parameter Mapping ──▶ Alignment        │
    │         │                 │                 │              │
    │         │                 │                 ▼              │
    │         │                 │         Shared Parameters     │
    │         │                 │                 │              │
    │         ▼                 ▼                 ▼              │
    │  ┌──────────────────────────────────────────────────┐      │
    │  │           Unified Parameter Space                │      │
    │  │   (Only includes parameters present in all      │      │
    │  │    clients AND baseline model)                   │      │
    │  └──────────────────────────────────────────────────┘      │
    │                         │                                   │
    │                         ▼                                   │
    │  ┌──────────────────────────────────────────────────┐      │
    │  │              Aggregation (FedAvg/Robust)         │      │
    │  └──────────────────────────────────────────────────┘      │
    │                         │                                   │
    │                         ▼                                   │
    │              Aggregated Delta                               │
    │                         │                                   │
    │                         ▼                                   │
    │              Update Baseline Model                          │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.parameter_mapper = ParameterMapper()
        
        # Configuration
        self.mapping_method = config.get('heterogeneous', {}).get('mapping_method', 'average')
        self.allow_partial = config.get('heterogeneous', {}).get('allow_partial_updates', True)
        self.min_param_overlap = config.get('heterogeneous', {}).get('min_param_overlap', 0.5)
    
    def get_shared_parameters(
        self,
        baseline_param_names: List[str],
        client_param_names: List[str]
    ) -> List[str]:
        """Find shared parameters between baseline and client model."""
        baseline_set = set(baseline_param_names)
        client_set = set(client_param_names)
        
        shared = baseline_set & client_set
        overlap_ratio = len(shared) / len(baseline_set) if baseline_set else 0
        
        if overlap_ratio < self.min_param_overlap:
            self.logger.warning(
                f"Low parameter overlap: {overlap_ratio:.2%}. "
                f"Baseline: {len(baseline_set)}, Client: {len(client_set)}"
            )
        
        return list(shared)
    
    def normalize_client_update(
        self,
        client_update: Dict[str, Any],
        baseline_param_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Normalize client update to match baseline parameter structure.
        
        Args:
            client_update: Client update with 'delta' or 'params' key containing model weights
            baseline_param_names: List of parameter names in baseline model
        
        Returns:
            Dictionary mapping parameter names to numpy arrays
        """
        # Get client parameters
        if 'params' in client_update:
            raw_params = client_update['params']
        elif 'delta' in client_update:
            raw_params = client_update['delta']
        elif 'local_updates' in client_update:
            # Handle encoded updates
            import base64
            decoded = base64.b64decode(client_update['local_updates'])
            raw_params = {'encoded': np.frombuffer(decoded, dtype=np.float32)}
        else:
            raise ValueError("Invalid client update format")
        
        # If encoded, we need special handling
        if 'encoded' in raw_params:
            # For encoded updates, we assume they're already aligned
            # This is a limitation - clients should send parameter names
            return {'encoded': raw_params['encoded']}
        
        # Map parameters to baseline structure
        aligned = {}
        shared_params = self.get_shared_parameters(
            baseline_param_names,
            list(raw_params.keys())
        )
        
        for param_name in shared_params:
            if param_name in raw_params:
                aligned[param_name] = np.array(raw_params[param_name], dtype=np.float32)
        
        return aligned
    
    def aggregate(
        self,
        baseline_param_names: List[str],
        client_updates: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        method: str = 'fedavg'
    ) -> Dict[str, np.ndarray]:
        """Aggregate heterogeneous client updates to update baseline model.
        
        Args:
            baseline_param_names: Parameter names in baseline model
            client_updates: List of client updates (each with parameter dict)
            weights: Optional weights for each client (default: equal)
            method: Aggregation method ('fedavg', 'robust', 'hybrid')
        
        Returns:
            Aggregated delta for each parameter
        """
        if not client_updates:
            return {}
        
        # Default weights
        if weights is None:
            weights = [1.0 / len(client_updates)] * len(client_updates)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Normalize all client updates to baseline structure
        normalized_updates = []
        for i, update in enumerate(client_updates):
            try:
                normalized = self.normalize_client_update(update, baseline_param_names)
                normalized_updates.append((normalized, weights[i]))
            except Exception as e:
                self.logger.warning(f"Failed to normalize client {i} update: {e}")
                continue
        
        if not normalized_updates:
            self.logger.error("No valid client updates after normalization")
            return {}
        
        # Find shared parameters across all valid updates
        all_param_names = set()
        for normalized, _ in normalized_updates:
            if 'encoded' in normalized:
                # Handle encoded updates
                all_param_names.add('encoded')
            else:
                all_param_names.update(normalized.keys())
        
        # Aggregate each parameter
        aggregated = {}
        
        for param_name in all_param_names:
            param_values = []
            param_weights = []
            
            for normalized, weight in normalized_updates:
                if param_name in normalized:
                    param_values.append(normalized[param_name])
                    param_weights.append(weight)
            
            if not param_values:
                continue
            
            param_weights = np.array(param_weights)
            param_weights = param_weights / param_weights.sum()
            
            if method == 'fedavg':
                aggregated[param_name] = self._weighted_average(param_values, param_weights)
            elif method == 'robust':
                aggregated[param_name] = self._robust_aggregate(param_values)
            elif method == 'hybrid':
                aggregated[param_name] = self._hybrid_aggregate(param_values, param_weights)
            else:
                aggregated[param_name] = self._weighted_average(param_values, param_weights)
        
        return aggregated
    
    def _weighted_average(
        self,
        values: List[np.ndarray],
        weights: np.ndarray
    ) -> np.ndarray:
        """Compute weighted average."""
        result = None
        for value, weight in zip(values, weights):
            if result is None:
                result = weight * value
            else:
                result = result + weight * value
        return result
    
    def _robust_aggregate(self, values: List[np.ndarray]) -> np.ndarray:
        """Compute robust aggregation using coordinate-wise median."""
        if len(values) == 1:
            return values[0].copy()
        
        values_array = np.array(values)
        
        # Replace NaN/Inf
        values_array = np.nan_to_num(values_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Coordinate-wise median
        aggregated = np.median(values_array, axis=0)
        
        return aggregated
    
    def _hybrid_aggregate(
        self,
        values: List[np.ndarray],
        weights: np.ndarray
    ) -> np.ndarray:
        """Hybrid aggregation: trimmed mean with weights."""
        if len(values) <= 2:
            return self._weighted_average(values, weights)
        
        # Compute trimmed mean
        trim_ratio = self.config.get('robust', {}).get('trim_ratio', 0.1)
        
        values_array = np.array(values)
        n = len(values)
        trim_count = max(1, int(n * trim_ratio))
        
        # Sort and trim
        sorted_indices = np.argsort(values_array, axis=0)
        trimmed = values_array.copy()
        
        for i in range(values_array.shape[1]):
            lower = sorted_indices[:trim_count, i]
            upper = sorted_indices[-(trim_count):, i]
            
            mask = np.ones(n, dtype=bool)
            mask[lower] = False
            mask[upper] = False
            
            trimmed[:, i] = np.where(mask, values_array[:, i], np.nan)
        
        # Weighted mean of trimmed values
        trimmed_mean = np.nanmean(trimmed, axis=0)
        trimmed_mean = np.nan_to_num(trimmed_mean, nan=0.0)
        
        return trimmed_mean


class ModelMetadata:
    """Metadata for model variants to support heterogeneous aggregation."""
    
    def __init__(
        self,
        model_id: str,
        architecture: str,
        param_count: int,
        param_names: List[str],
        layer_types: Dict[str, str]
    ):
        self.model_id = model_id
        self.architecture = architecture
        self.param_count = param_count
        self.param_names = param_names
        self.layer_types = layer_types
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'architecture': self.architecture,
            'param_count': self.param_count,
            'param_names': self.param_names,
            'layer_types': self.layer_types
        }
    
    @classmethod
    def from_model(cls, model_id: str, model: Any) -> 'ModelMetadata':
        """Create metadata from a PyTorch model."""
        import torch
        
        param_names = []
        layer_types = {}
        param_count = 0
        
        mapper = ParameterMapper()
        
        for name, param in model.named_parameters():
            param_names.append(name)
            layer_types[name] = mapper.get_layer_type(name)
            param_count += param.numel()
        
        # Infer architecture from model class
        architecture = model.__class__.__name__
        
        return cls(
            model_id=model_id,
            architecture=architecture,
            param_count=param_count,
            param_names=param_names,
            layer_types=layer_types
        )
    
    def get_compatibility(self, other: 'ModelMetadata') -> float:
        """Calculate compatibility score with another model."""
        shared = set(self.param_names) & set(other.param_names)
        
        if not self.param_names:
            return 0.0
        
        return len(shared) / len(self.param_names)


class BaselineModelManager:
    """Manages the baseline model for heterogeneous aggregation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.baseline_model = None
        self.baseline_metadata: Optional[ModelMetadata] = None
        self.model_variants: Dict[str, ModelMetadata] = {}
    
    def set_baseline(self, model: Any, model_id: str = 'baseline') -> ModelMetadata:
        """Set the baseline model."""
        self.baseline_model = model
        self.baseline_metadata = ModelMetadata.from_model(model_id, model)
        
        self.logger.info(
            f"Baseline model set: {self.baseline_metadata.architecture}, "
            f"params: {self.baseline_metadata.param_count}"
        )
        
        return self.baseline_metadata
    
    def register_variant(self, model: Any, model_id: str) -> ModelMetadata:
        """Register a model variant."""
        metadata = ModelMetadata.from_model(model_id, model)
        self.model_variants[model_id] = metadata
        
        compatibility = self.baseline_metadata.get_compatibility(metadata) if self.baseline_metadata else 0.0
        self.logger.info(
            f"Registered variant: {model_id}, "
            f"architecture: {metadata.architecture}, "
            f"compatibility: {compatibility:.2%}"
        )
        
        return metadata
    
    def get_baseline_param_names(self) -> List[str]:
        """Get baseline parameter names."""
        if self.baseline_metadata:
            return self.baseline_metadata.param_names
        return []
    
    def can_aggregate_with(self, model_id: str) -> Tuple[bool, float]:
        """Check if a variant can be aggregated with baseline."""
        if model_id not in self.model_variants:
            return False, 0.0
        
        compatibility = self.baseline_metadata.get_compatibility(
            self.model_variants[model_id]
        ) if self.baseline_metadata else 0.0
        
        min_compatibility = self.config.get('heterogeneous', {}).get('min_param_overlap', 0.5)
        
        return compatibility >= min_compatibility, compatibility


def create_heterogeneous_aggregator(config: Dict[str, Any]) -> HeterogeneousAggregator:
    """Factory function to create heterogeneous aggregator."""
    return HeterogeneousAggregator(config)
