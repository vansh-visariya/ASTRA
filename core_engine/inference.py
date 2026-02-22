"""
Privacy-Preserving Inference Module for Federated Learning.

Provides modular inference capabilities without exposing the full model:
1. Server-side inference (data leaves client, model stays)
2. Parameter-efficient transfer (only deltas/adapters transferred)
3. Distilled model (smaller model for inference)
4. Client-side inference (full model downloaded)

This module ensures federated learning principles are preserved.
"""

import base64
import io
import json
import logging
import pickle
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class InferenceMethod(ABC):
    """Abstract base class for inference methods."""
    
    @abstractmethod
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Run inference and return results."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the method name."""
        pass


@dataclass
class InferenceResult:
    """Result of inference operation."""
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    confidence: float
    method: str
    metadata: Dict[str, Any]


class ServerSideInference(InferenceMethod):
    """Server-side inference where data is sent to server.
    
    Privacy: Raw data leaves client but model stays on server.
    Use case: Clients with limited compute, trusted server.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, input_data: Any) -> InferenceResult:
        """Run inference on server."""
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                tensor = torch.from_numpy(input_data)
            elif isinstance(input_data, torch.Tensor):
                tensor = input_data
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            tensor = tensor.to(self.device)
            
            # Handle batch dimension
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            confidence = probabilities.max().item()
            
            return InferenceResult(
                predictions=predictions.cpu().numpy(),
                probabilities=probabilities.cpu().numpy(),
                confidence=confidence,
                method='server_side',
                metadata={'device': str(self.device)}
            )
    
    def get_method_name(self) -> str:
        return "Server-Side Inference"


class ParameterEfficientInference(InferenceMethod):
    """Parameter-efficient inference using adapters/deltas.
    
    Privacy: Only small deltas transferred, not full model.
    Use case: Bandwidth-constrained clients.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        adapter_weights: Dict[str, np.ndarray],
        device: str = 'cpu'
    ):
        self.base_model = base_model
        self.device = torch.device(device)
        self.base_model.to(self.device)
        self.base_model.eval()
        
        # Apply adapter weights
        self._apply_adapters(adapter_weights)
    
    def _apply_adapters(self, adapter_weights: Dict[str, np.ndarray]):
        """Apply adapter weights to base model."""
        param_idx = 0
        for param in self.base_model.parameters():
            param_shape = param.shape
            param_size = np.prod(param_shape)
            
            if param_idx < len(adapter_weights):
                delta = adapter_weights[param_idx]
                if delta.size == param_size:
                    delta_reshaped = delta.reshape(param_shape)
                    param.data.add_(torch.from_numpy(delta_reshaped).float())
            
            param_idx += 1
    
    def predict(self, input_data: Any) -> InferenceResult:
        """Run inference with parameter-efficient transfer."""
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                tensor = torch.from_numpy(input_data)
            elif isinstance(input_data, torch.Tensor):
                tensor = input_data
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            tensor = tensor.to(self.device)
            
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            confidence = probabilities.max().item()
            
            return InferenceResult(
                predictions=predictions.cpu().numpy(),
                probabilities=probabilities.cpu().numpy(),
                confidence=confidence,
                method='parameter_efficient',
                metadata={'device': str(self.device), 'adapter_size': len(self.base_model.state_dict())}
            )
    
    def get_method_name(self) -> str:
        return "Parameter-Efficient Inference"


class DistilledModelInference(InferenceMethod):
    """Inference using a distilled (smaller) model.
    
    Privacy: Smaller model doesn't reveal full model details.
    Use case: Privacy-critical deployments.
    """
    
    def __init__(self, distilled_model: nn.Module, device: str = 'cpu'):
        self.model = distilled_model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, input_data: Any) -> InferenceResult:
        """Run inference with distilled model."""
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                tensor = torch.from_numpy(input_data)
            elif isinstance(input_data, torch.Tensor):
                tensor = input_data
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            tensor = tensor.to(self.device)
            
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            confidence = probabilities.max().item()
            
            return InferenceResult(
                predictions=predictions.cpu().numpy(),
                probabilities=probabilities.cpu().numpy(),
                confidence=confidence,
                method='distilled',
                metadata={'device': str(self.device)}
            )
    
    def get_method_name(self) -> str:
        return "Distilled Model Inference"


class ClientSideInference(InferenceMethod):
    """Full model inference on client side.
    
    Privacy: Full model downloaded, but data stays local.
    Use case: Clients with sufficient compute.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, input_data: Any) -> InferenceResult:
        """Run full inference on client."""
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                tensor = torch.from_numpy(input_data)
            elif isinstance(input_data, torch.Tensor):
                tensor = input_data
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            tensor = tensor.to(self.device)
            
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)
            confidence = probabilities.max().item()
            
            return InferenceResult(
                predictions=predictions.cpu().numpy(),
                probabilities=probabilities.cpu().numpy(),
                confidence=confidence,
                method='client_side',
                metadata={'device': str(self.device)}
            )
    
    def get_method_name(self) -> str:
        return "Client-Side Inference"


class InferenceModule:
    """Main inference module coordinating different inference methods.
    
    Provides a unified API for privacy-preserving inference.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.inference_methods: Dict[str, InferenceMethod] = {}
        self.default_method = 'server_side'
        
        # Device selection
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def register_method(
        self,
        method_name: str,
        method: InferenceMethod
    ) -> None:
        """Register an inference method."""
        self.inference_methods[method_name] = method
        self.logger.info(f"Registered inference method: {method_name}")
    
    def set_default_method(self, method_name: str) -> bool:
        """Set the default inference method."""
        if method_name in self.inference_methods:
            self.default_method = method_name
            return True
        return False
    
    def predict(
        self,
        input_data: Any,
        method: Optional[str] = None,
        **kwargs
    ) -> InferenceResult:
        """Run inference using specified or default method."""
        method_name = method or self.default_method
        
        if method_name not in self.inference_methods:
            raise ValueError(f"Unknown method: {method_name}")
        
        return self.inference_methods[method_name].predict(input_data)
    
    def predict_batch(
        self,
        input_data: List[Any],
        method: Optional[str] = None,
        batch_size: int = 32
    ) -> List[InferenceResult]:
        """Run batch inference."""
        results = []
        
        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i + batch_size]
            batch_tensor = torch.stack([
                torch.from_numpy(d) if isinstance(d, np.ndarray) else d
                for d in batch
            ])
            
            result = self.predict(batch_tensor, method)
            results.append(result)
        
        return results
    
    def create_server_side(
        self,
        model: nn.Module
    ) -> ServerSideInference:
        """Create server-side inference instance."""
        method = ServerSideInference(model, device=str(self.device))
        self.register_method('server_side', method)
        return method
    
    def create_client_side(
        self,
        model: nn.Module
    ) -> ClientSideInference:
        """Create client-side inference instance."""
        method = ClientSideInference(model, device=str(self.device))
        self.register_method('client_side', method)
        return method
    
    def create_parameter_efficient(
        self,
        base_model: nn.Module,
        adapter_weights: Dict[str, np.ndarray]
    ) -> ParameterEfficientInference:
        """Create parameter-efficient inference instance."""
        method = ParameterEfficientInference(
            base_model,
            adapter_weights,
            device=str(self.device)
        )
        self.register_method('parameter_efficient', method)
        return method
    
    def get_available_methods(self) -> List[str]:
        """Get list of available inference methods."""
        return list(self.inference_methods.keys())


class ModelDistiller:
    """Creates distilled models for privacy-preserving inference."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def distill(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: DataLoader,
        epochs: int = 10,
        temperature: float = 4.0,
        alpha: float = 0.5
    ) -> nn.Module:
        """Distill knowledge from teacher to student.
        
        Args:
            teacher_model: Original larger model
            student_model: Smaller model to train
            train_loader: Training data
            epochs: Number of distillation epochs
            temperature: Softmax temperature
            alpha: Balance between hard and soft labels
        
        Returns:
            Trained student model
        """
        teacher_model.eval()
        student_model.train()
        
        device = next(teacher_model.parameters()).device
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                # Teacher predictions
                with torch.no_grad():
                    teacher_output = teacher_model(data)
                    soft_teacher = torch.softmax(teacher_output / temperature, dim=1)
                
                # Student predictions
                student_output = student_model(data)
                soft_student = torch.softmax(student_output / temperature, dim=1)
                
                # Hard loss
                hard_loss = nn.functional.cross_entropy(student_output, target)
                
                # Soft loss
                soft_loss = nn.functional.kl_div(
                    soft_student.log(),
                    soft_teacher,
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                # Combined loss
                loss = alpha * hard_loss + (1 - alpha) * soft_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            self.logger.info(f"Distillation epoch {epoch + 1}/{epochs}, loss: {total_loss / len(train_loader):.4f}")
        
        student_model.eval()
        return student_model
    
    def create_student(
        self,
        teacher_param_count: int,
        compression_ratio: float = 0.1
    ) -> nn.Module:
        """Create a smaller student model.
        
        Args:
            teacher_param_count: Number of parameters in teacher
            compression_ratio: Target ratio of student to teacher size
        
        Returns:
            Student model
        """
        from core_engine.model_zoo import SimpleCNN, SimpleMLP
        
        target_params = int(teacher_param_count * compression_ratio)
        
        # Use smaller MLP as student
        student = SimpleMLP(
            input_dim=784,
            num_classes=10,
            hidden_dim=max(32, target_params // 1000)
        )
        
        return student


class InferenceAPI:
    """REST API wrapper for inference operations."""
    
    def __init__(self, inference_module: InferenceModule):
        self.inference_module = inference_module
    
    def predict_from_encoded(
        self,
        encoded_data: str,
        method: str = 'server_side'
    ) -> Dict[str, Any]:
        """Run inference from base64-encoded input."""
        try:
            # Decode base64 to numpy array
            decoded = base64.b64decode(encoded_data)
            input_data = np.frombuffer(decoded, dtype=np.float32)
            
            # Reshape based on expected input size
            # This is a placeholder - actual implementation needs input shape info
            result = self.inference_module.predict(input_data, method)
            
            return {
                'success': True,
                'predictions': result.predictions.tolist(),
                'probabilities': result.probabilities.tolist() if result.probabilities is not None else None,
                'confidence': result.confidence,
                'method': result.method
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict_from_file(
        self,
        file_path: str,
        method: str = 'server_side'
    ) -> Dict[str, Any]:
        """Run inference from image file."""
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Load and preprocess image
            image = Image.open(file_path).convert('L')  # Grayscale for MNIST
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            input_tensor = transform(image).unsqueeze(0)
            
            result = self.inference_module.predict(input_tensor, method)
            
            return {
                'success': True,
                'predictions': result.predictions.tolist(),
                'confidence': result.confidence,
                'method': result.method
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


# Factory function
def create_inference_module(
    config: Dict[str, Any],
    model: Optional[nn.Module] = None
) -> InferenceModule:
    """Create and configure inference module."""
    module = InferenceModule(config)
    
    if model:
        module.create_server_side(model)
        module.create_client_side(model)
    
    return module
