"""
API-Based Model Recommendation System using Gemini.

Analyzes client metadata and recommends appropriate model architectures
for heterogeneous federated learning.

Privacy: Only metadata is sent to the API, never raw data.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class ClientMetadata:
    """Client metadata for model recommendation."""
    dataset_size: int
    num_classes: int
    class_distribution: Dict[int, float]
    has_gpu: bool
    gpu_memory_mb: Optional[int] = None
    cpu_cores: Optional[int] = None
    memory_mb: Optional[int] = None
    network_bandwidth_mbps: Optional[float] = None
    preferred_model_type: Optional[str] = None


@dataclass
class ModelRecommendation:
    """Model recommendation from Gemini."""
    id: str = ""
    model_type: str = "cnn"  # 'cnn', 'mlp', 'transformer'
    model_size: str = "medium"  # 'small', 'medium', 'large'
    estimated_params: int = 100000
    architecture: Dict[str, Any] = None
    expected_accuracy: float = 0.8
    reasoning: str = ""
    config: Dict[str, Any] = None
    source: str = "gemini"  # 'gemini', 'builtin', 'huggingface'
    model_id: str = ""  # registry model_id if available
    model_name: str = ""  # display name
    hf_url: str = ""  # HuggingFace URL if from HF

    def __post_init__(self):
        if self.architecture is None:
            self.architecture = {}
        if self.config is None:
            self.config = {}
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())[:8]


class GeminiRecommender:
    """Model recommendation using Gemini API."""
    
    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        self.model_name = 'gemini-2.0-flash'
        
        if not self.api_key:
            self.logger.warning("No Gemini API key provided. Using fallback recommendations.")
    
    def _build_prompt(self, metadata: ClientMetadata) -> str:
        """Build prompt for Gemini API."""
        
        distribution_str = ", ".join([
            f"class {k}: {v:.1%}"
            for k, v in list(metadata.class_distribution.items())[:5]
        ])
        
        prompt = f"""You are a federated learning model recommendation system.
Given the following client metadata, recommend an appropriate model architecture for training in a federated learning setting.

IMPORTANT constraints:
- Model must be compatible with federated aggregation
- Prioritize privacy and efficiency
- Consider client hardware limitations
- Support heterogeneous model updates

Client Metadata:
- Dataset size: {metadata.dataset_size} samples
- Number of classes: {metadata.num_classes}
- Class distribution: {distribution_str}
- Has GPU: {metadata.has_gpu}
- GPU Memory: {metadata.gpu_memory_mb} MB (if available)
- CPU Cores: {metadata.cpu_cores}
- Memory: {metadata.memory_mb} MB (if available)
- Network bandwidth: {metadata.network_bandwidth_mbps} Mbps (if available)
- Preferred model type: {metadata.preferred_model_type or 'any'}

Respond with ONLY valid JSON (no markdown formatting):
{{
  "model_type": "cnn|mlp|transformer|efficientnet",
  "model_size": "small|medium|large",
  "estimated_params": <number>,
  "architecture": {{
    "conv_layers": <number or null>,
    "fc_layers": <number or null>,
    "hidden_dim": <number or null>,
    "dropout": <number>
  }},
  "expected_accuracy": <0.0-1.0>,
  "reasoning": "<2-3 sentence explanation>",
  "config": {{
    "batch_size": <number>,
    "learning_rate": <number>,
    "local_epochs": <number>
  }}
}}"""
        
        return prompt
    
    def _call_api(self, prompt: str) -> Optional[str]:
        """Call Gemini API."""
        if not self.api_key:
            return None
        
        try:
            url = f"{self.API_BASE_URL}/models/{self.model_name}:generateContent"
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            payload = {
                'contents': [{
                    'parts': [{'text': prompt}]
                }],
                'generationConfig': {
                    'temperature': 0.7,
                    'maxOutputTokens': 1024,
                    'topP': 0.95,
                    'topK': 40
                }
            }
            
            response = requests.post(
                f"{url}?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                self.logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {e}")
            return None
    
    def _parse_response(self, response_text: str) -> Optional[ModelRecommendation]:
        """Parse Gemini response into ModelRecommendation."""
        try:
            # Try to extract JSON from response
            # Sometimes Gemini returns markdown-wrapped JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            data = json.loads(response_text.strip())
            
            return ModelRecommendation(
                model_type=data.get('model_type', 'cnn'),
                model_size=data.get('model_size', 'medium'),
                estimated_params=data.get('estimated_params', 100000),
                architecture=data.get('architecture', {}),
                expected_accuracy=data.get('expected_accuracy', 0.8),
                reasoning=data.get('reasoning', ''),
                config=data.get('config', {})
            )
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Gemini response: {e}")
            return None
    
    def _fallback_recommendation(self, metadata: ClientMetadata) -> ModelRecommendation:
        """Provide fallback recommendation when API unavailable."""
        
        # Determine model size based on hardware
        if metadata.has_gpu and (metadata.gpu_memory_mb or 0) > 4000:
            model_size = 'large'
            estimated_params = 500000
        elif metadata.has_gpu or (metadata.cpu_cores or 0) > 4:
            model_size = 'medium'
            estimated_params = 200000
        else:
            model_size = 'small'
            estimated_params = 50000
        
        # Adjust based on dataset size
        if metadata.dataset_size < 1000:
            estimated_params = min(estimated_params, 30000)
            model_size = 'small'
        elif metadata.dataset_size > 10000:
            estimated_params = max(estimated_params, 100000)
        
        # Determine model type
        if metadata.preferred_model_type:
            model_type = metadata.preferred_model_type
        elif metadata.dataset_size > 5000 and (metadata.has_gpu or (metadata.cpu_cores or 0) > 2):
            model_type = 'cnn'
        else:
            model_type = 'mlp'
        
        return ModelRecommendation(
            model_type=model_type,
            model_size=model_size,
            estimated_params=estimated_params,
            architecture={
                'conv_layers': 2 if model_type == 'cnn' else 0,
                'fc_layers': 2,
                'hidden_dim': estimated_params // 1000,
                'dropout': 0.5
            },
            expected_accuracy=0.85 if model_size == 'large' else 0.75,
            reasoning="Fallback recommendation based on hardware constraints.",
            config={
                'batch_size': 32 if not metadata.has_gpu else 64,
                'learning_rate': 0.01,
                'local_epochs': 2
            }
        )
    
    def recommend(
        self,
        metadata: ClientMetadata
    ) -> ModelRecommendation:
        """Get model recommendation for client."""
        
        # Try Gemini API first
        if self.api_key:
            prompt = self._build_prompt(metadata)
            response = self._call_api(prompt)
            
            if response:
                recommendation = self._parse_response(response)
                if recommendation:
                    self.logger.info(f"Gemini recommendation: {recommendation.model_type} ({recommendation.model_size})")
                    return recommendation
        
        # Fallback to rule-based recommendation
        self.logger.info("Using fallback recommendation")
        return self._fallback_recommendation(metadata)
    
    def get_multiple_recommendations(
        self,
        metadata: ClientMetadata,
        count: int = 5
    ) -> List[ModelRecommendation]:
        """Get multiple model recommendations (up to count)."""
        recommendations = []
        
        # Try Gemini for top recommendation
        if self.api_key:
            prompt = self._build_prompt(metadata)
            response = self._call_api(prompt)
            
            if response:
                rec = self._parse_response(response)
                if rec:
                    rec.source = "gemini"
                    recommendations.append(rec)
        
        # If we don't have enough, add fallback recommendations
        if len(recommendations) < count:
            fallback_opts = self._get_fallback_options(metadata)
            for opt in fallback_opts:
                if len(recommendations) >= count:
                    break
                # Avoid duplicates
                if not any(r.model_type == opt.model_type and r.model_size == opt.model_size for r in recommendations):
                    recommendations.append(opt)
        
        return recommendations[:count]
    
    def _get_fallback_options(self, metadata: ClientMetadata) -> List[ModelRecommendation]:
        """Get multiple fallback options."""
        options = []
        
        # Small option
        if metadata.dataset_size < 5000 or not metadata.has_gpu:
            options.append(ModelRecommendation(
                model_type='mlp',
                model_size='small',
                estimated_params=30000,
                architecture={'fc_layers': 2, 'hidden_dim': 64, 'dropout': 0.5},
                expected_accuracy=0.7,
                reasoning="Lightweight option for limited resources",
                config={'batch_size': 16, 'learning_rate': 0.01, 'local_epochs': 2},
                source='builtin'
            ))
        
        # Medium option
        options.append(ModelRecommendation(
            model_type='cnn',
            model_size='medium',
            estimated_params=200000,
            architecture={'conv_layers': 2, 'fc_layers': 2, 'hidden_dim': 128, 'dropout': 0.5},
            expected_accuracy=0.8,
            reasoning="Balanced option for most scenarios",
            config={'batch_size': 32, 'learning_rate': 0.01, 'local_epochs': 2},
            source='builtin'
        ))
        
        # Large option
        if metadata.has_gpu and (metadata.gpu_memory_mb or 0) > 4000:
            options.append(ModelRecommendation(
                model_type='cnn',
                model_size='large',
                estimated_params=500000,
                architecture={'conv_layers': 3, 'fc_layers': 2, 'hidden_dim': 256, 'dropout': 0.5},
                expected_accuracy=0.85,
                reasoning="High-capacity option for powerful GPUs",
                config={'batch_size': 64, 'learning_rate': 0.01, 'local_epochs': 3},
                source='builtin'
            ))
        
        # Transformer option if enough resources
        if metadata.has_gpu and metadata.dataset_size > 5000:
            options.append(ModelRecommendation(
                model_type='transformer',
                model_size='medium',
                estimated_params=400000,
                architecture={'num_heads': 4, 'num_layers': 3, 'hidden_dim': 128, 'dropout': 0.3},
                expected_accuracy=0.82,
                reasoning="Transformer architecture for complex patterns",
                config={'batch_size': 32, 'learning_rate': 0.0001, 'local_epochs': 2},
                source='builtin'
            ))
        
        return options
    
    @staticmethod
    def parse_huggingface_url(url: str) -> Optional[str]:
        """Parse HuggingFace model URL to get model name."""
        import re
        # Handle various HF URL formats
        patterns = [
            r'huggingface\.co/([^/]+/[^/]+)',
            r'^([^/]+/[^/]+)$',  # Direct model name
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None


class RecommendationCache:
    """Caches recommendations to reduce API calls."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[ModelRecommendation, float]] = {}
        self.ttl = ttl_seconds
        self.logger = logging.getLogger(__name__)
    
    def _get_key(self, metadata: ClientMetadata) -> str:
        """Generate cache key from metadata."""
        key_parts = [
            str(metadata.dataset_size),
            str(metadata.num_classes),
            str(metadata.has_gpu),
            str(metadata.gpu_memory_mb or 0),
            str(metadata.cpu_cores or 0),
            metadata.preferred_model_type or 'any'
        ]
        return "|".join(key_parts)
    
    def get(self, metadata: ClientMetadata) -> Optional[ModelRecommendation]:
        """Get cached recommendation if valid."""
        import time
        
        key = self._get_key(metadata)
        
        if key in self.cache:
            recommendation, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.logger.debug("Using cached recommendation")
                return recommendation
            else:
                del self.cache[key]
        
        return None
    
    def set(self, metadata: ClientMetadata, recommendation: ModelRecommendation):
        """Cache a recommendation."""
        import time
        key = self._get_key(metadata)
        self.cache[key] = (recommendation, time.time())
    
    def clear(self):
        """Clear all cached recommendations."""
        self.cache.clear()


class ModelRecommendationService:
    """Service managing model recommendations."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.recommender = GeminiRecommender(api_key)
        self.cache = RecommendationCache()
        self.logger = logging.getLogger(__name__)
        
        # Recommendation history
        self.history: List[Dict[str, Any]] = []
    
    def get_recommendation(
        self,
        metadata: ClientMetadata,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> ModelRecommendation:
        """Get model recommendation with caching."""
        
        # Check cache first
        if use_cache and not force_refresh:
            cached = self.cache.get(metadata)
            if cached:
                return cached
        
        # Get recommendation
        recommendation = self.recommender.recommend(metadata)
        
        # Cache result
        if use_cache:
            self.cache.set(metadata, recommendation)
        
        # Log to history
        self.history.append({
            'timestamp': None,  # Will be added by DB
            'metadata': {
                'dataset_size': metadata.dataset_size,
                'num_classes': metadata.num_classes,
                'has_gpu': metadata.has_gpu
            },
            'recommendation': {
                'model_type': recommendation.model_type,
                'model_size': recommendation.model_size,
                'estimated_params': recommendation.estimated_params
            }
        })
        
        return recommendation
    
    def approve_recommendation(
        self,
        recommendation: ModelRecommendation,
        admin_user_id: int
    ) -> Dict[str, Any]:
        """Admin approves a recommendation for group use."""
        self.logger.info(
            f"Admin {admin_user_id} approved recommendation: "
            f"{recommendation.model_type} ({recommendation.model_size})"
        )
        
        return {
            'status': 'approved',
            'model_config': {
                'type': recommendation.model_type,
                'size': recommendation.model_size,
                'params': recommendation.estimated_params,
                'architecture': recommendation.architecture,
                'training_config': recommendation.config
            }
        }
    
    def reject_recommendation(
        self,
        recommendation: ModelRecommendation,
        admin_user_id: int,
        reason: str
    ) -> Dict[str, Any]:
        """Admin rejects a recommendation."""
        self.logger.info(
            f"Admin {admin_user_id} rejected recommendation: {reason}"
        )
        
        return {
            'status': 'rejected',
            'reason': reason
        }
    
    def get_history(
        self,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recommendation history."""
        return self.history[-limit:]
    
    def get_all_recommendations(
        self,
        metadata: ClientMetadata,
        builtin_models: List[Dict[str, Any]],
        count: int = 5
    ) -> List[ModelRecommendation]:
        """Get unified recommendations from all sources."""
        recommendations = []
        seen_types = set()
        
        # 1. Get Gemini/recommendations
        gemini_recs = self.recommender.get_multiple_recommendations(metadata, count)
        for rec in gemini_recs:
            rec.source = "gemini"
            recommendations.append(rec)
            seen_types.add((rec.model_type, rec.model_size))
        
        # 2. Add matching builtin models
        for model in builtin_models:
            if len(recommendations) >= count:
                break
            model_key = (model.get('model_type', 'cnn'), 'medium')
            if model_key in seen_types:
                continue
            
            rec = ModelRecommendation(
                model_id=model.get('model_id', ''),
                model_type=model.get('model_type', 'vision'),
                model_size='medium',
                estimated_params=model.get('total_params', 100000),
                architecture={},
                expected_accuracy=0.8,
                reasoning=f"Built-in model: {model.get('architecture', 'Custom')}",
                config={},
                source='builtin',
                model_name=model.get('architecture', model.get('model_id', ''))
            )
            recommendations.append(rec)
            seen_types.add(model_key)
        
        return recommendations[:count]
    
    def add_huggingface_model(
        self,
        model_url: str,
        use_peft: bool = False,
        peft_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Add a model from HuggingFace URL to registry."""
        model_name = self.recommender.parse_huggingface_url(model_url)
        
        if not model_name:
            return {'success': False, 'error': 'Invalid HuggingFace URL or model name'}
        
        try:
            # Import here to avoid circular imports
            from model_registry.registry import get_registry
            registry = get_registry()
            
            model_info = registry.register_hf_model(
                model_name=model_name,
                use_peft=use_peft,
                peft_config=peft_config
            )
            
            return {
                'success': True,
                'model': model_info.to_dict(),
                'message': f'Successfully registered {model_name}'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


# Global service instance
_recommendation_service: Optional[ModelRecommendationService] = None


def get_recommendation_service() -> ModelRecommendationService:
    """Get the global recommendation service."""
    global _recommendation_service
    if _recommendation_service is None:
        _recommendation_service = ModelRecommendationService()
    return _recommendation_service


def init_recommendation_service(api_key: str) -> ModelRecommendationService:
    """Initialize the recommendation service with API key."""
    global _recommendation_service
    _recommendation_service = ModelRecommendationService(api_key)
    return _recommendation_service


# Utility functions
def metadata_from_dict(data: Dict[str, Any]) -> ClientMetadata:
    """Create ClientMetadata from dictionary."""
    return ClientMetadata(
        dataset_size=data.get('dataset_size', 0),
        num_classes=data.get('num_classes', 10),
        class_distribution=data.get('class_distribution', {}),
        has_gpu=data.get('has_gpu', False),
        gpu_memory_mb=data.get('gpu_memory_mb'),
        cpu_cores=data.get('cpu_cores'),
        memory_mb=data.get('memory_mb'),
        network_bandwidth_mbps=data.get('network_bandwidth_mbps'),
        preferred_model_type=data.get('preferred_model_type')
    )
