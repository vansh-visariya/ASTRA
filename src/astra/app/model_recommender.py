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
from typing import Any

import requests


@dataclass
class ClientMetadata:
    """Client metadata for model recommendation."""

    dataset_size: int
    num_classes: int
    class_distribution: dict[int, float]
    has_gpu: bool
    gpu_memory_mb: int | None = None
    cpu_cores: int | None = None
    memory_mb: int | None = None
    network_bandwidth_mbps: float | None = None
    preferred_model_type: str | None = None
    data_type: str | None = None  # 'image', 'text', 'audio', 'tabular', 'multimodal'


@dataclass
class ModelRecommendation:
    """Model recommendation from Gemini."""

    id: str = ""
    model_type: str = "cnn"  # 'cnn', 'mlp', 'transformer'
    model_size: str = "medium"  # 'small', 'medium', 'large'
    estimated_params: int = 100000
    architecture: dict[str, Any] | None = None
    expected_accuracy: float = 0.8
    reasoning: str = ""
    config: dict[str, Any] | None = None
    source: str = "gemini"  # 'gemini', 'builtin', 'huggingface'
    model_id: str = ""  # registry model_id if available
    model_name: str = ""  # display name
    hf_url: str = ""  # HuggingFace URL if from HF

    def __post_init__(self):
        import uuid

        if self.architecture is None:
            self.architecture = {}
        if self.config is None:
            self.config = {}
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.source:
            self.source = "builtin"
        if not self.model_id:
            self.model_id = f"rec_{self.model_type}_{self.model_size}_{uuid.uuid4().hex[:6]}"
        if not self.model_name:
            self.model_name = f"{self.model_type.title()} ({self.model_size})"


class GeminiRecommender:
    """Model recommendation using Gemini API."""

    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str | None = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = "gemini-2.5-flash"

        if not self.api_key:
            self.logger.warning("No Gemini API key provided. Using fallback recommendations.")

    def _build_prompt(self, metadata: ClientMetadata) -> str:
        """Build prompt for Gemini API."""

        prompt = f"""FL model recommender. Return ONLY a flat JSON object (no nesting beyond 1 level, no arrays). No markdown.

Client: {metadata.dataset_size} samples, {metadata.num_classes} classes, GPU={metadata.has_gpu}, CPU={metadata.cpu_cores}, data={metadata.data_type or "unknown"}

Return: {{"model_type":"cnn|mlp|transformer|efficientnet","model_size":"small|medium|large","params":N,"accuracy":0.X,"reason":"1 sentence","batch":N,"lr":0.X,"epochs":N}}"""

        return prompt

    def _call_api(self, prompt: str) -> str | None:
        """Call Gemini API."""
        if not self.api_key:
            return None

        try:
            url = f"{self.API_BASE_URL}/models/{self.model_name}:generateContent"

            headers = {"Content-Type": "application/json"}

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 2046,
                    "topP": 0.95,
                    "topK": 40,
                },
            }

            response = requests.post(
                f"{url}?key={self.api_key}", headers=headers, json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                self.logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"Gemini API call failed: {e}")
            return None

    def _parse_response(self, response_text: str) -> ModelRecommendation | None:
        """Parse Gemini response into ModelRecommendation."""
        import re

        try:
            # Clean markdown wrappers
            cleaned = response_text.strip()
            for pattern in [r'```json\s*', r'```\s*', r'`json\s*']:
                cleaned = re.sub(pattern, '', cleaned)
            cleaned = re.sub(r'```\s*$', '', cleaned).strip()

            # Try direct parse
            data = self._try_json_parse(cleaned)
            if data:
                return self._dict_to_recommendation(data)

            # Extract JSON object with balanced braces
            brace_count = 0
            start = -1
            for i, ch in enumerate(cleaned):
                if ch == '{':
                    if brace_count == 0:
                        start = i
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0 and start >= 0:
                        json_str = cleaned[start:i + 1]
                        data = self._try_json_parse(json_str)
                        if data:
                            return self._dict_to_recommendation(data)
                        start = -1

            self.logger.warning(f"Could not extract JSON from Gemini response: {response_text[:200]}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to parse Gemini response: {e}")
            return None

    def _try_json_parse(self, text: str) -> dict | None:
        """Try multiple JSON parse strategies including truncated recovery."""
        import re
        for attempt in [text, text.replace('True', 'true').replace('False', 'false').replace('None', 'null')]:
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
        # Try fixing missing commas (common Gemini bug)
        try:
            fixed = re.sub(r'"\s*\n\s*"', '",\n"', text)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        # Truncation recovery: close dangling strings and braces
        try:
            fixed = re.sub(r'(:\s*"[^"]*)$', r'\1"}', text.strip())
            fixed = re.sub(r'(:\s*[-\d.]+)$', r'\1}', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        return None

    def _dict_to_recommendation(self, data: dict) -> ModelRecommendation:
        """Convert parsed dict to ModelRecommendation. Handles flat (new prompt) and nested formats."""
        model_type = data.get("model_type", "cnn")
        model_size = data.get("model_size", "medium")
        params = data.get("estimated_params", data.get("params", 100000))
        accuracy = data.get("expected_accuracy", data.get("accuracy", 0.8))
        reason = data.get("reasoning", data.get("reason", ""))
        config = data.get("config", {})
        if not config and "batch" in data:
            config = {"batch_size": data["batch"], "learning_rate": data["lr"], "local_epochs": data["epochs"]}

        return ModelRecommendation(
            id=f"gemini_{model_type}_{model_size}",
            model_type=model_type,
            model_size=model_size,
            estimated_params=params,
            architecture=data.get("architecture", {}),
            expected_accuracy=accuracy,
            reasoning=reason,
            config=config,
            source="gemini",
            model_id=f"gemini_{model_type}_{model_size}",
            model_name=f"Gemini: {model_type.title()} ({model_size})",
        )

    def _fallback_recommendation(self, metadata: ClientMetadata) -> ModelRecommendation:
        """Provide fallback recommendation when API unavailable."""

        # Determine model size based on hardware
        if metadata.has_gpu and (metadata.gpu_memory_mb or 0) > 4000:
            model_size = "large"
            estimated_params = 500000
        elif metadata.has_gpu or (metadata.cpu_cores or 0) > 4:
            model_size = "medium"
            estimated_params = 200000
        else:
            model_size = "small"
            estimated_params = 50000

        # Adjust based on dataset size
        if metadata.dataset_size < 1000:
            estimated_params = min(estimated_params, 30000)
            model_size = "small"
        elif metadata.dataset_size > 10000:
            estimated_params = max(estimated_params, 100000)

        # Determine model type
        if metadata.preferred_model_type:
            model_type = metadata.preferred_model_type
        elif metadata.dataset_size > 5000 and (metadata.has_gpu or (metadata.cpu_cores or 0) > 2):
            model_type = "cnn"
        else:
            model_type = "mlp"

        return ModelRecommendation(
            model_type=model_type,
            model_size=model_size,
            estimated_params=estimated_params,
            architecture={
                "conv_layers": 2 if model_type == "cnn" else 0,
                "fc_layers": 2,
                "hidden_dim": estimated_params // 1000,
                "dropout": 0.5,
            },
            expected_accuracy=0.85 if model_size == "large" else 0.75,
            reasoning="Fallback recommendation based on hardware constraints.",
            config={
                "batch_size": 32 if not metadata.has_gpu else 64,
                "learning_rate": 0.01,
                "local_epochs": 2,
            },
        )

    def recommend(self, metadata: ClientMetadata) -> ModelRecommendation:
        """Get model recommendation for client."""

        # Try Gemini API first
        if self.api_key:
            prompt = self._build_prompt(metadata)
            response = self._call_api(prompt)

            if response:
                recommendation = self._parse_response(response)
                if recommendation:
                    self.logger.info(
                        f"Gemini recommendation: {recommendation.model_type}"
                        f" ({recommendation.model_size})"
                    )
                    return recommendation

        # Fallback to rule-based recommendation
        self.logger.info("Using fallback recommendation")
        return self._fallback_recommendation(metadata)

    def get_multiple_recommendations(
        self, metadata: ClientMetadata, count: int = 5
    ) -> list[ModelRecommendation]:
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

        # Add fallback recommendations NOT overlapping with Gemini
        if len(recommendations) < count:
            fallback_opts = self._get_fallback_options(metadata)
            for opt in fallback_opts:
                if len(recommendations) >= count:
                    break
                if not any(
                    r.model_type == opt.model_type and r.model_size == opt.model_size
                    for r in recommendations
                ):
                    recommendations.append(opt)

        return recommendations[:count]

    def _get_fallback_options(self, metadata: ClientMetadata) -> list[ModelRecommendation]:
        """Get multiple fallback options, tailored to data type."""
        options = []
        dtype = metadata.data_type or "unknown"

        if dtype == "image":
            options.append(
                ModelRecommendation(
                    model_type="cnn",
                    model_size="small",
                    estimated_params=100000,
                    architecture={"conv_layers": 2, "fc_layers": 1, "hidden_dim": 64, "dropout": 0.5},
                    expected_accuracy=0.75,
                    reasoning="Lightweight CNN for small image datasets",
                    config={"batch_size": 32, "learning_rate": 0.01, "local_epochs": 2},
                    source="builtin",
                    model_id="recommended_cnn_small_image",
                    model_name="Small CNN (Image)",
                )
            )
            options.append(
                ModelRecommendation(
                    model_type="efficientnet",
                    model_size="medium",
                    estimated_params=5000000,
                    architecture={"variant": "efficientnet-b0", "pretrained": True},
                    expected_accuracy=0.85,
                    reasoning="EfficientNet-B0 — strong image baseline with pretrained weights",
                    config={"batch_size": 32, "learning_rate": 0.001, "local_epochs": 3},
                    source="builtin",
                    model_id="recommended_efficientnet",
                    model_name="EfficientNet-B0",
                )
            )
            if metadata.has_gpu and metadata.dataset_size > 5000:
                options.append(
                    ModelRecommendation(
                        model_type="vit",
                        model_size="large",
                        estimated_params=86000000,
                        architecture={"variant": "vit-base-patch16-224", "pretrained": True},
                        expected_accuracy=0.92,
                        reasoning="Vision Transformer — state-of-the-art for large image datasets",
                        config={"batch_size": 16, "learning_rate": 0.0001, "local_epochs": 2},
                        source="builtin",
                        model_id="recommended_vit",
                        model_name="ViT-Base (Image)",
                    )
                )
        elif dtype == "text":
            options.append(
                ModelRecommendation(
                    model_type="transformer",
                    model_size="small",
                    estimated_params=2000000,
                    architecture={"variant": "distilbert", "num_layers": 6, "hidden_dim": 768},
                    expected_accuracy=0.78,
                    reasoning="DistilBERT — efficient text model, good for FL with limited bandwidth",
                    config={"batch_size": 16, "learning_rate": 0.0001, "local_epochs": 2},
                    source="builtin",
                    model_id="recommended_distilbert",
                    model_name="DistilBERT (Text)",
                )
            )
            options.append(
                ModelRecommendation(
                    model_type="transformer",
                    model_size="medium",
                    estimated_params=110000000,
                    architecture={"variant": "bert-base", "num_layers": 12, "hidden_dim": 768},
                    expected_accuracy=0.87,
                    reasoning="BERT-Base — strong transformer for text classification",
                    config={"batch_size": 8, "learning_rate": 0.00005, "local_epochs": 2},
                    source="builtin",
                    model_id="recommended_bert",
                    model_name="BERT-Base (Text)",
                )
            )
        elif dtype == "tabular":
            options.append(
                ModelRecommendation(
                    model_type="mlp",
                    model_size="small",
                    estimated_params=30000,
                    architecture={"fc_layers": 3, "hidden_dim": 64, "dropout": 0.3},
                    expected_accuracy=0.72,
                    reasoning="MLP — simple and effective for tabular data in FL settings",
                    config={"batch_size": 64, "learning_rate": 0.01, "local_epochs": 2},
                    source="builtin",
                    model_id="recommended_mlp_small",
                    model_name="Small MLP (Tabular)",
                )
            )
            options.append(
                ModelRecommendation(
                    model_type="mlp",
                    model_size="medium",
                    estimated_params=150000,
                    architecture={"fc_layers": 4, "hidden_dim": 256, "dropout": 0.3, "batch_norm": True},
                    expected_accuracy=0.8,
                    reasoning="Deep MLP with batch norm — handles complex tabular relationships",
                    config={"batch_size": 64, "learning_rate": 0.005, "local_epochs": 3},
                    source="builtin",
                    model_id="recommended_mlp_medium",
                    model_name="Medium MLP (Tabular)",
                )
            )
        else:
            # Generic fallback for unknown, audio, or multimodal
            generic = self._get_fallback_options_legacy(metadata)
            options.extend(generic)

        return options

    def _get_fallback_options_legacy(self, metadata: ClientMetadata) -> list[ModelRecommendation]:
        """Original generic fallback (kept for backwards compat)."""
        options = []

        # Small option
        if metadata.dataset_size < 5000 or not metadata.has_gpu:
            options.append(
                ModelRecommendation(
                    model_type="mlp",
                    model_size="small",
                    estimated_params=30000,
                    architecture={"fc_layers": 2, "hidden_dim": 64, "dropout": 0.5},
                    expected_accuracy=0.7,
                    reasoning="Lightweight option for limited resources",
                    config={"batch_size": 16, "learning_rate": 0.01, "local_epochs": 2},
                    source="builtin",
                    model_id="recommended_mlp_small_legacy",
                    model_name="Small MLP (General)",
                )
            )

        # Medium option
        options.append(
            ModelRecommendation(
                model_type="cnn",
                model_size="medium",
                estimated_params=200000,
                architecture={"conv_layers": 2, "fc_layers": 2, "hidden_dim": 128, "dropout": 0.5},
                expected_accuracy=0.8,
                reasoning="Balanced option for most scenarios",
                config={"batch_size": 32, "learning_rate": 0.01, "local_epochs": 2},
                source="builtin",
                model_id="recommended_cnn_medium_legacy",
                model_name="Medium CNN (General)",
            )
        )

        # Large option
        if metadata.has_gpu and (metadata.gpu_memory_mb or 0) > 4000:
            options.append(
                ModelRecommendation(
                    model_type="cnn",
                    model_size="large",
                    estimated_params=500000,
                    architecture={
                        "conv_layers": 3,
                        "fc_layers": 2,
                        "hidden_dim": 256,
                        "dropout": 0.5,
                    },
                    expected_accuracy=0.85,
                    reasoning="High-capacity option for powerful GPUs",
                    config={"batch_size": 64, "learning_rate": 0.01, "local_epochs": 3},
                    source="builtin",
                    model_id="recommended_cnn_large_legacy",
                    model_name="Large CNN (General)",
                )
            )

        # Transformer option if enough resources
        if metadata.has_gpu and metadata.dataset_size > 5000:
            options.append(
                ModelRecommendation(
                    model_type="transformer",
                    model_size="medium",
                    estimated_params=400000,
                    architecture={
                        "num_heads": 4,
                        "num_layers": 3,
                        "hidden_dim": 128,
                        "dropout": 0.3,
                    },
                    expected_accuracy=0.82,
                    reasoning="Transformer architecture for complex patterns",
                    config={"batch_size": 32, "learning_rate": 0.0001, "local_epochs": 2},
                    source="builtin",
                    model_id="recommended_transformer_legacy",
                    model_name="Transformer (General)",
                )
            )

        return options

    @staticmethod
    def parse_huggingface_url(url: str) -> str | None:
        """Parse HuggingFace model URL to get model name."""
        import re

        # Handle various HF URL formats
        patterns = [
            r"huggingface\.co/([^/]+/[^/]+)",
            r"^([^/]+/[^/]+)$",  # Direct model name
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None


class RecommendationCache:
    """Caches recommendations to reduce API calls."""

    def __init__(self, ttl_seconds: int = 3600):
        self.cache: dict[str, tuple[ModelRecommendation, float]] = {}
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
            metadata.preferred_model_type or "any",
        ]
        return "|".join(key_parts)

    def get(self, metadata: ClientMetadata) -> ModelRecommendation | None:
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

    def __init__(self, api_key: str | None = None):
        self.recommender = GeminiRecommender(api_key)
        self.cache = RecommendationCache()
        self.logger = logging.getLogger(__name__)

        # Recommendation history
        self.history: list[dict[str, Any]] = []

    def get_recommendation(
        self, metadata: ClientMetadata, use_cache: bool = True, force_refresh: bool = False
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
        self.history.append(
            {
                "timestamp": None,  # Will be added by DB
                "metadata": {
                    "dataset_size": metadata.dataset_size,
                    "num_classes": metadata.num_classes,
                    "has_gpu": metadata.has_gpu,
                },
                "recommendation": {
                    "model_type": recommendation.model_type,
                    "model_size": recommendation.model_size,
                    "estimated_params": recommendation.estimated_params,
                },
            }
        )

        return recommendation

    def approve_recommendation(
        self, recommendation: ModelRecommendation, admin_user_id: int
    ) -> dict[str, Any]:
        """Admin approves a recommendation for group use."""
        self.logger.info(
            f"Admin {admin_user_id} approved recommendation: "
            f"{recommendation.model_type} ({recommendation.model_size})"
        )

        return {
            "status": "approved",
            "model_config": {
                "type": recommendation.model_type,
                "size": recommendation.model_size,
                "params": recommendation.estimated_params,
                "architecture": recommendation.architecture,
                "training_config": recommendation.config,
            },
        }

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recommendation history."""
        return self.history[-limit:]

    def get_all_recommendations(
        self, metadata: ClientMetadata, builtin_models: list[dict[str, Any]], count: int = 5
    ) -> list[ModelRecommendation]:
        """Get unified recommendations from all sources."""
        recommendations = []
        seen_types = set()

        # 1. Get Gemini/Fallback recommendations
        gemini_recs = self.recommender.get_multiple_recommendations(metadata, count)
        for rec in gemini_recs:
            recommendations.append(rec)
            seen_types.add((rec.model_type, rec.model_size))

        # 2. Add matching builtin models
        for model in builtin_models:
            if len(recommendations) >= count:
                break
            model_key = (model.get("model_type", "cnn"), "medium")
            if model_key in seen_types:
                continue

            rec = ModelRecommendation(
                model_id=model.get("model_id", ""),
                model_type=model.get("model_type", "vision"),
                model_size="medium",
                estimated_params=model.get("total_params", 100000),
                architecture={},
                expected_accuracy=0.8,
                reasoning=f"Built-in model: {model.get('architecture', 'Custom')}",
                config={},
                source="builtin",
                model_name=model.get("architecture", model.get("model_id", "")),
            )
            recommendations.append(rec)
            seen_types.add(model_key)

        return recommendations[:count]

    def add_huggingface_model(
        self, model_url: str, use_peft: bool = False, peft_config: dict | None = None
    ) -> dict[str, Any]:
        """Add a model from HuggingFace URL to registry."""
        model_name = self.recommender.parse_huggingface_url(model_url)

        if not model_name:
            return {"success": False, "error": "Invalid HuggingFace URL or model name"}

        try:
            # Import here to avoid circular imports
            from astra.infra.registry import get_registry

            registry = get_registry()

            model_info = registry.register_hf_model(
                model_name=model_name, use_peft=use_peft, peft_config=peft_config
            )

            return {
                "success": True,
                "model": model_info.to_dict(),
                "message": f"Successfully registered {model_name}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global service instance
_recommendation_service: ModelRecommendationService | None = None


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
def metadata_from_dict(data: dict[str, Any]) -> ClientMetadata:
    """Create ClientMetadata from dictionary."""
    return ClientMetadata(
        dataset_size=data.get("dataset_size", 0),
        num_classes=data.get("num_classes", 10),
        class_distribution=data.get("class_distribution", {}),
        has_gpu=data.get("has_gpu", False),
        gpu_memory_mb=data.get("gpu_memory_mb"),
        cpu_cores=data.get("cpu_cores"),
        memory_mb=data.get("memory_mb"),
        network_bandwidth_mbps=data.get("network_bandwidth_mbps"),
        preferred_model_type=data.get("preferred_model_type"),
        data_type=data.get("data_type"),
    )
