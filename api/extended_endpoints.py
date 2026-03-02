"""
Extended API Endpoints for Federated Learning Platform.

This module adds new endpoints for:
- User signup and authentication
- Secure group joining with token request/approval
- Notifications
- Trust scores
- Model recommendations
- Inference

These endpoints integrate with the existing server API.
"""

import base64
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic.fields import Field

from api.auth_system import AuthManager, get_auth_manager
from api.integration import (
    FLPlatformIntegration,
    get_platform_integration,
    init_platform_integration
)


# ============================================================================
# Request/Response Models
# ============================================================================

class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    role: str = Field(default='client', pattern='^(admin|client|observer)$')
    email: Optional[str] = None
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class JoinGroupRequest(BaseModel):
    group_id: str
    metadata: Optional[Dict[str, Any]] = None


class ApproveJoinRequest(BaseModel):
    request_id: int
    group_id: str
    join_token: str


class RejectJoinRequest(BaseModel):
    request_id: int
    reason: Optional[str] = None


class ClientMetadataRequest(BaseModel):
    dataset_size: int
    num_classes: int = 10
    class_distribution: Dict[int, float] = {}
    has_gpu: bool = False
    gpu_memory_mb: Optional[int] = None
    cpu_cores: Optional[int] = None
    memory_mb: Optional[int] = None
    network_bandwidth_mbps: Optional[float] = None
    preferred_model_type: Optional[str] = None


class InferenceRequest(BaseModel):
    input_data: str  # Base64 encoded
    method: str = 'server_side'


class ModelRecommendationRequest(BaseModel):
    dataset_size: int = 1000
    num_classes: int = 10
    class_distribution: Dict[int, float] = {}
    has_gpu: bool = False
    gpu_memory_mb: Optional[int] = None
    cpu_cores: Optional[int] = None
    memory_mb: Optional[int] = None
    network_bandwidth_mbps: Optional[float] = None
    preferred_model_type: Optional[str] = None


class AddHuggingFaceModelRequest(BaseModel):
    model_url: str
    use_peft: bool = False
    peft_method: str = "lora"
    lora_rank: int = 8
    lora_alpha: int = 16


class TrustScoreUpdate(BaseModel):
    user_id: int
    group_id: str = 'default'
    score: float = Field(..., ge=0.0, le=1.0)
    quarantined: bool = False


# ============================================================================
# Authentication Dependencies
# ============================================================================

def get_current_user(
    authorization: str = Header(None),
    platform: FLPlatformIntegration = Depends(get_platform_integration),
) -> Dict[str, Any]:
    """Get current authenticated user from token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="No authorization header")

    token = authorization.replace("Bearer ", "")
    payload = platform.verify_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    return payload


def require_admin(
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    """Require admin role."""
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def require_client(
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    """Require client or admin role."""
    if current_user.get('role') not in ('client', 'admin'):
        raise HTTPException(status_code=403, detail="Client access required")
    return current_user


# ============================================================================
# Authentication Endpoints
# ============================================================================

def create_auth_router(platform: FLPlatformIntegration) -> APIRouter:
    """Create authentication router."""
    
    router = APIRouter(prefix="/api/auth", tags=["Authentication"])
    
    @router.post("/signup")
    async def signup(request: SignupRequest):
        """Register a new user account."""
        result = platform.signup(
            username=request.username,
            password=request.password,
            role=request.role,
            email=request.email,
            full_name=request.full_name
        )
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error', 'Signup failed'))
        
        return result
    
    @router.post("/login")
    async def login(request: LoginRequest):
        """Authenticate user and get JWT token."""
        result = platform.login(request.username, request.password)
        
        if not result.get('success'):
            raise HTTPException(status_code=401, detail=result.get('error', 'Login failed'))
        
        return result
    
    @router.get("/me")
    async def get_me(current_user: Dict = Depends(get_current_user)):
        """Get current user info."""
        return {
            'username': current_user.get('sub'),
            'role': current_user.get('role'),
            'user_id': current_user.get('user_id')
        }
    
    @router.get("/users")
    async def list_users(
        role: Optional[str] = None,
        admin: Dict = Depends(require_admin)
    ):
        """List all users (admin only)."""
        users = platform.get_all_users(role)
        return {"users": users, "count": len(users)}
    
    return router


# ============================================================================
# Group Joining Endpoints
# ============================================================================

def create_group_join_router(platform: FLPlatformIntegration) -> APIRouter:
    """Create group joining router."""
    
    router = APIRouter(prefix="/api/groups", tags=["Group Joining"])
    
    @router.post("/join-request")
    async def request_join_group(
        request: JoinGroupRequest,
        current_user: Dict = Depends(require_client)
    ):
        """Request to join a group (client request)."""
        result = platform.request_join_group(
            user_id=current_user['user_id'],
            group_id=request.group_id,
            metadata=request.metadata
        )
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error'))
        
        return result
    
    @router.get("/join-requests")
    async def get_join_requests(
        group_id: Optional[str] = None,
        admin: Dict = Depends(require_admin)
    ):
        """Get pending join requests (admin only)."""
        requests = platform.get_join_requests(
            token=None,  # Already verified by require_admin
            group_id=group_id
        )
        return {"requests": requests, "count": len(requests)}
    
    @router.post("/join-requests/approve")
    async def approve_join_request(
        request: ApproveJoinRequest,
        authorization: str = Header(None)
    ):
        """Approve a join request and deliver token (admin only)."""
        # Verify admin role
        from fastapi import HTTPException
        if not authorization:
            raise HTTPException(status_code=401, detail="No authorization header")
        
        token = authorization.replace("Bearer ", "")
        
        # Verify admin role through platform
        admin_payload = platform.require_role(token, ['admin'])
        if not admin_payload:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        result = platform.approve_join_request(
            token=token,
            request_id=request.request_id,
            group_id=request.group_id,
            join_token=request.join_token
        )
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error'))
        
        return result
    
    @router.post("/join-requests/reject")
    async def reject_join_request(
        request: RejectJoinRequest,
        authorization: str = Header(None)
    ):
        """Reject a join request (admin only)."""
        from fastapi import HTTPException
        if not authorization:
            raise HTTPException(status_code=401, detail="No authorization header")
        
        token = authorization.replace("Bearer ", "")
        
        admin_payload = platform.require_role(token, ['admin'])
        if not admin_payload:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        result = platform.reject_join_request(
            token=token,
            request_id=request.request_id,
            group_id=request.group_id,
            reason=request.reason
        )
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error'))
        
        return result
    
    @router.get("/my-requests/{group_id}")
    async def get_my_request_status(
        group_id: str,
        current_user: Dict = Depends(require_client)
    ):
        """Get user's join request status for a group."""
        status = platform.get_user_join_status(
            user_id=current_user['user_id'],
            group_id=group_id
        )
        
        if not status:
            return {"status": "none", "message": "No pending request"}
        
        return status
    
    @router.post("/validate-token")
    async def validate_join_token(token: str):
        """Validate a join token."""
        is_valid = platform.validate_join_token(token)
        return {"valid": is_valid}
    
    return router


# ============================================================================
# Notification Endpoints
# ============================================================================

def create_notification_router(platform: FLPlatformIntegration) -> APIRouter:
    """Create notification router."""
    
    router = APIRouter(prefix="/api/notifications", tags=["Notifications"])
    
    @router.get("")
    async def get_notifications(
        limit: int = 50,
        unread_only: bool = False,
        current_user: Dict = Depends(get_current_user)
    ):
        """Get user notifications."""
        notifications = platform.get_notifications(
            user_id=current_user['user_id'],
            limit=limit,
            unread_only=unread_only
        )
        return {"notifications": notifications, "count": len(notifications)}
    
    @router.get("/unread-count")
    async def get_unread_count(
        current_user: Dict = Depends(get_current_user)
    ):
        """Get unread notification count."""
        count = platform.get_unread_notification_count(current_user['user_id'])
        return {"count": count}
    
    @router.post("/{notification_id}/read")
    async def mark_read(
        notification_id: int,
        current_user: Dict = Depends(get_current_user)
    ):
        """Mark notification as read."""
        success = platform.mark_notification_read(notification_id)
        return {"success": success}
    
    @router.post("/read-all")
    async def mark_all_read(
        current_user: Dict = Depends(get_current_user)
    ):
        """Mark all notifications as read."""
        count = platform.mark_all_notifications_read(current_user['user_id'])
        return {"marked_count": count}
    
    return router


# ============================================================================
# Trust Score Endpoints
# ============================================================================

def create_trust_router(platform: FLPlatformIntegration) -> APIRouter:
    """Create trust score router."""
    
    router = APIRouter(prefix="/api/trust", tags=["Trust Scores"])
    
    @router.get("/scores")
    async def get_all_trust_scores(
        group_id: Optional[str] = None,
        admin: Dict = Depends(require_admin)
    ):
        """Get all trust scores (admin only)."""
        scores = platform.get_all_trust_scores(group_id)
        return {"scores": scores}
    
    @router.get("/scores/{user_id}")
    async def get_user_trust(
        user_id: int,
        group_id: str = 'default'
    ):
        """Get trust score for a user."""
        score = platform.get_trust_score(user_id, group_id)
        return {"user_id": user_id, "group_id": group_id, "score": score}
    
    @router.post("/scores")
    async def update_trust_score(
        request: TrustScoreUpdate,
        admin: Dict = Depends(require_admin)
    ):
        """Update trust score (admin only)."""
        platform.update_trust_score(
            user_id=request.user_id,
            group_id=request.group_id,
            score=request.score,
            quarantined=request.quarantined
        )
        return {"success": True}
    
    return router


# ============================================================================
# Model Recommendation Endpoints
# ============================================================================

def create_recommendation_router(platform: FLPlatformIntegration) -> APIRouter:
    """Create model recommendation router."""
    
    router = APIRouter(prefix="/api/recommendations", tags=["Model Recommendations"])
    
    @router.post("/model")
    async def recommend_model(
        request: ClientMetadataRequest,
        current_user: Dict = Depends(get_current_user)
    ):
        """Get model recommendation based on client metadata."""
        metadata = request.dict()
        result = platform.recommend_model(metadata)
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail="Recommendation failed")
        
        return result
    
    @router.post("/approve")
    async def approve_recommendation(
        recommendation: Dict[str, Any],
        admin: Dict = Depends(require_admin)
    ):
        """Admin approves a model recommendation."""
        result = get_platform_integration().recommendation_service.approve_recommendation(
            recommendation,
            admin['user_id']
        )
        return result
    
    @router.get("/history")
    async def get_recommendation_history(
        limit: int = 50,
        admin: Dict = Depends(require_admin)
    ):
        """Get recommendation history."""
        history = get_platform_integration().recommendation_service.get_history(limit)
        return {"history": history}
    
    @router.post("/unified")
    async def get_unified_recommendations(
        request: ModelRecommendationRequest,
        current_user: Dict = Depends(get_current_user)
    ):
        """Get unified recommendations (Gemini + builtin + custom HF models)."""
        from api.model_recommender import metadata_from_dict, ClientMetadata
        
        metadata = metadata_from_dict(request.dict())
        
        # Get builtin models from registry
        from model_registry.registry import get_registry
        registry = get_registry()
        builtin_models = registry.list_models()
        
        # Get unified recommendations
        recommendations = get_platform_integration().recommendation_service.get_all_recommendations(
            metadata=metadata,
            builtin_models=builtin_models,
            count=5
        )
        
        return {
            "recommendations": [
                {
                    "id": r.id,
                    "model_type": r.model_type,
                    "model_size": r.model_size,
                    "estimated_params": r.estimated_params,
                    "expected_accuracy": r.expected_accuracy,
                    "reasoning": r.reasoning,
                    "source": r.source,
                    "model_id": r.model_id,
                    "model_name": r.model_name,
                    "config": r.config
                }
                for r in recommendations
            ]
        }
    
    @router.post("/add-huggingface")
    async def add_huggingface_model(
        request: AddHuggingFaceModelRequest,
        current_user: Dict = Depends(get_current_user)
    ):
        """Add a model from HuggingFace URL."""
        peft_config = None
        if request.use_peft:
            peft_config = {
                'enabled': True,
                'method': request.peft_method,
                'lora_rank': request.lora_rank,
                'lora_alpha': request.lora_alpha,
                'target_modules': ['q_proj', 'v_proj']
            }
        
        result = get_platform_integration().recommendation_service.add_huggingface_model(
            model_url=request.model_url,
            use_peft=request.use_peft,
            peft_config=peft_config
        )
        
        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error', 'Failed to add model'))
        
        return result
    
    @router.get("/builtin")
    async def get_builtin_models(
        current_user: Dict = Depends(get_current_user)
    ):
        """Get all available builtin models."""
        from model_registry.registry import get_registry
        registry = get_registry()
        models = registry.list_models()
        return {"models": models, "count": len(models)}
    
    return router


# ============================================================================
# Inference Endpoints
# ============================================================================

def create_inference_router(platform: FLPlatformIntegration) -> APIRouter:
    """Create inference router."""
    
    router = APIRouter(prefix="/api/inference", tags=["Inference"])
    
    @router.post("/predict")
    async def predict(
        request: InferenceRequest,
        current_user: Dict = Depends(get_current_user)
    ):
        """Run inference on server."""
        try:
            # Decode base64 input
            decoded = base64.b64decode(request.input_data)
            input_data = json.loads(decoded)
            
            result = platform.run_inference(
                input_data=input_data,
                method=request.method
            )
            
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/methods")
    async def get_methods(
        current_user: Dict = Depends(get_current_user)
    ):
        """Get available inference methods."""
        methods = platform.inference_module.get_available_methods()
        return {"methods": methods, "default": platform.inference_module.default_method}
    
    return router


# ============================================================================
# Main Router Assembly
# ============================================================================

def create_all_routers(platform: FLPlatformIntegration) -> List[APIRouter]:
    """Create all API routers."""
    return [
        create_auth_router(platform),
        create_group_join_router(platform),
        create_notification_router(platform),
        create_trust_router(platform),
        create_recommendation_router(platform),
        create_inference_router(platform),
    ]


# ============================================================================
# Factory Function
# ============================================================================

def setup_extended_api(
    app,
    config: Dict[str, Any]
) -> FLPlatformIntegration:
    """Setup extended API with all new endpoints."""
    
    # Initialize platform integration
    platform = init_platform_integration(config)
    
    # Create and register routers
    routers = create_all_routers(platform)
    
    for router in routers:
        app.include_router(router)
    
    return platform
