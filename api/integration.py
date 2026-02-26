"""
Integration Module for Federated Learning Platform.

Combines:
- Authentication system (signup, login, role-based access)
- Notification system
- Trust score management
- Model recommendation (Gemini)
- Heterogeneous aggregation
- Inference module

This module extends the existing server API with new features.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from api.auth_system import (
    AuthManager,
    get_auth_manager,
    init_auth_manager,
    TokenManager,
    JoinRequestManager,
    TrustScoreManager
)
from api.notifications import (
    NotificationService,
    NotificationType,
    NotificationPriority,
    get_notification_service,
    init_notification_service
)
from api.model_recommender import (
    ModelRecommendationService,
    ClientMetadata,
    get_recommendation_service,
    init_recommendation_service
)
from core_engine.heterogeneous_aggregation import (
    HeterogeneousAggregator,
    BaselineModelManager,
    create_heterogeneous_aggregator
)
from core_engine.inference import (
    InferenceModule,
    create_inference_module,
    ServerSideInference,
    ClientSideInference
)


class FLPlatformIntegration:
    """Main integration class for the federated learning platform."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._init_auth()
        self._init_notifications()
        self._init_recommendations()
        self._init_aggregation()
        self._init_inference()
        
        self.logger.info("Platform integration initialized")
    
    def _init_auth(self):
        """Initialize authentication system."""
        db_path = self.config.get('auth', {}).get('db_path', './users.db')
        self.auth_manager: AuthManager = init_auth_manager(db_path)
        self.logger.info(f"Auth system initialized with DB: {db_path}")
    
    def _init_notifications(self):
        """Initialize notification system."""
        db_path = self.config.get('notifications', {}).get('db_path', './notifications.db')
        self.notification_service: NotificationService = init_notification_service(db_path)
        
        # Register notification handlers
        self._register_notification_handlers()
        
        self.logger.info(f"Notification system initialized with DB: {db_path}")
    
    def _register_notification_handlers(self):
        """Register handlers for notification events."""
        # Could add custom handlers here
        pass
    
    def _init_recommendations(self):
        """Initialize model recommendation service."""
        api_key = os.environ.get('GEMINI_API_KEY', '')
        if not api_key:
            api_key = self.config.get('gemini', {}).get('api_key', '')
        
        if api_key:
            self.recommendation_service: ModelRecommendationService = init_recommendation_service(api_key)
            self.logger.info("Model recommendation service initialized with Gemini API")
        else:
            self.recommendation_service = get_recommendation_service()
            self.logger.warning("No Gemini API key - using fallback recommendations")
    
    def _init_aggregation(self):
        """Initialize heterogeneous aggregation."""
        self.heterogeneous_aggregator = create_heterogeneous_aggregator(self.config)
        self.baseline_manager = BaselineModelManager(self.config)
        self.logger.info("Heterogeneous aggregation initialized")
    
    def _init_inference(self):
        """Initialize inference module."""
        self.inference_module = create_inference_module(self.config)
        self.logger.info("Inference module initialized")
    
    # ==================== Authentication Methods ====================
    
    def signup(
        self,
        username: str,
        password: str,
        role: str = 'client',
        email: Optional[str] = None,
        full_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register a new user."""
        user, error = self.auth_manager.signup(username, password, role, email, full_name)
        
        if error:
            return {'success': False, 'error': error}
        
        # Generate token
        token = self.auth_manager.token_manager.create_token(user)
        
        return {
            'success': True,
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'role': user.role,
                'full_name': user.full_name
            }
        }
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user."""
        result, error = self.auth_manager.login(username, password)
        
        if error:
            return {'success': False, 'error': error}
        
        return {'success': True, **result}
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        return self.auth_manager.verify_token(token)
    
    def require_role(self, token: str, allowed_roles: List[str]) -> Optional[Dict[str, Any]]:
        """Verify token and check role."""
        return self.auth_manager.require_role(token, allowed_roles)
    
    def get_all_users(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all users."""
        users = self.auth_manager.get_all_users(role)
        return [
            {
                'id': u.id,
                'username': u.username,
                'role': u.role,
                'full_name': u.full_name,
                'email': u.email
            }
            for u in users
        ]
    
    # ==================== Join Request Methods ====================
    
    def request_join_group(
        self,
        user_id: int,
        group_id: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Request to join a group."""
        nonce = self.auth_manager.join_request_manager.create_request(group_id, user_id, metadata)
        
        if not nonce:
            return {'success': False, 'error': 'Pending request already exists'}
        
        # Notify admin
        admin_users = self.auth_manager.get_all_users(role='admin')
        user = self.auth_manager.user_db.get_user_by_id(user_id)
        username = user.username if user else 'Unknown'
        
        for admin in admin_users:
            self.notification_service.notify_join_request(
                group_id=group_id,
                admin_user_id=admin.id,
                request_data={
                    'user_id': user_id,
                    'username': username,
                    'metadata': metadata or {}
                }
            )
        
        return {
            'success': True,
            'request_nonce': nonce,
            'status': 'pending'
        }
    
    def get_join_requests(
        self,
        token: str,
        group_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get pending join requests (admin only)."""
        payload = self.require_role(token, ['admin'])
        if not payload:
            return []
        
        return self.auth_manager.join_request_manager.get_pending_requests(group_id)
    
    def approve_join_request(
        self,
        token: str,
        request_id: int,
        group_id: str,
        join_token: str
    ) -> Dict[str, Any]:
        """Approve a join request."""
        admin_payload = self.require_role(token, ['admin'])
        if not admin_payload:
            return {'success': False, 'error': 'Unauthorized'}
        
        # Get request details
        requests = self.auth_manager.join_request_manager.get_pending_requests(group_id)
        request = next((r for r in requests if r['id'] == request_id), None)
        
        if not request:
            return {'success': False, 'error': 'Request not found'}
        
        # Create and validate token
        token, nonce = self.auth_manager.token_manager.create_join_token(group_id, request['user_id'])
        
        # Update request
        success = self.auth_manager.join_request_manager.approve_request(
            request_id,
            admin_payload['user_id'],
            token
        )
        
        if success:
            # Notify user
            self.notification_service.notify_join_approved(
                user_id=request['user_id'],
                group_id=group_id,
                token=token  # In production, this should be encrypted
            )
            
            return {'success': True, 'token': token}
        
        return {'success': False, 'error': 'Failed to approve request'}
    
    def reject_join_request(
        self,
        token: str,
        request_id: int,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reject a join request."""
        admin_payload = self.require_role(token, ['admin'])
        if not admin_payload:
            return {'success': False, 'error': 'Unauthorized'}
        
        success = self.auth_manager.join_request_manager.reject_request(
            request_id,
            admin_payload['user_id']
        )
        
        # Notify user
        if success:
            # Get request details for notification
            requests = self.auth_manager.join_request_manager.get_pending_requests()
            request = next((r for r in requests if r['id'] == request_id), None)
            
            if request:
                self.notification_service.notify_join_rejected(
                    user_id=request['user_id'],
                    group_id=request['group_id'],
                    reason=reason
                )
        
        return {'success': success}
    
    def get_user_join_status(self, user_id: int, group_id: str) -> Optional[Dict]:
        """Get user's join request status."""
        return self.auth_manager.join_request_manager.get_user_request_status(user_id, group_id)
    
    def validate_join_token(self, token: str) -> bool:
        """Validate a join token."""
        return self.auth_manager.token_manager.validate_join_token(token)
    
    # ==================== Notification Methods ====================
    
    def get_notifications(
        self,
        user_id: int,
        limit: int = 50,
        unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get user notifications."""
        notifications = self.notification_service.get_notifications(user_id, limit, unread_only)
        
        return [
            {
                'id': n.id,
                'type': n.notification_type.value,
                'priority': n.priority.value,
                'title': n.title,
                'message': n.message,
                'group_id': n.group_id,
                'data': n.data,
                'created_at': n.created_at.isoformat(),
                'read': n.read
            }
            for n in notifications
        ]
    
    def mark_notification_read(self, notification_id: int) -> bool:
        """Mark notification as read."""
        return self.notification_service.mark_read(notification_id)
    
    def mark_all_notifications_read(self, user_id: int) -> int:
        """Mark all notifications as read."""
        return self.notification_service.mark_all_read(user_id)
    
    def get_unread_notification_count(self, user_id: int) -> int:
        """Get unread notification count."""
        return self.notification_service.get_unread_count(user_id)
    
    # ==================== Trust Score Methods ====================
    
    def get_trust_score(self, user_id: int, group_id: str = 'default') -> float:
        """Get trust score for user."""
        return self.auth_manager.trust_score_manager.get_trust_score(user_id, group_id)
    
    def update_trust_score(
        self,
        user_id: int,
        group_id: str,
        score: float,
        quarantined: bool = False
    ):
        """Update trust score."""
        self.auth_manager.trust_score_manager.update_trust_score(user_id, group_id, score, quarantined)
    
    def get_all_trust_scores(self, group_id: Optional[str] = None) -> Dict[int, Dict]:
        """Get all trust scores."""
        return self.auth_manager.trust_score_manager.get_all_trust_scores(group_id)
    
    # ==================== Model Recommendation Methods ====================
    
    def recommend_model(
        self,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get model recommendation based on client metadata."""
        client_metadata = ClientMetadata(
            dataset_size=metadata.get('dataset_size', 0),
            num_classes=metadata.get('num_classes', 10),
            class_distribution=metadata.get('class_distribution', {}),
            has_gpu=metadata.get('has_gpu', False),
            gpu_memory_mb=metadata.get('gpu_memory_mb'),
            cpu_cores=metadata.get('cpu_cores'),
            memory_mb=metadata.get('memory_mb'),
            network_bandwidth_mbps=metadata.get('network_bandwidth_mbps'),
            preferred_model_type=metadata.get('preferred_model_type')
        )
        
        recommendation = self.recommendation_service.get_recommendation(client_metadata)
        
        return {
            'success': True,
            'model_type': recommendation.model_type,
            'model_size': recommendation.model_size,
            'estimated_params': recommendation.estimated_params,
            'architecture': recommendation.architecture,
            'expected_accuracy': recommendation.expected_accuracy,
            'reasoning': recommendation.reasoning,
            'config': recommendation.config
        }
    
    # ==================== Inference Methods ====================
    
    def run_inference(
        self,
        input_data: Any,
        method: str = 'server_side'
    ) -> Dict[str, Any]:
        """Run inference."""
        try:
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
    
    def register_inference_method(self, model: Any, method_name: str):
        """Register an inference method with a model."""
        if method_name == 'server_side':
            self.inference_module.create_server_side(model)
        elif method_name == 'client_side':
            self.inference_module.create_client_side(model)


# Global integration instance
_platform_integration: Optional[FLPlatformIntegration] = None


def get_platform_integration() -> FLPlatformIntegration:
    """Get the global platform integration instance."""
    global _platform_integration
    if _platform_integration is None:
        _platform_integration = FLPlatformIntegration({})
    return _platform_integration


def init_platform_integration(config: Dict[str, Any]) -> FLPlatformIntegration:
    """Initialize the platform integration with config."""
    global _platform_integration
    _platform_integration = FLPlatformIntegration(config)
    return _platform_integration
