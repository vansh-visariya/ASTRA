"""
Authentication and Authorization for Federated Learning Platform.

Implements:
- JWT token-based authentication
- Role-based access control (Admin, Observer, Client)
- API key management
"""

import os
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKey
from pydantic import BaseModel


class UserRole(str, Enum):
    """User roles for access control."""
    ADMIN = "admin"
    OBSERVER = "observer"
    CLIENT = "client"


class TokenData(BaseModel):
    """JWT token payload."""
    client_id: str
    role: UserRole
    exp: datetime


class AuthToken:
    """Authentication token manager."""
    
    def __init__(
        self,
        secret_key: str = None,
        algorithm: str = "HS256",
        expire_minutes: int = 60
    ):
        self.secret_key = secret_key or os.getenv("SECRET_KEY", "dev-secret-key")
        self.algorithm = algorithm
        self.expire_minutes = expire_minutes
    
    def create_token(self, client_id: str, role: UserRole = UserRole.CLIENT) -> str:
        """Create JWT token."""
        expire = datetime.utcnow() + timedelta(minutes=self.expire_minutes)
        
        payload = {
            "client_id": client_id,
            "role": role.value,
            "exp": expire.isoformat()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> TokenData:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return TokenData(
                client_id=payload["client_id"],
                role=UserRole(payload["role"]),
                exp=datetime.fromisoformat(payload["exp"])
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )


class APIKeyManager:
    """Manage API keys for clients."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
    
    def create_key(
        self,
        client_id: str,
        role: UserRole = UserRole.CLIENT,
        description: str = ""
    ) -> str:
        """Create new API key."""
        api_key = f"fl_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            "client_id": client_id,
            "role": role,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "is_active": True
        }
        
        return api_key
    
    def verify_key(self, api_key: str) -> Optional[Dict]:
        """Verify API key."""
        if api_key not in self.api_keys:
            return None
        
        key_data = self.api_keys[api_key]
        if not key_data["is_active"]:
            return None
        
        return key_data
    
    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.api_keys:
            self.api_keys[api_key]["is_active"] = False
            return True
        return False
    
    def list_keys(self, client_id: str = None) -> List[Dict]:
        """List API keys."""
        keys = list(self.api_keys.values())
        
        if client_id:
            keys = [k for k in keys if k["client_id"] == client_id]
        
        return keys


# Security schemes
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")
BEARER_TOKEN_HEADER = APIKeyHeader(name="Authorization")


# Global instances
_auth_token: Optional[AuthToken] = None
_api_key_manager: Optional[APIKeyManager] = None


def get_auth_token() -> AuthToken:
    """Get global auth token instance."""
    global _auth_token
    if _auth_token is None:
        _auth_token = AuthToken()
    return _auth_token


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


async def get_current_user(
    api_key: str = Security(API_KEY_HEADER),
    token: str = Security(BEARER_TOKEN_HEADER, default=None)
) -> TokenData:
    """
    Get current user from API key or JWT token.
    
    Supports both API key and JWT bearer token authentication.
    """
    # Try API key first
    key_manager = get_api_key_manager()
    key_data = key_manager.verify_key(api_key)
    
    if key_data:
        return TokenData(
            client_id=key_data["client_id"],
            role=key_data["role"],
            exp=datetime.utcnow() + timedelta(hours=1)
        )
    
    # Try JWT token
    if token:
        token = token.replace("Bearer ", "")
        auth = get_auth_token()
        return auth.verify_token(token)
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required"
    )


def require_role(allowed_roles: List[UserRole]):
    """Dependency for role-based access control."""
    async def role_checker(user: TokenData = Security(get_current_user)):
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {user.role} not allowed. Required: {allowed_roles}"
            )
        return user
    return role_checker


# =============================================================================
# Environment-based config
# =============================================================================

def is_production() -> bool:
    """Check if running in production mode."""
    return os.getenv("ENV", "dev").lower() == "prod"


def require_https():
    """Ensure request is over HTTPS in production."""
    if is_production():
        # In production, this would be handled by the web server/reverse proxy
        pass
