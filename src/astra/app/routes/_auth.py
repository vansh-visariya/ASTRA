"""Shared JWT verification helper for route modules."""

import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def verify_request_jwt(authorization: str | None) -> dict:
    """Validate JWT from Authorization header and return payload.

    Raises HTTPException(401) on any failure.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    try:
        from astra.app.integration import get_platform_integration

        payload = get_platform_integration().verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("JWT verify failed: %s", e)
        raise HTTPException(status_code=401, detail=f"Token verification failed: {e}") from None
