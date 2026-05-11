"""
Custom exceptions for the Federated Learning Framework.
"""

from typing import Optional


class FLBaseError(Exception):
    """Base exception for all FL platform errors."""
    pass


class RoundStateError(FLBaseError):
    """Raised when round state is invalid for the requested operation."""
    pass


class InsufficientClientsError(FLBaseError):
    """Raised when not enough clients are available for aggregation."""
    pass


class IntegrityError(FLBaseError):
    """Raised when data integrity check fails (e.g., checksum mismatch)."""
    pass


class PrivacyBudgetExhausted(FLBaseError):
    """Raised when differential privacy budget is exhausted."""
    pass


class ByzantineClientDetected(FLBaseError):
    """Raised when a Byzantine (malicious) client is detected."""
    pass


class TokenExpiredError(FLBaseError):
    """Raised when authentication token has expired."""
    pass


class InvalidUpdateError(FLBaseError):
    """Raised when client update is invalid or malformed."""
    pass


class ModelSerializationError(FLBaseError):
    """Raised when model serialization/deserialization fails."""
    pass


class AggregationError(FLBaseError):
    """Raised when aggregation fails."""
    pass


class ConfigurationError(FLBaseError):
    """Raised when configuration is invalid."""
    pass
