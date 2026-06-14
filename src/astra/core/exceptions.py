"""Custom exceptions for the Federated Learning Framework."""


class FLBaseError(Exception):
    """Base exception for all FL platform errors."""


class AggregationError(FLBaseError):
    """Raised when aggregation fails."""


class ConfigurationError(FLBaseError):
    """Raised when configuration is invalid."""
