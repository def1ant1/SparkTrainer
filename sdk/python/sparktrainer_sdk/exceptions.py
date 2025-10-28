"""
SparkTrainer SDK Exceptions
"""


class SparkTrainerException(Exception):
    """Base exception for SparkTrainer SDK"""
    pass


class AuthenticationError(SparkTrainerException):
    """Authentication failed"""
    pass


class NotFoundError(SparkTrainerException):
    """Resource not found"""
    pass


class RateLimitError(SparkTrainerException):
    """Rate limit exceeded"""
    pass


class ValidationError(SparkTrainerException):
    """Request validation failed"""
    pass


class ServerError(SparkTrainerException):
    """Server error occurred"""
    pass
