"""Exceptions for the Free.ai SDK."""


class FreeAIError(Exception):
    """Base exception for Free.ai SDK errors."""

    def __init__(self, message: str, status_code: int = None, error_id: str = None):
        self.message = message
        self.status_code = status_code
        self.error_id = error_id
        super().__init__(message)


class AuthenticationError(FreeAIError):
    """Raised when the API key is invalid or missing."""
    pass


class RateLimitError(FreeAIError):
    """Raised when rate limits are exceeded."""
    pass


class InsufficientCreditsError(FreeAIError):
    """Raised when the account has insufficient tokens/credits."""
    pass
