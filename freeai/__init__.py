"""Free.ai Python SDK — access 400+ AI tools from your code."""

__version__ = "0.1.0"

from .client import FreeAI
from .exceptions import FreeAIError, AuthenticationError, RateLimitError, InsufficientCreditsError
from .models import ChatResponse, ImageResponse, TTSResponse, TranslateResponse, STTResponse

__all__ = [
    "FreeAI",
    "FreeAIError",
    "AuthenticationError",
    "RateLimitError",
    "InsufficientCreditsError",
    "ChatResponse",
    "ImageResponse",
    "TTSResponse",
    "TranslateResponse",
    "STTResponse",
]
