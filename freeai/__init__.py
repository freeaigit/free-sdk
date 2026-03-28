"""Free.ai Python SDK — access 400+ AI tools from your code."""

__version__ = "0.2.0"

from .client import FreeAI, AsyncFreeAI
from .exceptions import FreeAIError, AuthenticationError, RateLimitError, InsufficientCreditsError
from .models import (
    ChatResponse,
    ChatStreamChunk,
    ImageResponse,
    TTSResponse,
    TranslateResponse,
    STTResponse,
    OCRResponse,
    MusicResponse,
    VideoResponse,
)

__all__ = [
    "FreeAI",
    "AsyncFreeAI",
    "FreeAIError",
    "AuthenticationError",
    "RateLimitError",
    "InsufficientCreditsError",
    "ChatResponse",
    "ChatStreamChunk",
    "ImageResponse",
    "TTSResponse",
    "TranslateResponse",
    "STTResponse",
    "OCRResponse",
    "MusicResponse",
    "VideoResponse",
]
