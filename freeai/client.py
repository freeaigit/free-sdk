"""Main FreeAI client class."""

import os
from typing import Optional, Dict, Any, List

import requests

from .exceptions import FreeAIError, AuthenticationError, RateLimitError, InsufficientCreditsError
from .models import (
    ChatResponse,
    ImageResponse,
    TTSResponse,
    STTResponse,
    TranslateResponse,
    MusicResponse,
    VideoResponse,
)

# Provider base URLs for BYOK
_PROVIDER_URLS = {
    "freeai": "https://api.free.ai",
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "google": "https://generativelanguage.googleapis.com",
    "openrouter": "https://openrouter.ai/api",
}

_DEFAULT_BASE_URL = "https://api.free.ai"


class FreeAI:
    """Free.ai Python SDK client.

    Usage:
        # Anonymous (daily free limits)
        ai = FreeAI()

        # With Free.ai API key
        ai = FreeAI(api_key="sk-free-xxx")

        # BYOK — use your own provider key, zero markup
        ai = FreeAI(provider="openai", api_key="sk-proj-xxx")

    All methods return typed response objects with a .raw attribute
    containing the full JSON response.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "freeai",
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        self.api_key = api_key or os.environ.get("FREEAI_API_KEY", "")
        self.provider = provider or os.environ.get("FREEAI_PROVIDER", "freeai")
        self.base_url = (base_url or _PROVIDER_URLS.get(self.provider, _DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider != "freeai" and self.api_key:
            headers["X-BYOK-Provider"] = self.provider
        return headers

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("headers", self._headers())

        try:
            resp = self._session.request(method, url, **kwargs)
        except requests.RequestException as e:
            raise FreeAIError(f"Request failed: {e}")

        if resp.status_code == 401:
            raise AuthenticationError("Invalid or missing API key", status_code=401)
        if resp.status_code == 402:
            raise InsufficientCreditsError("Insufficient tokens — buy more at https://free.ai/pricing/", status_code=402)
        if resp.status_code == 429:
            raise RateLimitError("Rate limit exceeded — wait or upgrade your plan", status_code=429)
        if resp.status_code >= 400:
            try:
                body = resp.json()
                msg = body.get("error", body.get("detail", resp.text))
            except Exception:
                msg = resp.text
            raise FreeAIError(f"API error {resp.status_code}: {msg}", status_code=resp.status_code)

        return resp.json()

    # ── Chat ──────────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        model: str = "qwen7b",
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> ChatResponse:
        """Chat with an AI model.

        Args:
            message: The user message (ignored if messages is provided).
            model: Model name (e.g. "qwen7b", "openai/gpt-4o", "anthropic/claude-sonnet-4").
            messages: Full message history (overrides message param).
            system: System prompt.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Max tokens in response.
            stream: Enable streaming (not yet supported in SDK, use raw API).

        Returns:
            ChatResponse with .text, .model, .usage, .raw attributes.
        """
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": message})

        payload: Dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        data = self._request("POST", "/v1/chat/", json=payload)
        return ChatResponse.from_dict(data)

    # ── Image Generation ──────────────────────────────────────────────

    def image(
        self,
        prompt: str,
        model: str = "flux-schnell",
        aspect_ratio: str = "1:1",
        negative_prompt: str = "",
    ) -> ImageResponse:
        """Generate an image from a text prompt.

        Args:
            prompt: Description of the image to generate.
            model: Model name (e.g. "flux-schnell", "sdxl", "kandinsky").
            aspect_ratio: Aspect ratio (e.g. "1:1", "16:9", "9:16").
            negative_prompt: Things to avoid in the image.

        Returns:
            ImageResponse with .url, .save(path), .usage, .raw attributes.
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "aspect_ratio": aspect_ratio,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        data = self._request("POST", "/v1/image/generate/", json=payload)
        return ImageResponse.from_dict(data)

    # ── Text to Speech ────────────────────────────────────────────────

    def tts(
        self,
        text: str,
        model: str = "kokoro",
        voice: str = "af_heart",
        language: str = "en",
    ) -> TTSResponse:
        """Convert text to speech.

        Args:
            text: Text to convert to speech.
            model: TTS model (e.g. "kokoro", "piper", "melotts", "chatterbox").
            voice: Voice name.
            language: Language code.

        Returns:
            TTSResponse with .url, .save(path), .usage, .raw attributes.
        """
        payload = {
            "text": text,
            "model": model,
            "voice": voice,
            "language": language,
        }
        data = self._request("POST", "/v1/tts/", json=payload)
        return TTSResponse.from_dict(data)

    # ── Speech to Text ────────────────────────────────────────────────

    def stt(
        self,
        url: Optional[str] = None,
        file_path: Optional[str] = None,
        model: str = "whisper",
        language: Optional[str] = None,
    ) -> STTResponse:
        """Transcribe audio to text.

        Args:
            url: URL of the audio file to transcribe.
            file_path: Local path to an audio file (uploaded to API).
            model: STT model (e.g. "whisper").
            language: Language hint (optional).

        Returns:
            STTResponse with .text, .language, .usage, .raw attributes.
        """
        payload: Dict[str, Any] = {"model": model}
        if url:
            payload["url"] = url
        if language:
            payload["language"] = language

        # TODO: file upload support
        data = self._request("POST", "/v1/stt/transcribe/", json=payload)
        return STTResponse.from_dict(data)

    # ── Translation ───────────────────────────────────────────────────

    def translate(
        self,
        text: str,
        to: str = "es",
        source: Optional[str] = None,
    ) -> TranslateResponse:
        """Translate text between languages.

        Args:
            text: Text to translate.
            to: Target language code (e.g. "es", "fr", "de", "ja").
            source: Source language code (auto-detected if omitted).

        Returns:
            TranslateResponse with .text, .source_language, .target_language, .usage, .raw attributes.
        """
        payload: Dict[str, Any] = {"text": text, "target": to}
        if source:
            payload["source"] = source

        data = self._request("POST", "/v1/translate/", json=payload)
        return TranslateResponse.from_dict(data)

    # ── Music Generation ──────────────────────────────────────────────

    def music(
        self,
        prompt: str,
        duration: int = 10,
        model: str = "audioldm2",
    ) -> MusicResponse:
        """Generate music from a text description.

        Args:
            prompt: Description of the music to generate.
            duration: Duration in seconds.
            model: Music model name.

        Returns:
            MusicResponse with .url, .save(path), .usage, .raw attributes.
        """
        payload = {"prompt": prompt, "duration": duration, "model": model}
        data = self._request("POST", "/v1/music/generate/", json=payload)
        return MusicResponse.from_dict(data)

    # ── Video Generation ──────────────────────────────────────────────

    def video(
        self,
        prompt: str,
        model: str = "cogvideox",
    ) -> VideoResponse:
        """Generate a video from a text prompt.

        Args:
            prompt: Description of the video to generate.
            model: Video model name.

        Returns:
            VideoResponse with .url, .save(path), .usage, .raw attributes.
        """
        payload = {"prompt": prompt, "model": model}
        data = self._request("POST", "/v1/video/generate/", json=payload)
        return VideoResponse.from_dict(data)

    # ── Image Tools ───────────────────────────────────────────────────

    def enhance_image(self, url: str, scale: int = 2) -> ImageResponse:
        """Upscale an image 2x or 4x."""
        data = self._request("POST", "/v1/image/enhance/", json={"url": url, "scale": scale})
        return ImageResponse.from_dict(data)

    def remove_background(self, url: str) -> ImageResponse:
        """Remove the background from an image."""
        data = self._request("POST", "/v1/image/remove-bg/", json={"url": url})
        return ImageResponse.from_dict(data)

    # ── Utility ───────────────────────────────────────────────────────

    def models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        data = self._request("GET", "/v1/models")
        return data.get("models", data.get("data", []))

    def health(self) -> Dict[str, Any]:
        """Check API health."""
        return self._request("GET", "/health")
