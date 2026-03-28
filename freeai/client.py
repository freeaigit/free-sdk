"""Main FreeAI client class — sync and async."""

import os
import json
import mimetypes
from typing import Optional, Dict, Any, List, Iterator

import requests

from .exceptions import FreeAIError, AuthenticationError, RateLimitError, InsufficientCreditsError
from .models import (
    ChatResponse,
    ChatStreamChunk,
    ImageResponse,
    TTSResponse,
    STTResponse,
    TranslateResponse,
    MusicResponse,
    VideoResponse,
    OCRResponse,
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


def _handle_error(status_code: int, body_text: str):
    """Raise the appropriate exception for an HTTP error status."""
    if status_code == 401:
        raise AuthenticationError("Invalid or missing API key", status_code=401)
    if status_code == 402:
        raise InsufficientCreditsError("Insufficient tokens — buy more at https://free.ai/pricing/", status_code=402)
    if status_code == 429:
        raise RateLimitError("Rate limit exceeded — wait or upgrade your plan", status_code=429)
    if status_code >= 400:
        try:
            body = json.loads(body_text)
            msg = body.get("error", body.get("detail", body_text))
        except Exception:
            msg = body_text
        raise FreeAIError(f"API error {status_code}: {msg}", status_code=status_code)


def _build_headers(api_key: str, provider: str, content_type: str = "application/json") -> Dict[str, str]:
    """Build common request headers."""
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if provider != "freeai" and api_key:
        headers["X-BYOK-Provider"] = provider
    return headers


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

    def _headers(self, content_type: str = "application/json") -> Dict[str, str]:
        return _build_headers(self.api_key, self.provider, content_type)

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("headers", self._headers())

        try:
            resp = self._session.request(method, url, **kwargs)
        except requests.RequestException as e:
            raise FreeAIError(f"Request failed: {e}")

        _handle_error(resp.status_code, resp.text)
        return resp.json()

    def _request_stream(self, method: str, path: str, **kwargs) -> Iterator[Dict[str, Any]]:
        """Make a streaming request and yield parsed SSE data dicts."""
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("headers", self._headers())
        kwargs["stream"] = True

        try:
            resp = self._session.request(method, url, **kwargs)
        except requests.RequestException as e:
            raise FreeAIError(f"Request failed: {e}")

        _handle_error(resp.status_code, "")

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue

    def _upload_file(self, path: str, endpoint: str, extra_fields: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Upload a file via multipart form POST."""
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "application/octet-stream"
        headers = _build_headers(self.api_key, self.provider, content_type=None)

        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, mime)}
            data = extra_fields or {}
            try:
                resp = self._session.post(
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                raise FreeAIError(f"Upload failed: {e}")

        _handle_error(resp.status_code, resp.text)
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
            stream: Enable streaming (use chat_stream() for iterator-based streaming).

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

    def chat_stream(
        self,
        message: str,
        model: str = "qwen7b",
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Iterator[ChatStreamChunk]:
        """Stream a chat response, yielding ChatStreamChunk objects.

        Usage:
            for chunk in ai.chat_stream("Write a poem"):
                print(chunk.text, end="")
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
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        for data in self._request_stream("POST", "/v1/chat/", json=payload):
            yield ChatStreamChunk.from_sse(data)

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
        if file_path:
            extra = {"model": model}
            if language:
                extra["language"] = language
            data = self._upload_file(file_path, "/v1/stt/transcribe/", extra_fields=extra)
            return STTResponse.from_dict(data)

        payload: Dict[str, Any] = {"model": model}
        if url:
            payload["url"] = url
        if language:
            payload["language"] = language

        data = self._request("POST", "/v1/stt/transcribe/", json=payload)
        return STTResponse.from_dict(data)

    def transcribe(
        self,
        file_path: str,
        model: str = "whisper",
        language: Optional[str] = None,
    ) -> STTResponse:
        """Convenience alias: transcribe a local audio file.

        Args:
            file_path: Local path to an audio file (mp3, wav, etc.).
            model: STT model.
            language: Language hint (optional).

        Returns:
            STTResponse with .text, .language, .usage, .raw attributes.
        """
        return self.stt(file_path=file_path, model=model, language=language)

    # ── OCR ───────────────────────────────────────────────────────────

    def ocr(
        self,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        language: str = "eng",
    ) -> OCRResponse:
        """Extract text from an image or PDF via OCR.

        Args:
            file_path: Local path to a PDF or image file.
            url: URL of the document to OCR.
            language: OCR language (e.g. "eng", "fra", "deu").

        Returns:
            OCRResponse with .text, .pages, .usage, .raw attributes.
        """
        if file_path:
            data = self._upload_file(file_path, "/v1/ocr/", extra_fields={"language": language})
            return OCRResponse.from_dict(data)

        payload: Dict[str, Any] = {"language": language}
        if url:
            payload["url"] = url
        data = self._request("POST", "/v1/ocr/", json=payload)
        return OCRResponse.from_dict(data)

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


# ════════════════════════════════════════════════════════════════════════
# Async Client
# ════════════════════════════════════════════════════════════════════════

try:
    import httpx as _httpx
    _HAS_HTTPX = True
except ImportError:
    _httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False


class AsyncFreeAI:
    """Async Free.ai Python SDK client (requires ``httpx``).

    Usage::

        from freeai import AsyncFreeAI

        ai = AsyncFreeAI(api_key="sk-free-xxx")
        response = await ai.chat("What is Python?")

        # Streaming
        async for chunk in ai.chat_stream("Write a poem"):
            print(chunk.text, end="")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "freeai",
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        if not _HAS_HTTPX:
            raise ImportError(
                "AsyncFreeAI requires the 'httpx' package. "
                "Install it with: pip install 'free-dot-ai[async]'"
            )
        self.api_key = api_key or os.environ.get("FREEAI_API_KEY", "")
        self.provider = provider or os.environ.get("FREEAI_PROVIDER", "freeai")
        self.base_url = (base_url or _PROVIDER_URLS.get(self.provider, _DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout
        self._client: Optional[Any] = None

    async def _get_client(self):
        if self._client is None:
            self._client = _httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _headers(self, content_type: str = "application/json") -> Dict[str, str]:
        return _build_headers(self.api_key, self.provider, content_type)

    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        client = await self._get_client()
        url = f"{self.base_url}{path}"
        kwargs.setdefault("headers", self._headers())

        try:
            resp = await client.request(method, url, **kwargs)
        except Exception as e:
            raise FreeAIError(f"Request failed: {e}")

        _handle_error(resp.status_code, resp.text)
        return resp.json()

    async def _upload_file(self, path: str, endpoint: str, extra_fields: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Upload a file via multipart form POST (async)."""
        client = await self._get_client()
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "application/octet-stream"
        headers = _build_headers(self.api_key, self.provider, content_type=None)

        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, mime)}
            data = extra_fields or {}
            try:
                resp = await client.post(
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    files=files,
                    data=data,
                )
            except Exception as e:
                raise FreeAIError(f"Upload failed: {e}")

        _handle_error(resp.status_code, resp.text)
        return resp.json()

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ── Chat ──────────────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        model: str = "qwen7b",
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """Chat with an AI model (async)."""
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

        data = await self._request("POST", "/v1/chat/", json=payload)
        return ChatResponse.from_dict(data)

    async def chat_stream(
        self,
        message: str,
        model: str = "qwen7b",
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Stream a chat response asynchronously, yielding ChatStreamChunk objects.

        Usage::

            async for chunk in ai.chat_stream("Write a poem"):
                print(chunk.text, end="")
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
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        client = await self._get_client()
        url = f"{self.base_url}/v1/chat/"
        headers = self._headers()

        try:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                _handle_error(resp.status_code, "")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        yield ChatStreamChunk.from_sse(data)
                    except json.JSONDecodeError:
                        continue
        except FreeAIError:
            raise
        except Exception as e:
            raise FreeAIError(f"Stream request failed: {e}")

    # ── Image Generation ──────────────────────────────────────────────

    async def image(
        self,
        prompt: str,
        model: str = "flux-schnell",
        aspect_ratio: str = "1:1",
        negative_prompt: str = "",
    ) -> ImageResponse:
        """Generate an image from a text prompt (async)."""
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "aspect_ratio": aspect_ratio,
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        data = await self._request("POST", "/v1/image/generate/", json=payload)
        return ImageResponse.from_dict(data)

    # ── Text to Speech ────────────────────────────────────────────────

    async def tts(
        self,
        text: str,
        model: str = "kokoro",
        voice: str = "af_heart",
        language: str = "en",
    ) -> TTSResponse:
        """Convert text to speech (async)."""
        payload = {
            "text": text,
            "model": model,
            "voice": voice,
            "language": language,
        }
        data = await self._request("POST", "/v1/tts/", json=payload)
        return TTSResponse.from_dict(data)

    # ── Speech to Text ────────────────────────────────────────────────

    async def stt(
        self,
        url: Optional[str] = None,
        file_path: Optional[str] = None,
        model: str = "whisper",
        language: Optional[str] = None,
    ) -> STTResponse:
        """Transcribe audio to text (async)."""
        if file_path:
            extra = {"model": model}
            if language:
                extra["language"] = language
            data = await self._upload_file(file_path, "/v1/stt/transcribe/", extra_fields=extra)
            return STTResponse.from_dict(data)

        payload: Dict[str, Any] = {"model": model}
        if url:
            payload["url"] = url
        if language:
            payload["language"] = language

        data = await self._request("POST", "/v1/stt/transcribe/", json=payload)
        return STTResponse.from_dict(data)

    async def transcribe(
        self,
        file_path: str,
        model: str = "whisper",
        language: Optional[str] = None,
    ) -> STTResponse:
        """Convenience alias: transcribe a local audio file (async)."""
        return await self.stt(file_path=file_path, model=model, language=language)

    # ── OCR ───────────────────────────────────────────────────────────

    async def ocr(
        self,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        language: str = "eng",
    ) -> OCRResponse:
        """Extract text from an image or PDF via OCR (async)."""
        if file_path:
            data = await self._upload_file(file_path, "/v1/ocr/", extra_fields={"language": language})
            return OCRResponse.from_dict(data)

        payload: Dict[str, Any] = {"language": language}
        if url:
            payload["url"] = url
        data = await self._request("POST", "/v1/ocr/", json=payload)
        return OCRResponse.from_dict(data)

    # ── Translation ───────────────────────────────────────────────────

    async def translate(
        self,
        text: str,
        to: str = "es",
        source: Optional[str] = None,
    ) -> TranslateResponse:
        """Translate text between languages (async)."""
        payload: Dict[str, Any] = {"text": text, "target": to}
        if source:
            payload["source"] = source

        data = await self._request("POST", "/v1/translate/", json=payload)
        return TranslateResponse.from_dict(data)

    # ── Music Generation ──────────────────────────────────────────────

    async def music(
        self,
        prompt: str,
        duration: int = 10,
        model: str = "audioldm2",
    ) -> MusicResponse:
        """Generate music from a text description (async)."""
        payload = {"prompt": prompt, "duration": duration, "model": model}
        data = await self._request("POST", "/v1/music/generate/", json=payload)
        return MusicResponse.from_dict(data)

    # ── Video Generation ──────────────────────────────────────────────

    async def video(
        self,
        prompt: str,
        model: str = "cogvideox",
    ) -> VideoResponse:
        """Generate a video from a text prompt (async)."""
        payload = {"prompt": prompt, "model": model}
        data = await self._request("POST", "/v1/video/generate/", json=payload)
        return VideoResponse.from_dict(data)

    # ── Image Tools ───────────────────────────────────────────────────

    async def enhance_image(self, url: str, scale: int = 2) -> ImageResponse:
        """Upscale an image 2x or 4x (async)."""
        data = await self._request("POST", "/v1/image/enhance/", json={"url": url, "scale": scale})
        return ImageResponse.from_dict(data)

    async def remove_background(self, url: str) -> ImageResponse:
        """Remove the background from an image (async)."""
        data = await self._request("POST", "/v1/image/remove-bg/", json={"url": url})
        return ImageResponse.from_dict(data)

    # ── Utility ───────────────────────────────────────────────────────

    async def models(self) -> List[Dict[str, Any]]:
        """List all available models (async)."""
        data = await self._request("GET", "/v1/models")
        return data.get("models", data.get("data", []))

    async def health(self) -> Dict[str, Any]:
        """Check API health (async)."""
        return await self._request("GET", "/health")
