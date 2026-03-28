"""Response types for the Free.ai SDK."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import requests


@dataclass
class Usage:
    """Token usage info returned with every response."""
    tokens_used: int = 0
    tokens_charged: int = 0
    source: str = ""
    model: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Usage":
        if not data:
            return cls()
        return cls(
            tokens_used=data.get("tokens_used", 0),
            tokens_charged=data.get("tokens_charged", 0),
            source=data.get("source", ""),
            model=data.get("model", ""),
        )


@dataclass
class ChatResponse:
    """Response from a chat completion request."""
    text: str
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatResponse":
        text = ""
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            text = msg.get("content", "")
        return cls(
            text=text,
            model=data.get("model", ""),
            usage=Usage.from_dict(data.get("free_ai_usage", {})),
            raw=data,
        )


@dataclass
class ImageResponse:
    """Response from an image generation request."""
    url: str
    usage: Usage = field(default_factory=Usage)
    raw: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> str:
        """Download and save the image to a local file."""
        r = requests.get(self.url, timeout=120)
        r.raise_for_status()
        p = Path(path)
        p.write_bytes(r.content)
        return str(p)

    @classmethod
    def from_dict(cls, data: dict) -> "ImageResponse":
        return cls(
            url=data.get("image_url", data.get("url", "")),
            usage=Usage.from_dict(data.get("free_ai_usage", {})),
            raw=data,
        )


@dataclass
class TTSResponse:
    """Response from a text-to-speech request."""
    url: str
    usage: Usage = field(default_factory=Usage)
    raw: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> str:
        """Download and save the audio to a local file."""
        r = requests.get(self.url, timeout=120)
        r.raise_for_status()
        p = Path(path)
        p.write_bytes(r.content)
        return str(p)

    @classmethod
    def from_dict(cls, data: dict) -> "TTSResponse":
        return cls(
            url=data.get("audio_url", data.get("url", "")),
            usage=Usage.from_dict(data.get("free_ai_usage", {})),
            raw=data,
        )


@dataclass
class STTResponse:
    """Response from a speech-to-text request."""
    text: str
    language: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "STTResponse":
        return cls(
            text=data.get("text", ""),
            language=data.get("language", ""),
            usage=Usage.from_dict(data.get("free_ai_usage", {})),
            raw=data,
        )


@dataclass
class TranslateResponse:
    """Response from a translation request."""
    text: str
    source_language: str = ""
    target_language: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "TranslateResponse":
        return cls(
            text=data.get("translated_text", data.get("text", "")),
            source_language=data.get("source_language", data.get("source", "")),
            target_language=data.get("target_language", data.get("target", "")),
            usage=Usage.from_dict(data.get("free_ai_usage", {})),
            raw=data,
        )


@dataclass
class MusicResponse:
    """Response from a music generation request."""
    url: str
    usage: Usage = field(default_factory=Usage)
    raw: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> str:
        """Download and save the audio to a local file."""
        r = requests.get(self.url, timeout=120)
        r.raise_for_status()
        p = Path(path)
        p.write_bytes(r.content)
        return str(p)

    @classmethod
    def from_dict(cls, data: dict) -> "MusicResponse":
        return cls(
            url=data.get("audio_url", data.get("url", "")),
            usage=Usage.from_dict(data.get("free_ai_usage", {})),
            raw=data,
        )


@dataclass
class VideoResponse:
    """Response from a video generation request."""
    url: str
    usage: Usage = field(default_factory=Usage)
    raw: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> str:
        """Download and save the video to a local file."""
        r = requests.get(self.url, timeout=300)
        r.raise_for_status()
        p = Path(path)
        p.write_bytes(r.content)
        return str(p)

    @classmethod
    def from_dict(cls, data: dict) -> "VideoResponse":
        return cls(
            url=data.get("video_url", data.get("url", "")),
            usage=Usage.from_dict(data.get("free_ai_usage", {})),
            raw=data,
        )
