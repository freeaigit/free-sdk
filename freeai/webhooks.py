"""Webhook signature verification for Free.ai webhooks."""

import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

from .exceptions import FreeAIError


class WebhookVerificationError(FreeAIError):
    """Raised when a webhook signature is invalid."""
    pass


def verify_webhook(
    payload: bytes,
    signature: str,
    secret: str,
    tolerance: int = 300,
) -> Dict[str, Any]:
    """Verify a Free.ai webhook signature and return the parsed payload.

    Free.ai webhooks use HMAC-SHA256 signatures in the format:
        ``t=<timestamp>,v1=<hex_signature>``

    Args:
        payload: The raw request body bytes.
        signature: The ``X-FreeAI-Signature`` header value.
        secret: Your webhook secret (from the developer dashboard).
        tolerance: Max age in seconds before rejecting (default 300 = 5 min).

    Returns:
        Parsed webhook payload as a dict.

    Raises:
        WebhookVerificationError: If the signature is invalid or the
            timestamp is outside the tolerance window.

    Example::

        from freeai.webhooks import verify_webhook

        # In your Flask/Django/FastAPI handler:
        payload = verify_webhook(
            payload=request.body,
            signature=request.headers["X-FreeAI-Signature"],
            secret="whsec_...",
        )
        print(payload["event"])  # e.g. "generation.complete"
    """
    if not payload:
        raise WebhookVerificationError("Empty payload")
    if not signature:
        raise WebhookVerificationError("Missing signature header")
    if not secret:
        raise WebhookVerificationError("Missing webhook secret")

    # Parse the signature header: t=<timestamp>,v1=<sig>
    parts: Dict[str, str] = {}
    for item in signature.split(","):
        if "=" in item:
            key, _, value = item.partition("=")
            parts[key.strip()] = value.strip()

    timestamp_str = parts.get("t")
    received_sig = parts.get("v1")

    if not timestamp_str or not received_sig:
        raise WebhookVerificationError(
            "Invalid signature format — expected 't=<timestamp>,v1=<signature>'"
        )

    # Check timestamp tolerance
    try:
        timestamp = int(timestamp_str)
    except ValueError:
        raise WebhookVerificationError("Invalid timestamp in signature")

    now = int(time.time())
    if abs(now - timestamp) > tolerance:
        raise WebhookVerificationError(
            f"Webhook timestamp too old ({abs(now - timestamp)}s > {tolerance}s tolerance)"
        )

    # Compute expected signature: HMAC-SHA256(secret, "timestamp.payload")
    signed_content = f"{timestamp_str}.".encode() + payload
    expected_sig = hmac.new(
        secret.encode(),
        signed_content,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected_sig, received_sig):
        raise WebhookVerificationError("Signature mismatch — payload may have been tampered with")

    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise WebhookVerificationError(f"Invalid JSON payload: {e}")
