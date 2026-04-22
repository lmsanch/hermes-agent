"""Secret redaction preprocessor for all user-facing Hermes output.

Applied at the outbound message chokepoints (Telegram send, email send,
TTS text synthesis) to strip common secret patterns before they hit any
user-visible surface.

Motivating incident: toryx-private#807 Issue 5 — Christopher leaked raw
Alpaca API keys into a Telegram approval prompt. This module closes that
class by running outbound text through a last-step regex scrub.

See toryx-private#818 for the tracking issue.

Design notes
------------
- Regex-based only. We do NOT attempt to deduce secrets from entropy or
  context; false positives would be worse than missed secrets at this
  layer (other layers catch things we miss).
- Replacement is structured: ``[REDACTED:<kind>]`` so the reader understands
  what was stripped and can escalate if it was supposed to be visible.
- Idempotent: running redact() twice is a no-op on the second pass.
- Fast: all patterns are precompiled module-level.
"""

from __future__ import annotations

import re
from typing import Final

# Pattern → human-readable kind. Ordering matters only for overlapping matches;
# longer / more-specific patterns should come first.
_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # PEM private keys — multi-line match
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ED25519 )?PRIVATE KEY-----.*?-----END (?:RSA |EC |DSA |OPENSSH |ED25519 )?PRIVATE KEY-----", re.DOTALL), "pem-private-key"),
    # Provider API keys — prefix-anchored
    (re.compile(r"\bsk-ant-api\d{2}-[A-Za-z0-9_\-]{40,}"), "anthropic-key"),
    (re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{40,}"), "openai-project-key"),
    (re.compile(r"\bsk-[A-Za-z0-9]{40,}"), "openai-key"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{30,}"), "groq-key"),
    (re.compile(r"\bfw_[A-Za-z0-9]{20,}"), "fireworks-key"),
    (re.compile(r"\btxk_[A-Za-z0-9]{20,}"), "toryx-key"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}"), "github-fine-grained-pat"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"), "github-pat"),
    (re.compile(r"\bghs_[A-Za-z0-9]{30,}"), "github-server-token"),
    (re.compile(r"\bxox[abpr]-[A-Za-z0-9\-]{20,}"), "slack-token"),
    # Alpaca — labeled-header style (from the #807 incident specifically)
    (re.compile(r"APCA-API-KEY-ID:\s*[A-Z0-9]{16,}", re.IGNORECASE), "alpaca-key-id"),
    (re.compile(r"APCA-API-SECRET-KEY:\s*[A-Za-z0-9]{30,}", re.IGNORECASE), "alpaca-secret-key"),
    # Bearer tokens — catch-all, generic
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-]{25,}", re.IGNORECASE), "bearer-token"),
    # AWS
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws-access-key-id"),
    (re.compile(r"\baws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}", re.IGNORECASE), "aws-secret-access-key"),
)


def redact(text: str) -> str:
    """Replace known secret patterns with ``[REDACTED:<kind>]`` markers.

    Safe to call on None / empty input (returns input unchanged).
    Idempotent — applying to already-redacted text is a no-op.
    """
    if not text:
        return text
    out = text
    for pattern, kind in _PATTERNS:
        out = pattern.sub(f"[REDACTED:{kind}]", out)
    return out


def contains_secret(text: str) -> bool:
    """Return True if any known secret pattern matches.

    Useful for pre-send guard rails that want to log or alert rather than
    silently redact.
    """
    if not text:
        return False
    return any(pattern.search(text) for pattern, _ in _PATTERNS)
