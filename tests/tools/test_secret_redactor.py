"""Unit tests for tools.secret_redactor — see toryx-private#818.

All test vectors use OBVIOUSLY SYNTHETIC values so they don't trip
secret-scanning on push: strings of AAAA..., 1234..., or explicitly
non-production patterns. Never use real keys in test code.
"""

from __future__ import annotations

import pytest

from tools.secret_redactor import contains_secret, redact


@pytest.mark.parametrize(
    "raw,kind",
    [
        # Synthetic Anthropic — prefix + 40+ chars of AAAA
        ("sk-ant-api03-" + "A" * 60, "anthropic-key"),
        # Synthetic Groq
        ("gsk_" + "A" * 40, "groq-key"),
        # Synthetic Fireworks
        ("fw_" + "A" * 25, "fireworks-key"),
        # Synthetic Toryx
        ("txk_" + "A" * 25, "toryx-key"),
        # Synthetic GitHub PAT
        ("ghp_" + "A" * 35, "github-pat"),
        # Synthetic Slack bot token
        ("xoxb-111111111111-222222222222-" + "A" * 20, "slack-token"),
        # Synthetic Alpaca labeled headers
        ("APCA-API-KEY-ID: " + "A" * 20, "alpaca-key-id"),
        ("APCA-API-SECRET-KEY: " + "A" * 40, "alpaca-secret-key"),
        # Synthetic Bearer
        ("Authorization: Bearer " + "A" * 30, "bearer-token"),
        # Synthetic AWS access key
        ("AKIA" + "A" * 16, "aws-access-key-id"),
    ],
)
def test_known_patterns_redacted(raw: str, kind: str) -> None:
    out = redact(raw)
    assert raw not in out, f"raw secret survived redaction: {raw!r}"
    assert f"[REDACTED:{kind}]" in out, f"expected [REDACTED:{kind}] in {out!r}"


def test_pem_private_key_redacted() -> None:
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "AAAA" * 32 + "\n"
        "AAAA" * 32 + "\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = redact(pem)
    assert "BEGIN RSA PRIVATE KEY" not in out
    assert "[REDACTED:pem-private-key]" in out


def test_safe_text_unchanged() -> None:
    safe = "Here is a normal message about macro debt cycles and Argentina."
    assert redact(safe) == safe
    assert not contains_secret(safe)


def test_empty_and_none_safe() -> None:
    assert redact("") == ""
    assert redact(None) is None  # type: ignore[arg-type]
    assert not contains_secret("")
    assert not contains_secret(None)  # type: ignore[arg-type]


def test_idempotent() -> None:
    raw = "token=fw_" + "A" * 25 + " more=txk_" + "A" * 25
    once = redact(raw)
    twice = redact(once)
    assert once == twice


def test_contains_secret_positive() -> None:
    assert contains_secret("here is fw_" + "A" * 25 + " inline")
    assert contains_secret("Bearer " + "A" * 30)


def test_contains_secret_negative() -> None:
    assert not contains_secret("no secrets here")
    # Short strings that match a prefix but not the length threshold
    assert not contains_secret("fw_short")
    assert not contains_secret("gsk_short")


def test_multiple_secrets_in_one_text() -> None:
    raw = "first: fw_" + "A" * 25 + " and then txk_" + "B" * 25
    out = redact(raw)
    assert "fw_" not in out
    assert "txk_" not in out
    assert out.count("[REDACTED:") == 2


def test_alpaca_incident_shape_from_807() -> None:
    """Regression coverage for the #807 Alpaca leak shape — SYNTHETIC values only."""
    incident_shape = (
        'curl -H "APCA-API-KEY-ID: ' + "A" * 20 + '" '
        '-H "APCA-API-SECRET-KEY: ' + "B" * 40 + '" ...'
    )
    out = redact(incident_shape)
    assert "A" * 20 not in out
    assert "B" * 40 not in out
    assert "[REDACTED:alpaca-key-id]" in out
    assert "[REDACTED:alpaca-secret-key]" in out
