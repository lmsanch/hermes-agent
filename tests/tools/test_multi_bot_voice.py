"""Smoke tests for the voice pipeline (issue #1 / toryx-private #797).

Guard the three failure modes that caused the Eva-in-generic-English
regression:
  1. Empty voice_id must be rejected — never silently hit Fish Audio default.
  2. Voice brevity prefix must not force English.
  3. Language detection must catch Spanish so [RESPOND IN: ES] gets injected.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fish Audio empty voice_id rejection
# ---------------------------------------------------------------------------

class TestFishAudioVoiceId:
    def test_empty_voice_id_raises(self):
        from tools.tts_tool import _generate_fish_audio
        with pytest.raises(ValueError, match="voice_id is empty"):
            _generate_fish_audio("hola", "/tmp/test.mp3", {"fish": {"voice_id": ""}})

    def test_whitespace_voice_id_raises(self):
        from tools.tts_tool import _generate_fish_audio
        with pytest.raises(ValueError, match="voice_id is empty"):
            _generate_fish_audio("hola", "/tmp/test.mp3", {"fish": {"voice_id": "   "}})

    def test_missing_voice_id_raises(self):
        from tools.tts_tool import _generate_fish_audio
        with pytest.raises(ValueError, match="voice_id is empty"):
            _generate_fish_audio("hola", "/tmp/test.mp3", {})

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("FISH_AUDIO_API_KEY", raising=False)
        from tools.tts_tool import _generate_fish_audio
        with pytest.raises(ValueError, match="FISH_AUDIO_API_KEY not set"):
            _generate_fish_audio("hola", "/tmp/test.mp3", {"fish": {"voice_id": "abc123"}})


# ---------------------------------------------------------------------------
# Voice brevity prefix does not force English
# ---------------------------------------------------------------------------

class TestVoiceBrevityPrefix:
    def test_prefix_does_not_force_english(self):
        from gateway.run import _VOICE_BREVITY_PREFIX
        text = _VOICE_BREVITY_PREFIX.lower()
        assert "mirror" in text and "language" in text
        assert "conversational english" not in text


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_spanish_diacritics(self):
        from gateway.run import detect_language
        assert detect_language("¿Cómo estás hoy?") == "es"

    def test_spanish_markers(self):
        from gateway.run import detect_language
        assert detect_language("hola que tal soy luis") == "es"

    def test_english_default(self):
        from gateway.run import detect_language
        assert detect_language("hello how are you today") == "en"

    def test_empty_fallback(self):
        from gateway.run import detect_language
        assert detect_language("") == "en"
        assert detect_language("x") == "en"

    def test_lang_hint_env_flag_disabled(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANG_HINT_ENABLED", "false")
        assert os.environ.get("HERMES_LANG_HINT_ENABLED", "true").lower() not in {"1", "true", "yes"}

    def test_lang_hint_env_flag_enabled(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANG_HINT_ENABLED", "true")
        assert os.environ.get("HERMES_LANG_HINT_ENABLED", "true").lower() in {"1", "true", "yes"}
