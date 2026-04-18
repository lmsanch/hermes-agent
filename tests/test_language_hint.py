"""Tests for agent.language_hint — deterministic language detection + hint."""

import pytest

from agent.language_hint import (
    apply_hint_if_enabled,
    detect,
    format_respond_in,
    maybe_prefix,
)


class TestDetectLatin:
    def test_english_full_sentence(self):
        text = "What is the status of the deployment and why is it failing?"
        lang, conf = detect(text)
        assert lang == "en"
        assert conf >= 0.65

    def test_spanish_with_accents(self):
        text = "¿Cuál es el estado del despliegue y por qué está fallando?"
        lang, conf = detect(text)
        assert lang == "es"
        assert conf >= 0.65

    def test_spanish_without_accents(self):
        text = "Cual es el estado del despliegue y por que esta fallando"
        lang, conf = detect(text)
        assert lang == "es"
        assert conf >= 0.65

    def test_portuguese(self):
        text = "Qual é o status do deploy e por que está falhando?"
        lang, conf = detect(text)
        assert lang == "pt"
        assert conf >= 0.60

    def test_french(self):
        text = "Quel est le statut du déploiement et pourquoi ne fonctionne pas?"
        lang, conf = detect(text)
        assert lang == "fr"
        assert conf >= 0.65

    def test_german(self):
        text = "Wie ist der Status der Bereitstellung und warum ist sie fehlgeschlagen?"
        lang, conf = detect(text)
        assert lang == "de"
        assert conf >= 0.65

    def test_italian(self):
        text = "Qual è lo stato del deployment e perché non funziona?"
        lang, conf = detect(text)
        assert lang == "it"
        assert conf >= 0.60


class TestDetectNonLatin:
    def test_japanese_hiragana_katakana(self):
        text = "デプロイの状態はどうですか、なぜ失敗していますか？"
        lang, conf = detect(text)
        assert lang == "ja"
        assert conf >= 0.70

    def test_russian_cyrillic(self):
        text = "Каков статус развертывания и почему оно не работает?"
        lang, conf = detect(text)
        assert lang == "ru"
        assert conf >= 0.70

    def test_arabic(self):
        text = "ما هي حالة النشر ولماذا يفشل؟ هذا سؤال مهم جدا."
        lang, conf = detect(text)
        assert lang == "ar"
        assert conf >= 0.70

    def test_korean_hangul(self):
        text = "배포 상태는 어떻고 왜 실패하고 있습니까? 이것은 중요합니다."
        lang, conf = detect(text)
        assert lang == "ko"
        assert conf >= 0.70

    def test_chinese(self):
        text = "部署的状态如何，为什么失败了？这个问题很重要。"
        lang, conf = detect(text)
        assert lang == "zh"
        assert conf >= 0.70


class TestNoDetection:
    def test_empty_string(self):
        lang, conf = detect("")
        assert lang is None
        assert conf == 0.0

    def test_whitespace_only(self):
        lang, conf = detect("   \n\t  ")
        assert lang is None
        assert conf == 0.0

    def test_too_short(self):
        lang, conf = detect("ok")
        assert lang is None

    def test_pure_numbers(self):
        lang, conf = detect("12345 67890 99.5%")
        assert lang is None

    def test_pure_punctuation(self):
        lang, conf = detect("!!! ??? ... ---")
        assert lang is None

    def test_code_snippet_no_false_positive(self):
        text = "def foo(x): return x * 2"
        lang, conf = detect(text)
        assert (lang, conf) == (None, 0.0) or conf < 0.65


class TestFormatRespondIn:
    def test_basic_format(self):
        out = format_respond_in("en")
        assert out.startswith("[RESPOND IN: en]")
        assert out.endswith("\n\n")

    def test_spanish_format(self):
        out = format_respond_in("es")
        assert "[RESPOND IN: es]" in out


class TestMaybePrefix:
    def test_prefixes_confident_english(self):
        text = "What is the status of the deployment and the current errors?"
        prefixed, lang = maybe_prefix(text)
        assert lang == "en"
        assert prefixed.startswith("[RESPOND IN: en]")
        assert text in prefixed

    def test_prefixes_confident_spanish(self):
        text = "¿Cuál es el estado del despliegue y cuáles son los errores actuales?"
        prefixed, lang = maybe_prefix(text)
        assert lang == "es"
        assert prefixed.startswith("[RESPOND IN: es]")

    def test_skips_low_confidence(self):
        text = "ok"
        prefixed, lang = maybe_prefix(text, min_confidence=0.65)
        assert lang is None
        assert prefixed == text

    def test_skips_empty(self):
        prefixed, lang = maybe_prefix("", min_confidence=0.65)
        assert lang is None
        assert prefixed == ""

    def test_skips_on_high_threshold_when_confidence_low(self):
        text = "Quick question"
        prefixed, lang = maybe_prefix(text, min_confidence=0.95)
        assert lang is None
        assert prefixed == text

    def test_preserves_original_text_after_prefix(self):
        text = "Please summarize the recent deploy logs and errors from today."
        prefixed, lang = maybe_prefix(text)
        assert lang == "en"
        assert prefixed.endswith(text)


class TestApplyHintIfEnabled:
    def test_default_off_passes_through(self, monkeypatch):
        monkeypatch.delenv("HERMES_LANG_HINT_ENABLED", raising=False)
        text = "¿Cuál es el estado del despliegue y cuáles son los errores?"
        out = apply_hint_if_enabled(text)
        assert out == text

    def test_enabled_applies_spanish(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANG_HINT_ENABLED", "true")
        text = "¿Cuál es el estado del despliegue y cuáles son los errores?"
        out = apply_hint_if_enabled(text)
        assert out.startswith("[RESPOND IN: es]")
        assert text in out

    def test_enabled_applies_english(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANG_HINT_ENABLED", "1")
        text = "What is the status of the deployment and why is it failing?"
        out = apply_hint_if_enabled(text)
        assert out.startswith("[RESPOND IN: en]")

    def test_enabled_but_low_confidence_passes_through(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANG_HINT_ENABLED", "yes")
        text = "ok"
        out = apply_hint_if_enabled(text)
        assert out == text

    def test_already_prefixed_is_not_double_prefixed(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANG_HINT_ENABLED", "true")
        text = "[RESPOND IN: en]\n\nWhat is the deploy status please?"
        out = apply_hint_if_enabled(text)
        assert out == text
        assert out.count("[RESPOND IN:") == 1

    def test_empty_string_passes_through_when_enabled(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANG_HINT_ENABLED", "true")
        assert apply_hint_if_enabled("") == ""

    def test_whitespace_only_passes_through_when_enabled(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANG_HINT_ENABLED", "true")
        assert apply_hint_if_enabled("   \n\t  ") == "   \n\t  "

    def test_truthy_variants_all_enable(self, monkeypatch):
        text = "What is the status of the deployment and errors today?"
        for val in ("1", "true", "TRUE", "yes", "YES", "on", " on "):
            monkeypatch.setenv("HERMES_LANG_HINT_ENABLED", val)
            out = apply_hint_if_enabled(text)
            assert out.startswith("[RESPOND IN: en]"), f"val={val!r} did not enable"

    def test_falsy_variants_stay_off(self, monkeypatch):
        text = "What is the status of the deployment and errors today?"
        for val in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("HERMES_LANG_HINT_ENABLED", val)
            out = apply_hint_if_enabled(text)
            assert out == text, f"val={val!r} unexpectedly enabled"

    def test_high_threshold_skips_marginal_detection(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANG_HINT_ENABLED", "true")
        monkeypatch.setenv("HERMES_LANG_HINT_MIN_CONFIDENCE", "0.95")
        # Spanish sample that scores in the ~0.82-0.85 range — above
        # the default 0.65 threshold but below this test's 0.95 floor.
        text = "el coche es grande y la casa roja"
        out = apply_hint_if_enabled(text)
        assert out == text, (
            "a high HERMES_LANG_HINT_MIN_CONFIDENCE should gate moderate-"
            "confidence detections out"
        )

    def test_bad_threshold_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("HERMES_LANG_HINT_ENABLED", "true")
        monkeypatch.setenv("HERMES_LANG_HINT_MIN_CONFIDENCE", "not-a-number")
        text = "What is the status of the deployment and errors today?"
        out = apply_hint_if_enabled(text)
        assert out.startswith("[RESPOND IN: en]")
