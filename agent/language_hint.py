"""Deterministic language detection + response-language hint injector.

Purpose
-------
Belt-and-suspenders companion to the SOUL-level ``## Language Mirroring``
directive. The prompt-side directive works for stronger models but under
context pressure (long tool outputs, KMS harvest hits, prior non-user
turns) weaker or default-quantization models drift. A deterministic,
code-level detector that prepends ``[RESPOND IN: <lang>]`` to the user
turn closes that gap without requiring model strength.

Design
------
Zero dependencies (stdlib only). Two-stage detection:

1. **Unicode script fast-path.** If the message is dominated by one of
   a small set of distinctive scripts (CJK, Hangul, Cyrillic, Arabic,
   Devanagari, Hebrew, Thai), return the corresponding ISO code
   immediately with high confidence.
2. **Function-word scoring** for Latin-script text. Short, frequent
   words are language-defining. We score the six most common Hermes
   languages (en, es, pt, fr, de, it) by counting hits from a curated
   function-word set. The winner needs a minimum absolute hit count
   and a minimum margin over the runner-up.

Shared concerns:
- Minimum text length (too-short inputs are not detectable).
- Confidence is a number in [0.0, 1.0]; callers pass ``min_confidence``.

This module is intentionally small and auditable. It is not a
replacement for a proper language model for edge cases (mixed-language
turns, rare languages, very short messages) — for those it returns
``(None, 0.0)`` and the caller should skip the hint.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_MIN_ALPHA_CHARS = 12

_SCRIPT_RANGES = {
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF)],
    "ja": [(0x3040, 0x309F), (0x30A0, 0x30FF)],
    "ko": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
    "ru": [(0x0400, 0x04FF)],
    "ar": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "hi": [(0x0900, 0x097F)],
    "he": [(0x0590, 0x05FF)],
    "th": [(0x0E00, 0x0E7F)],
}

_SCRIPT_DOMINANCE_RATIO = 0.30

_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset({
        "the", "and", "of", "to", "in", "is", "it", "that", "for", "on",
        "with", "as", "are", "was", "be", "this", "have", "has", "from",
        "but", "not", "or", "an", "at", "by", "we", "you", "they", "he",
        "she", "what", "when", "where", "which", "how", "why", "if",
        "about", "into", "than", "then", "there", "their", "will",
        "would", "could", "should", "can", "do", "does", "did", "been",
    }),
    "es": frozenset({
        "el", "la", "los", "las", "de", "del", "y", "que", "en", "un",
        "una", "es", "por", "con", "para", "no", "se", "su", "al", "lo",
        "mi", "tu", "pero", "más", "como", "este", "esta", "estos",
        "estas", "eso", "son", "fue", "ser", "está", "están", "qué",
        "cuál", "dónde", "cómo", "porqué", "cuando", "también",
        "sobre", "porque", "hasta", "desde",
    }),
    "pt": frozenset({
        "o", "a", "os", "as", "de", "do", "da", "dos", "das", "e", "que",
        "em", "um", "uma", "é", "por", "com", "para", "não", "se", "mas",
        "mais", "como", "este", "esta", "estes", "estas", "isso", "são",
        "foi", "ser", "está", "estão", "qual", "onde", "quando", "porque",
        "também", "sobre", "até", "pelo", "pela", "você", "eu",
    }),
    "fr": frozenset({
        "le", "la", "les", "de", "des", "du", "et", "que", "en", "un",
        "une", "est", "par", "avec", "pour", "ne", "pas", "se", "son",
        "sa", "ses", "au", "aux", "mais", "plus", "comme", "ce", "cette",
        "ces", "sont", "a", "à", "être", "où", "quand", "comment",
        "pourquoi", "aussi", "sur", "parce", "jusqu", "depuis", "vous",
    }),
    "de": frozenset({
        "der", "die", "das", "den", "dem", "des", "und", "ist", "in", "zu",
        "ein", "eine", "einer", "einen", "von", "mit", "für", "auf", "nicht",
        "sich", "sein", "aber", "mehr", "wie", "dieser", "diese", "dieses",
        "sind", "war", "waren", "wo", "wann", "warum", "weil", "über",
        "auch", "bis", "seit", "du", "sie", "er", "wir", "ich",
    }),
    "it": frozenset({
        "il", "la", "lo", "i", "gli", "le", "di", "del", "della", "dei",
        "e", "che", "in", "un", "una", "è", "per", "con", "non", "si",
        "suo", "sua", "ma", "più", "come", "questo", "questa", "questi",
        "queste", "sono", "era", "essere", "dove", "quando", "come",
        "perché", "anche", "fino", "da", "voi", "tu", "noi",
    }),
}

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+", re.UNICODE)


def _script_vote(text: str) -> Optional[Tuple[str, float]]:
    """Count characters by script and return dominant non-Latin script.

    Returns ``(lang, confidence)`` when a single non-Latin script accounts
    for at least ``_SCRIPT_DOMINANCE_RATIO`` of alphabetic characters,
    else ``None``. Confidence scales from the ratio threshold up to 1.0.
    """
    counts: dict[str, int] = {}
    total_alpha = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total_alpha += 1
        cp = ord(ch)
        for lang, ranges in _SCRIPT_RANGES.items():
            for lo, hi in ranges:
                if lo <= cp <= hi:
                    counts[lang] = counts.get(lang, 0) + 1
                    break

    if total_alpha < _MIN_ALPHA_CHARS or not counts:
        return None

    lang, hits = max(counts.items(), key=lambda kv: kv[1])
    ratio = hits / total_alpha
    if ratio < _SCRIPT_DOMINANCE_RATIO:
        return None
    confidence = min(1.0, 0.5 + ratio * 0.5)
    return (lang, confidence)


def _strip_accents(word: str) -> str:
    """NFKD-normalize then drop combining marks so 'está' matches 'esta'."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", word)
        if not unicodedata.combining(c)
    )


def _function_word_vote(text: str) -> Optional[Tuple[str, float]]:
    """Score Latin-script text against per-language function-word sets.

    Returns ``(lang, confidence)`` when one language wins with enough
    absolute hits AND a margin over the runner-up, else ``None``.
    """
    tokens = [w.lower() for w in _WORD_RE.findall(text)]
    if len(tokens) < 3:
        return None

    scores: dict[str, int] = {lang: 0 for lang in _FUNCTION_WORDS}
    for tok in tokens:
        for lang, fw in _FUNCTION_WORDS.items():
            if tok in fw:
                scores[lang] += 1
        stripped = _strip_accents(tok)
        if stripped != tok:
            for lang, fw in _FUNCTION_WORDS.items():
                if stripped in fw:
                    scores[lang] += 1

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_lang, top = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0

    if top < 2:
        return None
    margin = top - runner
    if margin < 1 and top < 4:
        return None

    denom = top + runner if (top + runner) > 0 else 1
    confidence = min(1.0, 0.55 + (margin / denom) * 0.45)
    return (top_lang, confidence)


def detect(text: str) -> Tuple[Optional[str], float]:
    """Detect the language of ``text``.

    Returns ``(iso_code, confidence)`` when a language is identifiable
    with non-trivial confidence, else ``(None, 0.0)``. Confidence is in
    ``[0.0, 1.0]``; typical thresholds are 0.6–0.7.
    """
    if not text or not text.strip():
        return (None, 0.0)

    sv = _script_vote(text)
    if sv is not None:
        return sv

    fv = _function_word_vote(text)
    if fv is not None:
        return fv

    return (None, 0.0)


def format_respond_in(lang: str) -> str:
    """Return the canonical ``[RESPOND IN: <lang>]`` prefix line.

    One-line prefix followed by a blank line, intended to be prepended
    to the user's message before it reaches the agent. The trailing
    newlines keep the hint visually distinct from the user text.
    """
    return f"[RESPOND IN: {lang}]\n\n"


def maybe_prefix(
    text: str, min_confidence: float = 0.65
) -> Tuple[str, Optional[str]]:
    """Prepend ``[RESPOND IN: <lang>]`` to ``text`` when detection is confident.

    Returns ``(possibly_prefixed_text, detected_lang)``. When confidence
    is below ``min_confidence`` or no language was detected, the text is
    returned unchanged and ``detected_lang`` is ``None``.
    """
    lang, conf = detect(text)
    if lang is None or conf < min_confidence:
        return (text, None)
    return (format_respond_in(lang) + text, lang)


_TRUTHY = frozenset({"1", "true", "yes", "on"})


def apply_hint_if_enabled(user_message: str) -> str:
    """Env-gated pre-processor: prepend ``[RESPOND IN: <lang>]`` in place.

    Default behavior is unchanged (returns ``user_message`` as-is).
    Set ``HERMES_LANG_HINT_ENABLED=true`` (typically in a per-profile
    ``.env``) to enable deterministic language-mirror reinforcement.
    Minimum detection confidence is tunable via
    ``HERMES_LANG_HINT_MIN_CONFIDENCE`` (default 0.65).

    Skips cleanly in safe cases:
    - Env flag off (default).
    - Message is empty / whitespace.
    - Message already carries a ``[RESPOND IN:`` prefix.
    - Detection below confidence threshold.
    - Any exception during detection (the hint is belt-and-suspenders;
      it must never fail the turn).
    """
    if os.getenv("HERMES_LANG_HINT_ENABLED", "").strip().lower() not in _TRUTHY:
        return user_message
    if not user_message or not user_message.strip():
        return user_message
    if user_message.lstrip().startswith("[RESPOND IN:"):
        return user_message
    try:
        raw = os.getenv("HERMES_LANG_HINT_MIN_CONFIDENCE", "0.65")
        try:
            min_conf = float(raw)
        except ValueError:
            min_conf = 0.65
        prefixed, lang = maybe_prefix(user_message, min_confidence=min_conf)
        if lang is not None:
            logger.debug("language hint applied: lang=%s", lang)
        return prefixed
    except Exception as exc:
        logger.warning(
            "language hint failed, passing message through: %s", exc
        )
        return user_message
