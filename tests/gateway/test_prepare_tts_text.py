"""Tests for BasePlatformAdapter.prepare_tts_text.

Regression coverage for the "empty"/garbled auto-TTS voice replies caused
by code blocks, tables, and regex/notation being read to the TTS engine
verbatim. See toryx-private CLAUDE.md (SGX/dnnalpha migration session,
2026-07-18) for the incident this was found from.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.platforms.base import BasePlatformAdapter


def prepare(text: str) -> str:
    # prepare_tts_text doesn't touch ``self``, so the unbound method can be
    # called directly without constructing a concrete adapter subclass.
    return BasePlatformAdapter.prepare_tts_text(None, text)


def test_plain_text_passes_through_unchanged():
    text = "This is a normal sentence with no markdown at all."
    assert prepare(text) == text


def test_strips_basic_markdown_emphasis():
    assert prepare("This is **bold** and _italic_ and `code`.") == "This is bold and italic and ."


def test_drops_fenced_code_block_entirely():
    text = "Here's the fix:\n```python\ndef f(x):\n    return x * 2\n```\nThat should work."
    result = prepare(text)
    assert "def f(x)" not in result
    assert "return x" not in result
    assert "Here's the fix:" in result
    assert "That should work." in result


def test_drops_inline_code_span_content_not_just_backticks():
    text = "Use `re.compile(r\"[.!?]+\\s*\")` to split sentences."
    result = prepare(text)
    assert "re.compile" not in result
    assert "[.!?]" not in result
    assert "Use" in result
    assert "to split sentences." in result


def test_drops_markdown_table():
    text = "Summary:\n| Col A | Col B |\n|---|---|\n| 1 | 2 |\nDone."
    result = prepare(text)
    assert "|" not in result
    assert "Col A" not in result
    assert "Summary:" in result
    assert "Done." in result


def test_drops_horizontal_rule():
    text = "Section one.\n\n---\n\nSection two."
    result = prepare(text)
    assert "---" not in result
    assert "Section one." in result
    assert "Section two." in result


def test_strips_leading_list_markers():
    text = "- first item\n- second item\n1. numbered item"
    result = prepare(text)
    assert result.splitlines()[0] == "first item"
    assert "second item" in result
    assert "numbered item" in result


def test_spells_out_math_notation():
    result = prepare("target_chars = target_tokens × 4")
    assert "times" in result
    assert "×" not in result


def test_truncates_to_4000_chars_after_cleanup():
    text = "a" * 5000
    assert len(prepare(text)) == 4000


def test_realistic_technical_reply_has_no_leftover_code_smell():
    text = (
        "Here is the brief.\n\n---\n\n"
        "**Algorithms**\n\n"
        "1. Token-to-character approximation\n"
        "`target_chars = target_tokens × 4`\n\n"
        "2. Sentence boundary detection\n"
        "`SENTENCE_END = re.compile(r\"[.!?]+\\s*\")`\n\n"
        "| Format | Chunk size |\n|---|---|\n| PDF | 1200 |\n"
    )
    result = prepare(text)
    for smell in ("re.compile", "SENTENCE_END", "target_chars", "```", "|", "`", "×"):
        assert smell not in result, f"{smell!r} leaked into TTS-bound text"
