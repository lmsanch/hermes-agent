import pytest


def _run_normalizer(messages):
    from run_agent import _normalize_messages_for_groq
    return _normalize_messages_for_groq(messages)


def test_tool_content_list_becomes_string():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is AAPL at?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "get_stock_quotes", "arguments": '{"symbol":"AAPL"}'}}]},
        {"role": "tool", "content": [{"type": "text", "text": '{"result": "AAPL $211.38"}'}], "tool_call_id": "tc1"},
    ]
    result = _run_normalizer(messages)
    assert isinstance(result[3]["content"], str)
    assert result[3]["content"] == '{"result": "AAPL $211.38"}'


def test_tool_content_string_stays_string():
    messages = [
        {"role": "tool", "content": '{"result": "AAPL $211.38"}', "tool_call_id": "tc1"},
    ]
    result = _run_normalizer(messages)
    assert isinstance(result[0]["content"], str)
    assert result[0]["content"] == '{"result": "AAPL $211.38"}'


def test_user_content_list_becomes_string():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    ]
    result = _run_normalizer(messages)
    assert isinstance(result[0]["content"], str)
    assert result[0]["content"] == "Hello"


def test_system_content_string_untouched():
    messages = [
        {"role": "system", "content": "You are helpful."},
    ]
    result = _run_normalizer(messages)
    assert result[0]["content"] == "You are helpful."


def test_multiple_content_blocks_joined():
    messages = [
        {"role": "tool", "content": [{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}], "tool_call_id": "tc1"},
    ]
    result = _run_normalizer(messages)
    assert isinstance(result[0]["content"], str)
    assert result[0]["content"] == "Part 1\nPart 2"


def test_empty_content_list_becomes_empty_string():
    messages = [
        {"role": "tool", "content": [], "tool_call_id": "tc1"},
    ]
    result = _run_normalizer(messages)
    assert result[0]["content"] == ""


def test_none_content_untouched():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": []},
    ]
    result = _run_normalizer(messages)
    assert result[0]["content"] is None


def test_non_dict_content_block():
    messages = [
        {"role": "tool", "content": ["raw string", {"type": "text", "text": "dict block"}], "tool_call_id": "tc1"},
    ]
    result = _run_normalizer(messages)
    assert isinstance(result[0]["content"], str)
