"""consult_colleague tool — thin adapter that delegates to the spec-complete
implementation at agent/consult_colleague.py (shipped via PR #898).

The OpenAI tool schema (agent, query) and JSON-stringified return shape
({"colleague", "answer", "latency_ms", "terminal_status"}) are unchanged
from the MVP shim so the LLM-side contract does NOT change.

When the full spec shipped (#724), this adapter replaced the subprocess-only
MVP shim with no caller-side change.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_PROFILE_ALIASES = {"hillary": "hilary"}


CONSULT_COLLEAGUE_SCHEMA = {
    "name": "consult_colleague",
    "description": (
        "Ask another Toryx Managing Director a single focused question and "
        "return their answer. Same-host subprocess (no email, no network). "
        "Use for cross-desk coordination when a question falls outside your "
        "domain and a peer has a relevant skill. Colleague must be one of: "
        "scarlett, christopher, hilary, elon, eva. The response is the "
        "colleague's final reply as a single string; integrate it into your "
        "own answer and credit them (e.g. 'Christopher says: …'). Do not "
        "chain multiple consults — call this ONCE per question. If the "
        "colleague fails (timeout, error, or empty response), do NOT retry "
        "the consult. Instead, use your own tools (web_search, mcp_kms_*, "
        "terminal) to answer the question yourself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": (
                    "Which colleague to consult. One of: "
                    "scarlett, christopher, hilary, elon, eva."
                ),
                "enum": ["scarlett", "christopher", "hilary", "elon", "eva", "hillary"],
            },
            "query": {
                "type": "string",
                "description": (
                    "The full, self-contained question. The colleague will "
                    "NOT see the surrounding conversation, so include any "
                    "needed context in the question itself."
                ),
            },
        },
        "required": ["agent", "query"],
    },
}


def consult_colleague(agent: str, query: str, **_: Any) -> str:
    agent = (agent or "").strip().lower()
    agent = _PROFILE_ALIASES.get(agent, agent)
    if not query or not query.strip():
        return tool_error("query must be a non-empty string")

    depth = int(os.getenv("HERMES_CONSULT_DEPTH", "0") or "0")
    if depth >= 1:
        return tool_error(
            "consult chain already at max depth (1). Answer from your own "
            "skills or refuse."
        )

    caller = os.getenv("HERMES_PROFILE", "unknown")
    if agent == caller:
        return tool_error(
            f"Cannot consult self ('{agent}'). Answer from your own skills instead."
        )

    from agent.consult_colleague import consult_colleague as impl
    result = impl(
        colleague=_PROFILE_ALIASES.get(agent, agent),
        question=query,
        context="",
        urgency="normal",
    )

    if result.terminal_status == "refused":
        if result.error and "Daily consult cap" in result.error:
            return tool_error("daily_cap_exceeded", detail=result.error)
        if result.error and "Unknown colleague" in result.error:
            return tool_error(result.error)
        if result.error:
            return tool_error(result.error)
        return tool_error("consult refused")

    if result.terminal_status == "timeout":
        return tool_error(
            f"consult to '{agent}' timed out after {result.latency_ms}ms. "
            "Do NOT retry this consult — use your own tools to answer instead."
        )

    if result.terminal_status == "error":
        return tool_error(result.error or f"consult to '{agent}' failed")

    if not result.answer:
        return tool_error(
            f"'{agent}' returned empty answer (ran for {result.latency_ms} ms). "
            "Do NOT retry this consult — use your own tools to answer instead."
        )

    return json.dumps({
        "colleague": result.colleague,
        "answer": result.answer,
        "latency_ms": result.latency_ms,
        "terminal_status": result.terminal_status,
    })


registry.register(
    name="consult_colleague",
    toolset="delegation",
    schema=CONSULT_COLLEAGUE_SCHEMA,
    handler=lambda args, **kw: consult_colleague(
        agent=args.get("agent"),
        query=args.get("query"),
    ),
    emoji="🤝",
)
