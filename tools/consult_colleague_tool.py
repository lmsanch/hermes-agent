"""consult_colleague tool — thin wrapper that shells to another MD via hermes CLI.

This is the minimum-viable version. It matches the signature the LLMs already
naturally emit (agent + query) so the tool call resolves cleanly instead of
dying as a malformed JSON fragment.

When the full 2026-04-18 spec ships (#724 — routing matrix, depth guard,
audit DB, daily cap), this wrapper is replaced with no caller-side change.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Any, Dict

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_VALID_PROFILES = {"scarlett", "christopher", "hilary", "elon", "eva"}
_PROFILE_ALIASES = {"hillary": "hilary"}  # one-L vs two-L typo accommodation


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
        "chain multiple consults — call this once per question."
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


def consult_colleague(agent: str, query: str, **_: Any) -> Dict[str, Any]:
    agent = (agent or "").strip().lower()
    agent = _PROFILE_ALIASES.get(agent, agent)
    if agent not in _VALID_PROFILES:
        return tool_error(
            f"Unknown colleague '{agent}'. Must be one of: "
            + ", ".join(sorted(_VALID_PROFILES))
        )
    if not query or not query.strip():
        return tool_error("query must be a non-empty string")

    caller = os.getenv("HERMES_PROFILE", "unknown")
    if agent == caller:
        return tool_error(
            f"Cannot consult self ('{agent}'). Answer from your own skills instead."
        )

    # Depth guard — one level only. Prevents A→B→A loops and fan-out.
    depth = int(os.getenv("HERMES_CONSULT_DEPTH", "0") or "0")
    if depth >= 1:
        return tool_error(
            "consult chain already at max depth (1). Answer from your own "
            "skills or refuse."
        )

    env = os.environ.copy()
    env["HERMES_CONSULT_DEPTH"] = str(depth + 1)
    env["HERMES_CONSULT_CALLER"] = caller

    hermes_bin = os.getenv(
        "HERMES_BIN",
        "/home/luis/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main",
    ).split()

    cmd = hermes_bin + [
        "-p", agent, "chat",
        "-q", f"[Consult from {caller}] {query}",
        "-Q",
        "--max-turns", "30",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return tool_error(
            f"consult to '{agent}' timed out after 180s. Retry once with a "
            "tighter question, otherwise refuse and notify the user."
        )
    latency_ms = int((time.time() - t0) * 1000)

    answer = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return tool_error(
            f"consult to '{agent}' failed with exit {proc.returncode}; "
            f"stderr tail: {(proc.stderr or '')[-400:]}"
        )
    if not answer:
        return tool_error(
            f"'{agent}' returned empty stdout (ran for {latency_ms} ms). "
            "Retry once; if still empty, refuse."
        )

    return {
        "colleague": agent,
        "answer": answer,
        "latency_ms": latency_ms,
        "terminal_status": "ok",
    }


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
