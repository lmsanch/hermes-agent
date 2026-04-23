#!/usr/bin/env python3
"""Retrieval deduplication and adaptive budget for the Hermes agent loop.

Two functions:
  1. dedup_retrieval_calls() — collapse near-synonym retrieval queries
     within a single planning step using Jaccard token overlap (threshold 0.6).
     Replaced calls get a stub result so the model sees the merge happened.

  2. StallDetector — tracks "useful-result rate" over the last K iterations.
     When below threshold, injects a clarifying question instead of spinning.

See toryx-private#817.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

RETRIEVAL_TOOLS: frozenset = frozenset({
    "session_search", "search_files", "skill_view", "skills_list",
    "memory", "recall_search", "recall", "web_search", "web_extract",
})

JACCARD_THRESHOLD: float = 0.6


def _tokenize(text: str) -> Set[str]:
    return set(text.lower().split())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _primary_query(tool_name: str, args: dict) -> Optional[str]:
    mapping = {
        "session_search": "query",
        "search_files": "pattern",
        "skill_view": "name",
        "skills_list": "category",
        "memory": "content",
        "recall_search": "query",
        "recall": "query",
        "web_search": "query",
        "web_extract": "urls",
    }
    key = mapping.get(tool_name)
    if not key:
        return None
    val = args.get(key)
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return None


def dedup_retrieval_calls(
    tool_calls: list,
) -> Tuple[list, int]:
    """Deduplicate near-synonym retrieval tool calls within a single turn.

    For each group of overlapping queries, keep the broadest (most tokens)
    and replace the rest with a stub that tells the model the query was merged.

    Returns (filtered_tool_calls, removed_count).
    """
    if not tool_calls:
        return tool_calls, 0

    retrieval_indices: List[int] = []
    queries: List[Optional[str]] = []
    tokens: List[Optional[Set[str]]] = []

    for i, tc in enumerate(tool_calls):
        name = getattr(tc.function, "name", "")
        if name not in RETRIEVAL_TOOLS:
            retrieval_indices.append(-1)
            queries.append(None)
            tokens.append(None)
            continue
        try:
            args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {})
        except (json.JSONDecodeError, TypeError):
            args = {}
        q = _primary_query(name, args)
        retrieval_indices.append(i)
        queries.append(q)
        tokens.append(_tokenize(q) if q else None)

    # Cluster by Jaccard overlap
    n = len(tool_calls)
    merged_into: List[Optional[int]] = [None] * n
    removed = 0

    for i in range(n):
        if tokens[i] is None or retrieval_indices[i] == -1:
            continue
        if merged_into[i] is not None:
            continue
        for j in range(i + 1, n):
            if tokens[j] is None or retrieval_indices[j] == -1:
                continue
            if merged_into[j] is not None:
                continue
            sim = _jaccard(tokens[i], tokens[j])
            if sim >= JACCARD_THRESHOLD:
                if len(tokens[i]) >= len(tokens[j]):
                    merged_into[j] = i
                else:
                    merged_into[i] = j
                removed += 1

    if removed == 0:
        return tool_calls, 0

    filtered = []
    for i in range(n):
        if merged_into[i] is not None:
            target = merged_into[i]
            target_q = queries[target] or "(query)"
            name = getattr(tool_calls[i].function, "name", "unknown")
            stub_content = json.dumps({
                "success": True,
                "deduped": True,
                "message": f"Query merged with similar call ({target_q[:60]}). "
                           f"Jaccard similarity above {JACCARD_THRESHOLD}.",
            })
            tc = tool_calls[i]
            filtered.append(_make_stub_tool_call(tc, stub_content))
        else:
            filtered.append(tool_calls[i])

    logger.info("Retrieval dedup: removed %d near-synonym queries from %d calls", removed, n)
    return filtered, removed


def _make_stub_tool_call(original_tc, stub_content: str):
    """Create a lightweight wrapper that returns a dedup stub when executed.

    We cannot modify the tool_call object directly (it may be frozen),
    so we return the original and let the dedup handler intercept it.
    Instead, we tag the arguments so the executor knows to skip it.
    """
    try:
        args = json.loads(original_tc.function.arguments) if isinstance(original_tc.function.arguments, str) else (original_tc.function.arguments or {})
    except (json.JSONDecodeError, TypeError):
        args = {}
    args["_hermes_dedup_stub"] = True
    args["_hermes_dedup_reason"] = stub_content

    class _DedupedFunction:
        name = original_tc.function.name
        arguments = json.dumps(args, ensure_ascii=False)

    class _DedupedTC:
        id = original_tc.id
        function = _DedupedFunction()
        type = getattr(original_tc, "type", "function")

    return _DedupedTC()


class StallDetector:
    """Detect when the agent loop is making no progress.

    Tracks "useful result rate" over a sliding window of K iterations.
    A tool call is "useful" if it produced a non-empty, non-error result
    that was not a dedup stub. When the useful rate drops below threshold
    for N consecutive checks, emit a clarifying question.

    Usage in the agent loop:
        detector = StallDetector(window=5, threshold=0.3, stall_limit=3)
        # After each iteration:
        detector.record(tool_name, result_content)
        if detector.should_intervene():
            # inject clarifying question into messages
    """

    def __init__(
        self,
        window: int = 5,
        threshold: float = 0.3,
        stall_limit: int = 3,
    ):
        self.window = window
        self.threshold = threshold
        self.stall_limit = stall_limit
        self._history: List[bool] = []
        self._stall_streak: int = 0
        self._intervened: bool = False
        self._retrieval_count: int = 0
        self._useful_count: int = 0
        self._total_count: int = 0
        self._dedup_count: int = 0

    def record(self, tool_name: str, result_content: str) -> None:
        self._total_count += 1
        if tool_name in RETRIEVAL_TOOLS:
            self._retrieval_count += 1

        useful = True
        if not result_content:
            useful = False
        elif isinstance(result_content, str):
            try:
                data = json.loads(result_content)
                if data.get("deduped"):
                    self._dedup_count += 1
                    useful = False
                elif data.get("success") is False and data.get("error"):
                    useful = False
            except (json.JSONDecodeError, TypeError):
                pass

        if useful:
            self._useful_count += 1
        self._history.append(useful)
        if len(self._history) > self.window:
            self._history.pop(0)

        if not useful:
            self._stall_streak += 1
        else:
            self._stall_streak = 0

    def useful_rate(self) -> float:
        if not self._history:
            return 1.0
        return sum(self._history) / len(self._history)

    def should_intervene(self) -> bool:
        if self._intervened:
            return False
        if len(self._history) < self.window:
            return False
        if self.useful_rate() < self.threshold and self._stall_streak >= self.stall_limit:
            self._intervened = True
            return True
        return False

    def intervention_message(self) -> str:
        rate = self.useful_rate()
        return (
            f"[System] You have been searching and calling tools for {self._total_count} iterations "
            f"with only a {rate:.0%} useful-result rate. Please summarize what you know so far, "
            f"state what is still unclear, and ask the user a specific clarifying question "
            f"instead of continuing to search. Do NOT call any more tools in this response."
        )

    def report(self) -> dict:
        return {
            "total_tool_calls": self._total_count,
            "retrieval_calls": self._retrieval_count,
            "useful_calls": self._useful_count,
            "deduped_calls": self._dedup_count,
            "useful_rate": round(self.useful_rate(), 2),
            "intervened": self._intervened,
        }
