"""Helpers for optional cheap-vs-strong model routing.

Supports two routing modes (selected via ``routing_config.mode``):

- ``cheap_model`` (or absent) — keyword-based routing for simple turns.
- ``thompson_sampling`` — Beta-Bernoulli bandit across a model pool.

Thompson Sampling mode samples one model per user turn from a configurable
pool of arms.  The sampled model is then ``turn-sticky``: the gateway caches
it in ``AIAgent.model`` for the remainder of the turn so every LLM call
within that turn uses the same model.  Outcomes (success/failure) are
recorded back to a persistent JSON state file so the Beta posteriors
converge toward the true best arm over time.

Kill-switches (env vars):

- ``HERMES_TS_DISABLED=true`` — completely disables TS; returns None.
- ``HERMES_TS_DRY_RUN=true`` — samples and logs but does not route or
  mutate state; returns None (falls through to primary).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from utils import is_truthy_value

logger = logging.getLogger(__name__)

_COMPLEX_KEYWORDS = {
    "debug",
    "debugging",
    "implement",
    "implementation",
    "refactor",
    "patch",
    "traceback",
    "stacktrace",
    "exception",
    "error",
    "analyze",
    "analysis",
    "investigate",
    "architecture",
    "design",
    "compare",
    "benchmark",
    "optimize",
    "optimise",
    "review",
    "terminal",
    "shell",
    "tool",
    "tools",
    "pytest",
    "test",
    "tests",
    "plan",
    "planning",
    "delegate",
    "subagent",
    "cron",
    "docker",
    "kubernetes",
}

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    return is_truthy_value(value, default=default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def choose_cheap_model_route(
    user_message: str, routing_config: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Return the configured cheap-model route when a message looks simple.

    Conservative by design: if the message has signs of code/tool/debugging/
    long-form work, keep the primary model.
    """
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    cheap_model = cfg.get("cheap_model") or {}
    if not isinstance(cheap_model, dict):
        return None
    provider = str(cheap_model.get("provider") or "").strip().lower()
    model = str(cheap_model.get("model") or "").strip()
    if not provider or not model:
        return None

    text = (user_message or "").strip()
    if not text:
        return None

    max_chars = _coerce_int(cfg.get("max_simple_chars"), 160)
    max_words = _coerce_int(cfg.get("max_simple_words"), 28)

    if len(text) > max_chars:
        return None
    if len(text.split()) > max_words:
        return None
    if text.count("\n") > 1:
        return None
    if "```" in text or "`" in text:
        return None
    if _URL_RE.search(text):
        return None

    lowered = text.lower()
    words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}
    if words & _COMPLEX_KEYWORDS:
        return None

    route = dict(cheap_model)
    route["provider"] = provider
    route["model"] = model
    route["routing_reason"] = "simple_turn"
    return route


def _primary_route(primary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": primary.get("model"),
        "runtime": {
            "api_key": primary.get("api_key"),
            "base_url": primary.get("base_url"),
            "provider": primary.get("provider"),
            "api_mode": primary.get("api_mode"),
            "command": primary.get("command"),
            "args": list(primary.get("args") or []),
            "credential_pool": primary.get("credential_pool"),
        },
        "label": None,
        "signature": (
            primary.get("model"),
            primary.get("provider"),
            primary.get("base_url"),
            primary.get("api_mode"),
            primary.get("command"),
            tuple(primary.get("args") or ()),
        ),
    }


def choose_thompson_sampling_route(
    user_message: str,
    routing_config: Optional[Dict[str, Any]],
    primary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return a TS-sampled route, or None if TS is disabled/misconfigured.

    Honors kill-switches:
      - env HERMES_TS_DISABLED=true → returns None (fall through to primary).
      - env HERMES_TS_DRY_RUN=true → samples + logs but returns None.

    Each arm is a dict with keys: name, provider, model, api_key (optional,
    resolves via env by default), base_url (optional), api_mode (optional).
    """
    if os.getenv("HERMES_TS_DISABLED", "").strip().lower() in ("true", "1", "yes"):
        return None

    cfg = routing_config or {}
    ts_cfg = cfg.get("thompson_sampling") or {}
    arms = ts_cfg.get("arms") or []
    if not isinstance(arms, list) or len(arms) < 2:
        return None

    arm_keys = [str(a.get("name") or a.get("model") or "") for a in arms]
    if not all(arm_keys):
        return None

    from hermes_constants import get_hermes_home

    state_path = Path(
        ts_cfg.get("state_path") or str(get_hermes_home() / "ts_hermes_mds.json")
    )

    from agent.ts_state import thompson_sample

    dry_run = os.getenv("HERMES_TS_DRY_RUN", "").strip().lower() in ("true", "1", "yes")

    chosen_key = thompson_sample(arm_keys, state_path)

    if dry_run:
        logger.info("TS dry run: sampled %s but not routing", chosen_key)
        return None

    chosen_arm = None
    for arm in arms:
        key = str(arm.get("name") or arm.get("model") or "")
        if key == chosen_key:
            chosen_arm = arm
            break
    if not chosen_arm:
        return None

    from hermes_cli.runtime_provider import resolve_runtime_provider

    explicit_api_key = None
    api_key_env = str(chosen_arm.get("api_key_env") or "").strip()
    if api_key_env:
        explicit_api_key = os.getenv(api_key_env) or None

    try:
        runtime = resolve_runtime_provider(
            requested=chosen_arm.get("provider"),
            explicit_api_key=explicit_api_key,
            explicit_base_url=chosen_arm.get("base_url"),
        )
    except Exception:
        logger.warning(
            "TS arm %s: credential resolution failed, falling back to primary",
            chosen_key,
        )
        return None

    if not runtime.get("api_key"):
        logger.warning(
            "TS arm %s: no api_key resolved, falling back to primary", chosen_key
        )
        return None

    return {
        "model": chosen_arm.get("model"),
        "runtime": {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "command": runtime.get("command"),
            "args": list(runtime.get("args") or []),
            "credential_pool": runtime.get("credential_pool"),
        },
        "label": f"thompson sampling → {chosen_arm.get('model')} ({runtime.get('provider')})",
        "signature": (
            chosen_arm.get("model"),
            runtime.get("provider"),
            runtime.get("base_url"),
            runtime.get("api_mode"),
            runtime.get("command"),
            tuple(runtime.get("args") or ()),
        ),
    }


def resolve_turn_route(
    user_message: str, routing_config: Optional[Dict[str, Any]], primary: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve the effective model/runtime for one turn.

    Returns a dict with model/runtime/signature/label fields.
    Dispatches on ``routing_config.mode``:
      - ``"thompson_sampling"`` → ``choose_thompson_sampling_route``
      - ``"cheap_model"`` or absent → ``choose_cheap_model_route``
    """
    mode = (routing_config or {}).get("mode", "cheap_model")

    if mode == "thompson_sampling":
        route = choose_thompson_sampling_route(user_message, routing_config, primary)
        if route:
            return {
                "model": route["model"],
                "runtime": route["runtime"],
                "label": route["label"],
                "signature": route["signature"],
            }
        return _primary_route(primary)

    route = choose_cheap_model_route(user_message, routing_config)
    if not route:
        return _primary_route(primary)

    from hermes_cli.runtime_provider import resolve_runtime_provider

    explicit_api_key = None
    api_key_env = str(route.get("api_key_env") or "").strip()
    if api_key_env:
        explicit_api_key = os.getenv(api_key_env) or None

    try:
        runtime = resolve_runtime_provider(
            requested=route.get("provider"),
            explicit_api_key=explicit_api_key,
            explicit_base_url=route.get("base_url"),
        )
    except Exception:
        return _primary_route(primary)

    return {
        "model": route.get("model"),
        "runtime": {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "command": runtime.get("command"),
            "args": list(runtime.get("args") or []),
            "credential_pool": runtime.get("credential_pool"),
        },
        "label": f"smart route → {route.get('model')} ({runtime.get('provider')})",
        "signature": (
            route.get("model"),
            runtime.get("provider"),
            runtime.get("base_url"),
            runtime.get("api_mode"),
            runtime.get("command"),
            tuple(runtime.get("args") or ()),
        ),
    }
