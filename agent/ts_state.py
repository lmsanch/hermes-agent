"""Beta-Bernoulli Thompson Sampling state store for model routing.

Schema (JSON file at ``<path>``):
::

    {
      "arms": {
        "anthropic:claude-sonnet-4-6": {"a": 1.0, "b": 1.0, "wins": 0, "losses": 0, "last_updated": "2026-04-17T21:00:00Z"},
        ...
      },
      "version": 1
    }

Invariants:
- ``a`` and ``b`` are Beta-distribution shape parameters (prior: Beta(1,1) = uniform).
- ``wins`` / ``losses`` are integer counts of observed outcomes.
- ``a = 1 + wins``, ``b = 1 + losses`` at all times.
- ``last_updated`` is ISO-8601 with trailing ``Z``.
- File locking uses ``fcntl.flock`` on a sidecar ``<path>.lock`` file (POSIX only).
"""

from __future__ import annotations

import fcntl
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ARM_DEFAULT = {"a": 1.0, "b": 1.0, "wins": 0, "losses": 0}


def load_state(path: Path, arm_keys: Optional[list[str]] = None) -> dict:
    """Read state JSON from *path*. Returns ``{}`` when file is missing.

    If *arm_keys* is given, any key not already present in ``arms`` is
    auto-initialised with the default prior.
    """
    try:
        with open(path, "r") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        state = {"arms": {}, "version": 1}
    if "arms" not in state or not isinstance(state["arms"], dict):
        state["arms"] = {}
    if arm_keys:
        for key in arm_keys:
            if key not in state["arms"]:
                state["arms"][key] = dict(_ARM_DEFAULT, last_updated="")
    return state


def _acquire_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_fd = open(lock_path, "w")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    return lock_fd


def _release_lock(lock_fd):
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        lock_fd.close()


def thompson_sample(
    arm_keys: list[str], path: Path, rng: Optional[random.Random] = None
) -> str:
    """Sample one arm via Beta(a, b) per arm; return the arm_key with the highest draw.

    *rng* is optional — inject a ``random.Random`` for deterministic testing.
    """
    state = load_state(path, arm_keys=arm_keys)
    best_key = arm_keys[0]
    best_val = -1.0
    _betavariate = rng.betavariate if rng else random.betavariate
    for key in arm_keys:
        arm = state["arms"].get(key, _ARM_DEFAULT)
        sample = _betavariate(arm["a"], arm["b"])
        if sample > best_val:
            best_val = sample
            best_key = key
    return best_key


def record_outcome(arm_key: str, success: bool, path: Path) -> None:
    """Increment a/b and wins/losses for *arm_key*, persist to disk with flock."""
    lock_fd = _acquire_lock(path)
    try:
        state = load_state(path, arm_keys=[arm_key])
        arm = state["arms"][arm_key]
        if success:
            arm["a"] += 1.0
            arm["wins"] += 1
        else:
            arm["b"] += 1.0
            arm["losses"] += 1
        arm["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    finally:
        _release_lock(lock_fd)
