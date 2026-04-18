"""Tests for agent.ts_state — Beta-Bernoulli Thompson Sampling store."""

import json
import math
from datetime import datetime
from pathlib import Path

import pytest

from agent.ts_state import load_state, record_outcome, thompson_sample


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "ts_state.json"


class TestThompsonSample:
    def test_flat_prior_uniform_sampling(self, state_path):
        rng = __import__("random").Random(42)
        arms = ["arm_a", "arm_b", "arm_c"]
        counts = {k: 0 for k in arms}
        n = 300
        for _ in range(n):
            chosen = thompson_sample(arms, state_path, rng=rng)
            counts[chosen] += 1
        expected = n / len(arms)
        std = math.sqrt(expected * (1 - 1 / len(arms)))
        three_sigma = 3 * std
        for arm in arms:
            assert expected - three_sigma <= counts[arm] <= expected + three_sigma, (
                f"{arm} count {counts[arm]} outside 3-sigma [{expected - three_sigma:.0f}, {expected + three_sigma:.0f}]"
            )

    def test_record_outcome_updates_counters(self, state_path):
        arm = "test_arm"
        for _ in range(5):
            record_outcome(arm, True, state_path)
        for _ in range(3):
            record_outcome(arm, False, state_path)
        state = load_state(state_path, arm_keys=[arm])
        entry = state["arms"][arm]
        assert entry["a"] == 6.0
        assert entry["b"] == 4.0
        assert entry["wins"] == 5
        assert entry["losses"] == 3
        datetime.strptime(entry["last_updated"], "%Y-%m-%dT%H:%M:%SZ")

    def test_state_persistence_roundtrip(self, state_path):
        record_outcome("x", True, state_path)
        record_outcome("y", False, state_path)
        fresh = load_state(state_path)
        assert fresh["arms"]["x"]["wins"] == 1
        assert fresh["arms"]["y"]["losses"] == 1
