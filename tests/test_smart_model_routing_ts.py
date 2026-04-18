"""Tests for agent.smart_model_routing — Thompson Sampling mode."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.smart_model_routing import choose_thompson_sampling_route, resolve_turn_route


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "ts_state.json"


@pytest.fixture
def ts_config(state_path):
    return {
        "mode": "thompson_sampling",
        "thompson_sampling": {
            "state_path": str(state_path),
            "arms": [
                {
                    "name": "arm_a",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                },
                {
                    "name": "arm_b",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "api_key_env": "ANTHROPIC_API_KEY",
                },
            ],
        },
    }


PRIMARY = {
    "model": "primary-model",
    "api_key": "pk-1",
    "base_url": "https://primary.example.com",
    "provider": "primary",
    "api_mode": "chat",
    "command": None,
    "args": [],
    "credential_pool": None,
}


class TestChooseThompsonSamplingRoute:
    def test_ts_disabled_env_returns_none(self, ts_config):
        with patch.dict(os.environ, {"HERMES_TS_DISABLED": "true"}):
            result = choose_thompson_sampling_route("hello", ts_config, PRIMARY)
        assert result is None

    def test_ts_dry_run_env_returns_none_after_sampling(self, ts_config, state_path):
        from agent.ts_state import load_state

        state_before = load_state(state_path, arm_keys=["arm_a", "arm_b"])
        with patch.dict(os.environ, {"HERMES_TS_DRY_RUN": "true"}):
            result = choose_thompson_sampling_route("hello", ts_config, PRIMARY)
        assert result is None
        state_after = load_state(state_path, arm_keys=["arm_a", "arm_b"])
        for arm in ("arm_a", "arm_b"):
            assert state_after["arms"][arm]["last_updated"] == state_before["arms"].get(
                arm, {}
            ).get("last_updated", "")

    def test_missing_arm_credentials_returns_none(self, state_path):
        config = {
            "mode": "thompson_sampling",
            "thompson_sampling": {
                "state_path": str(state_path),
                "arms": [
                    {
                        "name": "no_creds",
                        "provider": "nonexistent_provider",
                        "model": "fake-model",
                    },
                ],
            },
        }
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            side_effect=Exception("no provider"),
        ):
            result = choose_thompson_sampling_route("hello", config, PRIMARY)
        assert result is None


class TestResolveTurnRoute:
    def test_turn_stickiness_simulation(self, ts_config, state_path):
        runtime_stub = {
            "api_key": "sk-test",
            "base_url": "https://api.openai.com/v1",
            "provider": "openai",
            "api_mode": "chat",
            "command": None,
            "args": [],
            "credential_pool": None,
        }
        with (
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=runtime_stub,
            ),
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}),
        ):
            route = resolve_turn_route("hello", ts_config, PRIMARY)
        chosen_model = route["model"]
        for _ in range(5):
            assert route["model"] == chosen_model

    def test_ts_mode_dispatches_to_ts(self, ts_config, state_path):
        runtime_stub = {
            "api_key": "sk-test",
            "base_url": "https://api.openai.com/v1",
            "provider": "openai",
            "api_mode": "chat",
            "command": None,
            "args": [],
            "credential_pool": None,
        }
        with (
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=runtime_stub,
            ),
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}),
            patch("agent.ts_state.thompson_sample", return_value="arm_a"),
        ):
            route = resolve_turn_route("hello", ts_config, PRIMARY)
        assert route["model"] == "gpt-4o-mini"
        assert "thompson sampling" in (route["label"] or "")

    def test_cheap_model_mode_unchanged(self):
        config = {
            "mode": "cheap_model",
            "enabled": True,
            "cheap_model": {"provider": "openai", "model": "gpt-4o-mini"},
        }
        with patch("agent.smart_model_routing.choose_cheap_model_route") as mock_cmr:
            mock_cmr.return_value = None
            resolve_turn_route("hello", config, PRIMARY)
        mock_cmr.assert_called_once_with("hello", config)

    def test_no_mode_defaults_to_cheap_model(self):
        config = {
            "enabled": True,
            "cheap_model": {"provider": "openai", "model": "gpt-4o-mini"},
        }
        with patch("agent.smart_model_routing.choose_cheap_model_route") as mock_cmr:
            mock_cmr.return_value = None
            resolve_turn_route("hello", config, PRIMARY)
        mock_cmr.assert_called_once_with("hello", config)
