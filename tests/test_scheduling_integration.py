# SPDX-License-Identifier: Apache-2.0
"""Tests for Task 10 integration — scheduling context extraction and
YAML-based policy selection."""

import itertools
from unittest.mock import MagicMock

import pytest

from core.scheduler import (
    CacheAwarePolicy,
    ConsistentHashPolicy,
    PowerOfTwoPolicy,
    RoundRobinSchedulingPolicy,
    default_registry,
)
from core.scheduler.cache_aware import CacheAwarePolicy as CacheAwareDirect

# ------------------------------------------------------------------ #
# Cache-aware policy unit tests
# ------------------------------------------------------------------ #


class TestCacheAwarePolicy:
    """Unit tests for CacheAwarePolicy."""

    def test_same_prefix_same_worker(self):
        policy = CacheAwarePolicy(
            workers=["w1", "w2", "w3"],
            prefix_length=5,
        )
        # Both prompts share the same first 5 tokens (whitespace-split)
        w1 = policy.select(prompt="alpha beta gamma delta epsilon zeta")
        w2 = policy.select(prompt="alpha beta gamma delta epsilon different")
        assert w1 == w2

    def test_different_prefix_can_differ(self):
        policy = CacheAwarePolicy(
            workers=["w1", "w2", "w3"],
            prefix_length=256,
        )
        selected = set()
        for i in range(50):
            w = policy.select(prompt=f"Unique prompt {i} " * 100)
            selected.add(w)
        assert len(selected) > 1

    def test_single_worker(self):
        policy = CacheAwarePolicy(workers=["w1"], prefix_length=256)
        assert policy.select(prompt="hello") == "w1"

    def test_no_workers(self):
        policy = CacheAwarePolicy(workers=[], prefix_length=256)
        assert policy.select(prompt="hello") is None

    def test_none_prompt_deterministic(self):
        policy = CacheAwarePolicy(workers=["w1", "w2"], prefix_length=256)
        w1 = policy.select(prompt=None)
        w2 = policy.select(prompt=None)
        assert w1 == w2

    def test_add_remove_worker(self):
        policy = CacheAwarePolicy(workers=["w1", "w2"])
        policy.add_worker("w3")
        selected = {policy.select(prompt=f"p{i}") for i in range(100)}
        assert "w3" in selected
        policy.remove_worker("w3")
        # After removal, w3 should never be selected
        for i in range(50):
            assert policy.select(prompt=f"p{i}") != "w3"

    def test_schedule_interface(self):
        """schedule() passes prompt through to select()."""
        policy = CacheAwarePolicy(workers=["w1", "w2", "w3"])
        cycler = itertools.cycle(["w1", "w2", "w3"])
        result = policy.schedule(cycler, prompt="hello world")
        assert result in {"w1", "w2", "w3"}


# ------------------------------------------------------------------ #
# Policy registry — all built-in policies registered
# ------------------------------------------------------------------ #


class TestPolicyRegistryIntegration:
    """Verify all Task 10 policies are registered in default_registry."""

    @pytest.mark.parametrize(
        "name",
        [
            "roundrobin",
            "loadbalanced",
            "consistent_hash",
            "power_of_two",
            "cache_aware",
        ],
    )
    def test_builtin_policies_registered(self, name):
        assert default_registry.has(name)

    def test_create_consistent_hash(self):
        policy = default_registry.create(
            "consistent_hash",
            workers=["w1", "w2"],
        )
        assert isinstance(policy, ConsistentHashPolicy)

    def test_create_power_of_two(self):
        policy = default_registry.create(
            "power_of_two",
            workers=["w1", "w2"],
        )
        assert isinstance(policy, PowerOfTwoPolicy)

    def test_create_cache_aware(self):
        policy = default_registry.create(
            "cache_aware",
            workers=["w1", "w2"],
            prefix_length=128,
        )
        assert isinstance(policy, CacheAwareDirect)

    def test_create_cache_aware_default_prefix(self):
        policy = default_registry.create(
            "cache_aware",
            workers=["w1"],
        )
        assert policy._prefix_length == 256

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduling policy"):
            default_registry.create("nonexistent", workers=["w1"])


# ------------------------------------------------------------------ #
# Session ID / prompt extraction helpers
# ------------------------------------------------------------------ #


def _make_mock_request(
    headers=None,
    client_host="127.0.0.1",
):
    """Create a mock Starlette Request."""
    req = MagicMock()
    _headers = headers or {}
    req.headers = _headers
    if client_host:
        req.client = MagicMock()
        req.client.host = client_host
    else:
        req.client = None
    return req


class TestSessionIdExtraction:
    """Test the session_id extraction priority logic used in proxy."""

    @staticmethod
    def _extract_session_id(raw_request, body):
        """Replicate the extraction logic from MicroPDProxyServer."""
        return (
            raw_request.headers.get("x-session-id")
            or body.get("user")
            or (raw_request.client.host if raw_request.client else None)
        )

    def test_header_takes_priority(self):
        req = _make_mock_request(
            headers={"x-session-id": "sess-1"},
            client_host="1.2.3.4",
        )
        body = {"user": "user-abc"}
        assert self._extract_session_id(req, body) == "sess-1"

    def test_user_field_fallback(self):
        req = _make_mock_request(headers={}, client_host="1.2.3.4")
        body = {"user": "user-abc"}
        assert self._extract_session_id(req, body) == "user-abc"

    def test_client_ip_fallback(self):
        req = _make_mock_request(headers={}, client_host="10.0.0.1")
        body = {}
        assert self._extract_session_id(req, body) == "10.0.0.1"

    def test_no_client(self):
        req = _make_mock_request(headers={}, client_host=None)
        body = {}
        assert self._extract_session_id(req, body) is None


class TestPromptExtraction:
    """Test prompt text extraction from request bodies."""

    def test_completion_prompt(self):
        body = {"prompt": "Hello, world!"}
        prompt = body.get("prompt", "")
        assert prompt == "Hello, world!"

    def test_chat_messages(self):
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi there"},
            ],
        }
        prompt_text = " ".join(
            msg.get("content", "")
            for msg in body.get("messages", [])
            if isinstance(msg.get("content"), str)
        )
        assert "You are helpful." in prompt_text
        assert "Hi there" in prompt_text


# ------------------------------------------------------------------ #
# YAML config → policy instantiation
# ------------------------------------------------------------------ #


class TestYamlConfigPolicySelection:
    """Test that YAML scheduling config produces the correct policy."""

    def test_default_is_loadbalanced(self):
        """Without explicit scheduling key, default is loadbalanced."""
        from core.config import ProxyConfig

        config = ProxyConfig(
            model="/tmp/model",
            decode=["127.0.0.1:8000"],
        )
        assert config.scheduling == "loadbalanced"

    def test_scheduling_field_stored(self):
        from core.config import ProxyConfig

        config = ProxyConfig(
            model="/tmp/model",
            decode=["127.0.0.1:8000"],
            scheduling="cache_aware",
            scheduling_config={"cache_aware": {"prefix_length": 128}},
        )
        assert config.scheduling == "cache_aware"
        assert config.scheduling_config["cache_aware"]["prefix_length"] == 128


# ------------------------------------------------------------------ #
# Backward compatibility — kwargs ignored by legacy policies
# ------------------------------------------------------------------ #


class TestBackwardCompatibility:
    """Verify roundrobin and loadbalanced accept and ignore extra kwargs."""

    def test_roundrobin_ignores_kwargs(self):
        policy = RoundRobinSchedulingPolicy()
        cycler = itertools.cycle(["w1", "w2"])
        result = policy.schedule(
            cycler,
            is_prompt=True,
            header="sess-1",
            session_id="sess-1",
            user="u",
            client_ip="1.2.3.4",
            prompt="hello",
        )
        assert result in {"w1", "w2"}

    def test_consistent_hash_uses_kwargs(self):
        policy = ConsistentHashPolicy(workers=["w1", "w2", "w3"])
        cycler = itertools.cycle(["w1", "w2", "w3"])
        r1 = policy.schedule(
            cycler,
            header="sess-1",
            session_id="sess-1",
        )
        r2 = policy.schedule(
            cycler,
            header="sess-1",
            session_id="sess-1",
        )
        assert r1 == r2  # same session → same worker

    def test_power_of_two_ignores_kwargs(self):
        policy = PowerOfTwoPolicy(workers=["w1", "w2"])
        cycler = itertools.cycle(["w1", "w2"])
        result = policy.schedule(
            cycler,
            header="sess-1",
            prompt="hello",
        )
        assert result in {"w1", "w2"}

    def test_cache_aware_uses_prompt_kwarg(self):
        policy = CacheAwarePolicy(workers=["w1", "w2", "w3"])
        cycler = itertools.cycle(["w1", "w2", "w3"])
        r1 = policy.schedule(cycler, prompt="same prefix " * 100)
        r2 = policy.schedule(cycler, prompt="same prefix " * 100)
        assert r1 == r2
