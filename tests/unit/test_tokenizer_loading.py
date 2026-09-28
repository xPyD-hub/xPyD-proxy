# SPDX-License-Identifier: Apache-2.0
"""Tests for per-model tokenizer loading and scheduler fallback."""

import itertools
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xpyd.proxy import Proxy
from xpyd.registry import InstanceRegistry
from xpyd.scheduler import RoundRobinSchedulingPolicy


def _proxy(tokenizer_path=None):
    proxy = Proxy.__new__(Proxy)
    proxy.tokenizer_path = tokenizer_path
    proxy._tokenizers = {}
    proxy._round_robin_models = set()
    proxy.tokenizer = None
    return proxy


def test_local_tokenizer_uses_model_named_subdirectory(tmp_path):
    model_dir = tmp_path / "org" / "model"
    model_dir.mkdir(parents=True)
    tokenizer = MagicMock()
    proxy = _proxy(str(tmp_path))

    with patch(
        "xpyd.proxy.AutoTokenizer.from_pretrained",
        return_value=tokenizer,
    ) as load:
        assert proxy.ensure_tokenizer("org/model") is True

    load.assert_called_once_with(
        str(model_dir),
        local_files_only=True,
    )
    assert proxy.get_tokenizer("org/model") is tokenizer


def test_local_tokenizer_missing_directory_has_guidance(tmp_path):
    proxy = _proxy(str(tmp_path))

    with pytest.raises(
        ValueError,
        match=r"Expected directory: .*org/model.*Set tokenizer_path",
    ):
        proxy.ensure_tokenizer("org/model")


def test_local_tokenizer_invalid_files_have_guidance(tmp_path):
    model_dir = tmp_path / "org" / "model"
    model_dir.mkdir(parents=True)
    proxy = _proxy(str(tmp_path))

    with (
        patch(
            "xpyd.proxy.AutoTokenizer.from_pretrained",
            side_effect=OSError("missing tokenizer_config.json"),
        ),
        pytest.raises(
            ValueError,
            match="model-named subdirectory with valid Hugging Face",
        ),
    ):
        proxy.ensure_tokenizer("org/model")


def test_remote_tokenizer_success_keeps_default_scheduler():
    tokenizer = MagicMock()
    proxy = _proxy()

    with patch(
        "xpyd.proxy.AutoTokenizer.from_pretrained",
        return_value=tokenizer,
    ) as load:
        assert proxy.ensure_tokenizer("org/model") is True

    load.assert_called_once_with("org/model", local_files_only=False)
    assert proxy.uses_round_robin_fallback("org/model") is False


def test_remote_tokenizer_failure_falls_back_to_round_robin():
    proxy = _proxy()

    with (
        patch(
            "xpyd.proxy.AutoTokenizer.from_pretrained",
            side_effect=OSError("offline"),
        ),
        patch("xpyd.proxy.logger.warning") as warning,
    ):
        assert proxy.ensure_tokenizer("org/model") is False

    assert proxy.uses_round_robin_fallback("org/model") is True
    assert "Falling back to roundrobin" in warning.call_args.args[0]


def test_pd_model_with_failed_tokenizer_uses_round_robin():
    registry = InstanceRegistry()
    registry.add("prefill", "10.0.0.1:8000", model="org/model")
    registry.add("decode", "10.0.0.2:8000", model="org/model")
    registry.mark_healthy("10.0.0.1:8000")
    registry.mark_healthy("10.0.0.2:8000")
    proxy = _proxy()
    proxy.registry = registry
    proxy.prefill_instances = ["10.0.0.1:8000"]
    proxy.decode_instances = ["10.0.0.2:8000"]
    from xpyd.scheduler import Scheduler

    proxy._scheduler = Scheduler(registry)
    proxy._round_robin_models.add("org/model")
    proxy._round_robin_policy = RoundRobinSchedulingPolicy(registry=registry)
    proxy.scheduling_policy = MagicMock()

    selected = Proxy.schedule(
        proxy,
        itertools.cycle(["10.0.0.1:8000"]),
        is_prompt=True,
        request_len=0,
        max_tokens=1,
        model="org/model",
    )
    Proxy.schedule_completion(
        proxy,
        prefill_instance=selected,
        req_len=0,
    )

    assert selected == "10.0.0.1:8000"
    proxy.scheduling_policy.select_node.assert_not_called()
    proxy.scheduling_policy.on_request_finished.assert_not_called()


@pytest.mark.asyncio
async def test_backend_tokenizer_used_when_local_tokenizer_unavailable():
    registry = InstanceRegistry()
    registry.add("prefill", "10.0.0.1:8000", model="org/model")
    registry.mark_healthy("10.0.0.1:8000")
    proxy = _proxy()
    proxy.registry = registry
    proxy.prefill_instances = ["10.0.0.1:8000"]

    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={"tokens": [1, 2, 3]})
    request_context = MagicMock()
    request_context.__aenter__ = AsyncMock(return_value=response)
    request_context.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.post.return_value = request_context
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "xpyd.proxy.aiohttp.ClientSession",
        return_value=session_context,
    ):
        tokens = await Proxy.tokenize_on_backend(
            proxy,
            {
                "model": "org/model",
                "messages": [{"role": "user", "content": "hello"}],
            },
            is_chat=True,
        )

    assert tokens == [1, 2, 3]
    session.post.assert_called_once_with(
        "http://10.0.0.1:8000/tokenize",
        json={
            "model": "org/model",
            "messages": [{"role": "user", "content": "hello"}],
            "add_generation_prompt": True,
        },
    )
