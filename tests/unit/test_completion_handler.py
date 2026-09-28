"""Unit tests for the unified completion handler helpers."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.responses import JSONResponse

from xpyd.metrics import REGISTRY
from xpyd.routes.completions import (
    _chat_completion_nonstream,
    _chat_completion_stream,
    _RequestReservation,
    _zmq_nonstream_generator,
    _zmq_stream_generator,
    build_kv_prepare_request,
    build_zmq_decode_request,
    build_zmq_prepare_request,
    extract_prompt_info,
    handle_completion,
    tokenize_zmq_prompt,
    validate_completion_request,
)
from xpyd.scheduler import RoundRobinSchedulingPolicy, Scheduler, SchedulingContext


def reservations(request_len=5):
    runtime = Scheduler()
    policy = RoundRobinSchedulingPolicy()
    return [
        runtime.reserve(
            policy,
            SchedulingContext(role=role, request_len=request_len),
            [f"{role}:8000"],
        )
        for role in ("prefill", "decode")
    ]


@pytest.fixture
def server():
    """Create a mock server with minimal attributes."""
    srv = MagicMock()
    srv.get_total_token_length = MagicMock(
        side_effect=lambda x, model="": len(x) if isinstance(x, str) else 0
    )
    srv.get_tokenizer = MagicMock(side_effect=lambda model="": srv.tokenizer)
    srv._is_aggregated_model = MagicMock(return_value=False)
    return srv


class TestValidateCompletionRequest:
    """Tests for validate_completion_request."""

    def test_completion_valid(self):
        result = validate_completion_request({"prompt": "hello"}, is_chat=False)
        assert result is None

    def test_completion_missing_prompt(self):
        result = validate_completion_request({}, is_chat=False)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400

    def test_chat_valid(self):
        result = validate_completion_request(
            {"messages": [{"role": "user", "content": "hi"}]}, is_chat=True
        )
        assert result is None

    def test_chat_missing_messages(self):
        result = validate_completion_request({}, is_chat=True)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400

    def test_chat_messages_not_list(self):
        result = validate_completion_request({"messages": "bad"}, is_chat=True)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400


class TestExtractPromptInfo:
    """Tests for extract_prompt_info."""

    def test_completion_string_prompt(self, server):
        total_length, max_tokens, prompt_text = extract_prompt_info(
            {"prompt": "hello world", "max_tokens": 50}, is_chat=False, server=server
        )
        assert total_length == 11  # len("hello world")
        assert max_tokens == 50
        assert prompt_text == "hello world"

    def test_completion_list_prompt(self, server):
        total_length, max_tokens, prompt_text = extract_prompt_info(
            {"prompt": [1, 2, 3], "max_tokens": 10}, is_chat=False, server=server
        )
        assert total_length == 0  # MagicMock returns 0 for non-str
        assert max_tokens == 10
        assert prompt_text == "[1, 2, 3]"

    def test_completion_default_max_tokens(self, server):
        _, max_tokens, _ = extract_prompt_info(
            {"prompt": "test"}, is_chat=False, server=server
        )
        assert max_tokens == 0

    def test_chat_basic(self, server):
        total_length, max_tokens, prompt_text = extract_prompt_info(
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 100,
            },
            is_chat=True,
            server=server,
        )
        assert total_length == len("You are helpful") + len("Hello")
        assert max_tokens == 100
        assert "You are helpful" in prompt_text
        assert "Hello" in prompt_text

    def test_chat_max_completion_tokens_priority(self, server):
        _, max_tokens, _ = extract_prompt_info(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 200,
                "max_tokens": 100,
            },
            is_chat=True,
            server=server,
        )
        assert max_tokens == 200

    def test_chat_fallback_to_max_tokens(self, server):
        _, max_tokens, _ = extract_prompt_info(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
            is_chat=True,
            server=server,
        )
        assert max_tokens == 100

    def test_chat_mixed_content_types(self, server):
        """Non-string content should be excluded from prompt_text."""
        _, _, prompt_text = extract_prompt_info(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": None},
                    {"role": "user", "content": "world"},
                ],
            },
            is_chat=True,
            server=server,
        )
        assert "hello" in prompt_text
        assert "world" in prompt_text
        assert "None" not in prompt_text

    def test_chat_null_content_zero_length(self, server):
        """Messages with None content should contribute 0 to total_length."""
        total_length, _, _ = extract_prompt_info(
            {
                "messages": [
                    {"role": "assistant", "content": None},
                    {"role": "user", "content": "hi"},
                ],
            },
            is_chat=True,
            server=server,
        )
        assert total_length == 2  # len("hi") via mock

    def test_chat_multimodal_content_array(self, server):
        """Multimodal content (list of parts) should extract text parts only."""
        total_length, _, prompt_text = extract_prompt_info(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/img.png"},
                            },
                        ],
                    },
                ],
            },
            is_chat=True,
            server=server,
        )
        assert total_length == len("What is this?")
        assert "What is this?" in prompt_text

    def test_completion_token_ids(self, server):
        """Flat list of ints (already tokenized) should return its length."""
        server.get_total_token_length = MagicMock(
            side_effect=lambda x, model="": (
                len(x) if isinstance(x, (str, list)) else 0
            )
        )
        total_length, _, _ = extract_prompt_info(
            {"prompt": [101, 102, 103]},
            is_chat=False,
            server=server,
        )
        assert total_length == 3


class TestGetTotalTokenLength:
    """Tests for get_total_token_length in core.utils."""

    @pytest.fixture
    def tokenizer(self):
        return MagicMock(side_effect=lambda text: {"input_ids": list(range(len(text)))})

    def test_none_input(self, tokenizer):
        from xpyd.utils import get_total_token_length

        assert get_total_token_length(tokenizer, None) == 0

    def test_empty_list(self, tokenizer):
        from xpyd.utils import get_total_token_length

        assert get_total_token_length(tokenizer, []) == 0

    def test_flat_int_list(self, tokenizer):
        """Single flat list of ints — already tokenized token IDs."""
        from xpyd.utils import get_total_token_length

        assert get_total_token_length(tokenizer, [101, 102, 103]) == 3

    def test_multimodal_dict_list(self, tokenizer):
        """List of dicts with text parts — multimodal content."""
        from xpyd.utils import get_total_token_length

        result = get_total_token_length(
            tokenizer,
            [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "http://example.com"}},
            ],
        )
        assert result == 5  # len("hello") via mock tokenizer


class TestBuildKvPrepareRequest:
    """Tests for build_kv_prepare_request."""

    def test_completion(self):
        req = {"prompt": "test", "max_tokens": 50}
        result = build_kv_prepare_request(req, is_chat=False)
        assert result["max_tokens"] == 1
        assert "max_completion_tokens" not in result
        assert req["max_tokens"] == 50

    def test_chat(self):
        req = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50}
        result = build_kv_prepare_request(req, is_chat=True)
        assert result["max_tokens"] == 1
        assert result["max_completion_tokens"] == 1
        assert req["max_tokens"] == 50

    def test_nixl_remote_decode(self):
        req = {
            "prompt": "test",
            "max_tokens": 50,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        result = build_kv_prepare_request(
            req,
            is_chat=False,
            disaggregated_mode="nixl",
        )

        assert result["stream"] is False
        assert "stream_options" not in result
        assert result["kv_transfer_params"] == {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        assert req["stream"] is True


class TestBuildZmqPrepareRequest:
    """Tests for LMCache ZMQ prefill request construction."""

    @pytest.fixture
    def receiver(self):
        receiver = MagicMock()
        receiver.host = "127.0.0.1"
        receiver.init_ports = [7300]
        receiver.alloc_ports = [7400]
        return receiver

    @pytest.mark.parametrize("source", ["prefill", "decode"])
    def test_first_token_source(self, receiver, source):
        result = build_zmq_prepare_request(
            {"prompt": "test", "max_tokens": 10, "stream": True},
            [1, 2, 3],
            "request-1",
            receiver,
            source,
        )

        params = result["kv_transfer_params"]
        assert ("ret_first_tok" in params) is (source == "prefill")
        assert params["disagg_spec"]["req_id"] == "request-1"
        assert result["prompt"] == [1, 2, 3]
        assert result["max_tokens"] == 1
        assert result["stream"] is False

    def test_chat_becomes_completion_request(self, receiver):
        result = build_zmq_prepare_request(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 10,
                "stream_options": {"include_usage": True},
            },
            [1, 2, 3],
            "request-1",
            receiver,
            "decode",
            is_chat=True,
        )

        assert "messages" not in result
        assert "max_completion_tokens" not in result
        assert "stream_options" not in result
        assert result["prompt"] == [1, 2, 3]


class TestZmqChatHelpers:
    def test_chat_template_tokenization(self, server):
        server.tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }
        request = {
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template": "custom",
            "chat_template_kwargs": {"foo": "bar"},
            "tools": [{"type": "function"}],
        }

        assert tokenize_zmq_prompt(request, True, server) == [1, 2, 3]
        server.tokenizer.apply_chat_template.assert_called_once_with(
            request["messages"],
            foo="bar",
            tokenize=True,
            add_generation_prompt=True,
            chat_template="custom",
            tools=request["tools"],
        )

    def test_continue_final_message_disables_generation_prompt(self, server):
        server.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        request = {
            "messages": [{"role": "assistant", "content": "partial"}],
            "continue_final_message": True,
        }

        tokenize_zmq_prompt(request, True, server)

        server.tokenizer.apply_chat_template.assert_called_once_with(
            request["messages"],
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
        )

    def test_chat_template_single_batch_is_unwrapped(self, server):
        server.tokenizer.apply_chat_template.return_value = {
            "input_ids": [[1, 2, 3]],
        }

        result = tokenize_zmq_prompt(
            {"messages": [{"role": "user", "content": "hi"}]},
            True,
            server,
        )

        assert result == [1, 2, 3]

    @pytest.mark.parametrize(
        ("first_token", "expected_prompt", "expected_max"),
        [(None, [1, 2, 3], 4), (9, [1, 2, 3, 9], 3)],
    )
    def test_decode_request(self, first_token, expected_prompt, expected_max):
        result = build_zmq_decode_request(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 4,
                "max_tokens": 4,
                "stream": True,
            },
            [1, 2, 3],
            first_token,
            4,
            is_chat=True,
        )

        assert result["prompt"] == expected_prompt
        assert result["max_tokens"] == expected_max
        assert "messages" not in result
        assert "max_completion_tokens" not in result

    @pytest.mark.asyncio
    async def test_stream_conversion_handles_split_sse_chunks(self):
        async def source():
            yield b'data: {"id":"cmpl","object":"text_completion","choices":['
            yield (
                b'{"index":0,"text":"hello","finish_reason":null}]}\n\n'
                b"data: [DONE]\n\n"
            )

        chunks = [chunk async for chunk in _chat_completion_stream(source())]
        first = json.loads(chunks[0].decode().removeprefix("data: "))
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"] == {
            "content": "hello",
            "role": "assistant",
        }
        assert chunks[1] == b"data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_nonstream_conversion(self):
        async def source():
            yield json.dumps(
                {
                    "id": "cmpl",
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": "hello", "finish_reason": "stop"}],
                }
            ).encode()

        chunks = [chunk async for chunk in _chat_completion_nonstream(source())]
        output = json.loads(chunks[0])
        assert output["object"] == "chat.completion"
        assert output["choices"][0]["message"] == {
            "role": "assistant",
            "content": "hello",
        }

    @pytest.mark.asyncio
    async def test_prefill_first_nonstream_merges_chat_and_usage(self, server):
        async def decode():
            yield json.dumps(
                {
                    "id": "cmpl-d",
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": "B", "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 1,
                        "total_tokens": 5,
                    },
                }
            ).encode()

        prefill = {
            "id": "cmpl-p",
            "created": 1,
            "model": "model",
            "choices": [{"text": "A"}],
        }
        chunks = [
            chunk
            async for chunk in _zmq_nonstream_generator(
                prefill, decode(), server, "p", "d", 3, is_chat=True
            )
        ]
        output = json.loads(chunks[0])
        assert output["choices"][0]["message"]["content"] == "AB"
        assert output["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }

    @pytest.mark.asyncio
    async def test_prefill_first_stream_has_chat_shape(self, server):
        async def decode():
            yield (
                b'data: {"id":"cmpl-d","created":2,"model":"model",'
                b'"choices":[{"index":0,"text":"B","finish_reason":null}]}\n\n'
            )
            yield (
                b'data: {"id":"cmpl-d","created":2,"model":"model",'
                b'"choices":[],"usage":{"prompt_tokens":4,'
                b'"completion_tokens":1,"total_tokens":5}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        prefill = {
            "id": "cmpl-p",
            "created": 1,
            "model": "model",
            "choices": [{"text": "A"}],
        }
        chunks = [
            chunk
            async for chunk in _zmq_stream_generator(
                prefill, decode(), server, "p", "d", 3, is_chat=True
            )
        ]
        head = json.loads(chunks[0].decode().removeprefix("data: "))
        body = json.loads(chunks[1].decode().removeprefix("data: "))
        assert head["choices"][0]["delta"] == {
            "content": "A",
            "role": "assistant",
        }
        assert body["id"] == "cmpl-p"
        assert body["object"] == "chat.completion.chunk"
        assert body["choices"][0]["delta"] == {"content": "B"}
        usage = json.loads(chunks[2].decode().removeprefix("data: "))
        assert usage["choices"] == []
        assert usage["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }


class TestHandleCompletion:
    """Integration-level tests for handle_completion."""

    @pytest.mark.asyncio
    async def test_invalid_json(self, server):
        raw_request = AsyncMock()
        raw_request.json = AsyncMock(side_effect=ValueError("bad json"))

        with (
            patch("xpyd.routes.completions.track_request_start", return_value=0),
            patch("xpyd.routes.completions.track_request_end") as track_end,
        ):
            result = await handle_completion(
                "/v1/completions", raw_request, server, is_chat=False
            )

        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        track_end.assert_called_once_with("/v1/completions", 0)

    @pytest.mark.asyncio
    async def test_second_selection_failure_releases_prefill(self, server):
        request = MagicMock(headers={}, client=None)
        request.json = AsyncMock(return_value={"prompt": "hello"})
        leases = reservations()
        leases[1].release()
        server.reserve.side_effect = [leases[0], RuntimeError("selection failed")]
        result = await handle_completion("/v1/completions", request, server, False)
        assert result.status_code == 500
        assert leases[0]._released

    @pytest.mark.asyncio
    async def test_prefill_cancellation_releases_both_nodes(self, server):
        request = MagicMock(headers={}, client=None)
        request.json = AsyncMock(return_value={"model": "model", "prompt": "hello"})
        leases = reservations()
        server.reserve.side_effect = leases
        server.disaggregated_mode = "direct"

        async def forward(*args, **kwargs):
            raise asyncio.CancelledError()
            yield b""  # pragma: no cover

        server.forward_request = forward
        with pytest.raises(asyncio.CancelledError):
            await handle_completion("/v1/completions", request, server, False)
        assert all(lease._released for lease in leases)
        server.registry.record_success.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_required_field(self, server):
        raw_request = AsyncMock()
        raw_request.json = AsyncMock(return_value={})

        with (
            patch("xpyd.routes.completions.track_request_start", return_value=0),
            patch("xpyd.routes.completions.track_request_end") as track_end,
        ):
            result = await handle_completion(
                "/v1/completions", raw_request, server, is_chat=False
            )

        assert isinstance(result, JSONResponse)
        assert result.status_code == 400
        track_end.assert_called_once_with("/v1/completions", 0)

    @pytest.mark.asyncio
    async def test_no_available_instance(self, server):
        raw_request = AsyncMock()
        raw_request.json = AsyncMock(return_value={"prompt": "hello"})
        raw_request.headers = {}
        raw_request.client = None

        server.reserve = MagicMock(return_value=None)
        server.prefill_cycler = MagicMock()
        server.decode_cycler = MagicMock()
        server.exception_handler = MagicMock()
        labels = {
            "instance": "unknown",
            "error_type": "no_available_instance",
            "model": "unknown",
        }
        before = REGISTRY.get_sample_value("proxy_instance_errors_total", labels) or 0

        with (
            patch("xpyd.routes.completions.track_request_start", return_value=0),
            patch("xpyd.routes.completions.track_request_end") as track_end,
            patch("xpyd.routes.completions.logger"),
        ):
            result = await handle_completion(
                "/v1/completions", raw_request, server, is_chat=False
            )

        assert isinstance(result, JSONResponse)
        assert result.status_code == 503
        server.exception_handler.assert_not_called()
        track_end.assert_called_once_with("/v1/completions", 0)
        assert (
            REGISTRY.get_sample_value("proxy_instance_errors_total", labels)
            == before + 1
        )

    def test_request_reservation_releases_each_node_once(self, server):
        leases = reservations(10)
        reservation = _RequestReservation(*leases)

        reservation.release_all()
        reservation.release_all()

        assert all(lease._released for lease in leases)
        assert all(value == 0 for value in leases[0]._runtime._active.values())

    @pytest.mark.asyncio
    async def test_nixl_prefill_parse_error_releases_reservations(self, server):
        raw_request = AsyncMock()
        raw_request.json = AsyncMock(
            return_value={
                "model": "model",
                "prompt": "hello",
                "max_tokens": 1,
            }
        )
        raw_request.headers = {}
        raw_request.client = None
        server.disaggregated_mode = "nixl"
        leases = reservations()
        server.reserve = MagicMock(side_effect=leases)
        server.prefill_cycler = MagicMock()
        server.decode_cycler = MagicMock()
        server.exception_handler = MagicMock()
        server._record_failure = MagicMock()

        async def forward(*_args, **_kwargs):
            yield b"not-json"

        server.forward_request = forward

        with pytest.raises(HTTPException, match="valid JSON"):
            await handle_completion(
                "/v1/completions", raw_request, server, is_chat=False
            )

        assert all(lease._released for lease in leases)
        server.registry.record_success.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_aggregated_model_ends_metrics(self, server):
        raw_request = AsyncMock()
        raw_request.json = AsyncMock(
            return_value={
                "model": "unknown-model",
                "prompt": "hello",
                "max_tokens": 1,
            }
        )
        raw_request.headers = {}
        raw_request.client = None

        server._is_aggregated_model.return_value = True
        server.reserve.return_value = None
        server.registry.get_registered_models.return_value = ["known-model"]

        with (
            patch("xpyd.routes.completions.track_request_start", return_value=0),
            patch("xpyd.routes.completions.track_request_end") as track_end,
        ):
            result = await handle_completion(
                "/v1/completions", raw_request, server, is_chat=False
            )

        assert isinstance(result, JSONResponse)
        assert result.status_code == 404
        track_end.assert_called_once_with("/v1/completions", 0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("source", ["decode", "prefill"])
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("max_tokens", [1, 2])
    async def test_zmq_chat_completion_modes(self, server, source, stream, max_tokens):
        from xpyd.proxy import D_first_token_generator

        request = {
            "model": "model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": max_tokens,
            "stream": stream,
        }
        raw_request = AsyncMock()
        raw_request.json = AsyncMock(return_value=request)
        raw_request.headers = {}
        raw_request.client = None

        receiver = MagicMock(host="receiver")
        receiver.init_ports = [7300]
        receiver.alloc_ports = [7400]
        server.disaggregated_mode = "zmq"
        server.first_token_source = source
        server.tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }
        leases = reservations(3)
        server.reserve = MagicMock(side_effect=leases)
        server.prefill_cycler = MagicMock()
        server.decode_cycler = MagicMock()
        server.zmq_config.receivers = {"decode:8000": receiver}
        server.zmq_notifications.register = AsyncMock()
        server.zmq_notifications.wait = AsyncMock()
        server.exception_handler = MagicMock()
        server._record_failure = MagicMock()
        server.registry = None
        server.generator = D_first_token_generator
        server.d_first_token_generator_class = D_first_token_generator
        forwarded = []

        async def forward(url, data, **kwargs):
            forwarded.append((url, data.copy()))
            if "prefill" in url:
                yield json.dumps(
                    {
                        "id": "cmpl-p",
                        "created": 1,
                        "model": "model",
                        "choices": [{"index": 0, "text": "A"}],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 1,
                            "total_tokens": 4,
                        },
                        "kv_transfer_params": {"first_tok": 9},
                    }
                ).encode()
            elif stream:
                yield (
                    b'data: {"id":"cmpl-d","created":2,"model":"model",'
                    b'"choices":[{"index":0,"text":"B",'
                    b'"finish_reason":null}]}\n\n'
                )
                yield b"data: [DONE]\n\n"
            else:
                yield json.dumps(
                    {
                        "id": "cmpl-d",
                        "object": "text_completion",
                        "created": 2,
                        "model": "model",
                        "choices": [
                            {
                                "index": 0,
                                "text": "B",
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 4 if source == "prefill" else 3,
                            "completion_tokens": (
                                1 if source == "prefill" else max_tokens
                            ),
                            "total_tokens": (
                                5 if source == "prefill" else 3 + max_tokens
                            ),
                        },
                    }
                ).encode()

        server.forward_request = forward
        with (
            patch("xpyd.routes.completions.track_request_start", return_value=0),
            patch("xpyd.routes.completions.track_request_end"),
            patch("xpyd.routes.completions.record_disaggregated_metrics"),
        ):
            response = await handle_completion(
                "/v1/chat/completions", raw_request, server, is_chat=True
            )
            body = b"".join([chunk async for chunk in response.body_iterator])

        expected_urls = ["http://prefill:8000/v1/completions"] + (
            ["http://decode:8000/v1/completions"]
            if source == "decode" or max_tokens > 1
            else []
        )
        assert [call[0] for call in forwarded] == expected_urls
        assert server.reserve.call_count == 2
        assert all(lease._released for lease in leases)
        assert "messages" not in forwarded[0][1]
        assert forwarded[0][1]["prompt"] == [1, 2, 3]
        if len(forwarded) == 2:
            assert "messages" not in forwarded[1][1]
            expected_prompt = [1, 2, 3, 9] if source == "prefill" else [1, 2, 3]
            assert forwarded[1][1]["prompt"] == expected_prompt
            expected_max = max_tokens - 1 if source == "prefill" else max_tokens
            assert forwarded[1][1]["max_tokens"] == expected_max

        if stream:
            events = [
                json.loads(event.removeprefix("data: "))
                for event in body.decode().split("\n\n")
                if event and event != "data: [DONE]"
            ]
            assert all(event["object"] == "chat.completion.chunk" for event in events)
            content = "".join(
                event["choices"][0]["delta"].get("content", "")
                for event in events
                if event["choices"]
            )
        else:
            output = json.loads(body)
            assert output["object"] == "chat.completion"
            content = output["choices"][0]["message"]["content"]
            if source == "prefill" and max_tokens > 1:
                assert output["usage"] == {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                }
            elif source == "decode":
                assert output["usage"]["completion_tokens"] == max_tokens
        expected_content = (
            "A"
            if source == "prefill" and max_tokens == 1
            else "AB" if source == "prefill" else "B"
        )
        assert content == expected_content
