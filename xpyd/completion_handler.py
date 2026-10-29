# SPDX-License-Identifier: Apache-2.0
"""Shared execution and helpers for completion and chat completion requests."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from asyncio import CancelledError
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from dataclasses import dataclass

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from xpyd.config import ZmqConfig, ZmqReceiverConfig
from xpyd.errors import INVALID_REQUEST, PROXY_ERROR, error_response
from xpyd.interfaces import (
    CountTokens,
    ForwardRequest,
    Notifications,
    TokenizeOnBackend,
    Tokenizer,
)
from xpyd.metrics import (
    FirstTokenTracker,
    proxy_decode_active_requests,
    proxy_decode_requests_total,
    proxy_instance_errors_total,
    proxy_prefill_active_requests,
    proxy_prefill_queue_depth,
    proxy_prefill_requests_total,
    record_disaggregated_metrics,
    track_request_end,
    track_request_start,
)
from xpyd.scheduler import Reservation, SchedulingContext

logger = logging.getLogger("xpyd.proxy")

FirstTokenGenerator = Callable[
    [
        AsyncIterator[bytes],
        AsyncIterator[bytes],
        Callable[[], None],
        Callable[[], None],
    ],
    AsyncGenerator[bytes, None],
]


@dataclass(frozen=True)
class CompletionOptions:
    disaggregated_mode: str = "direct"
    first_token_source: str = "decode"
    zmq: ZmqConfig | None = None


def _record_failures(
    record_failure: Callable[[str], None] | None, *instances: str
) -> None:
    if record_failure is not None:
        for instance in instances:
            record_failure(instance)


async def P_first_token_generator(
    generator_p: AsyncIterator[bytes],
    generator_d: AsyncIterator[bytes],
    release_prefill: Callable[[], None],
    release_decode: Callable[[], None],
) -> AsyncGenerator[bytes, None]:
    first_decode = True

    try:
        async for chunk in generator_p:
            yield chunk
    finally:
        release_prefill()

    try:
        async for chunk in generator_d:
            if first_decode:
                first_decode = False
                continue
            yield chunk
    finally:
        release_decode()


async def D_first_token_generator(
    generator_p: AsyncIterator[bytes],
    generator_d: AsyncIterator[bytes],
    release_prefill: Callable[[], None],
    release_decode: Callable[[], None],
) -> AsyncGenerator[bytes, None]:
    try:
        async for _ in generator_p:
            continue
    finally:
        release_prefill()

    try:
        async for chunk in generator_d:
            yield chunk
    finally:
        release_decode()


_CHAT_ONLY_REQUEST_FIELDS = {
    "messages",
    "max_completion_tokens",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "chat_template",
    "chat_template_kwargs",
    "add_generation_prompt",
    "continue_final_message",
}


class _RequestReservation:
    """Release each P/D scheduling reservation at most once."""

    def __init__(
        self,
        prefill: Reservation | None = None,
        decode: Reservation | None = None,
    ) -> None:
        self.prefill = prefill
        self.decode = decode

    def release_prefill(self) -> None:
        if self.prefill is not None:
            self.prefill.release()

    def release_decode(self) -> None:
        if self.decode is not None:
            self.decode.release()

    def release_all(self) -> None:
        self.release_prefill()
        self.release_decode()


# ---------------------------------------------------------------------------
# Request preparation helpers
# ---------------------------------------------------------------------------


def validate_completion_request(request: dict, is_chat: bool) -> JSONResponse | None:
    """Validate required fields. Returns JSONResponse on error, None on success."""
    if is_chat:
        if "messages" not in request:
            return error_response(
                "Missing required field: messages", INVALID_REQUEST, 400
            )
        if not isinstance(request["messages"], list):
            return error_response("Field messages must be a list", INVALID_REQUEST, 400)
    else:
        if "prompt" not in request:
            return error_response(
                "Missing required field: prompt", INVALID_REQUEST, 400
            )
    return None


def extract_prompt_info(
    request: dict, is_chat: bool, count_tokens: CountTokens
) -> tuple[int, int, str]:
    """Extract prompt metrics. Returns (total_length, max_tokens, prompt_text)."""
    model = request.get("model", "")
    if is_chat:
        total_length = 0
        prompt_parts = []
        for msg in request["messages"]:
            content = msg.get("content")
            if content is None:
                continue
            if isinstance(content, str):
                total_length += count_tokens(content, model)
                prompt_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        total_length += count_tokens(text, model)
                        prompt_parts.append(text)
        max_tokens = request.get("max_completion_tokens", 0)
        if max_tokens == 0:
            max_tokens = request.get("max_tokens", 0)
        prompt_text = " ".join(prompt_parts)
    else:
        prompt = request.get("prompt")
        total_length = count_tokens(prompt, model)
        max_tokens = request.get("max_tokens", 0)
        prompt_text = prompt if isinstance(prompt, str) else str(prompt)
    return total_length, max_tokens, prompt_text


def build_kv_prepare_request(
    request: dict,
    is_chat: bool,
    disaggregated_mode: str = "direct",
) -> dict:
    """Build the KV-prepare request with max_tokens=1."""
    kv_prepare_request = request.copy()
    kv_prepare_request["max_tokens"] = 1
    if is_chat:
        kv_prepare_request["max_completion_tokens"] = 1
    if disaggregated_mode == "nixl":
        kv_prepare_request["stream"] = False
        kv_prepare_request.pop("stream_options", None)
        kv_prepare_request["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
    return kv_prepare_request


def build_zmq_prepare_request(
    request: dict,
    prompt_tokens: list[int],
    request_id: str,
    receiver: ZmqReceiverConfig,
    first_token_source: str,
    is_chat: bool = False,
) -> dict:
    """Build an LMCache sender request for a preselected decode instance."""
    prepare = request.copy()
    if is_chat:
        for field in _CHAT_ONLY_REQUEST_FIELDS:
            prepare.pop(field, None)
    prepare["prompt"] = prompt_tokens
    prepare["max_tokens"] = 1
    prepare["stream"] = False
    prepare.pop("stream_options", None)
    transfer_params = {
        "disagg_spec": {
            "req_id": request_id,
            "receiver_host": receiver.host,
            "receiver_init_port": receiver.init_ports,
            "receiver_alloc_port": receiver.alloc_ports,
        }
    }
    if first_token_source == "prefill":
        transfer_params["ret_first_tok"] = True
    prepare["kv_transfer_params"] = transfer_params
    return prepare


def tokenize_zmq_prompt(
    request: dict, is_chat: bool, tokenizer: Tokenizer | None
) -> list[int]:
    """Tokenize the exact prompt used by the LMCache sender and receiver."""
    if tokenizer is None:
        raise ValueError(
            "ZMQ prompt tokenization requires a tokenizer for model "
            f"{request.get('model', '')!r}"
        )
    if not is_chat:
        prompt = request["prompt"]
        return list(prompt) if isinstance(prompt, list) else tokenizer.encode(prompt)

    template_kwargs = dict(request.get("chat_template_kwargs") or {})
    template_kwargs["tokenize"] = True
    continue_final_message = request.get("continue_final_message", False)
    template_kwargs["add_generation_prompt"] = request.get(
        "add_generation_prompt", not continue_final_message
    )
    if "continue_final_message" in request:
        template_kwargs["continue_final_message"] = continue_final_message
    if request.get("chat_template") is not None:
        template_kwargs["chat_template"] = request["chat_template"]
    if request.get("tools") is not None:
        template_kwargs["tools"] = request["tools"]
    tokenized = tokenizer.apply_chat_template(request["messages"], **template_kwargs)
    if isinstance(tokenized, Mapping):
        tokenized = tokenized.get("input_ids")
        if tokenized is None:
            raise ValueError("Chat template output did not contain input_ids")
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if (
        isinstance(tokenized, list)
        and len(tokenized) == 1
        and isinstance(tokenized[0], list)
    ):
        tokenized = tokenized[0]
    return list(tokenized)


def build_zmq_decode_request(
    request: dict,
    prompt_tokens: list[int],
    first_token: int | None,
    max_tokens: int,
    is_chat: bool,
) -> dict:
    """Build a completions request whose prompt exactly matches transferred KV."""
    decode = request.copy()
    if is_chat:
        for field in _CHAT_ONLY_REQUEST_FIELDS:
            decode.pop(field, None)
        if max_tokens > 0:
            decode["max_tokens"] = max_tokens
    decode["prompt"] = list(prompt_tokens)
    if first_token is not None:
        decode["prompt"].append(first_token)
        if max_tokens > 0:
            decode["max_tokens"] = max(1, max_tokens - 1)
    decode.pop("kv_transfer_params", None)
    return decode


def _completion_to_chat(output: dict) -> dict:
    """Convert a text completion response to OpenAI chat completion shape."""
    result = output.copy()
    result["object"] = "chat.completion"
    result["choices"] = [
        {
            "index": choice.get("index", index),
            "message": {
                "role": "assistant",
                "content": choice.get("text", ""),
            },
            "logprobs": choice.get("logprobs"),
            "finish_reason": choice.get("finish_reason"),
        }
        for index, choice in enumerate(output.get("choices", []))
    ]
    return result


def _completion_chunk_to_chat(
    output: dict,
    *,
    response_metadata: dict | None = None,
) -> dict:
    """Convert a text completion stream chunk to chat completion chunk shape."""
    result = output.copy()
    if response_metadata:
        result.update(response_metadata)
    result["object"] = "chat.completion.chunk"
    result["choices"] = [
        {
            "index": choice.get("index", index),
            "delta": {"content": choice.get("text", "")},
            "logprobs": choice.get("logprobs"),
            "finish_reason": choice.get("finish_reason"),
        }
        for index, choice in enumerate(output.get("choices", []))
    ]
    return result


async def _chat_completion_stream(
    completion_generator,
    *,
    response_metadata: dict | None = None,
    include_role: bool = True,
):
    """Convert arbitrarily chunked completion SSE data to chat completion SSE."""
    buffer = ""
    role_pending = include_role
    async for chunk in completion_generator:
        buffer += chunk.decode("utf-8")
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            for line in event.splitlines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    yield b"data: [DONE]\n\n"
                    continue
                output = _completion_chunk_to_chat(
                    json.loads(payload),
                    response_metadata=response_metadata,
                )
                if role_pending and output["choices"]:
                    output["choices"][0]["delta"]["role"] = "assistant"
                    role_pending = False
                yield (
                    f"data: {json.dumps(output, separators=(',', ':'))}\n\n"
                ).encode()
    if buffer.strip():
        raise ValueError("Decode response ended with an incomplete SSE event")


async def _chat_completion_nonstream(completion_generator):
    """Convert a non-streaming text completion response to chat shape."""
    value = b""
    async for chunk in completion_generator:
        value += chunk
    yield json.dumps(
        _completion_to_chat(json.loads(value)),
        separators=(",", ":"),
    ).encode()


def _merge_prefill_token_usage(usage: dict) -> None:
    """Move the appended prefill token from prompt usage to completion usage."""
    if usage.get("prompt_tokens", 0) > 0:
        usage["prompt_tokens"] -= 1
    usage["completion_tokens"] = usage.get("completion_tokens", 0) + 1
    usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
        "completion_tokens", 0
    )


async def _zmq_stream_usage_generator(decode_generator):
    """Correct usage events after the prefill token was appended to the prompt."""
    buffer = ""
    async for chunk in decode_generator:
        buffer += chunk.decode("utf-8")
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            lines = []
            for line in event.splitlines():
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload != "[DONE]":
                        output = json.loads(payload)
                        if isinstance(output.get("usage"), dict):
                            _merge_prefill_token_usage(output["usage"])
                        line = "data: " + json.dumps(output, separators=(",", ":"))
                lines.append(line)
            yield ("\n".join(lines) + "\n\n").encode()
    if buffer.strip():
        raise ValueError("Decode response ended with an incomplete SSE event")


async def _zmq_stream_generator(
    prefill_output: dict,
    decode_generator,
    reservation: _RequestReservation,
    is_chat: bool = False,
):
    """Return the P token followed by every streamed D token."""
    head: dict = {
        "id": prefill_output["id"],
        "object": "text_completion",
        "created": prefill_output["created"],
        "model": prefill_output["model"],
        "choices": [
            {
                "index": 0,
                "text": prefill_output["choices"][0]["text"],
                "logprobs": None,
                "finish_reason": None,
            }
        ],
        "usage": None,
    }
    if is_chat:
        head = _completion_chunk_to_chat(head)
        head["choices"][0]["delta"]["role"] = "assistant"
    yield f"data: {json.dumps(head, separators=(',', ':'))}\n\n".encode()
    try:
        decode_generator = _zmq_stream_usage_generator(decode_generator)
        generator = (
            _chat_completion_stream(
                decode_generator,
                response_metadata={
                    "id": prefill_output["id"],
                    "created": prefill_output["created"],
                    "model": prefill_output["model"],
                },
                include_role=False,
            )
            if is_chat
            else decode_generator
        )
        async for chunk in generator:
            yield chunk
    finally:
        reservation.release_all()


async def _zmq_nonstream_generator(
    prefill_output: dict,
    decode_generator,
    reservation: _RequestReservation,
    is_chat: bool = False,
):
    """Merge the P token into the non-streaming D response."""
    value = b""
    try:
        async for chunk in decode_generator:
            value += chunk
        output = json.loads(value)
        output["choices"][0]["text"] = (
            prefill_output["choices"][0]["text"] + output["choices"][0]["text"]
        )
        usage = output.get("usage")
        if isinstance(usage, dict):
            _merge_prefill_token_usage(usage)
        if is_chat:
            output = _completion_to_chat(output)
        yield json.dumps(output, separators=(",", ":")).encode()
    finally:
        reservation.release_all()


async def _zmq_prefill_only_generator(
    prefill_output: dict,
    reservation: _RequestReservation,
    is_chat: bool,
    stream: bool,
    include_usage: bool,
):
    """Return the prefill token when it exhausts the requested token budget."""
    output = prefill_output.copy()
    output.pop("kv_transfer_params", None)
    try:
        if not stream:
            if is_chat:
                output = _completion_to_chat(output)
            yield json.dumps(output, separators=(",", ":")).encode()
            return

        choice = output["choices"][0]
        head = {
            "id": output["id"],
            "object": "text_completion",
            "created": output["created"],
            "model": output["model"],
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "text": choice.get("text", ""),
                    "logprobs": choice.get("logprobs"),
                    "finish_reason": None,
                }
            ],
            "usage": None,
        }
        tail = {
            **head,
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "text": "",
                    "logprobs": None,
                    "finish_reason": choice.get("finish_reason") or "length",
                }
            ],
        }
        if is_chat:
            head = _completion_chunk_to_chat(head)
            head["choices"][0]["delta"]["role"] = "assistant"
            tail = _completion_chunk_to_chat(tail)
        for chunk in (head, tail):
            yield (f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n").encode()
        if include_usage and isinstance(output.get("usage"), dict):
            usage = {
                **tail,
                "choices": [],
                "usage": output["usage"],
            }
            yield (f"data: {json.dumps(usage, separators=(',', ':'))}\n\n").encode()
        yield b"data: [DONE]\n\n"
    finally:
        reservation.release_all()


async def handle_completion(
    raw_request: Request,
    *,
    endpoint: str,
    is_chat: bool,
    reserve: Callable[[SchedulingContext], Reservation | None],
    forward_request: ForwardRequest,
    count_tokens: CountTokens,
    get_tokenizer: Callable[[str], Tokenizer | None],
    tokenize_on_backend: TokenizeOnBackend,
    is_aggregated_model: Callable[[str], bool],
    options: CompletionOptions,
    get_registered_models: Callable[[], list[str]] | None = None,
    record_success: Callable[[str], None] | None = None,
    record_failure: Callable[[str], None] | None = None,
    notifications: Notifications | None = None,
    stream_generator: FirstTokenGenerator | None = None,
    nonstream_generator: FirstTokenGenerator = D_first_token_generator,
) -> JSONResponse | StreamingResponse:
    """Unified completion handler for both /v1/completions and /v1/chat/completions."""
    _metrics_start = track_request_start(endpoint)
    t_request_start = time.monotonic()
    handler_name = "create_chat_completion" if is_chat else "create_completion"
    t_prefill_done = None
    decode_tracker = None
    reservation = None
    try:
        try:
            request = await raw_request.json()
        except (json.JSONDecodeError, ValueError):
            track_request_end(endpoint, _metrics_start)
            return error_response("Invalid JSON in request body", INVALID_REQUEST, 400)

        error_resp = validate_completion_request(request, is_chat)
        if error_resp:
            track_request_end(endpoint, _metrics_start)
            return error_resp

        prefill_instance = None
        decode_instance = None

        start_time = time.time()
        total_length, max_tokens, prompt_text = extract_prompt_info(
            request, is_chat, count_tokens
        )
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        logger.info(
            "Completion request received",
            extra={
                "endpoint": endpoint,
                "prompt_length": total_length,
                "max_tokens": max_tokens,
                "tokenizer_ms": round(elapsed_ms, 2),
            },
        )

        requested_model = request.get("model", "")
        model_label = requested_model or "unknown"

        # Aggregated-role fast path: single forward, no P→D split
        if is_aggregated_model(requested_model):
            return await _handle_aggregated_completion(
                endpoint,
                request,
                raw_request,
                requested_model,
                total_length,
                max_tokens,
                prompt_text,
                _metrics_start,
                handler_name,
                reserve=reserve,
                forward_request=forward_request,
                get_registered_models=get_registered_models,
                record_success=record_success,
                record_failure=record_failure,
            )
        zmq_prompt_tokens = None
        if options.disaggregated_mode == "zmq":
            tokenizer = get_tokenizer(requested_model)
            if tokenizer is None:
                zmq_prompt_tokens = await tokenize_on_backend(request, is_chat)
            else:
                zmq_prompt_tokens = tokenize_zmq_prompt(request, is_chat, tokenizer)
            total_length = len(zmq_prompt_tokens)
        kv_prepare_request = build_kv_prepare_request(
            request,
            is_chat,
            options.disaggregated_mode,
        )
        upstream_headers = None
        if options.disaggregated_mode == "nixl":
            request_id = raw_request.headers.get("x-request-id") or str(uuid.uuid4())
            upstream_headers = {"X-Request-Id": request_id}

        _session_id = (
            raw_request.headers.get("x-session-id")
            or request.get("user")
            or (raw_request.client.host if raw_request.client else None)
        )
        _sched_kwargs = {
            "header": raw_request.headers.get("x-session-id"),
            "session_id": _session_id,
            "user": request.get("user"),
            "client_ip": (raw_request.client.host if raw_request.client else None),
            "prompt": prompt_text,
            "model": request.get("model", ""),
        }

        reservation = _RequestReservation()
        reservation.prefill = reserve(
            SchedulingContext(
                role="prefill",
                request_len=total_length,
                max_tokens=1,
                **_sched_kwargs,
            )
        )
        reservation.decode = reserve(
            SchedulingContext(
                role="decode",
                request_len=total_length,
                max_tokens=max_tokens,
                **_sched_kwargs,
            )
        )
        prefill_instance = reservation.prefill.address if reservation.prefill else None
        decode_instance = reservation.decode.address if reservation.decode else None

        if prefill_instance is None or decode_instance is None:
            logger.warning(
                "No available instance",
                extra={
                    "endpoint": endpoint,
                    "prompt_length": total_length,
                    "model": requested_model,
                },
            )
            track_request_end(endpoint, _metrics_start)
            reservation.release_all()
            # Check for unknown model first to return a clean 404 without
            # triggering error-path side effects (logging, metrics, etc.)
            if requested_model and get_registered_models is not None:
                known_models = get_registered_models()
                if requested_model not in known_models:
                    return error_response(
                        f"The model '{requested_model}' does not exist",
                        INVALID_REQUEST,
                        404,
                    )
            proxy_instance_errors_total.labels(
                instance=str(prefill_instance or decode_instance or "unknown"),
                error_type="no_available_instance",
                model=model_label,
            ).inc()
            return error_response(
                "No available instance can handle the request", PROXY_ERROR, 503
            )

        zmq_request_id = None
        zmq_wait_for_notification = False
        if options.disaggregated_mode == "zmq":
            if notifications is None or options.zmq is None:
                raise HTTPException(
                    status_code=503,
                    detail="ZMQ notification listener is not ready",
                )
            receiver = options.zmq.receivers[decode_instance]
            zmq_request_id = str(uuid.uuid4())
            assert zmq_prompt_tokens is not None
            kv_prepare_request = build_zmq_prepare_request(
                request,
                zmq_prompt_tokens,
                zmq_request_id,
                receiver,
                options.first_token_source,
                is_chat,
            )
            zmq_wait_for_notification = True
            await notifications.register(zmq_request_id)

        # Track per-instance request counters
        proxy_prefill_requests_total.labels(
            prefill_instance=prefill_instance,
            decode_instance=decode_instance,
            model=model_label,
        ).inc()
        proxy_prefill_queue_depth.labels(
            prefill_instance=prefill_instance,
            decode_instance=decode_instance,
            model=model_label,
        ).set(0)
        proxy_decode_requests_total.labels(
            prefill_instance=prefill_instance,
            decode_instance=decode_instance,
            model=model_label,
        ).inc()

        # Track active prefill requests
        proxy_prefill_active_requests.labels(
            prefill_instance=prefill_instance,
            decode_instance=decode_instance,
            model=model_label,
        ).inc()

        value = b""
        try:
            async for chunk in forward_request(
                f"http://{prefill_instance}"
                f"{'/v1/completions' if options.disaggregated_mode == 'zmq' else endpoint}",  # noqa: E501
                kv_prepare_request,
                extra_headers=upstream_headers,
            ):
                value += chunk
        except CancelledError:
            if zmq_wait_for_notification and zmq_request_id is not None:
                await notifications.discard(zmq_request_id)
            proxy_prefill_active_requests.labels(
                prefill_instance=prefill_instance,
                decode_instance=decode_instance,
                model=model_label,
            ).dec()
            raise
        except HTTPException as http_exc:
            if zmq_wait_for_notification and zmq_request_id is not None:
                await notifications.discard(zmq_request_id)
            reservation.release_all()
            _record_failures(record_failure, prefill_instance, decode_instance)
            proxy_instance_errors_total.labels(
                instance=prefill_instance,
                error_type="prefill_forward_error",
                model=model_label,
            ).inc()
            proxy_prefill_active_requests.labels(
                prefill_instance=prefill_instance,
                decode_instance=decode_instance,
                model=model_label,
            ).dec()
            raise http_exc

        t_prefill_done = time.monotonic()
        # Prefill complete — decrement active prefill, start active decode
        proxy_prefill_active_requests.labels(
            prefill_instance=prefill_instance,
            decode_instance=decode_instance,
            model=model_label,
        ).dec()
        proxy_decode_active_requests.labels(
            prefill_instance=prefill_instance,
            decode_instance=decode_instance,
            model=model_label,
        ).inc()

        value = (
            value.strip().decode("utf-8").removesuffix("data: [DONE]").encode("utf-8")
        )
        if options.disaggregated_mode == "nixl":
            try:
                prefill_output = json.loads(value)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=502,
                    detail="Prefill response did not contain valid JSON",
                ) from exc
            kv_transfer_params = prefill_output.get("kv_transfer_params")
            if not kv_transfer_params:
                raise HTTPException(
                    status_code=502,
                    detail="Prefill response did not contain kv_transfer_params",
                )
            request = request.copy()
            request["kv_transfer_params"] = kv_transfer_params
        elif options.disaggregated_mode == "zmq":
            prefill_output = None
            first_token = None
            if options.first_token_source == "prefill":
                try:
                    prefill_output = json.loads(value)
                    first_token = prefill_output["kv_transfer_params"]["first_tok"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    assert zmq_request_id is not None
                    await notifications.discard(zmq_request_id)
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            "Prefill response did not contain an LMCache " "first token"
                        ),
                    ) from exc
            receiver = options.zmq.receivers[decode_instance]
            if zmq_wait_for_notification:
                try:
                    await notifications.wait(
                        zmq_request_id,
                        len(receiver.init_ports),
                    )
                except TimeoutError as exc:
                    raise HTTPException(
                        status_code=504,
                        detail=(
                            "Timed out waiting for ZMQ notification "
                            f"{zmq_request_id}"
                        ),
                    ) from exc
            assert zmq_prompt_tokens is not None
            request = build_zmq_decode_request(
                request,
                zmq_prompt_tokens,
                first_token,
                max_tokens,
                is_chat,
            )

        if record_success is not None:
            record_success(prefill_instance)

        async def streaming_response(value):
            if value:
                yield value
            else:
                yield b""

        prefill_only = (
            options.disaggregated_mode == "zmq"
            and options.first_token_source == "prefill"
            and max_tokens == 1
        )
        if prefill_only:
            assert prefill_output is not None
            final_generator = _zmq_prefill_only_generator(
                prefill_output,
                reservation,
                is_chat,
                request.get("stream", False),
                request.get("stream_options", {}).get("include_usage", False),
            )
            first_token_from_p = True
        else:
            generator_p = streaming_response(value)
            # forward_request() is an async generator; HTTP errors
            # surface only when wrapped_generator() iterates it.
            generator_d_raw = forward_request(
                f"http://{decode_instance}"
                f"{'/v1/completions' if options.disaggregated_mode == 'zmq' and is_chat else endpoint}",  # noqa: E501
                request,
                extra_headers=upstream_headers,
            )
            decode_tracker = FirstTokenTracker(generator_d_raw)
            generator_d = decode_tracker

            if (
                options.disaggregated_mode == "zmq"
                and options.first_token_source == "prefill"
            ):
                assert prefill_output is not None
                if request.get("stream", False):
                    final_generator = _zmq_stream_generator(
                        prefill_output,
                        generator_d,
                        reservation,
                        is_chat,
                    )
                else:
                    final_generator = _zmq_nonstream_generator(
                        prefill_output,
                        generator_d,
                        reservation,
                        is_chat,
                    )
                first_token_from_p = True
            else:
                if options.disaggregated_mode == "zmq" and is_chat:
                    generator_d = (
                        _chat_completion_stream(generator_d)
                        if request.get("stream", False)
                        else _chat_completion_nonstream(generator_d)
                    )
                if request.get("stream", False):
                    generator_class = stream_generator or (
                        P_first_token_generator
                        if options.first_token_source == "prefill"
                        else D_first_token_generator
                    )
                else:
                    generator_class = nonstream_generator
                # Determine if user's first token comes from prefill or decode node.
                # P_first_token_generator yields P's token first;
                # D_first_token_generator
                # discards P's token and yields D's tokens only.
                first_token_from_p = (
                    options.first_token_source == "prefill"
                    and generator_class is P_first_token_generator
                )
                final_generator = generator_class(
                    generator_p,
                    generator_d,
                    reservation.release_prefill,
                    reservation.release_decode,
                )
        media_type = (
            "text/event-stream" if request.get("stream", False) else "application/json"
        )

        async def wrapped_generator():
            try:
                async for chunk in final_generator:
                    yield chunk
                if record_success is not None and not prefill_only:
                    record_success(decode_instance)
            except CancelledError:
                logger.warning(
                    "[0]Client disconnected during %s (CancelledError)",
                    handler_name,
                )
            except HTTPException as http_exc:
                reservation.release_all()
                _record_failures(record_failure, prefill_instance, decode_instance)
                proxy_instance_errors_total.labels(
                    instance=decode_instance,
                    error_type="decode_forward_error",
                    model=model_label,
                ).inc()
                logger.error(
                    "[1] HTTPException in wrapped_generator: %s", str(http_exc.detail)
                )
                raise
            except Exception as e:
                proxy_instance_errors_total.labels(
                    instance=decode_instance,
                    error_type="decode_forward_error",
                    model=model_label,
                ).inc()
                logger.error("[1] Exception in wrapped_generator: %s", str(e))
                raise
            finally:
                reservation.release_all()
                if (
                    prefill_instance
                    and decode_instance
                    and t_prefill_done is not None
                    and decode_tracker is not None
                ):
                    record_disaggregated_metrics(
                        prefill_instance=prefill_instance,
                        decode_instance=decode_instance,
                        model=model_label,
                        t_request_start=t_request_start,
                        t_prefill_done=t_prefill_done,
                        tracker=decode_tracker,
                        is_streaming=request.get("stream", False),
                        first_token_from_prefill=first_token_from_p,
                    )
                # Decrement decode active gauge (only if decode was started)
                if decode_instance and prefill_instance and t_prefill_done is not None:
                    proxy_decode_active_requests.labels(
                        prefill_instance=prefill_instance,
                        decode_instance=decode_instance,
                        model=model_label,
                    ).dec()
                track_request_end(endpoint, _metrics_start)

        return StreamingResponse(wrapped_generator(), media_type=media_type)
    except (HTTPException, CancelledError):
        if reservation is not None:
            reservation.release_all()
        if decode_instance and prefill_instance and t_prefill_done is not None:
            proxy_decode_active_requests.labels(
                prefill_instance=prefill_instance,
                decode_instance=decode_instance,
                model=model_label,
            ).dec()
        track_request_end(endpoint, _metrics_start)
        raise
    except Exception:
        if reservation is not None:
            reservation.release_all()
        if decode_instance and prefill_instance and t_prefill_done is not None:
            proxy_decode_active_requests.labels(
                prefill_instance=prefill_instance,
                decode_instance=decode_instance,
                model=model_label,
            ).dec()
        track_request_end(endpoint, _metrics_start)
        logger.error("Error in %s: %s", handler_name, sys.exc_info()[1])
        return JSONResponse(
            {"error": {"message": "Internal proxy error", "type": "proxy_error"}},
            status_code=500,
        )


async def _handle_aggregated_completion(
    endpoint: str,
    request: dict,
    raw_request: Request,
    model: str,
    total_length: int,
    max_tokens: int,
    prompt_text: str,
    metrics_start: float,
    handler_name: str,
    *,
    reserve: Callable[[SchedulingContext], Reservation | None],
    forward_request: ForwardRequest,
    get_registered_models: Callable[[], list[str]] | None,
    record_success: Callable[[str], None] | None,
    record_failure: Callable[[str], None] | None,
) -> JSONResponse | StreamingResponse:
    """Single-pass completion for aggregated-role instances."""
    reservation = reserve(
        SchedulingContext(
            role="aggregated",
            model=model,
            request_len=total_length,
            max_tokens=max_tokens,
            header=raw_request.headers.get("x-session-id"),
            session_id=request.get("session_id"),
            user=request.get("user"),
            client_ip=(raw_request.client.host if raw_request.client else None),
            prompt=prompt_text,
        )
    )
    instance = reservation.address if reservation is not None else None

    logger.info(
        "Aggregated %s request",
        handler_name,
        extra={
            "model": model,
            "prompt_length": total_length,
            "max_tokens": max_tokens,
            "prompt": prompt_text[:200] if prompt_text else "",
            "instance": instance,
        },
    )

    if instance is None:
        track_request_end(endpoint, metrics_start)
        # Check for unknown model
        if model and get_registered_models is not None:
            known_models = get_registered_models()
            if model not in known_models:
                return error_response(
                    f"The model '{model}' does not exist",
                    INVALID_REQUEST,
                    404,
                )
        return error_response(
            "No available instance can handle the request",
            PROXY_ERROR,
            503,
        )

    url = f"http://{instance}{endpoint}"

    if request.get("stream", False):
        generator = forward_request(url, request)

        async def wrapped():
            _ok = False
            try:
                async for chunk in generator:
                    yield chunk
                _ok = True
            except CancelledError:
                _ok = False
                logger.warning(
                    "Client disconnected during aggregated %s (CancelledError)",
                    handler_name,
                )
            except Exception as e:
                _ok = False
                logger.error("Exception in aggregated stream: %s", str(e))
                if record_failure is not None:
                    record_failure(instance)
                raise
            finally:
                reservation.release()
                if _ok and record_success is not None:
                    record_success(instance)
                track_request_end(endpoint, metrics_start)

        return StreamingResponse(wrapped(), media_type="text/event-stream")
    else:
        # Non-streaming: forward request using forward_request()
        # for consistent auth, timeouts, and connection handling.
        try:
            value = b""
            async for chunk in forward_request(url, request):
                value += chunk
            data = json.loads(value)
            # Detect error responses from upstream. forward_request yields
            # raw bytes without HTTP status, so inspect the parsed JSON.
            # OpenAI error format: {"error": {"message": ..., "type": ..., "code": ...}}
            # where "code" can be null, a string like "invalid_api_key", or an int.
            if "error" in data:
                err = data["error"]
                err_type = err.get("type", "")
                # Map error type to HTTP status code
                if err_type == "invalid_request_error":
                    status_code = 400
                elif err_type == "authentication_error":
                    status_code = 401
                elif err_type == "not_found_error":
                    status_code = 404
                elif err_type == "rate_limit_error":
                    status_code = 429
                else:
                    status_code = 502
            else:
                status_code = 200
            reservation.release()
            if status_code < 400 and record_success is not None:
                record_success(instance)
            elif status_code >= 400 and record_failure is not None:
                record_failure(instance)
            track_request_end(endpoint, metrics_start)
            return JSONResponse(data, status_code=status_code)
        except HTTPException as http_exc:
            reservation.release()
            if record_failure is not None:
                record_failure(instance)
            track_request_end(endpoint, metrics_start)
            return error_response(
                str(http_exc.detail),
                PROXY_ERROR,
                http_exc.status_code,
            )
        except CancelledError:
            reservation.release()
            raise
        except Exception as e:
            logger.error("Error in aggregated non-streaming: %s", str(e))
            reservation.release()
            if record_failure is not None:
                record_failure(instance)
            track_request_end(endpoint, metrics_start)
            return error_response(
                "Internal proxy error",
                PROXY_ERROR,
                500,
            )


async def validate_json_content_type(raw_request: Request) -> None:
    """Reject unsupported content types before invoking either completion handler."""
    # Dependencies must raise to short-circuit a request, including custom handlers.
    content_type = raw_request.headers.get("content-type", "").lower()
    if content_type != "application/json":
        raise HTTPException(
            status_code=415,
            detail="Unsupported Media Type: Only 'application/json' is allowed",
        )
