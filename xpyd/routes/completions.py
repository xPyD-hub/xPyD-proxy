# SPDX-License-Identifier: Apache-2.0
"""Completion route handlers."""

import json
import logging
import sys
import time
from asyncio import CancelledError

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from xpyd.errors import INVALID_REQUEST, PROXY_ERROR, error_response
from xpyd.metrics import track_request_end, track_request_start

logger = logging.getLogger("xpyd.proxy")


# ---------------------------------------------------------------------------
# Pure helper functions (no server dependency or explicit server param)
# ---------------------------------------------------------------------------


def validate_completion_request(request, is_chat):
    """Validate required fields. Returns JSONResponse on error, None on success."""
    if is_chat:
        if "messages" not in request:
            return error_response("Missing required field: messages", INVALID_REQUEST, 400)
        if not isinstance(request["messages"], list):
            return error_response("Field messages must be a list", INVALID_REQUEST, 400)
    else:
        if "prompt" not in request:
            return error_response("Missing required field: prompt", INVALID_REQUEST, 400)
    return None


def extract_prompt_info(request, is_chat, server):
    """Extract prompt metrics. Returns (total_length, max_tokens, prompt_text)."""
    if is_chat:
        total_length = 0
        prompt_parts = []
        for msg in request["messages"]:
            content = msg.get("content")
            if content is None:
                continue
            if isinstance(content, str):
                total_length += server.get_total_token_length(content)
                prompt_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        total_length += server.get_total_token_length(text)
                        prompt_parts.append(text)
        max_tokens = request.get("max_completion_tokens", 0)
        if max_tokens == 0:
            max_tokens = request.get("max_tokens", 0)
        prompt_text = " ".join(prompt_parts)
    else:
        prompt = request.get("prompt")
        total_length = server.get_total_token_length(prompt)
        max_tokens = request.get("max_tokens", 0)
        prompt_text = prompt if isinstance(prompt, str) else str(prompt)
    return total_length, max_tokens, prompt_text


def build_kv_prepare_request(request, is_chat):
    """Build the KV-prepare request with max_tokens=1."""
    kv_prepare_request = request.copy()
    kv_prepare_request["max_tokens"] = 1
    if is_chat:
        kv_prepare_request["max_completion_tokens"] = 1
    return kv_prepare_request


async def handle_completion(endpoint, raw_request, server, is_chat):
    """Unified completion handler for both /v1/completions and /v1/chat/completions."""
    _metrics_start = track_request_start(endpoint)
    handler_name = "create_chat_completion" if is_chat else "create_completion"
    try:
        try:
            request = await raw_request.json()
        except (json.JSONDecodeError, ValueError):
            return error_response("Invalid JSON in request body", INVALID_REQUEST, 400)

        error_resp = validate_completion_request(request, is_chat)
        if error_resp:
            return error_resp

        prefill_instance = None
        decode_instance = None

        kv_prepare_request = build_kv_prepare_request(request, is_chat)

        start_time = time.time()
        total_length, max_tokens, prompt_text = extract_prompt_info(
            request, is_chat, server
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

        _session_id = (
            raw_request.headers.get("x-session-id")
            or request.get("user")
            or (raw_request.client.host if raw_request.client else None)
        )
        _sched_kwargs = {
            "header": raw_request.headers.get("x-session-id"),
            "session_id": _session_id,
            "user": request.get("user"),
            "client_ip": (
                raw_request.client.host if raw_request.client else None
            ),
            "prompt": prompt_text,
            "model": request.get("model", ""),
        }

        prefill_instance = server.schedule(
            server.prefill_cycler,
            is_prompt=True,
            request_len=total_length,
            max_tokens=1,
            **_sched_kwargs,
        )

        decode_instance = server.schedule(
            server.decode_cycler,
            is_prompt=False,
            request_len=total_length,
            max_tokens=max_tokens,
            **_sched_kwargs,
        )

        if prefill_instance is None or decode_instance is None:
            logger.warning(
                "No available instance",
                extra={"endpoint": endpoint, "prompt_length": total_length},
            )
            server.exception_handler(
                prefill_instance=prefill_instance,
                decode_instance=decode_instance,
                req_len=total_length,
            )
            return error_response("No available instance can handle the request", PROXY_ERROR, 503)

        value = b""
        try:
            async for chunk in server.forward_request(
                f"http://{prefill_instance}{endpoint}", kv_prepare_request
            ):
                value += chunk
        except HTTPException as http_exc:
            server.exception_handler(prefill_instance, decode_instance, total_length)
            server._record_failure(prefill_instance, decode_instance)
            raise http_exc

        value = (
            value.strip().decode("utf-8").removesuffix("data: [DONE]").encode("utf-8")
        )

        async def streaming_response(value):
            if value:
                yield value
            else:
                yield b""

        generator_p = streaming_response(value)
        try:
            generator_d = server.forward_request(
                f"http://{decode_instance}{endpoint}", request
            )
        except HTTPException as http_exc:
            server.exception_handler(prefill_instance, decode_instance, total_length)
            server._record_failure(prefill_instance, decode_instance)
            raise http_exc

        if request.get("stream", False):
            generator_class = server.generator
        else:
            generator_class = server.d_first_token_generator_class
        final_generator = generator_class(
            generator_p,
            generator_d,
            server,
            prefill_instance,
            decode_instance,
            req_len=total_length,
        )
        media_type = (
            "text/event-stream"
            if request.get("stream", False)
            else "application/json"
        )

        async def wrapped_generator():
            try:
                async for chunk in final_generator:
                    yield chunk
            except CancelledError:
                logger.warning(
                    "[0]Client disconnected during %s (CancelledError)",
                    handler_name,
                )
            except Exception as e:
                logger.error("[1] Exception in wrapped_generator: %s", str(e))
                raise
            finally:
                track_request_end(endpoint, _metrics_start)

        return StreamingResponse(wrapped_generator(), media_type=media_type)
    except HTTPException:
        track_request_end(endpoint, _metrics_start)
        raise
    except Exception:
        track_request_end(endpoint, _metrics_start)
        logger.error("Error in %s: %s", handler_name, sys.exc_info()[1])
        return JSONResponse(
            {"error": {"message": "Internal proxy error", "type": "proxy_error"}},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register(router: APIRouter, server) -> None:
    """Register completion routes on *router*."""

    async def _validate_json(raw_request: Request):
        content_type = raw_request.headers.get("content-type", "").lower()
        if content_type != "application/json":
            raise HTTPException(
                status_code=415,
                detail="Unsupported Media Type: Only 'application/json' is allowed",
            )

    async def create_completion(raw_request: Request):
        return await handle_completion(
            "/v1/completions", raw_request, server, is_chat=False
        )

    async def create_chat_completion(raw_request: Request):
        return await handle_completion(
            "/v1/chat/completions", raw_request, server, is_chat=True
        )

    router.post(
        "/v1/completions",
        dependencies=[Depends(_validate_json)],
    )(
        server.custom_create_completion
        if server.custom_create_completion
        else create_completion
    )

    router.post(
        "/v1/chat/completions",
        dependencies=[Depends(_validate_json)],
    )(
        server.custom_create_chat_completion
        if server.custom_create_chat_completion
        else create_chat_completion
    )

    router.options("/v1/completions")(lambda: None)
    router.options("/v1/chat/completions")(lambda: None)
