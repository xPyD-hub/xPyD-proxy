"""Dummy prefill node compatible with the proxy in ``core/``.

This node intentionally implements only the subset of endpoints the proxy uses
for local debugging. It can serve both ``/v1/completions`` and
``/v1/chat/completions`` as well as ``/v1/models`` for startup validation.
"""

from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from dummy_nodes.common import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    CompletionChunk,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    StreamChoice,
    UsageInfo,
    build_models_response,
    count_prompt_tokens_from_messages,
    count_prompt_tokens_from_prompt,
    generate_id,
    get_effective_max_tokens,
    now_ts,
    render_dummy_text,
)

PREFILL_DELAY_PER_TOKEN: float = float(os.getenv("PREFILL_DELAY_PER_TOKEN", "0.001"))

app = FastAPI(title="Dummy Prefill Node")


async def _sleep_for_messages(request: ChatCompletionRequest) -> int:
    prompt_tokens = count_prompt_tokens_from_messages(request.messages)
    delay = prompt_tokens * PREFILL_DELAY_PER_TOKEN
    if delay > 0:
        await asyncio.sleep(delay)
    return prompt_tokens


async def _sleep_for_prompt(request: CompletionRequest) -> int:
    prompt_tokens = count_prompt_tokens_from_prompt(request.prompt)
    delay = prompt_tokens * PREFILL_DELAY_PER_TOKEN
    if delay > 0:
        await asyncio.sleep(delay)
    return prompt_tokens


def _build_chat_response(request: ChatCompletionRequest, request_id: str) -> ChatCompletionResponse:
    prompt_tokens = count_prompt_tokens_from_messages(request.messages)
    max_tokens = get_effective_max_tokens(request.max_completion_tokens, request.max_tokens)
    completion_tokens = min(max_tokens, len(render_dummy_text(max_tokens)))
    text = render_dummy_text(max_tokens)
    return ChatCompletionResponse(
        id=request_id,
        created=now_ts(),
        model=request.model,
        choices=[Choice(message=ChoiceMessage(content=text), finish_reason="stop")],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=max_tokens,
            total_tokens=prompt_tokens + max_tokens,
        ),
    )


def _build_completion_response(request: CompletionRequest, request_id: str) -> CompletionResponse:
    prompt_tokens = count_prompt_tokens_from_prompt(request.prompt)
    max_tokens = get_effective_max_tokens(request.max_tokens)
    text = render_dummy_text(max_tokens)
    return CompletionResponse(
        id=request_id,
        created=now_ts(),
        model=request.model,
        choices=[CompletionChoice(text=text, finish_reason="stop")],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=max_tokens,
            total_tokens=prompt_tokens + max_tokens,
        ),
    )


async def _chat_stream(request: ChatCompletionRequest, request_id: str):
    max_tokens = get_effective_max_tokens(request.max_completion_tokens, request.max_tokens)
    yield f"data: {ChatCompletionChunk(id=request_id, created=now_ts(), model=request.model, choices=[StreamChoice(delta=DeltaMessage(role='assistant'))]).model_dump_json()}\n\n"
    text = render_dummy_text(max_tokens)
    for token in text:
        chunk = ChatCompletionChunk(
            id=request_id,
            created=now_ts(),
            model=request.model,
            choices=[StreamChoice(delta=DeltaMessage(content=token))],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0)
    chunk = ChatCompletionChunk(
        id=request_id,
        created=now_ts(),
        model=request.model,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def _completion_stream(request: CompletionRequest, request_id: str):
    max_tokens = get_effective_max_tokens(request.max_tokens)
    text = render_dummy_text(max_tokens)
    for token in text:
        chunk = CompletionChunk(
            id=request_id,
            created=now_ts(),
            model=request.model,
            choices=[CompletionChoice(text=token)],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0)
    finish = CompletionChunk(
        id=request_id,
        created=now_ts(),
        model=request.model,
        choices=[CompletionChoice(text="", finish_reason="stop")],
    )
    yield f"data: {finish.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def get_models():
    return build_models_response()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    request_id = generate_id("chatcmpl")
    await _sleep_for_messages(request)
    if request.stream:
        return StreamingResponse(_chat_stream(request, request_id), media_type="text/event-stream")
    return _build_chat_response(request, request_id)


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    request_id = generate_id("cmpl")
    await _sleep_for_prompt(request)
    if request.stream:
        return StreamingResponse(_completion_stream(request, request_id), media_type="text/event-stream")
    return _build_completion_response(request, request_id)


@app.get("/health")
async def health():
    return {"status": "ok", "node_type": "prefill"}


@app.get("/ping", response_class=PlainTextResponse)
@app.post("/ping", response_class=PlainTextResponse)
async def ping():
    return "pong"
