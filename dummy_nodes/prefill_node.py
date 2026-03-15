"""Dummy Prefill Node – OpenAI-compatible FastAPI server.

Simulates the *prefill* phase of PD-separated inference:
  • Accepts a ChatCompletion request.
  • "Processes" the prompt (artificial latency proportional to prompt length).
  • Returns the **first token** plus a ``request_id`` that a decode node can
    pick up to continue generation.

Usage:
    uvicorn dummy_nodes.prefill_node:app --host 0.0.0.0 --port 8100
"""

import asyncio
import json
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from dummy_nodes.common import (
    DUMMY_TOKENS,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    DeltaMessage,
    StreamChoice,
    UsageInfo,
    count_prompt_tokens,
    generate_id,
    now_ts,
)

PREFILL_DELAY_PER_TOKEN: float = float(os.getenv("PREFILL_DELAY_PER_TOKEN", "0.001"))
"""Simulated per-prompt-token latency in seconds (default 1 ms)."""

app = FastAPI(title="Dummy Prefill Node")


def _build_non_stream_response(
    request: ChatCompletionRequest,
    request_id: str,
) -> ChatCompletionResponse:
    prompt_tokens = count_prompt_tokens(request.messages)
    max_tokens = request.max_tokens or 16
    completion_tokens = min(max_tokens, len(DUMMY_TOKENS))
    text = "".join(DUMMY_TOKENS[:completion_tokens])

    return ChatCompletionResponse(
        id=request_id,
        created=now_ts(),
        model=request.model,
        choices=[
            Choice(
                message=ChoiceMessage(content=text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _stream_generator(request: ChatCompletionRequest, request_id: str):
    """Yield SSE frames (``data: <json>\\n\\n``)."""
    max_tokens = request.max_tokens or 16

    # First chunk: role
    chunk = ChatCompletionChunk(
        id=request_id,
        created=now_ts(),
        model=request.model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"

    # Content chunks
    for i in range(min(max_tokens, len(DUMMY_TOKENS))):
        token = DUMMY_TOKENS[i]
        chunk = ChatCompletionChunk(
            id=request_id,
            created=now_ts(),
            model=request.model,
            choices=[StreamChoice(delta=DeltaMessage(content=token))],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        await asyncio.sleep(0)  # yield control

    # Final chunk
    chunk = ChatCompletionChunk(
        id=request_id,
        created=now_ts(),
        model=request.model,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    request_id = generate_id()

    # Simulate prefill latency
    prompt_tokens = count_prompt_tokens(request.messages)
    delay = prompt_tokens * PREFILL_DELAY_PER_TOKEN
    if delay > 0:
        await asyncio.sleep(delay)

    if request.stream:
        return StreamingResponse(
            _stream_generator(request, request_id),
            media_type="text/event-stream",
        )

    return _build_non_stream_response(request, request_id)


@app.get("/health")
async def health():
    return {"status": "ok", "node_type": "prefill"}
