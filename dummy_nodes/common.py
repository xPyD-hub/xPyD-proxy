"""Common models and helpers for dummy prefill/decode nodes.

The goal of these dummy nodes is compatibility with the proxy in ``core/``,
not perfect protocol coverage. The helpers below implement the subset of the
OpenAI-compatible API that the proxy depends on for local debugging:

- ``/v1/models``
- ``/v1/completions``
- ``/v1/chat/completions``
- ``/health``
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = "dummy"
    messages: list[ChatMessage]
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate.",
    )
    max_completion_tokens: Optional[int] = Field(
        default=None,
        description="Alias used by some chat-completions clients.",
    )
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str = "dummy"
    prompt: Any
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate.",
    )
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str = ""


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = "dummy"
    choices: list[Choice] = []
    usage: UsageInfo = UsageInfo()


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    id: str = ""
    object: str = "text_completion"
    created: int = 0
    model: str = "dummy"
    choices: list[CompletionChoice] = []
    usage: UsageInfo = UsageInfo()


# ---------------------------------------------------------------------------
# Streaming (SSE) response models
# ---------------------------------------------------------------------------

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = "dummy"
    choices: list[StreamChoice] = []


class CompletionStreamChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: Optional[str] = None


class CompletionChunk(BaseModel):
    id: str = ""
    object: str = "text_completion"
    created: int = 0
    model: str = "dummy"
    choices: list[CompletionStreamChoice] = []


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "dummy"
    max_model_len: int = 131072


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DUMMY_TOKENS = list("The quick brown fox jumps over the lazy dog. " * 20)
DEFAULT_MAX_TOKENS = 16


def generate_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def now_ts() -> int:
    return int(time.time())


def get_model_id() -> str:
    return os.getenv("DUMMY_MODEL_ID", "dummy")


def get_max_model_len() -> int:
    return int(os.getenv("DUMMY_MAX_MODEL_LEN", "131072"))


def get_effective_max_tokens(*values: Optional[int]) -> int:
    for value in values:
        if value is not None:
            return value
    return DEFAULT_MAX_TOKENS


def count_prompt_tokens_from_messages(messages: list[ChatMessage]) -> int:
    total_chars = 0
    for message in messages:
        total_chars += len(message.role)
        content = message.content
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    total_chars += len(str(item["text"]))
                else:
                    total_chars += len(str(item))
        else:
            total_chars += len(str(content))
    return max(1, total_chars // 4)


def count_prompt_tokens_from_prompt(prompt: Any) -> int:
    if isinstance(prompt, str):
        return max(1, len(prompt) // 4)
    if isinstance(prompt, list):
        if all(isinstance(item, str) for item in prompt):
            return max(1, sum(len(item) for item in prompt) // 4)
        if all(isinstance(item, int) for item in prompt):
            return len(prompt)
        if all(isinstance(item, list) for item in prompt):
            return sum(len(item) for item in prompt)
        if all(isinstance(item, dict) and "text" in item for item in prompt):
            return max(1, sum(len(str(item["text"])) for item in prompt) // 4)
    return max(1, len(str(prompt)) // 4)


def render_dummy_text(max_tokens: int) -> str:
    return "".join(DUMMY_TOKENS[: min(max_tokens, len(DUMMY_TOKENS))])


def build_models_response() -> ModelListResponse:
    return ModelListResponse(
        data=[
            ModelCard(
                id=get_model_id(),
                created=now_ts(),
                max_model_len=get_max_model_len(),
            )
        ]
    )
