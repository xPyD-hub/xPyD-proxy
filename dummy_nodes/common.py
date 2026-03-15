"""Common models and utilities for dummy prefill/decode nodes.

Provides OpenAI-compatible request/response models and shared helpers.
"""

import time
import uuid
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models (OpenAI Chat Completion API)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "dummy"
    messages: list[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, description="Maximum number of tokens to generate")
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False


# ---------------------------------------------------------------------------
# Non-streaming response models
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DUMMY_TOKENS = list("The quick brown fox jumps over the lazy dog. " * 20)


def generate_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def now_ts() -> int:
    return int(time.time())


def count_prompt_tokens(messages: list[ChatMessage]) -> int:
    """Rough token count: ~1 token per 4 characters."""
    total_chars = sum(len(m.content) + len(m.role) for m in messages)
    return max(1, total_chars // 4)
