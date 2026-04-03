"""Unit tests for the unified completion handler helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.responses import JSONResponse

from xpyd.routes.completions import (
    build_kv_prepare_request,
    extract_prompt_info,
    handle_completion,
    validate_completion_request,
)


@pytest.fixture
def server():
    """Create a mock server with minimal attributes."""
    srv = MagicMock()
    srv.get_total_token_length = MagicMock(
        side_effect=lambda x: len(x) if isinstance(x, str) else 0
    )
    srv._is_dual_model = MagicMock(return_value=False)
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
            side_effect=lambda x: len(x) if isinstance(x, (str, list)) else 0
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


class TestHandleCompletion:
    """Integration-level tests for handle_completion."""

    @pytest.mark.asyncio
    async def test_invalid_json(self, server):
        raw_request = AsyncMock()
        raw_request.json = AsyncMock(side_effect=ValueError("bad json"))

        with (
            patch("xpyd.routes.completions.track_request_start", return_value=0),
            patch("xpyd.routes.completions.track_request_end"),
        ):
            result = await handle_completion(
                "/v1/completions", raw_request, server, is_chat=False
            )

        assert isinstance(result, JSONResponse)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_required_field(self, server):
        raw_request = AsyncMock()
        raw_request.json = AsyncMock(return_value={})

        with (
            patch("xpyd.routes.completions.track_request_start", return_value=0),
            patch("xpyd.routes.completions.track_request_end"),
        ):
            result = await handle_completion(
                "/v1/completions", raw_request, server, is_chat=False
            )

        assert isinstance(result, JSONResponse)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_no_available_instance(self, server):
        raw_request = AsyncMock()
        raw_request.json = AsyncMock(return_value={"prompt": "hello"})
        raw_request.headers = {}
        raw_request.client = None

        server.schedule = MagicMock(return_value=None)
        server.prefill_cycler = MagicMock()
        server.decode_cycler = MagicMock()
        server.exception_handler = MagicMock()

        with (
            patch("xpyd.routes.completions.track_request_start", return_value=0),
            patch("xpyd.routes.completions.track_request_end"),
            patch("xpyd.routes.completions.logger"),
        ):
            result = await handle_completion(
                "/v1/completions", raw_request, server, is_chat=False
            )

        assert isinstance(result, JSONResponse)
        assert result.status_code == 503
        server.exception_handler.assert_called_once()
