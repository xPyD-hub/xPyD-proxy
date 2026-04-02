"""Unit tests for the unified completion handler helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.responses import JSONResponse

from core.MicroPDProxyServer import Proxy


@pytest.fixture
def server():
    """Create a Proxy with minimal mocking."""
    with patch("core.MicroPDProxyServer.Proxy.__init__", return_value=None):
        srv = Proxy.__new__(Proxy)
    srv.get_total_token_length = MagicMock(side_effect=lambda x: len(x) if isinstance(x, str) else 0)
    return srv


class TestValidateCompletionRequest:
    """Tests for _validate_completion_request."""

    def test_completion_valid(self, server):
        result = server._validate_completion_request({"prompt": "hello"}, is_chat=False)
        assert result is None

    def test_completion_missing_prompt(self, server):
        result = server._validate_completion_request({}, is_chat=False)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400

    def test_chat_valid(self, server):
        result = server._validate_completion_request(
            {"messages": [{"role": "user", "content": "hi"}]}, is_chat=True
        )
        assert result is None

    def test_chat_missing_messages(self, server):
        result = server._validate_completion_request({}, is_chat=True)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400

    def test_chat_messages_not_list(self, server):
        result = server._validate_completion_request({"messages": "bad"}, is_chat=True)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400


class TestExtractPromptInfo:
    """Tests for _extract_prompt_info."""

    def test_completion_string_prompt(self, server):
        total_length, max_tokens, prompt_text = server._extract_prompt_info(
            {"prompt": "hello world", "max_tokens": 50}, is_chat=False
        )
        assert total_length == 11  # len("hello world")
        assert max_tokens == 50
        assert prompt_text == "hello world"

    def test_completion_list_prompt(self, server):
        total_length, max_tokens, prompt_text = server._extract_prompt_info(
            {"prompt": [1, 2, 3], "max_tokens": 10}, is_chat=False
        )
        assert total_length == 0  # MagicMock returns 0 for non-str
        assert max_tokens == 10
        assert prompt_text == "[1, 2, 3]"

    def test_completion_default_max_tokens(self, server):
        _, max_tokens, _ = server._extract_prompt_info(
            {"prompt": "test"}, is_chat=False
        )
        assert max_tokens == 0

    def test_chat_basic(self, server):
        total_length, max_tokens, prompt_text = server._extract_prompt_info(
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 100,
            },
            is_chat=True,
        )
        assert total_length == len("You are helpful") + len("Hello")
        assert max_tokens == 100
        assert "You are helpful" in prompt_text
        assert "Hello" in prompt_text

    def test_chat_max_completion_tokens_priority(self, server):
        _, max_tokens, _ = server._extract_prompt_info(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 200,
                "max_tokens": 100,
            },
            is_chat=True,
        )
        assert max_tokens == 200

    def test_chat_fallback_to_max_tokens(self, server):
        _, max_tokens, _ = server._extract_prompt_info(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
            is_chat=True,
        )
        assert max_tokens == 100

    def test_chat_mixed_content_types(self, server):
        """Non-string content should be excluded from prompt_text."""
        _, _, prompt_text = server._extract_prompt_info(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": None},
                    {"role": "user", "content": "world"},
                ],
            },
            is_chat=True,
        )
        assert "hello" in prompt_text
        assert "world" in prompt_text
        assert "None" not in prompt_text


class TestBuildKvPrepareRequest:
    """Tests for _build_kv_prepare_request."""

    def test_completion(self, server):
        req = {"prompt": "test", "max_tokens": 50}
        result = server._build_kv_prepare_request(req, is_chat=False)
        assert result["max_tokens"] == 1
        assert "max_completion_tokens" not in result
        # Original should not be modified
        assert req["max_tokens"] == 50

    def test_chat(self, server):
        req = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50}
        result = server._build_kv_prepare_request(req, is_chat=True)
        assert result["max_tokens"] == 1
        assert result["max_completion_tokens"] == 1
        # Original should not be modified
        assert req["max_tokens"] == 50


class TestHandleCompletion:
    """Integration-level tests for _handle_completion."""

    @pytest.mark.asyncio
    async def test_invalid_json(self, server):
        raw_request = AsyncMock()
        raw_request.json = AsyncMock(side_effect=ValueError("bad json"))

        with patch("core.MicroPDProxyServer.track_request_start", return_value=0), \
             patch("core.MicroPDProxyServer.track_request_end"):
            result = await server._handle_completion("/v1/completions", raw_request, is_chat=False)

        assert isinstance(result, JSONResponse)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_required_field(self, server):
        raw_request = AsyncMock()
        raw_request.json = AsyncMock(return_value={})

        with patch("core.MicroPDProxyServer.track_request_start", return_value=0), \
             patch("core.MicroPDProxyServer.track_request_end"):
            result = await server._handle_completion("/v1/completions", raw_request, is_chat=False)

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

        with patch("core.MicroPDProxyServer.track_request_start", return_value=0), \
             patch("core.MicroPDProxyServer.track_request_end"), \
             patch("core.MicroPDProxyServer.log_info_green"), \
             patch("core.MicroPDProxyServer.log_info_red"):
            result = await server._handle_completion("/v1/completions", raw_request, is_chat=False)

        assert result is None
        server.exception_handler.assert_called_once()
