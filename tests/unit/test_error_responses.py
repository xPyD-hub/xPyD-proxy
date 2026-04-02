"""Unit tests for xpyd.errors module."""

from xpyd.errors import INVALID_REQUEST, PROXY_ERROR, SERVER_ERROR, error_response


class TestErrorResponse:
    """Tests for the error_response helper."""

    def test_default_server_error(self):
        resp = error_response("something broke")
        assert resp.status_code == 500
        assert resp.body is not None

    def test_custom_status_code(self):
        resp = error_response("not found", SERVER_ERROR, 404)
        assert resp.status_code == 404

    def test_invalid_request(self):
        resp = error_response("bad field", INVALID_REQUEST, 400)
        assert resp.status_code == 400

    def test_proxy_error(self):
        resp = error_response("no instance", PROXY_ERROR, 503)
        assert resp.status_code == 503

    def test_response_body_format(self):
        import json

        resp = error_response("test message", INVALID_REQUEST, 422)
        body = json.loads(resp.body)
        assert "error" in body
        assert body["error"]["message"] == "test message"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] is None

    def test_constants(self):
        assert INVALID_REQUEST == "invalid_request_error"
        assert SERVER_ERROR == "server_error"
        assert PROXY_ERROR == "proxy_error"
