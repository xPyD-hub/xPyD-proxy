"""A 4xx response alone must not pass the CPU forwarding smoke check."""

import io
import runpy
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

VALIDATE = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "examples/lib/validate_forwarding.py")
)["validate"]


@pytest.mark.parametrize("status", [200, 400, 404, 422])
def test_backend_access_log_proves_forwarding(tmp_path, status):
    log = tmp_path / "prefill-1.log"
    log.write_text("")

    def response(request, timeout):
        with log.open("a") as stream:
            stream.write(f'INFO: client - "POST /pooling HTTP/1.1" {status} Error\n')
        if status == 200:
            result = io.BytesIO(b"backend")
            result.status = status
            return result
        raise urllib.error.HTTPError(
            request.full_url, status, "response", {}, io.BytesIO(b"backend")
        )

    with patch("urllib.request.urlopen", response):
        VALIDATE("http://proxy", "/pooling", "{}", tmp_path)


@pytest.mark.parametrize("status", [400, 404, 422])
def test_proxy_error_and_old_log_do_not_prove_forwarding(tmp_path, status):
    (tmp_path / "prefill.log").write_text(
        f'INFO: client - "POST /pooling HTTP/1.1" {status} Error\n'
    )
    response = urllib.error.HTTPError(
        "http://proxy/pooling", status, "error", {}, io.BytesIO(b"proxy error")
    )
    with patch("urllib.request.urlopen", side_effect=response):
        with pytest.raises(AssertionError, match="without a new matching"):
            VALIDATE("http://proxy", "/pooling", "{}", tmp_path, timeout=0.01)


def test_missing_logs_fail_before_request(tmp_path):
    with patch("urllib.request.urlopen") as request:
        with pytest.raises(AssertionError, match="No backend access logs"):
            VALIDATE("http://proxy", "/pooling", "{}", tmp_path)
        request.assert_not_called()
