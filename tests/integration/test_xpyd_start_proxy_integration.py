"""Real integration tests for xpyd_start_proxy.sh with dummy nodes.

These tests do not stop at validating the generated command string. Instead,
they start dummy prefill/decode nodes locally, launch the proxy through the
shell script itself, and then send real requests through the proxy.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "xpyd" / "xpyd_start_proxy.sh"
PYTHON = sys.executable
TOKENIZER_DIR = str(REPO_ROOT / "tests" / "assets" / "dummy_tokenizer")
ENV_BASE = {
    **os.environ,
    "PYTHONPATH": str(REPO_ROOT),
    "model_path": TOKENIZER_DIR,
    "DUMMY_MODEL_ID": TOKENIZER_DIR,
    "DUMMY_MAX_MODEL_LEN": "262144",
    "PREFILL_DELAY_PER_TOKEN": "0",
    "DECODE_DELAY_PER_TOKEN": "0",
    "NO_PROXY": "127.0.0.1,localhost",
    "no_proxy": "127.0.0.1,localhost",
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_http_ok(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=1.5)
            if response.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.2)
    raise AssertionError(f"Timed out waiting for {url}; last_error={last_error}")


def _spawn_node(module: str, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            PYTHON,
            "-m",
            "uvicorn",
            module,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=REPO_ROOT,
        env=ENV_BASE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _drain_process_output(process: subprocess.Popen) -> str:
    stdout = ""
    stderr = ""
    try:
        if process.stdout:
            stdout = process.stdout.read() or ""
        if process.stderr:
            stderr = process.stderr.read() or ""
    except Exception:
        pass
    return f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"


def _launch_proxy_via_script(
    prefill_ports: list[int], decode_ports: list[int], proxy_port: int
):
    env = {
        **ENV_BASE,
        "XPYD_PREFILL_IPS": " ".join(["127.0.0.1"] * len(prefill_ports)),
        "XPYD_DECODE_IPS": " ".join(["127.0.0.1"] * len(decode_ports)),
        "XPYD_PROXY_PORT": str(proxy_port),
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "http_proxy": "",
        "https_proxy": "",
    }
    command = [
        "bash",
        str(SCRIPT),
        "-pn",
        str(len(prefill_ports)),
        "-pt",
        "8",
        "-pd",
        str(len(prefill_ports)),
        "-pw",
        "8",
        "-dn",
        str(len(decode_ports)),
        "-dt",
        "8",
        "-dd",
        str(len(decode_ports)),
        "-dw",
        "8",
        "--prefill-base-port",
        str(prefill_ports[0]),
        "--decode-base-port",
        str(decode_ports[0]),
    ]
    return subprocess.Popen(
        command,
        cwd=REPO_ROOT / "xpyd",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.mark.parametrize("prefill_count,decode_count", [(1, 2), (2, 2)])
def test_xpyd_start_proxy_launches_real_proxy_with_dummy_nodes(
    prefill_count: int, decode_count: int
):
    prefill_ports = [_free_port() for _ in range(prefill_count)]
    decode_ports = [_free_port() for _ in range(decode_count)]
    proxy_port = _free_port()

    with ExitStack() as stack:
        prefill_processes = []
        decode_processes = []
        for port in prefill_ports:
            process = _spawn_node("dummy_nodes.prefill_node:app", port)
            prefill_processes.append(process)
            stack.callback(_stop_process, process)
        for port in decode_ports:
            process = _spawn_node("dummy_nodes.decode_node:app", port)
            decode_processes.append(process)
            stack.callback(_stop_process, process)

        for port in prefill_ports:
            _wait_http_ok(f"http://127.0.0.1:{port}/v1/models")
        for port in decode_ports:
            _wait_http_ok(f"http://127.0.0.1:{port}/v1/models")

        proxy = _launch_proxy_via_script(prefill_ports, decode_ports, proxy_port)
        stack.callback(_stop_process, proxy)

        try:
            _wait_http_ok(f"http://127.0.0.1:{proxy_port}/status")
        except AssertionError:
            pytest.fail(_drain_process_output(proxy))

        status = requests.get(f"http://127.0.0.1:{proxy_port}/status", timeout=5).json()
        assert status["prefill_node_count"] == prefill_count
        assert status["decode_node_count"] == decode_count

        payload = {
            "model": TOKENIZER_DIR,
            "messages": [{"role": "user", "content": "integration via shell script"}],
            "max_tokens": 4,
            "stream": False,
        }
        response = requests.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json=payload,
            timeout=15,
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["usage"]["completion_tokens"] == 4

        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_response = requests.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            json=stream_payload,
            timeout=15,
        )
        assert stream_response.status_code == 200, stream_response.text
        assert "data: [DONE]" in stream_response.text

        for process in [*prefill_processes, *decode_processes, proxy]:
            if process.poll() not in (None, 0):
                pytest.fail(_drain_process_output(process))
