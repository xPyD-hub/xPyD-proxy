"""Integration tests for the proxy/dummy-node matrix from task_openclaw.md.

These tests intentionally exercise the real ``xpyd/proxy.py``
server with multiple dummy prefill/decode nodes and the requested proxy
configurations, without changing the core business logic.
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
PYTHON = sys.executable
TOKENIZER_DIR = str(REPO_ROOT / "tests" / "assets" / "dummy_tokenizer")
DUMMY_MODEL_ID = TOKENIZER_DIR
ENV = {
    **os.environ,
    "PYTHONPATH": str(REPO_ROOT),
    "DUMMY_MODEL_ID": DUMMY_MODEL_ID,
    "DUMMY_MAX_MODEL_LEN": "262144",
    "PREFILL_DELAY_PER_TOKEN": "0",
    "DECODE_DELAY_PER_TOKEN": "0",
}

MATRIX = [
    (1, 2, 1),
    (2, 2, 1),
    (1, 2, 2),
    (1, 2, 4),
    (1, 2, 8),
    (2, 2, 2),
    (2, 4, 1),
    (2, 4, 2),
]


_used_ports: set[int] = set()


def _free_port() -> int:
    """Find a free TCP port, avoiding previously allocated ports."""
    for _ in range(100):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in _used_ports:
            _used_ports.add(port)
            return port
    raise RuntimeError("Unable to find a unique free port")


def _wait_http_ok(url: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=1.5)
            if response.status_code == 200:
                return
        except Exception as exc:  # pragma: no cover - best effort polling
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
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _spawn_proxy(
    prefill_instances: list[str], decode_instances: list[str], port: int
) -> subprocess.Popen:
    # Generate a temporary YAML config for the proxy
    import tempfile

    import yaml

    config = {
        "model": TOKENIZER_DIR,
        "port": port,
        "decode": decode_instances,
    }
    if prefill_instances:
        config["prefill"] = prefill_instances

    config_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(REPO_ROOT)
    )
    yaml.dump(config, config_file)
    config_file.close()

    command = [
        PYTHON,
        "-m",
        "xpyd.proxy",
        "proxy",
        "--config",
        config_file.name,
    ]
    return subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=ENV,
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


@pytest.mark.parametrize("prefill_count,decode_count,tp_size", MATRIX)
def test_proxy_matrix(prefill_count: int, decode_count: int, tp_size: int):
    num_decode_ports = 8 // tp_size
    prefill_ports = [_free_port() for _ in range(prefill_count)]
    decode_ports = [_free_port() for _ in range(decode_count * num_decode_ports)]
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

        prefill_instances = [f"127.0.0.1:{port}" for port in prefill_ports]
        decode_instances = [f"127.0.0.1:{port}" for port in decode_ports]
        proxy = _spawn_proxy(prefill_instances, decode_instances, proxy_port)
        stack.callback(_stop_process, proxy)

        try:
            _wait_http_ok(f"http://127.0.0.1:{proxy_port}/status")
        except AssertionError:
            details = ["Proxy failed to start"]
            details.append(_drain_process_output(proxy))
            for process in [*prefill_processes, *decode_processes]:
                if process.poll() not in (None, 0):
                    details.append(_drain_process_output(process))
            pytest.fail("\n".join(details))

        status = requests.get(f"http://127.0.0.1:{proxy_port}/status", timeout=5).json()
        assert status["prefill_node_count"] == prefill_count
        assert status["decode_node_count"] == decode_count * num_decode_ports

        payload = {
            "model": DUMMY_MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": f"matrix {prefill_count}-{decode_count}-{tp_size}",
                }
            ],
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

        # Fail fast if any child process crashed during the request.
        for process in [*prefill_processes, *decode_processes, proxy]:
            if process.poll() not in (None, 0):
                pytest.fail(_drain_process_output(process))
