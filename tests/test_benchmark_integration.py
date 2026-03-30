"""Integration test that mimics vllm bench serve against proxy + dummy nodes.

Topology (matches benchmarks/run_benchmark.sh):
  - 2 prefill nodes  (ports 8100-8101)
  - 16 decode nodes  (ports 8200-8215)
  - 1 proxy          (port 8868)

Uses standard port range matching benchmarks/run_benchmark.sh.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_REPO_ROOT, "tokenizers", "DeepSeek-R1")

NUM_PREFILL = 2
NUM_DECODE = 16
PREFILL_BASE = 8100
DECODE_BASE = 8200
PROXY_PORT = 8868


def _wait_port(port: int, timeout: float = 20.0) -> bool:
    """Wait until a port is accepting connections."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def cluster():
    """Start dummy nodes + proxy, yield, then tear down."""
    env = os.environ.copy()
    env["DUMMY_MODEL_ID"] = MODEL_PATH
    procs = []

    # Start prefill nodes
    for i in range(NUM_PREFILL):
        port = PREFILL_BASE + i
        p = subprocess.Popen(
            [sys.executable, "-m", "uvicorn",
             "dummy_nodes.prefill_node:app",
             "--host", "127.0.0.1", "--port", str(port),
             "--log-level", "error"],
            env=env, cwd=_REPO_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(p)

    # Start decode nodes
    for i in range(NUM_DECODE):
        port = DECODE_BASE + i
        p = subprocess.Popen(
            [sys.executable, "-m", "uvicorn",
             "dummy_nodes.decode_node:app",
             "--host", "127.0.0.1", "--port", str(port),
             "--log-level", "error"],
            env=env, cwd=_REPO_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        procs.append(p)

    # Wait for nodes
    for i in range(NUM_PREFILL):
        assert _wait_port(PREFILL_BASE + i), f"Prefill {PREFILL_BASE + i} didn't start"
    for i in range(NUM_DECODE):
        assert _wait_port(DECODE_BASE + i), f"Decode {DECODE_BASE + i} didn't start"

    # Start proxy
    prefill_args = [f"127.0.0.1:{PREFILL_BASE + i}" for i in range(NUM_PREFILL)]
    decode_args = [f"127.0.0.1:{DECODE_BASE + i}" for i in range(NUM_DECODE)]

    proxy = subprocess.Popen(
        [sys.executable, "core/MicroPDProxyServer.py",
         "--model", MODEL_PATH,
         "--prefill", *prefill_args,
         "--decode", *decode_args,
         "--port", str(PROXY_PORT)],
        env=env, cwd=_REPO_ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    procs.append(proxy)
    assert _wait_port(PROXY_PORT), "Proxy didn't start"

    yield {"proxy_port": PROXY_PORT, "model": MODEL_PATH}

    # Teardown
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait(timeout=5)


CHAT_PAYLOAD = {
    "model": "",  # will be set in tests
    "messages": [{"role": "user", "content": "Hello world"}],
    "max_tokens": 5,
    "stream": False,
}


def test_models_endpoint(cluster):
    """Proxy /v1/models should return model info from backend nodes."""
    with httpx.Client(base_url=f"http://127.0.0.1:{cluster['proxy_port']}", timeout=10) as c:
        r = c.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 0  # proxy returns per-instance results


def test_chat_completions(cluster):
    """Non-streaming chat completions through proxy."""
    payload = {**CHAT_PAYLOAD, "model": cluster["model"]}
    with httpx.Client(base_url=f"http://127.0.0.1:{cluster['proxy_port']}", timeout=30) as c:
        r = c.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]


def test_chat_completions_streaming(cluster):
    """Streaming chat completions through proxy."""
    payload = {**CHAT_PAYLOAD, "model": cluster["model"], "stream": True}
    with httpx.Client(base_url=f"http://127.0.0.1:{cluster['proxy_port']}", timeout=30) as c:
        r = c.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        lines = r.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 2  # at least some chunks + [DONE]
        assert data_lines[-1] == "data: [DONE]"


def test_status_topology(cluster):
    """Proxy status should reflect correct topology."""
    with httpx.Client(base_url=f"http://127.0.0.1:{cluster['proxy_port']}", timeout=10) as c:
        r = c.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert data["prefill_node_count"] == NUM_PREFILL
        assert data["decode_node_count"] == NUM_DECODE


def test_concurrent_requests(cluster):
    """Multiple concurrent requests should all succeed."""
    import concurrent.futures

    payload = {**CHAT_PAYLOAD, "model": cluster["model"]}
    results = []

    def send_request(i):
        with httpx.Client(base_url=f"http://127.0.0.1:{cluster['proxy_port']}", timeout=30) as c:
            r = c.post("/v1/chat/completions", json=payload)
            return r.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(send_request, i) for i in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(code == 200 for code in results), f"Some requests failed: {results}"
