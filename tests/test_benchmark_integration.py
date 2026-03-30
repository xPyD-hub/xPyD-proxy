"""Integration test that mimics vllm bench serve against proxy + dummy nodes.

Topology (matches benchmarks/run_benchmark.sh):
  - 2 prefill nodes  (ports 18100-18101)
  - 16 decode nodes  (ports 18200-18215)
  - 1 proxy          (port 18868)

Uses higher port range (18xxx) to avoid clashing with manual runs.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import time
from typing import Generator

import httpx
import pytest
import uvicorn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_PREFILL = 2
NUM_DECODE = 16
PREFILL_BASE = 18100
DECODE_BASE = 18200
PROXY_PORT = 18868

MODEL_PATH = "tokenizers/DeepSeek-R1"


def _run_uvicorn(app_path: str, port: int) -> None:
    """Run a uvicorn server in a subprocess."""
    uvicorn.run(app_path, host="127.0.0.1", port=port, log_level="warning")


def _run_proxy() -> None:
    """Run the proxy server in a subprocess."""
    import sys, os

    # Ensure repo root is on sys.path so core/ is importable
    repo = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, repo)

    prefill = [f"127.0.0.1:{PREFILL_BASE + i}" for i in range(NUM_PREFILL)]
    decode = [f"127.0.0.1:{DECODE_BASE + i}" for i in range(NUM_DECODE)]

    # Build argv as if invoked from CLI
    sys.argv = [
        "MicroPDProxyServer.py",
        "--model", MODEL_PATH,
        "--prefill", *prefill,
        "--decode", *decode,
        "--port", str(PROXY_PORT),
    ]

    # Import and run — the proxy script typically calls uvicorn.run() in __main__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "proxy", os.path.join(repo, "core", "MicroPDProxyServer.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def _wait_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with httpx.Client() as c:
                r = c.get(f"http://127.0.0.1:{port}/ping", timeout=1)
                if r.status_code == 200:
                    return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"Port {port} not ready after {timeout}s")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cluster() -> Generator[None, None, None]:
    """Start dummy nodes + proxy, yield, then tear down."""
    procs: list[multiprocessing.Process] = []

    # Prefill nodes
    for i in range(NUM_PREFILL):
        p = multiprocessing.Process(
            target=_run_uvicorn,
            args=("dummy_nodes.prefill_node:app", PREFILL_BASE + i),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # Decode nodes
    for i in range(NUM_DECODE):
        p = multiprocessing.Process(
            target=_run_uvicorn,
            args=("dummy_nodes.decode_node:app", DECODE_BASE + i),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # Wait for nodes
    for i in range(NUM_PREFILL):
        _wait_port(PREFILL_BASE + i)
    for i in range(NUM_DECODE):
        _wait_port(DECODE_BASE + i)

    # Proxy
    proxy_proc = multiprocessing.Process(target=_run_proxy, daemon=True)
    proxy_proc.start()
    procs.append(proxy_proc)
    time.sleep(3)  # give proxy time to start

    yield

    for p in procs:
        p.terminate()
    for p in procs:
        p.join(timeout=5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

BASE_URL = f"http://127.0.0.1:{PROXY_PORT}"


def test_models_endpoint(cluster: None) -> None:
    """GET /v1/models should return at least one model."""
    with httpx.Client(base_url=BASE_URL, timeout=10) as c:
        r = c.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert "data" in data
        assert len(data["data"]) > 0


def test_completions_single(cluster: None) -> None:
    """POST /v1/completions with a single prompt."""
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        r = c.post("/v1/completions", json={
            "model": "dummy",
            "prompt": "Hello " * 500,
            "max_tokens": 50,
        })
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data
        assert len(data["choices"]) > 0


def test_completions_streaming(cluster: None) -> None:
    """POST /v1/completions with stream=true."""
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        with c.stream("POST", "/v1/completions", json={
            "model": "dummy",
            "prompt": "Hello " * 500,
            "max_tokens": 20,
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            chunks = []
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(line)
            assert len(chunks) > 0


def test_chat_completions(cluster: None) -> None:
    """POST /v1/chat/completions — basic smoke test."""
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "Hi " * 500}],
            "max_tokens": 50,
        })
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data


@pytest.mark.asyncio
async def test_concurrent_completions(cluster: None) -> None:
    """Send multiple concurrent requests to simulate benchmark load."""
    num_requests = 20

    async def _send(client: httpx.AsyncClient, idx: int) -> int:
        r = await client.post("/v1/completions", json={
            "model": "dummy",
            "prompt": f"Request {idx} " + "token " * 200,
            "max_tokens": 30,
        })
        return r.status_code

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as c:
        results = await asyncio.gather(*[_send(c, i) for i in range(num_requests)])
        assert all(s == 200 for s in results), f"Some requests failed: {results}"
