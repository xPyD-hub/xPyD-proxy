"""End-to-end benchmark: 1000 concurrent clients, 10000 requests, mixed lengths.

Topology: 2 prefill + 16 decode + 1 proxy (same as test_benchmark_integration).
Excluded from CI via --ignore. Run manually:

    pytest tests/test_benchmark_e2e.py -v -s

Uses pytest.mark.benchmark so it can also be collected via:

    pytest -m benchmark tests/test_benchmark_e2e.py
"""

from __future__ import annotations

import os
import random
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
import pytest
import yaml

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
MODEL_PATH = os.path.join(_REPO_ROOT, "tokenizers", "DeepSeek-R1")

NUM_PREFILL = 2
NUM_DECODE = 16
TOTAL_REQUESTS = 10_000
MAX_CONCURRENCY = 1_000


def _free_port() -> int:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_bench_config(model, prefill, decode, port):
    """Write a temporary YAML config for benchmark proxy launch."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump({"model": model, "prefill": prefill, "decode": decode, "port": port}, f)
    f.close()
    return f.name


def _wait_port(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _random_content(length: int) -> str:
    """Generate a random string of approximately *length* characters."""
    if length <= 0:
        return ""
    # Use words to produce roughly natural-looking content.
    words = ["hello", "world", "bench", "test", "proxy", "stream", "token", "data"]
    pieces: list[str] = []
    cur = 0
    while cur < length:
        w = random.choice(words)
        pieces.append(w)
        cur += len(w) + 1  # +1 for the space
    return " ".join(pieces)[:length]


def _build_payload(model: str, stream: bool) -> dict[str, Any]:
    """Build a chat completion payload with random prompt length 0-10k chars."""
    prompt_len = random.randint(0, 10_000)
    content = _random_content(prompt_len)
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": random.randint(1, 64),
        "stream": stream,
    }


# ---------------------------------------------------------------------------
# Cluster fixture (module-scoped — start once, reuse across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cluster():
    """Spin up dummy nodes + proxy, yield connection info, tear down."""
    env = os.environ.copy()
    # Speed up dummy nodes for benchmarking
    env["PREFILL_DELAY_PER_TOKEN"] = "0"
    env["DECODE_DELAY_PER_TOKEN"] = "0"
    procs: list[subprocess.Popen] = []

    prefill_ports = [_free_port() for _ in range(NUM_PREFILL)]
    decode_ports = [_free_port() for _ in range(NUM_DECODE)]
    proxy_port = _free_port()

    try:
        for port in prefill_ports:
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "sim_adapter:prefill_app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--log-level",
                        "error",
                    ],
                    env=env,
                    cwd=_REPO_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )

        for port in decode_ports:
            procs.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "sim_adapter:decode_app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--log-level",
                        "error",
                    ],
                    env=env,
                    cwd=_REPO_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )

        for port in prefill_ports + decode_ports:
            assert _wait_port(port), f"Node on port {port} failed to start"

        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "xpyd.proxy",
                    "proxy",
                    "--config",
                    _write_bench_config(
                        MODEL_PATH,
                        [f"127.0.0.1:{p}" for p in prefill_ports],
                        [f"127.0.0.1:{p}" for p in decode_ports],
                        proxy_port,
                    ),
                ],
                env=env,
                cwd=_REPO_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
        assert _wait_port(proxy_port, timeout=30), "Proxy failed to start"

        yield {
            "proxy_port": proxy_port,
            "model": MODEL_PATH,
        }

    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _send_non_streaming(base_url: str, payload: dict) -> dict:
    """Send a non-streaming request, return summary dict."""
    t0 = time.monotonic()
    with httpx.Client(base_url=base_url, timeout=60, trust_env=False) as c:
        r = c.post("/v1/chat/completions", json=payload)
    elapsed = time.monotonic() - t0
    return {"status": r.status_code, "elapsed": elapsed, "stream": False}


def _send_streaming(base_url: str, payload: dict) -> dict:
    """Send a streaming request, consume all SSE chunks, return summary."""
    t0 = time.monotonic()
    chunks = 0
    status = 0
    with httpx.Client(base_url=base_url, timeout=60, trust_env=False) as c:
        with c.stream("POST", "/v1/chat/completions", json=payload) as r:
            status = r.status_code
            for line in r.iter_lines():
                if line.startswith("data: "):
                    chunks += 1
    elapsed = time.monotonic() - t0
    return {"status": status, "elapsed": elapsed, "stream": True, "chunks": chunks}


def _send_request(base_url: str, model: str, idx: int) -> dict:
    """Build and send one request (randomly streaming or not)."""
    stream = random.choice([True, False])
    payload = _build_payload(model, stream=stream)
    try:
        if stream:
            return _send_streaming(base_url, payload)
        return _send_non_streaming(base_url, payload)
    except Exception as exc:
        return {"status": -1, "error": str(exc), "stream": stream, "elapsed": 0}


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.benchmark
def test_benchmark_10k_mixed(cluster):
    """Fire 10000 mixed (streaming + non-streaming) requests at 1000 concurrency.

    Validates that every request succeeds (HTTP 200) and prints a summary
    with throughput and latency percentiles.
    """
    base_url = f"http://127.0.0.1:{cluster['proxy_port']}"
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = [
            pool.submit(_send_request, base_url, cluster["model"], i)
            for i in range(TOTAL_REQUESTS)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    # ---- Assertions ----
    statuses = [r["status"] for r in results]
    success = statuses.count(200)
    failed = len(statuses) - success
    errors = [r for r in results if r["status"] != 200]

    # Print summary before asserting so we always see stats
    elapsed_all = sorted(r["elapsed"] for r in results if r["status"] == 200)
    stream_count = sum(1 for r in results if r.get("stream"))
    non_stream_count = len(results) - stream_count

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total requests : {TOTAL_REQUESTS}")
    print(f"Concurrency    : {MAX_CONCURRENCY}")
    print(f"Streaming      : {stream_count}")
    print(f"Non-streaming  : {non_stream_count}")
    print(f"Successful     : {success}")
    print(f"Failed         : {failed}")
    if elapsed_all:
        print(f"Latency p50    : {elapsed_all[len(elapsed_all) // 2]:.3f}s")
        print(f"Latency p90    : {elapsed_all[int(len(elapsed_all) * 0.9)]:.3f}s")
        print(f"Latency p99    : {elapsed_all[int(len(elapsed_all) * 0.99)]:.3f}s")
        print(f"Latency max    : {elapsed_all[-1]:.3f}s")
    print("=" * 60)

    if errors:
        sample = errors[:5]
        print(f"First {len(sample)} errors: {sample}")

    assert failed == 0, f"{failed}/{TOTAL_REQUESTS} requests failed"


@pytest.mark.benchmark
@pytest.mark.benchmark
def test_benchmark_streaming_only(cluster):
    """1000 concurrent streaming requests to verify SSE under load."""
    base_url = f"http://127.0.0.1:{cluster['proxy_port']}"
    count = 2000

    def send(idx: int) -> dict:
        payload = _build_payload(cluster["model"], stream=True)
        return _send_streaming(base_url, payload)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = [pool.submit(send, i) for i in range(count)]
        for f in as_completed(futures):
            results.append(f.result())

    success = sum(1 for r in results if r["status"] == 200)
    has_chunks = sum(1 for r in results if r.get("chunks", 0) >= 2)

    print(f"\nStreaming-only: {success}/{count} OK, {has_chunks} with >=2 chunks")
    assert success == count, f"{count - success} streaming requests failed"
    assert has_chunks == count, "Some streaming responses had fewer than 2 chunks"


@pytest.mark.benchmark
@pytest.mark.benchmark
def test_benchmark_burst_short_prompts(cluster):
    """Burst of 5000 short-prompt requests (< 100 chars) at full concurrency."""
    base_url = f"http://127.0.0.1:{cluster['proxy_port']}"
    count = 5000

    def send(idx: int) -> dict:
        payload = {
            "model": cluster["model"],
            "messages": [
                {"role": "user", "content": _random_content(random.randint(0, 100))}
            ],
            "max_tokens": 5,
            "stream": False,
        }
        return _send_non_streaming(base_url, payload)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = [pool.submit(send, i) for i in range(count)]
        for f in as_completed(futures):
            results.append(f.result())

    success = sum(1 for r in results if r["status"] == 200)
    elapsed = sorted(r["elapsed"] for r in results if r["status"] == 200)
    if elapsed:
        print(
            f"\nShort burst: {success}/{count} OK, "
            f"p50={elapsed[len(elapsed) // 2]:.3f}s, "
            f"p99={elapsed[int(len(elapsed) * 0.99)]:.3f}s"
        )
    assert success == count, f"{count - success} short-burst requests failed"


@pytest.mark.benchmark
@pytest.mark.benchmark
def test_benchmark_long_prompts(cluster):
    """500 requests with long prompts (5k-10k chars) at moderate concurrency."""
    base_url = f"http://127.0.0.1:{cluster['proxy_port']}"
    count = 500
    concurrency = 200

    def send(idx: int) -> dict:
        payload = {
            "model": cluster["model"],
            "messages": [
                {
                    "role": "user",
                    "content": _random_content(random.randint(5000, 10000)),
                }
            ],
            "max_tokens": 32,
            "stream": random.choice([True, False]),
        }
        if payload["stream"]:
            return _send_streaming(base_url, payload)
        return _send_non_streaming(base_url, payload)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send, i) for i in range(count)]
        for f in as_completed(futures):
            results.append(f.result())

    success = sum(1 for r in results if r["status"] == 200)
    elapsed = sorted(r["elapsed"] for r in results if r["status"] == 200)
    if elapsed:
        print(
            f"\nLong prompts: {success}/{count} OK, "
            f"p50={elapsed[len(elapsed) // 2]:.3f}s, "
            f"p99={elapsed[int(len(elapsed) * 0.99)]:.3f}s"
        )
    assert success == count, f"{count - success} long-prompt requests failed"
