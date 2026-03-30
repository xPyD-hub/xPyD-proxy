"""Integration test: proxy + dummy nodes end-to-end.

Topology (matches benchmarks/run_benchmark.sh):
  - 2 prefill nodes  (ports 8100-8101)
  - 16 decode nodes  (ports 8200-8215)
  - 1 proxy          (port 8868)
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

    # Wait for all nodes
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
    assert _wait_port(PROXY_PORT, timeout=30), "Proxy didn't start"

    yield {"proxy_port": PROXY_PORT, "model": MODEL_PATH}

    # Teardown
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait(timeout=5)


CHAT_PAYLOAD = {
    "model": "",
    "messages": [{"role": "user", "content": "Hello world"}],
    "max_tokens": 5,
    "stream": False,
}


def test_models_endpoint(cluster):
    """Proxy /v1/models returns per-instance aggregated response."""
    with httpx.Client(base_url=f"http://127.0.0.1:{cluster['proxy_port']}", timeout=10) as c:
        r = c.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        # Proxy returns per-instance aggregated response, not standard OpenAI format
        assert len(data) > 0, "No instances in /v1/models response"
        for instance, result in data.items():
            assert result["status"] == 200
            assert "data" in result["data"]
            assert len(result["data"]["data"]) > 0


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
        data_lines = [ln for ln in lines if ln.startswith("data: ")]
        assert len(data_lines) >= 2
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


def test_vllm_bench_serve(cluster):
    """Run vllm bench serve with 1000 prompts through the proxy.

    This test requires vllm to be installed. It is excluded from CI
    (--ignore in workflow) because vllm is heavy. Run manually:

        PYTHONPATH=core:dummy_nodes pytest tests/test_benchmark_integration.py::test_vllm_bench_serve -v

    Note: Uses --tokenizer gpt2 because the local DeepSeek-R1 tokenizer
    uses a custom tokenizer class that vllm cannot load directly.
    gpt2 is a lightweight standard tokenizer sufficient for benchmarking
    with random-generated prompts.
    """
    import shutil

    vllm_bin = shutil.which("vllm")
    if vllm_bin is None:
        pytest.skip("vllm not installed — skipping benchmark test")

    result = subprocess.run(
        [
            vllm_bin, "bench", "serve",
            "--host", "127.0.0.1",
            "--port", str(cluster["proxy_port"]),
            "--model", cluster["model"],
            # Use gpt2 tokenizer: lightweight, standard, avoids loading
            # DeepSeek-R1's custom TokenizersBackend class that vllm
            # cannot handle. For random-data benchmarks this is fine.
            "--tokenizer", "gpt2",
            "--dataset-name", "random",
            "--random-input-len", "3000",
            "--random-output-len", "200",
            "--num-prompts", "1000",
            "--burstiness", "100",
            "--request-rate", "3.6",
            "--endpoint", "/v1/completions",
        ],
        capture_output=True, text=True, timeout=600,
    )

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        # Filter out vllm startup warnings (Triton, CUDA)
        important = [line for line in result.stderr.split("\n")
                     if "error" in line.lower() and "triton" not in line.lower()]
        if important:
            print("STDERR:", "\n".join(important[-5:]))

    assert result.returncode == 0, f"vllm bench serve failed: {result.stderr[-500:]}"

    # Parse results with fallback assertion
    successful = None
    failed = None
    for line in result.stdout.strip().split("\n"):
        if "Successful requests:" in line:
            successful = int(line.split(":")[1].strip())
        if "Failed requests:" in line:
            failed = int(line.split(":")[1].strip())

    assert successful is not None, "Could not parse 'Successful requests' from output"
    assert failed is not None, "Could not parse 'Failed requests' from output"
    assert successful == 1000, f"Expected 1000 successful, got {successful}"
    assert failed == 0, f"Expected 0 failed, got {failed}"
