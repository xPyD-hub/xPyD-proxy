"""Integration test: proxy + dummy nodes end-to-end.

Topology (matches benchmarks/run_benchmark.sh):
  - 2 prefill nodes  (dynamically allocated ports)
  - 16 decode nodes  (dynamically allocated ports)
  - 1 proxy          (dynamically allocated port)

This test file is excluded from CI via --ignore in the workflow.
Run manually: pytest tests/test_benchmark_integration.py -v
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import pytest
import yaml

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
MODEL_PATH = os.path.join(_REPO_ROOT, "tokenizers", "DeepSeek-R1")

NUM_PREFILL = 2
NUM_DECODE = 16


def _free_port():
    """Allocate an ephemeral port."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 20.0) -> bool:
    """Wait until a port is accepting connections."""
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
    procs = []

    prefill_ports = [_free_port() for _ in range(NUM_PREFILL)]
    decode_ports = [_free_port() for _ in range(NUM_DECODE)]
    proxy_port = _free_port()

    try:
        # Start prefill nodes
        for port in prefill_ports:
            p = subprocess.Popen(
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
            procs.append(p)

        # Start decode nodes
        for port in decode_ports:
            p = subprocess.Popen(
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
            procs.append(p)

        # Wait for all nodes
        for port in prefill_ports:
            assert _wait_port(port), f"Prefill {port} didn't start"
        for port in decode_ports:
            assert _wait_port(port), f"Decode {port} didn't start"

        # Start proxy
        prefill_args = [f"127.0.0.1:{p}" for p in prefill_ports]
        decode_args = [f"127.0.0.1:{p}" for p in decode_ports]

        _cfg = {
            "model": MODEL_PATH,
            "prefill": prefill_args,
            "decode": decode_args,
            "port": proxy_port,
        }
        _cf = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(_cfg, _cf)
        _cf.close()
        proxy = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "xpyd.proxy",
                "proxy",
                "--config",
                _cf.name,
            ],
            env=env,
            cwd=_REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(proxy)
        assert _wait_port(proxy_port, timeout=30), "Proxy didn't start"

        yield {
            "proxy_port": proxy_port,
            "model": MODEL_PATH,
            "prefill_ports": prefill_ports,
            "decode_ports": decode_ports,
        }

    finally:
        # Teardown — always clean up, even if setup fails
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)


CHAT_PAYLOAD = {
    "model": "",
    "messages": [{"role": "user", "content": "Hello world"}],
    "max_tokens": 5,
    "stream": False,
}


def test_models_endpoint(cluster):
    """Proxy /v1/models returns OpenAI-compatible model listing."""
    with httpx.Client(
        base_url=f"http://127.0.0.1:{cluster['proxy_port']}",
        timeout=10,
        trust_env=False,
    ) as c:
        r = c.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0, "No models in /v1/models response"
        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"


def test_chat_completions(cluster):
    """Non-streaming chat completions through proxy."""
    payload = {**CHAT_PAYLOAD, "model": cluster["model"]}
    with httpx.Client(
        base_url=f"http://127.0.0.1:{cluster['proxy_port']}",
        timeout=30,
        trust_env=False,
    ) as c:
        r = c.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]


def test_chat_completions_streaming(cluster):
    """Streaming chat completions through proxy."""
    payload = {**CHAT_PAYLOAD, "model": cluster["model"], "stream": True}
    with httpx.Client(
        base_url=f"http://127.0.0.1:{cluster['proxy_port']}",
        timeout=30,
        trust_env=False,
    ) as c:
        r = c.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        lines = r.text.strip().split("\n")
        data_lines = [ln for ln in lines if ln.startswith("data: ")]
        assert len(data_lines) >= 2
        assert data_lines[-1] == "data: [DONE]"


def test_status_topology(cluster):
    """Proxy status should reflect correct topology."""
    with httpx.Client(
        base_url=f"http://127.0.0.1:{cluster['proxy_port']}",
        timeout=10,
        trust_env=False,
    ) as c:
        r = c.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert data["prefill_node_count"] == NUM_PREFILL
        assert data["decode_node_count"] == NUM_DECODE


def test_concurrent_requests(cluster):
    """Multiple concurrent requests should all succeed."""
    import concurrent.futures

    payload = {**CHAT_PAYLOAD, "model": cluster["model"]}

    def send_request(idx):
        with httpx.Client(
            base_url=f"http://127.0.0.1:{cluster['proxy_port']}",
            timeout=30,
            trust_env=False,
        ) as c:
            r = c.post("/v1/chat/completions", json=payload)
            return r.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(send_request, i) for i in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(code == 200 for code in results), f"Some requests failed: {results}"


@pytest.mark.skipif(
    os.environ.get("RUN_VLLM_BENCH") != "1",
    reason="Set RUN_VLLM_BENCH=1 to run this heavy benchmark test",
)
def test_vllm_bench_serve(cluster):
    """Run vllm bench serve with 1000 prompts through the proxy.

    This test is heavy (~5-10 min) and requires vllm. It is gated behind
    the RUN_VLLM_BENCH=1 env var and skipped by default.

    Run manually:
        RUN_VLLM_BENCH=1 \\
          pytest tests/test_benchmark_integration.py::test_vllm_bench_serve -v

    Note: Uses --tokenizer gpt2 because the local DeepSeek-R1 tokenizer
    uses a custom class (TokenizersBackend) that vllm cannot load.
    gpt2 is lightweight and sufficient for random-data benchmarks.
    """
    import shutil

    vllm_bin = shutil.which("vllm")
    if vllm_bin is None:
        pytest.skip("vllm not installed")

    result = subprocess.run(
        [
            vllm_bin,
            "bench",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(cluster["proxy_port"]),
            "--model",
            cluster["model"],
            "--tokenizer",
            "gpt2",
            "--dataset-name",
            "random",
            "--random-input-len",
            "3000",
            "--random-output-len",
            "200",
            "--num-prompts",
            "1000",
            "--burstiness",
            "100",
            "--request-rate",
            "3.6",
            "--endpoint",
            "/v1/completions",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        important = [
            line
            for line in result.stderr.split("\n")
            if "error" in line.lower() and "triton" not in line.lower()
        ]
        if important:
            print("STDERR:", "\n".join(important[-5:]))

    assert result.returncode == 0, f"vllm bench serve failed: {result.stderr[-500:]}"

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
