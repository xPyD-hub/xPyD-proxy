#!/usr/bin/env python3
"""Test real proxy scheduling against controllable CPU HTTP backends."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL = "dummy_tokenizer"
STRATEGIES = (
    "roundrobin",
    "loadbalanced",
    "consistent_hash",
    "power_of_two",
    "cache_aware",
)


def http(url, body=None, session="test"):
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Session-ID": session},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def wait_for(predicate, process):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        assert process.poll() is None, "Proxy exited; see scheduler semantics log"
        try:
            if predicate():
                return
        except urllib.error.URLError:
            # Proxy not accepting connections yet; retry until the deadline.
            pass
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for controlled scheduler state")


def assert_avoids_busy(selections, busy):
    assert len(selections) >= 3, "Need repeated decisions, not one lucky choice"
    for selected in selections:
        for role, address in busy.items():
            assert selected[role] != address, (role, selected, busy)


def affinity_cases(topology, strategy, instances):
    sys.path.insert(0, str(ROOT))
    from xpyd.scheduler import CacheAwarePolicy, ConsistentHashPolicy

    candidates = {
        role: {node["address"] for node in instances if node["role"] == role}
        for role in dict.fromkeys(node["role"] for node in instances)
    }
    workers = [node["address"] for node in instances]
    if strategy == "consistent_hash":
        policy = ConsistentHashPolicy(workers=workers)
    else:
        tokenizer = None
        if topology == "aggregated":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                ROOT / "tests/assets" / MODEL, local_files_only=True
            )
        policy = CacheAwarePolicy(workers=workers, tokenizer=tokenizer)
    uncovered = {
        (role, address) for role, nodes in candidates.items() for address in nodes
    }
    cases = []
    for index in range(4096):
        session = f"affinity-{index}"
        # Known vocabulary keeps prefixes distinct under both tokenization paths.
        words = ["hello" if index & (1 << bit) else "world" for bit in range(12)]
        prefix = (" ".join(words) + " ") * 32
        expected = {
            role: (
                policy.select_from(nodes, header=session)
                if strategy == "consistent_hash"
                else policy.select_from(nodes, prompt=prefix)
            )
            for role, nodes in candidates.items()
        }
        covered = set(expected.items()) & uncovered
        if covered:
            cases.append((session, prefix, expected))
            uncovered -= covered
        if not uncovered:
            return cases
    raise AssertionError(f"Could not construct affinity keys for nodes: {uncovered}")


def run(topology, strategy, actual_strategy=None):
    records = []
    entered = threading.Event()
    release = threading.Event()
    roles = ("aggregated",) if topology == "aggregated" else ("prefill", "decode")
    hold_role = roles[0]
    logs = ROOT / "examples" / "scheduler_semantics" / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def handler(role):
        class Backend(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self.reply({"ok": True})
                elif self.path == "/v1/models":
                    self.reply({"data": [{"id": MODEL, "max_model_len": 131072}]})
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                address = f"127.0.0.1:{self.server.server_port}"
                records.append((role, address))
                if role == hold_role and body.get("prompt") == "hold":
                    entered.set()
                    if not release.wait(30):
                        self.send_error(504, "Controlled request was not released")
                        return
                self.reply(
                    {
                        "id": "controlled",
                        "model": MODEL,
                        "choices": [
                            {"text": "ok", "finish_reason": "length", "index": 0}
                        ],
                        "usage": {"completion_tokens": body.get("max_tokens", 1)},
                    }
                )

        return Backend

    with ExitStack() as stack, tempfile.TemporaryDirectory(dir=logs) as temporary:
        instances = []
        for role in roles:
            for _ in range(2):
                server = ThreadingHTTPServer(("127.0.0.1", 0), handler(role))
                threading.Thread(target=server.serve_forever, daemon=True).start()
                stack.callback(server.server_close)
                stack.callback(server.shutdown)
                instances.append(
                    {
                        "role": role,
                        "model": MODEL,
                        "address": f"127.0.0.1:{server.server_port}",
                    }
                )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        config = {
            "instances": instances,
            "port": port,
            "scheduling": actual_strategy or strategy,
            "tokenizer_path": str(ROOT / "tests/assets"),
            "disaggregated_mode": "direct",
            "first_token_source": "decode",
            "health_check": {"enabled": True, "interval_seconds": 0.1},
            "startup": {"probe_interval_seconds": 1},
        }
        config_path = Path(temporary) / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with (logs / f"{topology}-{strategy}.log").open("wb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from xpyd.proxy import main; main()",
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            url = f"http://127.0.0.1:{port}"

            def idle():
                status = http(url + "/status/instances")
                return all(
                    item["active_requests"] == 0
                    for role in roles
                    for item in status[f"{role}_instances"]
                )

            def select(session, prompt):
                before = len(records)
                output = http(
                    url + "/v1/completions",
                    {
                        "model": MODEL,
                        "prompt": prompt,
                        "max_tokens": 1,
                    },
                    session,
                )
                assert output["choices"][0]["text"]
                selected = dict(records[before:])
                assert set(selected) == set(roles), selected
                return selected

            try:
                wait_for(
                    lambda: all(
                        len(http(url + "/status/instances")[f"{role}_instances"]) == 2
                        and all(
                            item["status"] == "healthy"
                            for item in http(url + "/status/instances")[
                                f"{role}_instances"
                            ]
                        )
                        for role in roles
                    ),
                    process,
                )
                if strategy in ("loadbalanced", "power_of_two"):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        held = pool.submit(select, "busy-session", "hold")
                        try:
                            assert entered.wait(10), "Busy backend was not reached"
                            status = http(url + "/status/instances")
                            busy = {
                                role: next(
                                    item["address"]
                                    for item in status[f"{role}_instances"]
                                    if item["active_requests"] > 0
                                )
                                for role in roles
                            }
                            selections = [
                                select(f"probe-{i}", f"probe {i}") for i in range(4)
                            ]
                            assert not held.done(), "Busy request ended too early"
                            assert_avoids_busy(selections, busy)
                        finally:
                            release.set()
                            held.result(timeout=10)
                elif strategy == "roundrobin":
                    selected = [select(str(i), f"request {i}") for i in range(6)]
                    for role in roles:
                        values = [item[role] for item in selected]
                        assert len(set(values[:2])) == 2, values
                        assert values == values[:2] * 3, values
                else:
                    for session, prefix, expected in affinity_cases(
                        topology, strategy, instances
                    ):
                        for index in range(3):
                            actual = select(
                                (
                                    session
                                    if strategy == "consistent_hash"
                                    else f"{session}-{index}"
                                ),
                                (
                                    f"different prompt {index}"
                                    if strategy == "consistent_hash"
                                    else prefix + str(index)
                                ),
                            )
                            assert actual == expected, (
                                "affinity-routing",
                                strategy,
                                expected,
                                actual,
                            )
                wait_for(idle, process)
                print(f"PASS {topology}/{strategy}: controlled scheduling semantics")
            finally:
                release.set()
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError(
                        "Scheduler test proxy required forced termination"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topology", choices=("aggregated", "disaggregated"), required=True
    )
    parser.add_argument("--scheduler", choices=STRATEGIES, required=True)
    options = parser.parse_args()
    run(options.topology, options.scheduler)
