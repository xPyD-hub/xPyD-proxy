#!/usr/bin/env python3
"""Run the complete first-token/API/scheduler matrix for an 8P8D scenario."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SOURCES = ("prefill", "decode")
APIS = ("completion", "chat")
SCHEDULERS = (
    "roundrobin",
    "loadbalanced",
    "consistent_hash",
    "power_of_two",
    "cache_aware",
)


def wait_ready(process: subprocess.Popen[bytes], timeout: float = 1200) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"xPyD exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8868/status/instances", timeout=2
            ):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("xPyD did not become ready")


def result_succeeded(path: Path, requests: int) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return result.get("requests") == requests and result.get("failed") == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-dir", type=Path, required=True)
    parser.add_argument("--transport", choices=("nixl", "zmq"), required=True)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source", action="append", choices=SOURCES)
    parser.add_argument("--api", action="append", choices=APIS)
    parser.add_argument("--scheduler", action="append", choices=SCHEDULERS)
    args = parser.parse_args()

    root = args.scenario_dir.resolve()
    harness = Path(__file__).resolve().parent / "disaggregated_matrix_bench.py"
    repo_root = Path(__file__).resolve().parents[3]
    failures: list[dict[str, object]] = []
    sources = tuple(args.source or SOURCES)
    apis = tuple(args.api or APIS)
    schedulers = tuple(args.scheduler or SCHEDULERS)

    for source in sources:
        for api in apis:
            for scheduler in schedulers:
                label = f"{source}/{api}/{scheduler}"
                result_path = root / "bench_results" / label
                result_path = result_path.with_suffix(".json")
                if args.resume and result_succeeded(result_path, args.requests):
                    print(f"=== {label}: already successful ===", flush=True)
                    continue

                print(f"=== {label} ===", flush=True)
                log_path = (root / "proxy_logs" / label).with_suffix(".log")
                metrics_path = (root / "metrics" / label).with_suffix(".prom")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                config = root / "proxy_configs" / source / f"{scheduler}.yaml"

                with log_path.open("wb") as log:
                    process = subprocess.Popen(
                        [
                            "bash",
                            str(repo_root / "examples/lib/run_proxy.sh"),
                            "proxy",
                            "--config",
                            str(config),
                        ],
                        env={**os.environ, "PYTHON": sys.executable},
                        cwd=repo_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        wait_ready(process)
                        subprocess.run(
                            [
                                sys.executable,
                                str(harness),
                                "--api",
                                api,
                                "--scheduler",
                                scheduler,
                                "--first-token-source",
                                source,
                                "--transport",
                                args.transport,
                                "--requests",
                                str(args.requests),
                                "--concurrency",
                                str(args.concurrency),
                                "--output",
                                str(result_path),
                            ],
                            check=True,
                        )
                        with urllib.request.urlopen(
                            "http://127.0.0.1:8868/metrics", timeout=10
                        ) as response:
                            metrics_path.write_bytes(response.read())
                    except Exception as exc:
                        failures.append({"scenario": label, "runner_error": str(exc)})
                        print(f"ERROR: {label}: {exc}", file=sys.stderr, flush=True)
                    finally:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    if result.get("failed"):
                        failures.append(
                            {
                                "scenario": label,
                                "request_failures": result["failed"],
                            }
                        )
                time.sleep(2)

    summary = {
        "transport": args.transport,
        "requests_per_combination": args.requests,
        "combinations": len(sources) * len(apis) * len(schedulers),
        "total_requests": args.requests * len(sources) * len(apis) * len(schedulers),
        "failures": failures,
    }
    (root / "matrix_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
