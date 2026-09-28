#!/usr/bin/env python3
"""Run the five xPyD scheduler configurations against ready backends."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEDULERS = (
    "roundrobin",
    "loadbalanced",
    "consistent_hash",
    "power_of_two",
    "cache_aware",
)


def wait_ready(process: subprocess.Popen, timeout: float = 180) -> None:
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


def main() -> None:
    (ROOT / "proxy_logs").mkdir(exist_ok=True)
    (ROOT / "metrics").mkdir(exist_ok=True)
    (ROOT / "bench_results").mkdir(exist_ok=True)

    for scheduler in SCHEDULERS:
        print(f"=== {scheduler} ===", flush=True)
        with (ROOT / "proxy_logs" / f"{scheduler}.log").open("wb") as log:
            process = subprocess.Popen(
                [
                    "bash",
                    str(ROOT.parents[1] / "lib/run_proxy.sh"),
                    "proxy",
                    "--config",
                    str(ROOT / f"xpyd_{scheduler}.yaml"),
                ],
                env={**os.environ, "PYTHON": sys.executable},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                wait_ready(process)
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scheduler_bench.py"),
                        "--scheduler",
                        scheduler,
                        "--requests",
                        "32",
                        "--concurrency",
                        "8",
                        "--output-dir",
                        str(ROOT / "bench_results"),
                    ],
                    check=True,
                )
                with urllib.request.urlopen(
                    "http://127.0.0.1:8868/metrics", timeout=10
                ) as response:
                    (ROOT / "metrics" / f"{scheduler}.prom").write_bytes(
                        response.read()
                    )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        time.sleep(2)


if __name__ == "__main__":
    main()
