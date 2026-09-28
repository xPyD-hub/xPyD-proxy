#!/usr/bin/env python3
"""Real CUDA/XPU backend lifecycle checks using the repository proxy."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAYOUTS = {
    "aggregated": ("aggregated", "aggregated"),
    "direct": ("prefill", "prefill", "decode", "decode"),
    "mixed": ("prefill", "decode", "aggregated", "aggregated"),
}


def instances(topology):
    return [
        {
            "address": f"127.0.0.1:{18200 + index}",
            "role": role,
            "model": (
                "lifecycle-aggregated" if role == "aggregated" else "lifecycle-pd"
            ),
        }
        for index, role in enumerate(LAYOUTS[topology])
    ]


def http(url, body=None):
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = response.read()
            return response.status, json.loads(data) if data else None
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def port_free(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def memory_mib(device, index):
    if device == "cuda":
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(index),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return float(output.strip())
    data = json.loads(
        subprocess.check_output(["xpu-smi", "stats", "-d", str(index), "-j"], text=True)
    )
    return sum(tile["current"] for tile in data["memory"]["used_mib"].values())


def wait_for(label, predicate, processes, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for name, process in processes.items():
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited with {process.returncode}: see logs")
        try:
            if predicate():
                return
        except urllib.error.URLError:
            # The service may still be starting; retry until the bounded deadline.
            pass
        time.sleep(1)
    raise AssertionError(f"Timed out: {label}")


def backend_command(model, node):
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--served-model-name",
        node["model"],
        "--host",
        "127.0.0.1",
        "--port",
        node["address"].split(":")[1],
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "4",
        "--enforce-eager",
        "--kv-cache-memory-bytes",
        "536870912",
    ]


def check_status(status, nodes, online):
    for role in ("prefill", "decode", "aggregated"):
        actual = status[f"{role}_instances"]
        expected = [node for node in nodes if node["role"] == role]
        assert {item["address"] for item in actual} == {
            node["address"] for node in expected
        }, status
        for item in actual:
            if (item["status"] == "healthy") != (item["address"] in online):
                return False
    return True


def heartbeat(topology, nodes, online):
    mode = "disaggregated" if topology == "direct" else topology
    fields = [f"mode={mode}"]
    for role, label in (
        ("prefill", "P"),
        ("decode", "D"),
        ("aggregated", "aggregated"),
    ):
        configured = [node for node in nodes if node["role"] == role]
        if configured:
            ready = sum(node["address"] in online for node in configured)
            fields.append(f"{label}={ready}/{len(configured)} online")
    return " | ".join(fields)


def stop(process):
    # Each child owns a new process group, including vLLM engine workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        # The owned process group has already exited.
        pass
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
        raise RuntimeError(f"Process {process.pid} required forced termination")


def run(args, topology):
    nodes = instances(topology)
    devices = args.devices[: len(nodes)]
    assert len(devices) == len(nodes), f"{topology} needs {len(nodes)} distinct devices"
    assert len(set(devices)) == len(devices), "Use a separate device per backend"
    ports = [18869] + [int(node["address"].split(":")[1]) for node in nodes]
    assert all(port_free(port) for port in ports), "Scenario ports are already occupied"
    baseline = {index: memory_mib(args.device, index) for index in devices}
    log_dir = args.log_dir / topology
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = {}
    url = "http://127.0.0.1:18869"
    online = set()
    proxy_log = log_dir / "proxy.log"
    config = yaml.safe_load((HERE / "xpyd.yaml").read_text())
    config["instances"] = nodes
    config_path = log_dir / "xpyd.yaml"
    config_path.write_text(yaml.safe_dump(config))

    def phase(label):
        message = f"\n===== {topology}: {label} ====="
        print(message, flush=True)
        with (log_dir / "scenario.log").open("a") as log:
            log.write(message + "\n")

    def start(name, command, env):
        with (log_dir / f"{name}.log").open("ab") as log:
            processes[name] = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT), **env},
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

    def converge():
        wait_for(
            "status/instances",
            lambda: check_status(http(url + "/status/instances")[1], nodes, online),
            processes,
        )
        expected = heartbeat(topology, nodes, online)
        wait_for(
            expected,
            lambda: proxy_log.read_text()
            .split("Node heartbeat | ")[-1]
            .splitlines()[0]
            .strip()
            == expected,
            processes,
        )

    def inference():
        for model in sorted({node["model"] for node in nodes}):
            roles = {node["role"] for node in nodes if node["model"] == model}
            ready = all(
                any(
                    node["address"] in online
                    for node in nodes
                    if node["role"] == role and node["model"] == model
                )
                for role in roles
            )
            status, body = http(
                url + "/v1/completions",
                {
                    "model": model,
                    "prompt": "The capital of France is",
                    "max_tokens": 8,
                    "temperature": 0,
                },
            )
            assert status == (200 if ready else 503), (model, status, body)
            if ready:
                assert body["model"] == model, body
                assert body["choices"][0]["text"], body

    def start_node(index):
        node = nodes[index]
        mask = "ZE_AFFINITY_MASK" if args.device == "xpu" else "CUDA_VISIBLE_DEVICES"
        start(
            f"backend-{index}",
            backend_command(args.model, node),
            {mask: str(devices[index]), "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        )
        wait_for(
            f"backend {index} health",
            lambda: http(f"http://{node['address']}/health")[0] == 200,
            processes,
            timeout=900,
        )
        online.add(node["address"])

    def stop_node(index):
        process = processes.pop(f"backend-{index}")
        stop(process)
        online.remove(nodes[index]["address"])
        wait_for(
            f"backend {index} port and accelerator memory release",
            lambda: port_free(ports[index + 1])
            and memory_mib(args.device, devices[index])
            <= baseline[devices[index]] + 64,
            processes,
        )

    def cleanup():
        phase("cleanup")
        errors = []
        for name, process in reversed(list(processes.items())):
            try:
                stop(process)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                errors.append(f"{name}: {error}")
        wait_for(
            "all ports and accelerator memory released",
            lambda: all(port_free(port) for port in ports)
            and all(
                memory_mib(args.device, index) <= baseline[index] + 64
                for index in devices
            ),
            {},
        )
        if errors:
            raise RuntimeError("; ".join(errors))
        phase("ports closed; accelerator memory returned to baseline (+64 MiB)")

    try:
        phase("proxy first; all models must return 503")
        start(
            "proxy",
            [
                sys.executable,
                "-c",
                "from xpyd.proxy import main; main()",
                "--config",
                str(config_path),
            ],
            {},
        )
        wait_for(
            "proxy starts without backends",
            lambda: http(url + "/status/instances")[0] == 200,
            processes,
        )
        converge()
        inference()
        phase("node discovery; all models infer")
        for index in range(len(nodes)):
            start_node(index)
        converge()
        inference()
        for role in dict.fromkeys(node["role"] for node in nodes):
            index = next(i for i, node in enumerate(nodes) if node["role"] == role)
            phase(f"stop one {role}; surviving topology and model isolation")
            stop_node(index)
            converge()
            inference()
            phase(f"reconnect {role} after port and memory release")
            start_node(index)
            converge()
            inference()
        phase("complete")
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cuda", "xpu"), required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--topology", choices=(*LAYOUTS, "all"), default="all")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=HERE / "logs" / time.strftime("%Y%m%dT%H%M%S"),
    )
    args = parser.parse_args()
    args.model = str(Path(args.model).resolve(strict=True))
    args.log_dir = args.log_dir.resolve()

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    topologies = LAYOUTS if args.topology == "all" else [args.topology]
    results = []
    args.log_dir.mkdir(parents=True, exist_ok=True)
    try:
        for topology in topologies:
            results.append({"topology": topology, "status": "failure"})
            run(args, topology)
            results[-1]["status"] = "success"
    finally:
        (args.log_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
