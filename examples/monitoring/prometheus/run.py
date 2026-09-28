#!/usr/bin/env python3
"""Run a real Prometheus scrape, traffic, and alert recovery demonstration."""

from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml
from prometheus_client.parser import text_string_to_metric_families

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROXY = "http://127.0.0.1:18868"
PROMETHEUS = "http://127.0.0.1:19090"


def http(url, body=None, timeout=10):
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def get_json(url):
    status, data = http(url)
    if status != 200:
        raise RuntimeError(f"{url}: HTTP {status}: {data[:300]!r}")
    return json.loads(data)


def samples(text):
    result = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            key = (sample.name, frozenset(sample.labels.items()))
            if key in result or not math.isfinite(sample.value):
                raise AssertionError(f"Invalid or duplicate metric: {key}")
            result[key] = sample.value
    return result


def metric_value(values, name, **labels):
    return sum(
        value
        for (metric, tags), value in values.items()
        if metric == name and dict(tags).items() >= labels.items()
    )


def assert_idle(values):
    found = False
    for (name, labels), value in values.items():
        if name in {
            "proxy_active_requests",
            "proxy_prefill_active_requests",
            "proxy_decode_active_requests",
        }:
            assert value == 0, (name, dict(labels), value)
            found = found or name == "proxy_active_requests"
    assert found, "proxy_active_requests is missing"


def scrape():
    status, data = http(PROXY + "/metrics")
    assert status == 200, status
    return data.decode()


def query(expression):
    result = get_json(
        PROMETHEUS + "/api/v1/query?" + urllib.parse.urlencode({"query": expression})
    )
    assert result["status"] == "success", result
    return result["data"]["result"]


def check_processes(label, processes):
    for process in processes:
        code = process.poll()
        if code is not None:
            raise RuntimeError(
                f"{label}: process {process.pid} exited with code {code}; see logs"
            )


def wait_for(label, predicate, processes, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check_processes(label, processes)
        try:
            if predicate():
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            # Endpoint not ready yet; retry until the deadline.
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Timed out: {label}")


def port_free(port, host="127.0.0.1"):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def group_running(group_id):
    if not Path("/proc/self/stat").is_file():
        raise RuntimeError("Process-group cleanup requires Linux /proc")
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            # Processes can exit while their group is being inspected.
            continue
        if int(fields[2]) == group_id and fields[0] not in ("Z", "X"):
            return True
    return False


def wait_group_exit(process, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if process.poll() is not None and not group_running(process.pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def stop(process, timeout=60, kill_timeout=10):
    # Every owned service starts a new group, including its engine workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        # The owned process group may already have exited.
        pass
    if wait_group_exit(process, timeout):
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        # The group can exit between the wait and escalation.
        pass
    if not wait_group_exit(process, kill_timeout):
        raise RuntimeError(f"Process group {process.pid} did not exit after SIGKILL")
    raise RuntimeError(f"Process group {process.pid} required forced termination")


def cleanup_owned_services(processes, logs, ports):
    failures = []
    for process in reversed(processes):
        try:
            stop(process)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            failures.append(str(exc))
    for log in logs:
        try:
            log.close()
        except OSError as exc:
            failures.append(str(exc))
    for host, port in ports:
        try:
            wait_for(
                f"cleanup port {host}:{port}",
                lambda host=host, port=port: port_free(port, host),
                [],
                timeout=30,
            )
        except TimeoutError as exc:
            failures.append(str(exc))
    return failures


def assert_pd_timing(before, after, model):
    timings = {}
    for name in (
        "proxy_prefill_duration_seconds",
        "proxy_ttft_seconds",
        "proxy_kv_transfer_duration_seconds",
    ):
        count = metric_value(after, name + "_count", model=model) - metric_value(
            before, name + "_count", model=model
        )
        assert count == 1, (name, count, "expected one streaming request")
        elapsed = metric_value(after, name + "_sum", model=model) - metric_value(
            before, name + "_sum", model=model
        )
        assert elapsed >= 0, (name, elapsed)
        timings[name] = elapsed
    prefill = timings["proxy_prefill_duration_seconds"]
    ttft = timings["proxy_ttft_seconds"]
    transfer = timings["proxy_kv_transfer_duration_seconds"]
    assert math.isclose(ttft - prefill, transfer, abs_tol=1e-6), (
        "Requires decode-first timing and no concurrent traffic",
        timings,
    )
    return timings


def traffic(model, log_dir, processes, mode="aggregated"):
    before_text = scrape()
    before = samples(before_text)
    payload = {
        "model": model,
        "prompt": "Continue counting: one two three",
        "max_tokens": 16,
        "temperature": 0,
        "ignore_eos": True,
    }
    for endpoint, body in (
        ("/v1/completions", payload),
        (
            "/v1/chat/completions",
            {
                **{key: value for key, value in payload.items() if key != "prompt"},
                "messages": [{"role": "user", "content": "Say hello briefly."}],
            },
        ),
    ):
        status, data = http(PROXY + endpoint, body, timeout=180)
        assert status == 200, (endpoint, status, data[:500])
        assert json.loads(data)["choices"], data

    wait_for(
        "nonstream cleanup",
        lambda: metric_value(samples(scrape()), "proxy_active_requests") == 0,
        processes,
    )
    stream_before = samples(scrape())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        stream = executor.submit(
            http,
            PROXY + "/v1/completions",
            {**payload, "max_tokens": 128, "stream": True},
            180,
        )
        active_seen = False
        deadline = time.monotonic() + 180
        while not stream.done() and time.monotonic() < deadline:
            active = metric_value(samples(scrape()), "proxy_active_requests")
            assert active >= 0, active
            active_seen = active_seen or active > 0
            time.sleep(0.05)
        status, data = stream.result(timeout=5)
    assert status == 200 and b"data: [DONE]" in data, (status, data[:500])
    assert active_seen, "No active streaming request was observed"
    (log_dir / "stream.txt").write_bytes(data)
    wait_for(
        "stream cleanup",
        lambda: metric_value(samples(scrape()), "proxy_active_requests") == 0,
        processes,
    )
    if mode == "disaggregated":
        timings = assert_pd_timing(stream_before, samples(scrape()), model)
        (log_dir / "pd-timing.json").write_text(json.dumps(timings, indent=2))
        print("PASS P/D streaming sample (proxy-side):")
        for name, value in timings.items():
            print(f"  {name}: {value * 1000:.3f} ms")
    status, data = http(PROXY + "/v1/completions", {"model": model})
    assert status == 400, (status, data)
    wait_for(
        "request cleanup",
        lambda: metric_value(samples(scrape()), "proxy_active_requests") == 0,
        processes,
    )
    after_text = scrape()
    after = samples(after_text)
    assert_idle(after)
    for endpoint, expected in (("/v1/completions", 3), ("/v1/chat/completions", 1)):
        for name in ("proxy_requests_total", "proxy_request_duration_seconds_count"):
            delta = metric_value(after, name, endpoint=endpoint) - metric_value(
                before, name, endpoint=endpoint
            )
            assert delta == expected, (name, endpoint, delta, expected)
    (log_dir / "before.prom").write_text(before_text)
    (log_dir / "after.prom").write_text(after_text)
    print("PASS traffic: completion, chat, streaming, invalid request; counter delta=4")
    print(
        "PASS gauges: active request observed; every exposed active series returned to 0"
    )


def main():
    global PROXY, PROMETHEUS
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Local model path; start an owned vLLM backend")
    source.add_argument(
        "--backend-url", help="Use an existing backend without stopping it"
    )
    source.add_argument(
        "--proxy-url", help="Observe an existing proxy without stopping it"
    )
    parser.add_argument("--served-model-name", default="observability-demo")
    parser.add_argument(
        "--mode",
        choices=("aggregated", "disaggregated"),
        default="aggregated",
        help="Existing proxy topology; disaggregated requires decode-first",
    )
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--serve", action="store_true", help="Keep demo running until Ctrl-C"
    )
    parser.add_argument(
        "--prometheus-host",
        type=ipaddress.IPv4Address,
        default="127.0.0.1",
        help="Prometheus listen IPv4 address; non-loopback access has no authentication",
    )
    args = parser.parse_args()
    prometheus_host = str(args.prometheus_host)
    client_host = (
        "127.0.0.1" if args.prometheus_host.is_unspecified else prometheus_host
    )
    PROMETHEUS = f"http://{client_host}:19090"
    if not args.prometheus_host.is_loopback:
        print(
            f"WARNING: Prometheus listens on {prometheus_host}:19090 without "
            "authentication. Restrict inbound access to trusted clients.",
            flush=True,
        )
    if args.model and not Path(args.model).is_dir():
        parser.error("--model must be an existing local directory")
    if args.device_id < 0:
        parser.error("--device-id must be nonnegative")
    if args.mode == "disaggregated" and not args.proxy_url:
        parser.error("disaggregated mode requires an existing decode-first --proxy-url")
    if args.proxy_url:
        target = urllib.parse.urlsplit(args.proxy_url)
        if (
            target.scheme != "http"
            or not target.hostname
            or target.path not in ("", "/")
            or target.username
            or target.query
        ):
            parser.error(
                "--proxy-url must be an HTTP base URL without credentials/query"
            )
        PROXY = args.proxy_url.rstrip("/")
    backend = args.backend_url or "http://127.0.0.1:18100"
    parsed = urllib.parse.urlsplit(backend)
    if parsed.scheme != "http" or not parsed.hostname or parsed.path not in ("", "/"):
        parser.error("--backend-url must be an HTTP base URL")
    ports = [(prometheus_host, 19090)] + (
        [] if args.proxy_url else [("127.0.0.1", 18868)]
    )
    if args.model:
        ports.append(("127.0.0.1", 18100))
    for host, port in ports:
        if not port_free(port, host):
            raise RuntimeError(
                f"Cannot bind {host}:{port}; address unavailable or port in use; "
                "nothing was stopped"
            )
    log_dir = HERE / "logs" / time.strftime("%Y%m%dT%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=False)
    tools = Path(os.environ["PROMETHEUS_DIR"])
    subprocess.run(
        [str(tools / "promtool"), "check", "config", "prometheus.yml"],
        cwd=HERE,
        check=True,
    )
    subprocess.run(
        [str(tools / "promtool"), "test", "rules", "alerts.test.yml"],
        cwd=HERE,
        check=True,
    )
    config = yaml.safe_load((HERE / "xpyd.yaml").read_text())
    config["instances"][0].update(address=parsed.netloc, model=args.served_model_name)
    config_path = log_dir / "xpyd.yaml"
    config_path.write_text(yaml.safe_dump(config))
    prometheus_config = yaml.safe_load((HERE / "prometheus.yml").read_text())
    prometheus_config["rule_files"] = [str(HERE / "alerts.yml")]
    prometheus_config["scrape_configs"][0]["static_configs"][0]["targets"] = [
        urllib.parse.urlsplit(PROXY).netloc
    ]
    prometheus_path = log_dir / "prometheus.yml"
    prometheus_path.write_text(yaml.safe_dump(prometheus_config))
    subprocess.run(
        [str(tools / "promtool"), "check", "config", str(prometheus_path)], check=True
    )
    processes = []
    logs = []

    def start(name, command, env=None):
        log = (log_dir / f"{name}.log").open("ab")
        logs.append(log)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(process)
        return process

    def start_proxy():
        return start(
            "proxy",
            [
                sys.executable,
                "-c",
                "from xpyd.proxy import main; main()",
                "--config",
                str(config_path),
            ],
            {**os.environ, "PYTHONPATH": str(ROOT)},
        )

    def ready():
        instances = get_json(PROXY + "/status/instances")["aggregated_instances"]
        return len(instances) == 1 and instances[0]["status"] == "healthy"

    def phase(text):
        print(f"=== {text} ===", flush=True)
        with (log_dir / "phases.log").open("a") as log:
            log.write(f"=== {text} ===\n")

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        phase("1: start Prometheus and connect to proxy")
        proxy = None if args.proxy_url else start_proxy()
        start(
            "prometheus",
            [
                str(tools / "prometheus"),
                "--config.file=" + str(prometheus_path),
                "--storage.tsdb.path=" + str(log_dir / "data"),
                "--storage.tsdb.retention.time=1d",
                f"--web.listen-address={prometheus_host}:19090",
            ],
        )
        wait_for(
            "proxy startup",
            lambda: http(PROXY + "/status/instances")[0] == 200,
            processes,
        )
        wait_for(
            "Prometheus startup",
            lambda: http(PROMETHEUS + "/-/ready")[0] == 200,
            processes,
        )
        if args.model:
            assert (
                http(
                    PROXY + "/v1/completions",
                    {
                        "model": args.served_model_name,
                        "prompt": "offline",
                        "max_tokens": 1,
                    },
                )[0]
                == 503
            )
            assert http(PROXY + "/health")[0] == 503
            phase("2: start local vLLM; proxy-first 503 confirmed")
            env = dict(os.environ)
            if args.device == "xpu":
                env["ZE_AFFINITY_MASK"] = str(args.device_id)
            elif args.device == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
            else:
                env.update(VLLM_TARGET_DEVICE="cpu", VLLM_CPU_KVCACHE_SPACE="1")
            start(
                "backend",
                [
                    sys.executable,
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    args.model,
                    "--served-model-name",
                    args.served_model_name,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "18100",
                    "--dtype",
                    "bfloat16",
                    "--max-model-len",
                    "1024",
                    "--max-num-seqs",
                    "4",
                    "--enforce-eager",
                    "--kv-cache-memory-bytes",
                    "536870912",
                ],
                env,
            )
        if args.proxy_url:
            assert http(PROXY + "/health")[0] == 200, "Existing proxy must be healthy"
        else:
            wait_for("backend discovery", ready, processes, timeout=900)
        wait_for(
            "scrape up=1",
            lambda: query('up{job="xpyd-demo"}')
            and float(query('up{job="xpyd-demo"}')[0]["value"][1]) == 1,
            processes,
        )
        status, body = http(
            PROXY + "/v1/completions",
            {
                "model": args.served_model_name,
                "prompt": "Warm up monitoring",
                "max_tokens": 1,
                "temperature": 0,
            },
            timeout=180,
        )
        assert status == 200, (status, body[:500])
        baseline_count = metric_value(
            samples(scrape()), "proxy_request_duration_seconds_count"
        )
        duration_count_query = (
            'sum(proxy_request_duration_seconds_count{job="xpyd-demo"})'
        )
        wait_for(
            "Prometheus histogram baseline",
            lambda: bool(query(duration_count_query))
            and float(query(duration_count_query)[0]["value"][1]) >= baseline_count,
            processes,
        )
        phase("3: verify traffic and metric deltas")
        traffic(args.served_model_name, log_dir, processes, args.mode)
        expressions = {
            "up": 'up{job="xpyd-demo"}',
            "request_rate": 'sum(rate(proxy_requests_total{job="xpyd-demo"}[1m]))',
            "latency_p95": "histogram_quantile(0.95, sum by (le) "
            '(rate(proxy_request_duration_seconds_bucket{job="xpyd-demo"}[1m])))',
            "active": 'proxy_active_requests{job="xpyd-demo"}',
        }
        if args.mode == "disaggregated":
            for name in (
                "proxy_prefill_duration_seconds",
                "proxy_ttft_seconds",
                "proxy_kv_transfer_duration_seconds",
            ):
                expressions[name + "_p95"] = (
                    "histogram_quantile(0.95, sum by (le, model) "
                    f'(rate({name}_bucket{{job="xpyd-demo"}}[1m])))'
                )
        wait_for(
            "PromQL request rate",
            lambda: bool(query(expressions["request_rate"]))
            and float(query(expressions["request_rate"])[0]["value"][1]) > 0,
            processes,
        )
        expected_count = metric_value(
            samples(scrape()), "proxy_request_duration_seconds_count"
        )
        wait_for(
            "Prometheus observed traffic histogram",
            lambda: bool(query(duration_count_query))
            and float(query(duration_count_query)[0]["value"][1]) >= expected_count,
            processes,
        )
        results = {name: query(expr) for name, expr in expressions.items()}
        for name, result in results.items():
            assert result and all(
                math.isfinite(float(item["value"][1])) for item in result
            ), (name, result)
        (log_dir / "queries.json").write_text(json.dumps(results, indent=2))
        print("PASS PromQL: up=1, request rate >0, finite p95, active gauge")
        if proxy is not None:
            phase("4: stop proxy; verify real alert firing")
            stop(proxy)
            processes.remove(proxy)
            wait_for("proxy port released", lambda: port_free(18868), processes)
            wait_for(
                "XPyDProxyDown firing",
                lambda: bool(
                    query('ALERTS{alertname="XPyDProxyDown",alertstate="firing"}')
                ),
                processes,
                timeout=30,
            )
            print("PASS alert: XPyDProxyDown fired after proxy stopped")
            phase("5: restart proxy; verify alert recovery")
            start_proxy()
            wait_for("proxy rediscovery", ready, processes)
            wait_for(
                "alert resolved",
                lambda: bool(query(expressions["up"]))
                and float(query(expressions["up"])[0]["value"][1]) == 1
                and not query('ALERTS{alertname="XPyDProxyDown"}'),
                processes,
                timeout=30,
            )
            traffic(args.served_model_name, log_dir, processes, args.mode)
            print("PASS recovery: target up=1, alert resolved, inference recovered")
        else:
            print("Existing proxy not stopped; live alert/recovery test not performed.")
        print(f"Logs and query snapshots: {log_dir}")
        print(f"Prometheus: {PROMETHEUS}/query")
        print(f"Targets: {PROMETHEUS}/targets")
        print(f"Alerts: {PROMETHEUS}/alerts")
        print(f"Metrics: {PROXY}/metrics")
        if args.serve:
            print(
                "Demo remains running; sends one request every 5s. Ctrl-C cleans up.",
                flush=True,
            )
            while True:
                time.sleep(5)
                check_processes("demo services", processes)
                status, body = http(
                    PROXY + "/v1/completions",
                    {
                        "model": args.served_model_name,
                        "prompt": "Say hello",
                        "max_tokens": 16,
                        "temperature": 0,
                        "stream": args.mode == "disaggregated",
                    },
                    timeout=180,
                )
                check_processes("demo services", processes)
                if status != 200:
                    raise RuntimeError(
                        f"Demo traffic failed: HTTP {status}: {body[:300]!r}"
                    )
                if args.mode == "disaggregated" and b"data: [DONE]" not in body:
                    raise RuntimeError("Demo streaming traffic ended without [DONE]")
    finally:
        failures = cleanup_owned_services(processes, logs, ports)
        if failures:
            raise RuntimeError("Service cleanup failed: " + "; ".join(failures))
    print("Owned services stopped; ports released. Existing backend was not stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user.")
