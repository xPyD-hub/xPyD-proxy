"""Regression checks for the standalone Prometheus example."""

import ctypes
import os
import runpy
import select
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

DEMO = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "examples/monitoring/prometheus/run.py")
)


@pytest.mark.parametrize("host", [None, "127.0.0.1"])
def test_proxy_listener_uses_configured_host(host):
    from xpyd.config import ProxyConfig
    from xpyd.proxy import ProxyServer

    path = (
        Path(__file__).resolve().parents[2] / "examples/monitoring/prometheus/xpyd.yaml"
    )
    config = ProxyConfig.from_yaml(path)
    assert config.host == "127.0.0.1"
    if host is None:
        config = ProxyConfig(**config.model_dump(exclude={"host"}))
    with patch("xpyd.proxy.uvicorn.Server") as server:
        ProxyServer(config).run_server()
    listener = server.call_args.args[0]
    assert listener.host == (host or "0.0.0.0")
    assert listener.port == 18868


def timing_samples(prefill, ttft, transfer, count=1):
    values = {}
    labels = frozenset({("model", "demo")})
    for name, value in (
        ("proxy_prefill_duration_seconds", prefill),
        ("proxy_ttft_seconds", ttft),
        ("proxy_kv_transfer_duration_seconds", transfer),
    ):
        values[(name + "_sum", labels)] = value
        values[(name + "_count", labels)] = count
    return values


def test_pd_timing_uses_same_request_deltas():
    before = timing_samples(10, 30, 20, count=100)
    after = timing_samples(10.2, 30.5, 20.3, count=101)
    result = DEMO["assert_pd_timing"](before, after, "demo")
    assert result["proxy_kv_transfer_duration_seconds"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "after",
    [
        timing_samples(0.2, 0.2, 0.3),
        timing_samples(0.2, 0.5, 0.1),
        timing_samples(0.2, 0.5, 0.3, count=2),
        timing_samples(0.2, 0.5, -0.3),
    ],
)
def test_pd_timing_rejects_wrong_semantics(after):
    with pytest.raises(AssertionError):
        DEMO["assert_pd_timing"]({}, after, "demo")


def test_gauges_cannot_cancel_each_other():
    values = {
        ("proxy_active_requests", frozenset()): 0,
        ("proxy_decode_active_requests", frozenset({("model", "a")})): 1,
        ("proxy_decode_active_requests", frozenset({("model", "b")})): -1,
    }
    with pytest.raises(AssertionError):
        DEMO["assert_idle"](values)


def test_missing_active_gauge_is_not_idle():
    with pytest.raises(AssertionError):
        DEMO["assert_idle"]({})


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-Inf"])
def test_nonfinite_samples_fail(value):
    with pytest.raises(AssertionError):
        DEMO["samples"](f"# TYPE demo gauge\ndemo {value}\n")


def test_listening_port_is_not_free():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen()
        assert not DEMO["port_free"](server.getsockname()[1])


def test_closed_connection_does_not_block_restarting_server():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen()
        with socket.create_connection(("127.0.0.1", port)) as client:
            connection, _ = server.accept()
            connection.close()
            assert client.recv(1) == b""
    assert DEMO["port_free"](port)


def test_explicit_listener_address_is_checked():
    with socket.socket() as server:
        server.bind(("127.0.0.2", 0))
        server.listen()
        port = server.getsockname()[1]
        assert DEMO["port_free"](port)
        assert not DEMO["port_free"](port, "127.0.0.2")


@pytest.mark.parametrize("host", ["http://127.0.0.1", "127.0.0.1:19090", "invalid"])
def test_invalid_prometheus_host_is_rejected(monkeypatch, host):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--backend-url",
            "http://127.0.0.1:18100",
            "--prometheus-host",
            host,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        DEMO["main"]()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("host", "client_host"),
    [("127.0.0.1", "127.0.0.1"), ("127.0.0.2", "127.0.0.2"), ("0.0.0.0", "127.0.0.1")],
)
def test_prometheus_client_and_port_check_follow_listener(
    monkeypatch, host, client_host
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--backend-url",
            "http://127.0.0.1:18100",
            "--prometheus-host",
            host,
        ],
    )
    with patch.dict(DEMO["main"].__globals__, port_free=lambda port, host: False):
        with pytest.raises(RuntimeError, match=f"Cannot bind {host}:19090"):
            DEMO["main"]()
        assert DEMO["main"].__globals__["PROMETHEUS"] == f"http://{client_host}:19090"


def test_cleanup_checks_explicit_listener_address():
    calls = []
    with patch.dict(
        DEMO["cleanup_owned_services"].__globals__,
        port_free=lambda port, host: calls.append((host, port)) or True,
    ):
        assert DEMO["cleanup_owned_services"]([], [], [("127.0.0.2", 19090)]) == []
    assert calls == [("127.0.0.2", 19090)]


@pytest.fixture
def owned_group():
    if sys.platform != "linux":
        pytest.skip("Process-group cleanup requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    groups = []

    def spawn(behavior):
        child_code = (
            "import signal,sys,time\n"
            + (
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                if behavior == "ignore"
                else "signal.signal(signal.SIGTERM, "
                "lambda *_: (time.sleep(0.2), sys.exit(0)))\n"
            )
            + "print('ready', flush=True)\ntime.sleep(60)\n"
        )
        parent_code = (
            "import subprocess,sys,time\n"
            "child=subprocess.Popen([sys.executable,'-c',sys.argv[1]],"
            "stdout=subprocess.PIPE,text=True)\n"
            "child.stdout.readline()\n"
            "print(child.pid,flush=True)\ntime.sleep(60)\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", parent_code, child_code],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        groups.append(parent)
        assert select.select([parent.stdout], [], [], 5)[0], "Child did not start"
        assert int(parent.stdout.readline()) > 0
        return parent

    try:
        yield spawn
    finally:
        for parent in groups:
            try:
                os.killpg(parent.pid, signal.SIGKILL)
            except ProcessLookupError:
                # The owned group may already have exited.
                pass
            parent.wait(timeout=5)
            while True:
                try:
                    os.waitpid(-parent.pid, 0)
                except ChildProcessError:
                    break
            parent.stdout.close()
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def test_stop_waits_for_workers_after_parent_exit(owned_group):
    parent = owned_group("delay")
    DEMO["stop"](parent, timeout=2, kill_timeout=2)
    assert parent.poll() is not None
    assert not DEMO["group_running"](parent.pid)
    DEMO["stop"](parent, timeout=2, kill_timeout=2)


@pytest.mark.parametrize("parent_exited", [False, True])
def test_stop_kills_unresponsive_workers_and_reports_failure(
    owned_group, parent_exited
):
    parent = owned_group("ignore")
    if parent_exited:
        parent.terminate()
        parent.wait(timeout=5)
    with pytest.raises(RuntimeError, match="required forced termination"):
        DEMO["stop"](parent, timeout=0.2, kill_timeout=2)
    assert not DEMO["group_running"](parent.pid)


def test_cleanup_continues_after_process_log_and_port_failures(monkeypatch):
    namespace = DEMO["cleanup_owned_services"].__globals__
    processes = [object(), object()]
    stop = Mock(side_effect=[PermissionError("signal denied"), None])
    monkeypatch.setitem(namespace, "stop", stop)
    monkeypatch.setitem(
        namespace, "wait_for", Mock(side_effect=TimeoutError("port still occupied"))
    )
    log = Mock()
    log.close.side_effect = OSError("log close failed")
    failures = DEMO["cleanup_owned_services"](processes, [log], [("127.0.0.1", 19090)])
    assert [call.args[0] for call in stop.call_args_list] == processes[::-1]
    assert failures == ["signal denied", "log close failed", "port still occupied"]


@pytest.fixture
def isolated_main(tmp_path, monkeypatch):
    namespace = DEMO["main"].__globals__
    for name in ("xpyd.yaml", "prometheus.yml"):
        shutil.copyfile(namespace["HERE"] / name, tmp_path / name)
    monkeypatch.setitem(namespace, "HERE", tmp_path)
    monkeypatch.setitem(namespace, "PROXY", "http://127.0.0.1:18868")
    monkeypatch.setitem(namespace, "PROMETHEUS", "http://127.0.0.1:19090")
    monkeypatch.setenv("PROMETHEUS_DIR", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--proxy-url", "http://127.0.0.1:18868", "--serve"],
    )
    process = Mock(pid=987654)
    process.poll.return_value = None
    start = Mock(return_value=process)
    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(subprocess, "run", Mock())
    monkeypatch.setattr(signal, "signal", Mock())
    monkeypatch.setitem(namespace, "port_free", lambda *args: True)
    monkeypatch.setitem(namespace, "wait_for", Mock())
    monkeypatch.setitem(namespace, "traffic", Mock())
    monkeypatch.setitem(namespace, "scrape", lambda: "")
    monkeypatch.setitem(namespace, "query", lambda expr: [{"value": [0, "1"]}])
    http = Mock(return_value=(200, b"{}"))
    monkeypatch.setitem(namespace, "http", http)
    errors = []

    def cleanup(processes, logs, ports):
        assert processes == [process]
        for log in logs:
            log.close()
        return errors

    cleanup_mock = Mock(side_effect=cleanup)
    monkeypatch.setitem(namespace, "cleanup_owned_services", cleanup_mock)
    yield namespace, process, http, errors, cleanup_mock
    assert start.call_count == 1
    assert start.call_args.kwargs["start_new_session"] is True
    assert cleanup_mock.call_count == 1


@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("startup failed")])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_main_preserves_errors_and_reports_cleanup_failure(
    isolated_main, monkeypatch, error, cleanup_fails
):
    namespace, _, _, failures, _ = isolated_main
    monkeypatch.setitem(namespace, "wait_for", Mock(side_effect=error))
    if cleanup_fails:
        failures.append("port did not close")
        with pytest.raises(RuntimeError, match="Service cleanup failed") as exc:
            DEMO["main"]()
        assert exc.value.__context__ is error
    else:
        with pytest.raises(type(error)) as exc:
            DEMO["main"]()
        assert exc.value is error


@pytest.mark.parametrize("when", ["sleep", "request"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_serve_detects_owned_process_exit(isolated_main, monkeypatch, when, exit_code):
    namespace, process, http, _, _ = isolated_main
    traffic_calls = []

    def sleep(seconds):
        if when == "sleep":
            process.poll.return_value = exit_code

    def request(url, body=None, **kwargs):
        if body and body.get("prompt") == "Say hello":
            traffic_calls.append(url)
            process.poll.return_value = exit_code
        return 200, b"{}"

    sleep_mock = Mock(side_effect=sleep)
    monkeypatch.setattr(namespace["time"], "sleep", sleep_mock)
    http.side_effect = request
    with pytest.raises(RuntimeError, match=f"exited with code {exit_code}"):
        DEMO["main"]()
    assert sleep_mock.call_count == 1
    assert len(traffic_calls) == (1 if when == "request" else 0)
