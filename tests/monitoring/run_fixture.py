#!/usr/bin/env python3
"""CPU-only monitoring integration: real proxy/Prometheus, synthetic backends."""

import json
import os
import runpy
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/monitoring/prometheus"
DEMO = runpy.run_path(str(EXAMPLE / "run.py"))
MODEL = "observability-demo"


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body):
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/health":
            self.reply({"status": "ok"})
        elif self.path == "/v1/models":
            self.reply({"data": [{"id": MODEL, "max_model_len": 1024}]})
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        count = body.get("max_tokens", 16)
        chat = self.path == "/v1/chat/completions"
        if self.path not in ("/v1/completions", "/v1/chat/completions"):
            self.send_error(404)
            return
        time.sleep(0.05)
        choice = {
            "index": 0,
            "finish_reason": "length",
            **(
                {"message": {"role": "assistant", "content": "fixture output"}}
                if chat
                else {"text": "fixture output"}
            ),
        }
        result = {
            "id": "fixture-request",
            "created": 0,
            "model": MODEL,
            "object": "chat.completion" if chat else "text_completion",
            "choices": [choice],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": count,
                "total_tokens": count + 8,
            },
        }
        if not body.get("stream"):
            self.reply(result)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for _ in range(20):
            chunk = {
                **result,
                "choices": [{"text": "x", "index": 0, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.05)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def start_backend(stack):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    stack.callback(server.server_close)
    stack.callback(server.shutdown)
    return f"127.0.0.1:{server.server_port}"


def main():
    logs = EXAMPLE / "logs" / ("fixture-" + time.strftime("%Y%m%dT%H%M%S"))
    logs.mkdir(parents=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"}
    with ExitStack() as stack:
        prefill = start_backend(stack)
        decode = start_backend(stack)
        subprocess.run(
            ["bash", str(EXAMPLE / "run_all.sh"), "--backend-url", f"http://{prefill}"],
            check=True,
            env=env,
        )
        config = yaml.safe_load((EXAMPLE / "xpyd.yaml").read_text())
        config.update(
            instances=[
                {"address": prefill, "role": "prefill", "model": MODEL},
                {"address": decode, "role": "decode", "model": MODEL},
            ],
            disaggregated_mode="direct",
            first_token_source="decode",
        )
        config_path = logs / "pd.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with (logs / "pd-proxy.log").open("wb") as log:
            proxy = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from xpyd.proxy import main; main()",
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                DEMO["wait_for"](
                    "fixture P/D readiness",
                    lambda: DEMO["http"]("http://127.0.0.1:18868/health")[0] == 200,
                    [proxy],
                )
                subprocess.run(
                    [
                        "bash",
                        str(EXAMPLE / "run_all.sh"),
                        "--proxy-url",
                        "http://127.0.0.1:18868",
                        "--mode",
                        "disaggregated",
                    ],
                    check=True,
                    env=env,
                )
                assert proxy.poll() is None, "Attached proxy must remain running"
            finally:
                DEMO["stop"](proxy)
    print(
        "CPU monitoring integration passed (synthetic backends, no real KV transfer)."
    )


if __name__ == "__main__":
    main()
