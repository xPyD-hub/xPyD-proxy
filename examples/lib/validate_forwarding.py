#!/usr/bin/env python3
"""Require a new backend access-log entry for a passthrough smoke request."""

import argparse
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def backend_logs(directory):
    return sorted(
        {
            path
            for pattern in ("vllm*.log", "prefill*.log", "decode*.log")
            for path in directory.glob(pattern)
        }
    )


def validate(url, path, body, log_dir, timeout=10):
    logs = backend_logs(log_dir)
    if not logs:
        raise AssertionError(f"No backend access logs found in {log_dir}")
    offsets = {log: log.stat().st_size for log in logs}
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        status, response_body = exc.code, exc.read()
    assert 200 <= status < 500, (path, status, response_body[:500])
    pattern = re.compile(
        rb'"POST '
        + re.escape(path.encode())
        + rb' HTTP/[\d.]+" '
        + str(status).encode()
        + rb"\b"
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for log, offset in offsets.items():
            assert log.stat().st_size >= offset, f"Backend log truncated: {log}"
            with log.open("rb") as stream:
                stream.seek(offset)
                if pattern.search(stream.read()):
                    print(
                        f"  {path} forwarded (HTTP {status}, backend log: {log.name})"
                    )
                    return
        time.sleep(0.05)
    raise AssertionError(
        f"{path}: HTTP {status} without a new matching backend access-log entry; "
        f"response={response_body[:300]!r}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    validate(args.url, args.path, args.body, args.log_dir)


if __name__ == "__main__":
    main()
