#!/usr/bin/env python3
"""Validate 4P2D scheduling semantics through real NIXL inference."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request

from prometheus_client.parser import text_string_to_metric_families

URL = "http://127.0.0.1:8868"
MODEL = "facebook/opt-125m"
PREFILL = {
    "127.0.0.1:8100",
    "127.0.0.1:8101",
    "127.0.0.1:8102",
    "127.0.0.1:8103",
}
DECODE = {"127.0.0.1:8200", "127.0.0.1:8201"}
ADMIN_KEY = "xpyd-scheduler-test-key"


def post(path: str, payload: dict, headers: dict | None = None) -> bytes:
    request = urllib.request.Request(
        URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def counters() -> dict[tuple[str, str], float]:
    with urllib.request.urlopen(URL + "/metrics", timeout=5) as response:
        families = text_string_to_metric_families(response.read().decode())
        return {
            (
                sample.labels["prefill_instance"],
                sample.labels["decode_instance"],
            ): sample.value
            for family in families
            for sample in family.samples
            if sample.name == "proxy_prefill_requests_total"
            and sample.labels["model"] == MODEL
        }


def request(session: str, prompt: str, max_tokens: int = 32) -> None:
    body = post(
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
        {"X-Session-ID": session},
    )
    output = json.loads(body)
    assert output["choices"][0]["text"], output


def selected_once(session: str, prompt: str) -> tuple[str, str]:
    before = counters()
    request(session, prompt)
    after = counters()
    changed = {
        pair for pair, value in after.items() if value - before.get(pair, 0) == 1
    }
    assert len(changed) == 1, changed
    pair = changed.pop()
    assert pair[0] in PREFILL and pair[1] in DECODE, pair
    return pair


def assert_round_robin() -> tuple[str, str]:
    pairs = [
        selected_once(f"round-{index}", f"round robin request {index}")
        for index in range(8)
    ]
    assert len({pair[0] for pair in pairs[:4]}) == 4, pairs
    assert [pair[0] for pair in pairs[:4]] == [pair[0] for pair in pairs[4:]], pairs
    assert len({pair[1] for pair in pairs[:2]}) == 2, pairs
    assert [pair[1] for pair in pairs[:2]] * 4 == [pair[1] for pair in pairs], pairs
    return pairs[0]


def assert_sticky(strategy: str) -> tuple[str, str]:
    if strategy == "consistent_hash":
        first = selected_once("stable-session", "hash request one")
        second = selected_once("stable-session", "hash request two")
    else:
        first = selected_once("cache-a", "shared cache prefix suffix-a")
        second = selected_once("cache-b", "shared cache prefix suffix-b")
    assert first == second, (strategy, first, second)
    return first


def assert_concurrent_inference(strategy: str) -> tuple[str, str]:
    before = counters()
    errors: list[BaseException] = []
    request_count = 4 if strategy == "loadbalanced" else 16
    barrier = threading.Barrier(request_count + 1)

    def run(index: int) -> None:
        try:
            barrier.wait()
            request(
                f"concurrent-{index}",
                f"concurrent scheduler request {index}",
                max_tokens=16,
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(index,)) for index in range(request_count)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=180)
        assert not thread.is_alive()
    assert not errors, errors

    after = counters()
    changed = {
        pair: value - before.get(pair, 0)
        for pair, value in after.items()
        if value > before.get(pair, 0)
    }
    assert sum(changed.values()) == request_count, changed
    assert changed, changed
    assert {pair[0] for pair in changed} <= PREFILL, changed
    assert {pair[1] for pair in changed} <= DECODE, changed
    return next(iter(changed))


def mutate(role: str, address: str, action: str) -> None:
    payload = {"type": role, "instance": address}
    if action == "remove":
        payload["timeout_seconds"] = 30
    post(
        f"/instances/{action}",
        payload,
        {"x-api-key": ADMIN_KEY},
    )


def assert_node_changes(pair: tuple[str, str]) -> None:
    removed_prefill, removed_decode = pair
    mutate("prefill", removed_prefill, "remove")
    mutate("decode", removed_decode, "remove")
    selected = selected_once("after-remove", "shared cache prefix after remove")
    assert selected[0] in PREFILL - {removed_prefill}, selected
    assert selected[1] in DECODE - {removed_decode}, selected

    mutate("prefill", removed_prefill, "add")
    mutate("decode", removed_decode, "add")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with urllib.request.urlopen(URL + "/status/instances", timeout=5) as response:
            status = json.load(response)
        healthy_p = {
            item["address"]
            for item in status["prefill_instances"]
            if item["status"] == "healthy"
        }
        healthy_d = {
            item["address"]
            for item in status["decode_instances"]
            if item["status"] == "healthy"
        }
        if healthy_p == PREFILL and healthy_d == DECODE:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("removed P/D instances did not recover after add")
    selected_once("after-add", "inference after add")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "strategy",
        choices=(
            "roundrobin",
            "loadbalanced",
            "consistent_hash",
            "power_of_two",
            "cache_aware",
        ),
    )
    strategy = parser.parse_args().strategy
    if strategy == "roundrobin":
        pair = assert_round_robin()
    elif strategy in {"consistent_hash", "cache_aware"}:
        pair = assert_sticky(strategy)
    else:
        pair = assert_concurrent_inference(strategy)
    assert_node_changes(pair)
    print(f"{strategy}: 4P2D scheduling semantics and node changes passed")


if __name__ == "__main__":
    main()
