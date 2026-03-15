# SPDX-License-Identifier: Apache-2.0
"""MicroPDProxyServer.

Ported from vllm-gaudi pd_xpyd/proxy_server.py with minimal dependencies.

The proxy routes incoming OpenAI-compatible requests through two phases:

  1. **Prefill** – sends a trimmed request (``stream=False``, ``max_tokens=1``)
     to a prefill node for KV-cache preparation.
  2. **Decode** – forwards the original request to a decode node for
     autoregressive generation.

The decode node's response is returned to the client (streaming or
non-streaming).
"""

import argparse
import ipaddress
import itertools
import json
import logging
import os
import sys
import threading
import uuid
from abc import ABC, abstractmethod
from typing import Optional

import httpx
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(_handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Scheduling policies
# ---------------------------------------------------------------------------


class SchedulingPolicy(ABC):
    """Base class for request scheduling across instances."""

    def __init__(self):
        self.lock = threading.Lock()

    @abstractmethod
    def schedule(self, cycler, is_prompt=None, request_len=None):
        raise NotImplementedError

    def schedule_completion(
        self, prefill_instance=None, decode_instance=None, req_len=None
    ):
        """Called when a request completes (for bookkeeping)."""


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    """Simple round-robin scheduling."""

    def __init__(self, prefill_instances=None, decode_instances=None):
        super().__init__()

    def schedule(self, cycler, is_prompt=None, request_len=None):
        with self.lock:
            return next(cycler)


class LoadBalancedScheduler(SchedulingPolicy):
    """Schedule to the least-loaded instance."""

    def __init__(self, prefill_instances, decode_instances):
        super().__init__()
        self.prefill_instances = list(prefill_instances)
        self.decode_instances = list(decode_instances)
        self.prefill_utils_counter = [0] * len(self.prefill_instances)
        self.prefill_bs_counter = [0] * len(self.prefill_instances)
        self.decode_kv_utils_counter = [0] * len(self.decode_instances)
        self.decode_bs_counter = [0] * len(self.decode_instances)

    def schedule(self, cycler, is_prompt=None, request_len=None):
        request_len = request_len or 0
        with self.lock:
            if is_prompt:
                min_val = min(self.prefill_utils_counter)
                idx = self.prefill_utils_counter.index(min_val)
                self.prefill_bs_counter[idx] += 1
                self.prefill_utils_counter[idx] += request_len
                return self.prefill_instances[idx]
            else:
                min_val = min(self.decode_bs_counter)
                if min_val == 0:
                    idx = self.decode_bs_counter.index(min_val)
                else:
                    candidates = [
                        i
                        for i, v in enumerate(self.decode_bs_counter)
                        if v == min_val
                    ]
                    idx = min(
                        candidates,
                        key=lambda i: self.decode_kv_utils_counter[i],
                    )
                self.decode_bs_counter[idx] += 1
                self.decode_kv_utils_counter[idx] += request_len
                return self.decode_instances[idx]

    def schedule_completion(
        self, prefill_instance=None, decode_instance=None, req_len=None
    ):
        req_len = req_len or 0
        with self.lock:
            if prefill_instance and prefill_instance in self.prefill_instances:
                idx = self.prefill_instances.index(prefill_instance)
                if self.prefill_bs_counter[idx] > 0:
                    self.prefill_bs_counter[idx] -= 1
                    if all(c == 0 for c in self.prefill_bs_counter):
                        self.prefill_utils_counter = [0] * len(
                            self.prefill_instances
                        )
                    else:
                        self.prefill_utils_counter[idx] -= req_len
            if decode_instance and decode_instance in self.decode_instances:
                idx = self.decode_instances.index(decode_instance)
                if self.decode_bs_counter[idx] > 0:
                    self.decode_bs_counter[idx] -= 1
                    if all(c == 0 for c in self.decode_bs_counter):
                        self.decode_kv_utils_counter = [0] * len(
                            self.decode_instances
                        )
                    else:
                        self.decode_kv_utils_counter[idx] -= req_len


# ---------------------------------------------------------------------------
# Lightweight token counting (no tokenizer library needed)
# ---------------------------------------------------------------------------


def _count_message_tokens(messages):
    """Rough token count: ~1 token per 4 characters (matches common.py)."""
    total_chars = sum(
        len(m.get("content", "")) + len(m.get("role", "")) for m in messages
    )
    return max(1, total_chars // 4)


def _count_prompt_tokens(prompt):
    """Count tokens for a prompt string or token list."""
    if isinstance(prompt, str):
        return max(1, len(prompt) // 4)
    if isinstance(prompt, list):
        if all(isinstance(p, int) for p in prompt):
            # Already tokenized (flat list of token IDs)
            return len(prompt)
        return sum(
            _count_prompt_tokens(p) if isinstance(p, str) else len(p)
            for p in prompt
        )
    return 100


# ---------------------------------------------------------------------------
# Async generator wrapper
# ---------------------------------------------------------------------------


async def _streaming_wrapper(
    generator, callback_owner=None, decode_instance=None, req_len=None
):
    """Wraps a response generator and calls on_done when finished."""
    try:
        async for chunk in generator:
            yield chunk
    finally:
        if callback_owner:
            callback_owner.on_done(
                decode_instance=decode_instance, req_len=req_len
            )


# ---------------------------------------------------------------------------
# Proxy core
# ---------------------------------------------------------------------------


class Proxy:
    """Routes requests between prefill and decode nodes."""

    def __init__(self, prefill_instances, decode_instances, scheduling_policy):
        self.prefill_instances = list(prefill_instances)
        self.decode_instances = list(decode_instances)
        self.prefill_cycler = itertools.cycle(self.prefill_instances)
        self.decode_cycler = itertools.cycle(self.decode_instances)
        self.scheduling_policy = scheduling_policy
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        self.router.post("/v1/chat/completions")(self.create_chat_completion)
        self.router.post("/v1/completions")(self.create_completion)
        self.router.get("/health")(self.get_health)
        self.router.get("/status")(self.get_status)

    def _schedule(self, cycler, is_prompt=None, request_len=None):
        return self.scheduling_policy.schedule(cycler, is_prompt, request_len)

    def on_done(
        self, prefill_instance=None, decode_instance=None, req_len=None
    ):
        self.scheduling_policy.schedule_completion(
            prefill_instance=prefill_instance,
            decode_instance=decode_instance,
            req_len=req_len,
        )

    @staticmethod
    def _headers(request_id):
        h = {"X-Request-Id": request_id}
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    # ---- endpoints ----

    async def get_health(self):
        return JSONResponse({"status": "ok", "node_type": "proxy"})

    async def get_status(self):
        return JSONResponse(
            {
                "prefill_node_count": len(self.prefill_instances),
                "decode_node_count": len(self.decode_instances),
                "prefill_nodes": self.prefill_instances,
                "decode_nodes": self.decode_instances,
            }
        )

    async def _send_prefill(self, instance, endpoint, data, request_id):
        """Send a trimmed request to the prefill node (KV preparation)."""
        pf = data.copy()
        pf["stream"] = False
        pf["max_tokens"] = 1
        pf.pop("stream_options", None)
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(
                f"http://{instance}{endpoint}",
                json=pf,
                headers=self._headers(request_id),
            )
            resp.raise_for_status()
            return resp

    async def _forward_decode_stream(
        self, instance, endpoint, data, request_id
    ):
        """Forward to decode node, yielding raw response bytes."""
        client = httpx.AsyncClient(timeout=None)
        try:
            async with client.stream(
                "POST",
                f"http://{instance}{endpoint}",
                json=data,
                headers=self._headers(request_id),
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
        finally:
            await client.aclose()

    async def _forward_decode(self, instance, endpoint, data, request_id):
        """Forward to decode node, return full response."""
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(
                f"http://{instance}{endpoint}",
                json=data,
                headers=self._headers(request_id),
            )
            resp.raise_for_status()
            return resp

    async def create_chat_completion(self, raw_request: Request):
        try:
            data = await raw_request.json()
            rid = str(uuid.uuid4())
            tok_len = _count_message_tokens(data.get("messages", []))

            pfill = self._schedule(
                self.prefill_cycler, is_prompt=True, request_len=tok_len
            )
            await self._send_prefill(
                pfill, "/v1/chat/completions", data, rid
            )

            dec = self._schedule(
                self.decode_cycler, is_prompt=False, request_len=tok_len
            )

            if data.get("stream", False):
                gen = self._forward_decode_stream(
                    dec, "/v1/chat/completions", data, rid
                )
                wrapped = _streaming_wrapper(
                    gen, self, dec, req_len=tok_len
                )
                return StreamingResponse(
                    wrapped, media_type="text/event-stream"
                )

            resp = await self._forward_decode(
                dec, "/v1/chat/completions", data, rid
            )
            self.on_done(decode_instance=dec, req_len=tok_len)
            return JSONResponse(
                content=resp.json(), status_code=resp.status_code
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error in create_chat_completion")
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def create_completion(self, raw_request: Request):
        try:
            data = await raw_request.json()
            rid = str(uuid.uuid4())
            tok_len = _count_prompt_tokens(data.get("prompt", ""))

            pfill = self._schedule(
                self.prefill_cycler, is_prompt=True, request_len=tok_len
            )
            await self._send_prefill(pfill, "/v1/completions", data, rid)

            dec = self._schedule(
                self.decode_cycler, is_prompt=False, request_len=tok_len
            )

            if data.get("stream", False):
                gen = self._forward_decode_stream(
                    dec, "/v1/completions", data, rid
                )
                wrapped = _streaming_wrapper(
                    gen, self, dec, req_len=tok_len
                )
                return StreamingResponse(
                    wrapped, media_type="text/event-stream"
                )

            resp = await self._forward_decode(
                dec, "/v1/completions", data, rid
            )
            self.on_done(decode_instance=dec, req_len=tok_len)
            return JSONResponse(
                content=resp.json(), status_code=resp.status_code
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error in create_completion")
            raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Instance-spec parsing
# ---------------------------------------------------------------------------


def parse_instance_spec(instance_spec):
    """Parse instance specs like ``host:port``, ``host:start-end``,
    or ``host:p1,p2``."""
    if ":" not in instance_spec:
        raise ValueError(
            f"Invalid instance specification '{instance_spec}'. "
            "Expected format: host:port"
        )

    host, port_spec = instance_spec.rsplit(":", 1)
    instances = []

    if "-" in port_spec:
        lo, hi = port_spec.split("-", 1)
        lo, hi = int(lo), int(hi)
        if lo > hi:
            raise ValueError(f"Invalid port range: {lo} > {hi}")
        for p in range(lo, hi + 1):
            instances.append(f"{host}:{p}")
    elif "," in port_spec:
        for p in port_spec.split(","):
            instances.append(f"{host}:{int(p.strip())}")
    else:
        instances.append(f"{host}:{int(port_spec)}")

    return instances


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    prefill_instances,
    decode_instances,
    scheduling_policy=None,
):
    """Create and return a configured FastAPI application."""
    if scheduling_policy is None:
        scheduling_policy = RoundRobinSchedulingPolicy()

    proxy = Proxy(prefill_instances, decode_instances, scheduling_policy)

    application = FastAPI(title="MicroPDProxyServer")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(proxy.router)
    return application


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MicroPDProxyServer – lightweight PD proxy server"
    )
    parser.add_argument(
        "--prefill",
        "-p",
        type=str,
        nargs="+",
        help="Prefill node URLs (host:port)",
    )
    parser.add_argument(
        "--decode",
        "-d",
        type=str,
        nargs="+",
        help="Decode node URLs (host:port)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Proxy server port"
    )
    parser.add_argument(
        "--roundrobin",
        action="store_true",
        help="Use round-robin scheduling (default: load-balanced)",
    )

    args = parser.parse_args()

    prefill_instances = []
    if args.prefill:
        for spec in args.prefill:
            prefill_instances.extend(parse_instance_spec(spec))

    decode_instances = []
    if args.decode:
        for spec in args.decode:
            decode_instances.extend(parse_instance_spec(spec))

    if not decode_instances:
        sys.exit("Error: specify at least one decode node with --decode")

    if args.roundrobin:
        policy = RoundRobinSchedulingPolicy()
    else:
        policy = LoadBalancedScheduler(prefill_instances, decode_instances)

    app = create_app(prefill_instances, decode_instances, policy)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
