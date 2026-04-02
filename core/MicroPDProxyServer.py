# SPDX-License-Identifier: Apache-2.0
"""MicroPDProxyServer.

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
import time
from typing import Callable, Optional

import aiohttp
import requests
import uvicorn
from colorlog.escape_codes import escape_codes
from fastapi import (APIRouter, Depends, FastAPI, Header, HTTPException,
                     Request, status)
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from transformers import AutoTokenizer
from asyncio import CancelledError
from fastapi.middleware.cors import CORSMiddleware
try:
    from .config import ProxyConfig
    from .discovery import NodeDiscovery
    from .metrics import get_metrics, track_request_end, track_request_start
    from .health_monitor import HealthMonitor
    from .registry import InstanceRegistry
    from .scheduler import (
        LoadBalancedScheduler,
        RoundRobinSchedulingPolicy,
        SchedulingPolicy,
        default_registry,
    )
except ImportError:
    from config import ProxyConfig
    from discovery import NodeDiscovery
    from metrics import get_metrics, track_request_end, track_request_start
    from health_monitor import HealthMonitor
    from registry import InstanceRegistry
    from scheduler import (
        LoadBalancedScheduler,
        RoundRobinSchedulingPolicy,
        SchedulingPolicy,
        default_registry,
    )

formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s",
                              "%Y-%m-%d %H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

def log_info_color(color, msg, *args):
    """Generic colored log with parameterized message."""
    msg_colored = f"{escape_codes[color]}{msg}{escape_codes['reset']}"
    logger.info(msg_colored, *args)

def log_info_blue(msg, *args):
    log_info_color('cyan', msg, *args)

def log_info_green(msg, *args):
    log_info_color('green', msg, *args)

def log_info_yellow(msg, *args):
    log_info_color('yellow', msg, *args)

def log_info_red(msg, *args):
    log_info_color('red', msg, *args)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=None,
                                        connect=None,
                                        sock_read=None,
                                        sock_connect=None)

def query_instance_model_len(instances, timeout=5.0):
    """
    Query each instance for its max_model_len.
    """
    model_lens = []
    for inst in instances:
        try:
            url = f"http://{inst}/v1/models"
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()["data"][0]
            max_len = data.get("max_model_len", 0)
            model_lens.append(max_len)
            logger.info("Instance %s model_len: %d", inst, max_len)
        except Exception as e:
            logger.warning("Failed to get model_len from %s: %s", inst, e)
            sys.exit(1)
    return model_lens

async def P_first_token_generator(generator_p,
                                  generator_d,
                                  callback_owner=None,
                                  prefill_instance: str = None,
                                  decode_instance: str = None,
                                  req_len: int = None):
    first_decode = True

    try:
        async for chunk in generator_p:
            yield chunk
    finally:
        if callback_owner:
            callback_owner.exception_handler(
                prefill_instance=prefill_instance,
                decode_instance=None,
                req_len=req_len
            )

    try:
        async for chunk in generator_d:
            if first_decode:
                first_decode = False
                continue
            yield chunk
    finally:
        if callback_owner:
            callback_owner.exception_handler(
                prefill_instance=None,
                decode_instance=decode_instance,
                req_len=req_len
            )

async def D_first_token_generator(generator_p,
                                  generator_d,
                                  callback_owner=None,
                                  prefill_instance: str = None,
                                  decode_instance: str = None,
                                  req_len: int = None):
    try:
        async for _ in generator_p:
            continue
    finally:
        if callback_owner:
            callback_owner.exception_handler(
                prefill_instance=prefill_instance,
                decode_instance=None,
                req_len=req_len
            )

    try:
        async for chunk in generator_d:
            yield chunk
    finally:
        if callback_owner:
            callback_owner.exception_handler(
                prefill_instance=None,
                decode_instance=decode_instance,
                req_len=req_len
            )

class Proxy:

    def __init__(self,
                 prefill_instances: list[str],
                 decode_instances: list[str],
                 model: str,
                 scheduling_policy: SchedulingPolicy,
                 custom_create_completion: Optional[Callable[
                     [Request], StreamingResponse]] = None,
                 custom_create_chat_completion: Optional[Callable[
                     [Request], StreamingResponse]] = None,
                 generator_on_p_node: bool = False,
                 registry: Optional[InstanceRegistry] = None):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances)
        self.model = model
        self.scheduling_policy = scheduling_policy
        self.registry = registry
        self.custom_create_completion = custom_create_completion
        self.custom_create_chat_completion = custom_create_chat_completion
        self.router = APIRouter()
        self.setup_routes()
        self.generator = (P_first_token_generator
                          if generator_on_p_node else D_first_token_generator)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def on_done(self,
                prefill_instance: str = None,
                decode_instance: str = None,
                req_len: int = None):
        self.schedule_completion(prefill_instance,
                                 decode_instance,
                                 req_len=req_len)

    def setup_routes(self):
        self.router.post(
            "/v1/completions",
            dependencies=[
                Depends(self.validate_json_request)
            ])(self.custom_create_completion if self.
               custom_create_completion else self.create_completion)
        self.router.post(
            "/v1/chat/completions",
            dependencies=[
                Depends(self.validate_json_request)
            ])(self.custom_create_chat_completion if self.
               custom_create_chat_completion else self.create_chat_completion)

        self.router.options("/v1/completions")(lambda: None)
        self.router.options("/v1/chat/completions")(lambda: None)
        self.router.options("/v1/models")(lambda: None)
        self.router.options("/status")(lambda: None)
        self.router.options("/health")(lambda: None)
        self.router.options("/ping")(lambda: None)
        self.router.options("/tokenize")(lambda: None)
        self.router.options("/detokenize")(lambda: None)
        self.router.options("/version")(lambda: None)
        self.router.options("/v1/embeddings")(lambda: None)
        self.router.options("/pooling")(lambda: None)
        self.router.options("/score")(lambda: None)
        self.router.options("/v1/score")(lambda: None)
        self.router.options("/rerank")(lambda: None)
        self.router.options("/v1/rerank")(lambda: None)
        self.router.options("/v2/rerank")(lambda: None)
        self.router.options("/invocations")(lambda: None)

        self.router.get("/status",
                        response_class=JSONResponse)(self.get_status)
        self.router.post("/instances/add",
                         dependencies=[Depends(self.api_key_authenticate)
                                       ])(self.add_instance_endpoint)
        self.router.get("/health", response_class=PlainTextResponse)(self.get_health)
        self.router.get("/ping", response_class=PlainTextResponse)(self.get_ping)
        self.router.post("/ping", response_class=PlainTextResponse)(self.get_ping)
        self.router.post("/tokenize", response_class=JSONResponse)(self.post_tokenize)
        self.router.post("/detokenize", response_class=JSONResponse)(self.post_detokenize)
        self.router.get("/v1/models", response_class=JSONResponse)(self.get_models)
        self.router.get("/version", response_class=JSONResponse)(self.get_version)
        self.router.post("/v1/embeddings", response_class=JSONResponse)(self.post_embeddings)
        self.router.post("/pooling", response_class=JSONResponse)(self.post_pooling)
        self.router.post("/score", response_class=JSONResponse)(self.post_score)
        self.router.post("/v1/score", response_class=JSONResponse)(self.post_scorev1)
        self.router.post("/rerank", response_class=JSONResponse)(self.post_rerank)
        self.router.post("/v1/rerank", response_class=JSONResponse)(self.post_rerankv1)
        self.router.post("/v2/rerank", response_class=JSONResponse)(self.post_rerankv2)
        self.router.post("/invocations", response_class=JSONResponse)(self.post_invocations)

        # Prometheus metrics
        self.router.get("/metrics")(self.get_metrics)

    @staticmethod
    async def get_metrics():
        return Response(
            content=get_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    async def get_from_instance(self, path: str, is_full_instancelist: int = 0):
        if not self.prefill_instances:
            return JSONResponse(content={"error": "No instances available"}, status_code=500)

        if is_full_instancelist == 0:
            instances = [self.prefill_instances[0]]
        else:
            instances = self.prefill_instances + self.decode_instances

        results = {}
        async with aiohttp.ClientSession() as session:
            for inst in instances:
                url = f"http://{inst}{path}"
                try:
                    async with session.get(url) as resp:
                        try:
                            data = await resp.json()
                            dtype = "json"
                        except aiohttp.ContentTypeError:
                            data = await resp.text()
                            dtype = "text"
                        results[inst] = {
                            "status": resp.status,
                            "type": dtype,
                            "data": data
                        }
                except Exception as e:
                    results[inst] = {
                        "status": 500,
                        "error": str(e)
                    }
                    print(f"Failed to fetch {url}: {e}, continue...")

        return JSONResponse(content=results, status_code=200)

    async def get_version(self):
        return await self.get_from_instance("/version")

    async def get_models(self):
        return await self.get_from_instance("/v1/models")

    async def get_health(self):
        return await self.get_from_instance("/health", is_full_instancelist=1)

    async def get_ping(self):
        return await self.get_from_instance("/ping", is_full_instancelist=1)

    async def post_to_instance(
        self,
        request: Request,
        path: str,
        json_template: dict
    ):
        body = await request.json()

        missing = [k for k in json_template.keys() if k not in body]
        if missing:
            return JSONResponse(
                {"error": f"Missing required fields: {', '.join(missing)}"},
                status_code=400
            )

        payload = json_template.copy()
        payload.update(body)

        url = f"http://{self.prefill_instances[0]}{path}"
        try:
            async with aiohttp.ClientSession() as session, \
                    session.post(url, json=payload) as resp:
                try:
                    content = await resp.json()
                except aiohttp.ContentTypeError:
                    content = {"raw": await resp.text()}
                return JSONResponse(content, status_code=resp.status)
        except Exception as e:
            return JSONResponse(
                {"error": f"Failed to fetch {url}, reason: {str(e)}"},
                status_code=500
            )

    async def post_detokenize(self, request: Request):
        json_template = {"model": "", "tokens": []}
        return await self.post_to_instance(request, "/detokenize", json_template)

    async def post_tokenize(self, request: Request):
        json_template = {"model": "", "prompt": ""}
        return await self.post_to_instance(request, "/tokenize", json_template)

    async def post_embeddings(self, request: Request):
        json_template = {"model": "", "input": ""}
        return await self.post_to_instance(request, "/v1/embeddings", json_template)

    async def post_pooling(self, request: Request):
        json_template = {"model": "", "messages": ""}
        return await self.post_to_instance(request, "/pooling", json_template)

    async def post_score(self, request: Request):
        json_template = {"model": "", "text_1": "", "text_2": "", "predictions": ""}
        return await self.post_to_instance(request, "/score", json_template)

    async def post_scorev1(self, request: Request):
        json_template = {"model": "", "text_1": "", "text_2": "", "predictions": ""}
        return await self.post_to_instance(request, "/v1/score", json_template)

    async def post_rerank(self, request: Request):
        json_template = {"model": "", "query": "", "documents": ""}
        return await self.post_to_instance(request, "/rerank", json_template)

    async def post_rerankv1(self, request: Request):
        json_template = {"model": "", "query": "", "documents": ""}
        return await self.post_to_instance(request, "/v1/rerank", json_template)

    async def post_rerankv2(self, request: Request):
        json_template = {"model": "", "query": "", "documents": ""}
        return await self.post_to_instance(request, "/v2/rerank", json_template)

    async def post_invocations(self, request: Request):
        json_template = {"model": "", "prompt": ""}
        return await self.post_to_instance(request, "/invocations", json_template)

    async def validate_json_request(self, raw_request: Request):
        content_type = raw_request.headers.get("content-type", "").lower()
        if content_type != "application/json":
            raise HTTPException(
                status_code=415,
                detail=
                "Unsupported Media Type: Only 'application/json' is allowed",
            )

    def api_key_authenticate(self, x_api_key: str = Header(...)):
        expected_api_key = os.environ.get("ADMIN_API_KEY")
        if not expected_api_key:
            logger.error("ADMIN_API_KEY is not set in the environment.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error.",
            )
        if x_api_key != expected_api_key:
            logger.warning("Unauthorized access attempt with API Key: %s",
                           x_api_key)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid API Key.",
            )

    async def validate_instance(self, instance: str) -> bool:
        url = f"http://{instance}/v1/models"
        try:
            async with aiohttp.ClientSession(
                    timeout=AIOHTTP_TIMEOUT) as client:
                logger.info("Verifying %s ...", instance)
                async with client.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and len(data["data"]) > 0:
                            model_cur = data["data"][0].get("id", "")
                            if model_cur == self.model:
                                logger.info("Instance: %s could be added.",
                                            instance)
                                return True
                            else:
                                logger.warning("Mismatch model %s : %s != %s",
                                               instance, model_cur, self.model)
                                return False
                        else:
                            return False
                    else:
                        return False
        except aiohttp.ClientError as e:
            logger.error(str(e))
            return False
        except Exception as e:
            logger.error(str(e))
            return False

    async def add_instance_endpoint(self, request: Request):
        try:
            data = await request.json()
            logger.warning(str(data))
            instance_type = data.get("type")
            instance = data.get("instance")
            if instance_type not in ["prefill", "decode"]:
                raise HTTPException(status_code=400,
                                    detail="Invalid instance type.")
            if not instance or ":" not in instance:
                raise HTTPException(status_code=400,
                                    detail="Invalid instance format.")
            host, port_str = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port_str)
                if not (0 < port < 65536):
                    raise HTTPException(status_code=400,
                                        detail="Invalid port number.")
            except Exception as e:
                raise HTTPException(status_code=400,
                                    detail="Invalid instance address.") from e

            is_valid = await self.validate_instance(instance)
            if not is_valid:
                raise HTTPException(status_code=400,
                                    detail="Instance validation failed.")

            if instance_type == "prefill":
                with self.scheduling_policy.lock:
                    if instance not in self.prefill_instances:
                        self.prefill_instances.append(instance)
                        self.prefill_cycler = itertools.cycle(
                            self.prefill_instances)
                    else:
                        raise HTTPException(status_code=400,
                                            detail="Instance already exists.")
            else:
                with self.scheduling_policy.lock:
                    if instance not in self.decode_instances:
                        self.decode_instances.append(instance)
                        self.decode_cycler = itertools.cycle(
                            self.decode_instances)
                    else:
                        raise HTTPException(status_code=400,
                                            detail="Instance already exists.")

            return JSONResponse(content={
                "message":
                f"Added {instance} to {instance_type}_instances."
            })
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error("Error in add_instance_endpoint: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    async def forward_request(self, url, data, use_chunked=True):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
            }
            try:
                async with session.post(url=url, json=data,
                                        headers=headers) as response:
                    if 200 <= response.status < 300 or 400 <= response.status < 500:  # noqa: E501
                        if use_chunked:
                            async for chunk_bytes in response.content.iter_chunked(  # noqa: E501
                                    1024):
                                yield chunk_bytes
                        else:
                            content = await response.read()
                            yield content
                    else:
                        error_content = await response.text()
                        try:
                            error_content = json.loads(error_content)
                        except json.JSONDecodeError:
                            error_content = error_content
                        logger.error("Request failed with status %s: %s",
                                     response.status, error_content)
                        raise HTTPException(
                            status_code=response.status,
                            detail=
                            f"Request failed with status {response.status}: "
                            f"{error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error("ClientError occurred: %s", str(e))
                raise HTTPException(
                    status_code=502,
                    detail=
                    "Bad Gateway: Error communicating with upstream server.",
                ) from e
            except Exception as e:
                logger.error("Unexpected error: %s", str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e

    def schedule(self,
                 cycler: itertools.cycle,
                 is_prompt: int = None,
                 request_len: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 **kwargs) -> str:
        return self.scheduling_policy.schedule(
            cycler, is_prompt, request_len, max_tokens, **kwargs,
        )

    def schedule_completion(self,
                            prefill_instance: str = None,
                            decode_instance: str = None,
                            req_len: int = None):
        self.scheduling_policy.schedule_completion(
            prefill_instance=prefill_instance,
            decode_instance=decode_instance,
            req_len=req_len)

    async def get_status(self):
        status = {
            "prefill_node_count": len(self.prefill_instances),
            "decode_node_count": len(self.decode_instances),
            "prefill_nodes": self.prefill_instances,
            "decode_nodes": self.decode_instances,
        }
        return status

    def get_total_token_length(self, prompt):
        fake_len = 100
        if prompt is None:
            return 0
        if isinstance(prompt, str):
            return len(self.tokenizer(prompt)["input_ids"])
        elif isinstance(prompt, list):
            if len(prompt) == 0:
                return 0
            # Single flat list of ints — already tokenized token IDs
            if all(isinstance(x, int) for x in prompt):
                return len(prompt)
            if all(isinstance(p, str) for p in prompt):
                return sum(len(self.tokenizer(p)["input_ids"]) for p in prompt)
            if all(
                isinstance(p, list) and all(isinstance(x, int) for x in p)
                for p in prompt
            ):
                # Nested list of ints — multiple already-tokenized sequences
                return sum(len(p) for p in prompt)
            if all(isinstance(p, dict) for p in prompt):
                # Multimodal content array — extract text parts only
                total = 0
                for p in prompt:
                    if "text" in p:
                        total += len(self.tokenizer(p["text"])["input_ids"])
                return total
            logger.error(
                "Unsupported prompt format: %s / nested types. Value: %r",
                type(prompt),
                prompt,
            )
            return fake_len
        else:
            logger.error("Unsupported prompt type: %s", type(prompt))
            return fake_len

    def exception_handler(self, prefill_instance=None, decode_instance=None, req_len=None):
        if prefill_instance or decode_instance:
            try:
                self.on_done(
                    prefill_instance=prefill_instance,
                    decode_instance=decode_instance,
                    req_len=req_len
                )
                # Record success with registry for circuit breaker tracking
                if self.registry is not None:
                    if prefill_instance:
                        self.registry.record_success(prefill_instance)
                    if decode_instance:
                        self.registry.record_success(decode_instance)
            except Exception as e:
                logger.error(f"Error releasing instances: {e}")
                raise

    def _record_failure(self, prefill_instance=None, decode_instance=None):
        """Record request failure with registry for circuit breaker tracking."""
        if self.registry is not None:
            if prefill_instance:
                self.registry.record_failure(prefill_instance)
            if decode_instance:
                self.registry.record_failure(decode_instance)

    async def create_completion(self, raw_request: Request):
        return await self._handle_completion("/v1/completions", raw_request, is_chat=False)

    async def create_chat_completion(self, raw_request: Request):
        return await self._handle_completion("/v1/chat/completions", raw_request, is_chat=True)

    def _validate_completion_request(self, request, is_chat):
        """Validate required fields. Returns JSONResponse on error, None on success."""
        if is_chat:
            if "messages" not in request:
                return JSONResponse(
                    {"error": {"message": "Missing required field: messages", "type": "invalid_request_error"}},
                    status_code=400,
                )
            if not isinstance(request["messages"], list):
                return JSONResponse(
                    {"error": {"message": "Field messages must be a list", "type": "invalid_request_error"}},
                    status_code=400,
                )
        else:
            if "prompt" not in request:
                return JSONResponse(
                    {"error": {"message": "Missing required field: prompt", "type": "invalid_request_error"}},
                    status_code=400,
                )
        return None

    def _extract_prompt_info(self, request, is_chat):
        """Extract prompt metrics. Returns (total_length, max_tokens, prompt_text)."""
        if is_chat:
            total_length = 0
            prompt_parts = []
            for msg in request["messages"]:
                content = msg.get("content")
                if content is None:
                    continue
                if isinstance(content, str):
                    total_length += self.get_total_token_length(content)
                    prompt_parts.append(content)
                elif isinstance(content, list):
                    # Multimodal content array — extract text parts
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            total_length += self.get_total_token_length(text)
                            prompt_parts.append(text)
            max_tokens = request.get("max_completion_tokens", 0)
            if max_tokens == 0:
                max_tokens = request.get("max_tokens", 0)
            prompt_text = " ".join(prompt_parts)
        else:
            prompt = request.get("prompt")
            total_length = self.get_total_token_length(prompt)
            max_tokens = request.get("max_tokens", 0)
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)
        return total_length, max_tokens, prompt_text

    def _build_kv_prepare_request(self, request, is_chat):
        """Build the KV-prepare request with max_tokens=1."""
        kv_prepare_request = request.copy()
        kv_prepare_request["max_tokens"] = 1
        if is_chat:
            kv_prepare_request["max_completion_tokens"] = 1
        return kv_prepare_request

    async def _handle_completion(self, endpoint, raw_request, is_chat):
        """Unified completion handler for both /v1/completions and /v1/chat/completions."""
        _metrics_start = track_request_start(endpoint)
        handler_name = "create_chat_completion" if is_chat else "create_completion"
        try:
            try:
                request = await raw_request.json()
            except (json.JSONDecodeError, ValueError):
                return JSONResponse(
                    {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
                    status_code=400,
                )

            error_resp = self._validate_completion_request(request, is_chat)
            if error_resp:
                return error_resp

            prefill_instance = None
            decode_instance = None

            kv_prepare_request = self._build_kv_prepare_request(request, is_chat)

            start_time = time.time()
            total_length, max_tokens, prompt_text = self._extract_prompt_info(request, is_chat)
            end_time = time.time()
            log_info_green(
                f"{handler_name} -- prompt length: {total_length}, "
                f"max tokens: {max_tokens}, "
                f"tokenizer took {(end_time - start_time) * 1000:.2f} ms"
            )

            # Extract scheduling context for advanced policies
            _session_id = (
                raw_request.headers.get("x-session-id")
                or request.get("user")
                or (raw_request.client.host if raw_request.client else None)
            )
            _sched_kwargs = {
                "header": raw_request.headers.get("x-session-id"),
                "session_id": _session_id,
                "user": request.get("user"),
                "client_ip": (
                    raw_request.client.host if raw_request.client else None
                ),
                "prompt": prompt_text,
            }

            prefill_instance = self.schedule(self.prefill_cycler,
                                             is_prompt=True,
                                             request_len=total_length,
                                             max_tokens=1,
                                             **_sched_kwargs)

            decode_instance = self.schedule(self.decode_cycler,
                                            is_prompt=False,
                                            request_len=total_length,
                                            max_tokens=max_tokens,
                                            **_sched_kwargs)

            if prefill_instance is None or decode_instance is None:
                log_info_red("No available instance can handle the request. ")
                self.exception_handler(
                    prefill_instance=prefill_instance,
                    decode_instance=decode_instance,
                    req_len=total_length
                )
                return JSONResponse(
                    {"error": {"message": "No available instance can handle the request", "type": "proxy_error"}},
                    status_code=503,
                )

            value = b''
            try:
                async for chunk in self.forward_request(
                        f"http://{prefill_instance}{endpoint}",
                        kv_prepare_request):
                    value += chunk
            except HTTPException as http_exc:
                self.exception_handler(prefill_instance, decode_instance, total_length)
                self._record_failure(prefill_instance, decode_instance)
                raise http_exc

            # Perform kv recv and decoding stage
            value = value.strip().decode("utf-8").removesuffix(
                "data: [DONE]").encode("utf-8")

            async def streaming_response(value):
                if value:
                    yield value
                else:
                    yield b""

            generator_p = streaming_response(value)
            try:
                generator_d = self.forward_request(
                    f"http://{decode_instance}{endpoint}", request)
            except HTTPException as http_exc:
                self.exception_handler(prefill_instance, decode_instance, total_length)
                self._record_failure(prefill_instance, decode_instance)
                raise http_exc

            if request.get("stream", False):
                generator_class = self.generator
            else:
                # For stream=False request, cannot use P first token
                generator_class = D_first_token_generator
            final_generator = generator_class(generator_p,
                                              generator_d,
                                              self,
                                              prefill_instance,
                                              decode_instance,
                                              req_len=total_length)
            media_type = (
                "text/event-stream"
                if request.get("stream", False)
                else "application/json"
            )
            async def wrapped_generator():
                try:
                    async for chunk in final_generator:
                        yield chunk
                except CancelledError:
                    logger.warning(
                        f"[0]Client disconnected during {handler_name} "
                        "(CancelledError)"
                    )
                except Exception as e:
                    logger.error("[1] Exception in wrapped_generator: %s", str(e))
                    raise
                finally:
                    track_request_end(endpoint, _metrics_start)
            return StreamingResponse(wrapped_generator(), media_type=media_type)
        except HTTPException:
            track_request_end(endpoint, _metrics_start)
            raise
        except Exception:
            track_request_end(endpoint, _metrics_start)
            logger.error("Error in %s: %s", handler_name, sys.exc_info()[1])
            return JSONResponse(
                {"error": {"message": "Internal proxy error", "type": "proxy_error"}},
                status_code=500,
            )

    def remove_instance_endpoint(self, instance_type, instance):
        return

def _create_scheduling_policy(
    config: ProxyConfig,
    scheduling_policy_cls: Optional[type] = None,
    registry: Optional[InstanceRegistry] = None,
) -> SchedulingPolicy:
    """Instantiate a scheduling policy from config or explicit class.

    When *scheduling_policy_cls* is provided (legacy path), it is used
    directly.  Otherwise the ``config.scheduling`` string selects the
    policy via :data:`default_registry`.
    """
    # Legacy explicit-class path (used by existing tests and CLI --roundrobin)
    if scheduling_policy_cls is not None:
        return scheduling_policy_cls(
            config.prefill, config.decode, registry=registry,
        )

    strategy = config.scheduling
    strategy_opts = config.scheduling_config.get(strategy, {})

    # Strategies that accept the legacy (prefill, decode) constructor
    if strategy == "loadbalanced":
        return LoadBalancedScheduler(
            config.prefill, config.decode, registry=registry,
        )
    if strategy == "roundrobin":
        return RoundRobinSchedulingPolicy(registry=registry)

    # Registry-based advanced strategies (all workers for role-aware routing)
    if default_registry.has(strategy):
        policy = default_registry.create(
            strategy,
            workers=list(config.prefill) + list(config.decode),
            registry=registry,
            **strategy_opts,
        )
        return policy

    # Fallback: try registry anyway
    policy = default_registry.create(strategy, registry=registry, **strategy_opts)
    return policy


class ProxyServer:

    def __init__(
        self,
        config: ProxyConfig,
        scheduling_policy: Optional[SchedulingPolicy] = None,
        create_completion: Optional[Callable[[Request],
                                             StreamingResponse]] = None,
        create_chat_completion: Optional[Callable[[Request],
                                                  StreamingResponse]] = None,
    ):
        self.config = config
        self.verify_model_config(config.prefill, config.model)
        self.verify_model_config(config.decode, config.model)
        self.port = config.port

        # Create instance registry and register all instances
        cb_cfg = config.circuit_breaker
        self.registry = InstanceRegistry(
            cb_enabled=cb_cfg.enabled,
            failure_threshold=cb_cfg.failure_threshold,
            success_threshold=cb_cfg.success_threshold,
            timeout_duration_seconds=cb_cfg.timeout_duration_seconds,
            window_duration_seconds=cb_cfg.window_duration_seconds,
        )
        _registered_prefill: set[str] = set()
        _registered_decode: set[str] = set()
        for addr in config.prefill:
            if addr not in _registered_prefill:
                self.registry.add("prefill", addr)
                _registered_prefill.add(addr)
        for addr in config.decode:
            if addr not in _registered_decode:
                self.registry.add("decode", addr)
                _registered_decode.add(addr)
        _registered = _registered_prefill | _registered_decode

        # Create health monitor if enabled
        self.health_monitor = None
        hc_cfg = config.health_check
        if hc_cfg.enabled:
            all_instances = config.prefill + config.decode
            self.health_monitor = HealthMonitor(
                nodes=all_instances,
                interval_seconds=hc_cfg.interval_seconds,
                timeout_seconds=hc_cfg.timeout_seconds,
                on_healthy=self.registry.mark_healthy,
                on_unhealthy=self.registry.mark_unhealthy,
            )
        else:
            # Without health monitoring, assume all instances are healthy
            # so they appear in get_available_instances().
            for addr in _registered:
                self.registry.mark_healthy(addr)

        self.proxy_instance = Proxy(
            prefill_instances=config.prefill,
            decode_instances=config.decode,
            model=config.model,
            scheduling_policy=_create_scheduling_policy(
                config, scheduling_policy, self.registry,
            ),
            custom_create_completion=create_completion,
            custom_create_chat_completion=create_chat_completion,
            generator_on_p_node=config.generator_on_p_node,
            registry=self.registry,
        )

    def verify_model_config(self, instances: list, model: str) -> None:
        for instance in instances:
            try:
                response = requests.get(f"http://{instance}/v1/models")
                if response.status_code == 200:
                    model_cur = response.json()["data"][0]["id"]
                    if model_cur != model:
                        raise ValueError(
                            f"{instance} serves a different model: "
                            f"{model_cur} != {model}")
                else:
                    raise ValueError(f"Cannot get model id from {instance}!")
            except requests.RequestException as e:
                raise ValueError(
                    f"Error communicating with {instance}: {str(e)}") from e

    def run_server(self):
        discovery = NodeDiscovery(
            prefill_instances=self.config.prefill,
            decode_instances=self.config.decode,
            probe_interval=self.config.probe_interval_seconds,
            wait_timeout=self.config.wait_timeout_seconds,
        )

        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.middleware("http")
        async def _check_readiness(request: Request, call_next):
            # Allow health/status/metrics endpoints through always
            path = request.url.path
            if path in ("/health", "/ping", "/status", "/metrics"):
                return await call_next(request)
            if not discovery.is_ready:
                return JSONResponse(
                    {"error": "waiting for backend nodes"},
                    status_code=503,
                )
            return await call_next(request)

        @app.on_event("startup")
        async def _start_discovery():
            await discovery.start()
            if self.health_monitor:
                await self.health_monitor.start()

        @app.on_event("shutdown")
        async def _stop_discovery():
            await discovery.stop()
            if self.health_monitor:
                await self.health_monitor.stop()

        @app.get("/status/instances")
        async def _instance_status():
            """Return per-instance health and circuit breaker state."""
            result: dict[str, list] = {
                "prefill_instances": [],
                "decode_instances": [],
            }
            for info in self.registry.get_all_instances():
                result[f"{info.role}_instances"].append({
                    "address": info.address,
                    "status": info.status.value,
                    "circuit": info.circuit_breaker_state.value,
                    "active_requests": info.active_request_count,
                    "last_check": info.last_health_check,
                })
            return JSONResponse(result)

        app.include_router(self.proxy_instance.router)
        config = uvicorn.Config(app,
                                host="0.0.0.0",
                                port=self.port,
                                log_level=self.config.log_level,
                                loop="uvloop")
        server = uvicorn.Server(config)
        server.run()


_VERSION = "0.1.0"


def _build_parser():
    """Build the argument parser for the proxy CLI."""
    parser = argparse.ArgumentParser(
        prog="xpyd",
        description="MicroPDProxy — lightweight PD proxy server",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_VERSION}")
    parser.add_argument("--config",
                        "-c",
                        type=str,
                        default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--validate-config",
                        type=str,
                        default=None,
                        metavar="FILE",
                        help="Validate YAML config and exit (no server start)")
    parser.add_argument("--model",
                        "-m",
                        type=str,
                        default=None,
                        help="Model name")

    parser.add_argument(
        "--prefill",
        "-p",
        type=str,
        nargs="+",
        help="List of prefill node URLs (host:port)",
    )

    parser.add_argument(
        "--decode",
        "-d",
        type=str,
        nargs="+",
        help="List of decode node URLs (host:port)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number",
    )

    parser.add_argument(
        "--generator_on_p_node",
        action="store_true",
        help="generate first token on P node or D node",
    )

    parser.add_argument(
        "--roundrobin",
        action="store_true",
        help="Use Round Robin scheduling for load balancing",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="warning",
        dest="log_level",
        help="Log level: debug, info, warning, error (default: warning)",
    )
    return parser


def _resolve_config_path(args):
    """Resolve the config file path: --config > XPYD_CONFIG env > ./xpyd.yaml."""
    if args.config:
        return args.config
    env_config = os.environ.get("XPYD_CONFIG")
    if env_config:
        return env_config
    default_path = os.path.join(os.getcwd(), "xpyd.yaml")
    if os.path.exists(default_path):
        return default_path
    return None


def main():
    """Entry point for the ``xpyd`` CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    # --validate-config mode: validate and exit
    if args.validate_config:
        args.config = args.validate_config
        try:
            config = ProxyConfig.from_args(args)
            print(f"Config is valid: {args.validate_config}")
            print(f"  model: {config.model}")
            print(f"  prefill: {len(config.prefill)} instances")
            print(f"  decode: {len(config.decode)} instances")
            print(f"  port: {config.port}")
            print(f"  log_level: {config.log_level}")
            sys.exit(0)
        except Exception as exc:
            print(f"Config validation failed: {exc}", file=sys.stderr)
            sys.exit(1)

    # Resolve config path with precedence
    args.config = _resolve_config_path(args)

    config = ProxyConfig.from_args(args)
    proxy_server = ProxyServer(config=config)
    proxy_server.run_server()


if __name__ == "__main__":
    main()
