# SPDX-License-Identifier: Apache-2.0
"""MicroDisaggregatedProxyServer.

The proxy routes incoming OpenAI-compatible requests through two phases:

  1. **Prefill** – sends a trimmed request (``stream=False``, ``max_tokens=1``)
     to a prefill node for KV-cache preparation.
  2. **Decode** – forwards the original request to a decode node for
     autoregressive generation.

The decode node's response is returned to the client (streaming or
non-streaming).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from xpyd.config import ProxyConfig
from xpyd.discovery import NodeDiscovery
from xpyd.errors import INVALID_REQUEST, PROXY_ERROR, SERVER_ERROR, error_response
from xpyd.health_monitor import HealthMonitor
from xpyd.registry import InstanceRegistry
from xpyd.routes import register_routes
from xpyd.scheduler import (
    Reservation,
    Scheduler,
    SchedulingContext,
    SchedulingPolicy,
    default_registry,
)


class _ExtraFormatter(logging.Formatter):
    """Formatter that appends ``extra`` fields as ``key=value`` pairs."""

    _SKIP = set(logging.LogRecord("", "", "", 0, "", (), None).__dict__) | {
        "taskName",
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in record.__dict__.items() if k not in self._SKIP}
        if extras:
            base += " | " + " ".join(f"{k}={v}" for k, v in extras.items())
        return base


formatter = _ExtraFormatter(
    "[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Use a fixed logger name so all modules (scheduler, routes, etc.) can
# reference the same configured logger regardless of import path.
_LOGGER_NAME = "xpyd.proxy"
logger = logging.getLogger(_LOGGER_NAME)
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(handler)
logger.propagate = False


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(
    total=None, connect=None, sock_read=None, sock_connect=None
)


async def P_first_token_generator(
    generator_p: AsyncGenerator[bytes, None],
    generator_d: AsyncGenerator[bytes, None],
    release_prefill: Callable[[], None],
    release_decode: Callable[[], None],
) -> AsyncGenerator[bytes, None]:
    first_decode = True

    try:
        async for chunk in generator_p:
            yield chunk
    finally:
        release_prefill()

    try:
        async for chunk in generator_d:
            if first_decode:
                first_decode = False
                continue
            yield chunk
    finally:
        release_decode()


async def D_first_token_generator(
    generator_p: AsyncGenerator[bytes, None],
    generator_d: AsyncGenerator[bytes, None],
    release_prefill: Callable[[], None],
    release_decode: Callable[[], None],
) -> AsyncGenerator[bytes, None]:
    try:
        async for _ in generator_p:
            continue
    finally:
        release_prefill()

    try:
        async for chunk in generator_d:
            yield chunk
    finally:
        release_decode()


class Proxy:

    def __init__(
        self,
        prefill_instances: list[str],
        decode_instances: list[str],
        model: str,
        scheduling_policy: SchedulingPolicy,
        custom_create_completion: Optional[
            Callable[[Request], StreamingResponse]
        ] = None,
        custom_create_chat_completion: Optional[
            Callable[[Request], StreamingResponse]
        ] = None,
        first_token_source: str = "decode",
        registry: Optional[InstanceRegistry] = None,
        aggregated_instances: Optional[dict[str, list[str]]] = None,
        model_schedulers: Optional[dict[str, str]] = None,
        tokenizer_path: Optional[str] = None,
        disaggregated_mode: str = "direct",
        zmq_config=None,
        scheduling_options: Optional[dict[str, Any]] = None,
        model_policy_options: Optional[dict[str, dict[str, Any]]] = None,
    ):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.model = model
        self.scheduling_policy = scheduling_policy
        self.registry = registry
        self.aggregated_instances = aggregated_instances or {}
        self.model_schedulers = model_schedulers or {}
        self.tokenizer_path = tokenizer_path
        self._tokenizers: dict[str, Any] = {}
        self._round_robin_models: set[str] = set()
        self._round_robin_policy = default_registry.build(
            "roundrobin", registry=registry
        )
        self.disaggregated_mode = disaggregated_mode
        self.first_token_source = first_token_source
        self.zmq_config = zmq_config
        self.zmq_notifications = None
        self._aggregated_policies: dict[str, SchedulingPolicy] = {}
        self._scheduling_options = scheduling_options or {}
        self._model_policy_options = model_policy_options or {}
        self._scheduler = Scheduler(registry)
        self.custom_create_completion = custom_create_completion
        self.custom_create_chat_completion = custom_create_chat_completion
        self.health_monitor = None
        self.discovery = None
        self._instance_mutation_lock = asyncio.Lock()
        self.router = APIRouter()
        self.setup_routes()
        self.generator = (
            P_first_token_generator
            if first_token_source == "prefill"
            else D_first_token_generator
        )
        self.d_first_token_generator_class = D_first_token_generator
        self.tokenizer = None

    def ensure_tokenizer(self, model: str) -> bool:
        """Load a model tokenizer or mark the model for round-robin fallback."""
        if not model or model in self._tokenizers:
            return bool(model and model in self._tokenizers)
        if model in self._round_robin_models:
            return False

        source = model
        local_only = False
        if self.tokenizer_path:
            root = Path(self.tokenizer_path).expanduser().resolve()
            source_path = root.joinpath(*model.split("/")).resolve()
            try:
                source_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid model name {model!r}: tokenizer directory must "
                    f"remain under tokenizer_path {str(root)!r}."
                ) from exc
            if not source_path.is_dir():
                raise ValueError(
                    f"Tokenizer for model {model!r} was not found. Expected "
                    f"directory: {source_path}. Set tokenizer_path to the "
                    "parent directory whose model-named subdirectories "
                    "contain Hugging Face tokenizer files."
                )
            source = str(source_path)
            local_only = True

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                local_files_only=local_only,
            )
        except Exception as exc:
            if self.tokenizer_path:
                raise ValueError(
                    f"Failed to load tokenizer for model {model!r} from "
                    f"{source!r}: {exc}. Verify that tokenizer_path contains "
                    "a model-named subdirectory with valid Hugging Face "
                    "tokenizer files."
                ) from exc
            self._round_robin_models.add(model)
            logger.warning(
                "Unable to download tokenizer for model %r: %s. "
                "Falling back to roundrobin scheduling for this model.",
                model,
                exc,
            )
            return False

        self._tokenizers[model] = tokenizer
        if self.tokenizer is None:
            self.tokenizer = tokenizer
        logger.info("Loaded tokenizer for model %r from %s", model, source)
        return True

    def get_tokenizer(self, model: str = ""):
        """Return the tokenizer loaded for a model, if available."""
        if model:
            return self.__dict__.get("_tokenizers", {}).get(model)
        return self.__dict__.get("tokenizer")

    def uses_round_robin_fallback(self, model: str) -> bool:
        """Return whether tokenizer loading forced round-robin for a model."""
        return model in self.__dict__.get("_round_robin_models", set())

    def _is_aggregated_model(self, model: str) -> bool:
        """Check if all instances for a model are aggregated-role."""
        return bool(Proxy._aggregated_instances_for_model(self, model))

    def _aggregated_instances_for_model(self, model: str) -> list[str]:
        """Return configured aggregated instances, including discovered models."""
        if self.registry is not None:
            discovered = [
                info.address
                for info in self.registry.get_all_instances()
                if info.role == "aggregated" and info.model == model
            ]
            if discovered:
                return discovered
        return list(self.aggregated_instances.get(model, []))

    def _get_aggregated_policy(
        self,
        model: str,
        instances: list[str],
    ) -> SchedulingPolicy:
        """Build a model policy through the same interface as the global policy."""
        policy = self._aggregated_policies.get(model)
        if policy is not None:
            return policy

        strategy = self.model_schedulers.get(model)
        options: dict[str, Any] = {
            "workers": instances,
            "registry": self.registry,
            "tokenizer": self.get_tokenizer(model),
        }
        if strategy:
            policy = default_registry.build(
                strategy, **options, **self._model_policy_options.get(strategy, {})
            )
        else:
            policy = type(self.scheduling_policy).from_config(
                **options, **self._scheduling_options
            )
        self._aggregated_policies[model] = policy
        return policy

    def reserve(self, context: SchedulingContext) -> Optional[Reservation]:
        """Acquire a node without depending on a concrete scheduling strategy."""
        with self._scheduler.lock:
            if context.role == "aggregated":
                instances = self._aggregated_instances_for_model(context.model)
                if not instances:
                    return None
                policy = self._get_aggregated_policy(context.model, instances)
            else:
                instances = (
                    self.prefill_instances
                    if context.role == "prefill"
                    else self.decode_instances
                )
                policy = self.scheduling_policy
            if self.uses_round_robin_fallback(context.model):
                policy = self._round_robin_policy
            return self._scheduler.reserve(policy, context, instances)

    def setup_routes(self) -> None:
        register_routes(self.router, self)

    async def forward_request(
        self,
        url: str,
        data: dict,
        use_chunked: bool = True,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> AsyncGenerator[bytes, None]:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
            if extra_headers:
                headers.update(extra_headers)
            try:
                async with session.post(
                    url=url, json=data, headers=headers
                ) as response:
                    if (
                        200 <= response.status < 300 or 400 <= response.status < 500
                    ):  # noqa: E501
                        if use_chunked:
                            async for (
                                chunk_bytes
                            ) in response.content.iter_chunked(  # noqa: E501
                                1024
                            ):
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
                        logger.error(
                            "Request failed with status %s: %s",
                            response.status,
                            error_content,
                        )
                        # HTTPException is intentional: forward_request is
                        # an async generator, so it cannot return a
                        # JSONResponse.  Callers catch HTTPException and
                        # convert it to error_response().
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Request failed with status {response.status}: "
                            f"{error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error("ClientError occurred: %s", str(e))
                # See comment above re: async generator context.
                raise HTTPException(
                    status_code=502,
                    detail="Bad Gateway: Error communicating with upstream server.",
                ) from e
            except Exception as e:
                logger.exception("Unexpected error in forward_request")
                # See comment above re: async generator context.
                raise HTTPException(
                    status_code=500,
                    detail="Internal proxy error",
                ) from e

    async def drain_and_remove_instance(
        self,
        role: str,
        address: str,
        timeout_seconds: float,
    ) -> None:
        """Stop new scheduling, drain active requests, then remove an instance."""
        if self.registry is None:
            raise RuntimeError("Instance registry is unavailable.")

        async with self._instance_mutation_lock:
            info = self.registry.get_instance_info(address)
            with self._scheduler.lock:
                self.registry.begin_draining(role, address)
            deadline = asyncio.get_running_loop().time() + timeout_seconds
            while self.registry.get_active_requests(address) > 0:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    active = self.registry.get_active_requests(address)
                    raise TimeoutError(
                        f"Timed out draining {address}; {active} requests remain."
                    )
                await asyncio.sleep(min(0.05, remaining))

            with self._scheduler.lock:
                if role == "aggregated":
                    instances = self._aggregated_instances_for_model(info.model)
                    policy = self._aggregated_policies.get(info.model)
                    if policy is not None:
                        policy.on_instance_removed(
                            role, address, instances.index(address)
                        )
                    for model, pool in list(self.aggregated_instances.items()):
                        if address in pool:
                            pool.remove(address)
                            if not pool:
                                del self.aggregated_instances[model]
                else:
                    instances = (
                        self.prefill_instances
                        if role == "prefill"
                        else self.decode_instances
                    )
                    index = instances.index(address)
                    self.scheduling_policy.on_instance_removed(role, address, index)
                    instances.remove(address)

                if self.health_monitor is not None:
                    self.health_monitor.remove_node(address)
                if self.discovery is not None:
                    self.discovery.remove_instance(role, address)
                self.registry.remove(address)

    async def add_instance(self, role: str, address: str) -> bool:
        """Validate and atomically register a runtime backend instance."""
        if self.registry is None:
            raise RuntimeError("Instance registry is unavailable.")
        details = await Proxy._validated_instance_details(self, address)
        if details is None:
            return False
        model, max_model_len = details

        async with self._instance_mutation_lock:
            with self._scheduler.lock:
                return self._register_validated_instance(
                    role, address, model, max_model_len
                )

    def _register_validated_instance(
        self, role: str, address: str, model: str, max_model_len: int
    ) -> bool:
        """Commit membership changes while selection and draining are excluded."""
        if self.registry is None:
            raise RuntimeError("Instance registry is unavailable.")
        try:
            self.registry.get_instance_info(address)
        except KeyError:
            pass
        else:
            raise ValueError("Instance already exists")

        if role == "aggregated":
            instances = self.aggregated_instances.setdefault(model, [])
        else:
            instances = (
                self.prefill_instances if role == "prefill" else self.decode_instances
            )
        self.registry.add(role, address, model=model)
        self.registry.mark_healthy(address)
        index = len(instances)
        policy = (
            self._aggregated_policies.get(model)
            if role == "aggregated"
            else self.scheduling_policy
        )
        policy_added = False
        try:
            if policy is not None:
                policy.on_instance_added(role, address, max_model_len)
                policy_added = True
            instances.append(address)
            if self.health_monitor is not None:
                self.health_monitor.add_node(address)
            if self.discovery is not None:
                self.discovery.add_instance(role, address)
        except Exception:
            if self.discovery is not None:
                self.discovery.remove_instance(role, address)
            if self.health_monitor is not None:
                self.health_monitor.remove_node(address)
            if policy_added and policy is not None:
                policy.on_instance_removed(role, address, index)
            if address in instances:
                instances.remove(address)
            if role == "aggregated" and not instances:
                self.aggregated_instances.pop(model, None)
            self.registry.remove(address)
            raise
        return True

    def get_total_token_length(self, prompt: Any, model: str = "") -> int:
        """Compute total token length — delegates to
        :func:`xpyd.utils.get_total_token_length`."""
        from xpyd.utils import get_total_token_length as _get_total_token_length

        tokenizer = Proxy.get_tokenizer(self, model)
        if tokenizer is None:
            return 0
        return _get_total_token_length(tokenizer, prompt)

    async def tokenize_on_backend(
        self,
        request: dict,
        is_chat: bool,
    ) -> list[int]:
        """Tokenize through a healthy prefill node when no local tokenizer exists."""
        model = request.get("model", "")
        instances = (
            self.registry.get_available_instances("prefill", model=model)
            if self.registry is not None
            else self.prefill_instances
        )
        if not instances:
            raise HTTPException(
                status_code=503,
                detail=f"No prefill instance can tokenize model {model!r}",
            )

        if is_chat:
            allowed = {
                "model",
                "messages",
                "add_generation_prompt",
                "continue_final_message",
                "chat_template",
                "chat_template_kwargs",
                "tools",
            }
            payload = {key: value for key, value in request.items() if key in allowed}
            continue_final_message = request.get("continue_final_message", False)
            payload["add_generation_prompt"] = request.get(
                "add_generation_prompt", not continue_final_message
            )
        else:
            prompt = request.get("prompt")
            if isinstance(prompt, list):
                return list(prompt)
            payload = {"model": model, "prompt": prompt}

        url = f"http://{instances[0]}/tokenize"
        try:
            async with (
                aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session,
                session.post(url, json=payload) as response,
            ):
                data = await response.json()
                if response.status != 200:
                    raise HTTPException(
                        status_code=502,
                        detail=(
                            f"Backend tokenizer returned HTTP "
                            f"{response.status}: {data}"
                        ),
                    )
        except aiohttp.ClientError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to tokenize through {url}: {exc}",
            ) from exc
        tokens = data.get("tokens")
        if not isinstance(tokens, list):
            raise HTTPException(
                status_code=502,
                detail=f"Backend tokenizer at {url} returned no tokens",
            )
        return tokens

    def _record_failure(
        self,
        prefill_instance: Optional[str] = None,
        decode_instance: Optional[str] = None,
    ) -> None:
        """Record request failure with registry for circuit breaker tracking."""
        if self.registry is not None:
            if prefill_instance:
                self.registry.record_failure(prefill_instance)
            if decode_instance:
                self.registry.record_failure(decode_instance)

    def _healthy_instances(self) -> set[str]:
        """Return every instance the registry currently considers usable.

        Single-instance probes such as ``/version`` must not be answered from a
        node that is known to be down while healthy nodes are available.
        """
        if self.registry is None:
            return set()
        healthy: set[str] = set()
        for role in ("prefill", "decode", "aggregated"):
            healthy.update(self.registry.get_available_instances(role))
        return healthy

    async def get_from_instance(
        self, path: str, is_full_instancelist: int = 0
    ) -> JSONResponse:
        """Fetch data from backend instance(s) via GET."""
        aggregated_instances = [
            instance
            for model_instances in self.aggregated_instances.values()
            for instance in model_instances
        ]
        instances = (
            self.prefill_instances + self.decode_instances + aggregated_instances
        )
        if not instances:
            return error_response("No instances available", SERVER_ERROR, 500)

        if is_full_instancelist == 0:
            healthy = self._healthy_instances()
            preferred = [inst for inst in instances if inst in healthy]
            instances = (preferred or instances)[:1]

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
                            "data": data,
                        }
                except Exception as e:
                    results[inst] = {
                        "status": 500,
                        "error": "Failed to connect to instance",
                    }
                    logger.warning("Failed to fetch %s from %s: %s", path, inst, e)

        reachable = any(
            "error" not in result and result["status"] < 500
            for result in results.values()
        )
        return JSONResponse(content=results, status_code=200 if reachable else 503)

    def auxiliary_instances(self, model: str = "") -> list[str]:
        """Return instances able to serve auxiliary (non-generation) endpoints.

        Prefill nodes hold the full model in disaggregated topologies, while
        aggregated deployments have no prefill role at all.  Fall through every
        role instead of assuming a prefill node is always present.
        """
        if self.registry is not None:
            for role in ("prefill", "aggregated", "decode"):
                instances = self.registry.get_available_instances(role, model=model)
                if instances:
                    return list(instances)
            if model:
                return []

        aggregated = [
            instance
            for model_instances in self.aggregated_instances.values()
            for instance in model_instances
        ]
        return list(self.prefill_instances or aggregated or self.decode_instances)

    async def post_to_instance(
        self, request: Request, path: str, json_template: dict
    ) -> JSONResponse:
        """Forward a POST request to a backend instance."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return error_response("Invalid JSON in request body", INVALID_REQUEST, 400)

        missing = [k for k in json_template.keys() if k not in body]
        if missing:
            return error_response(
                f"Missing required fields: {', '.join(missing)}",
                INVALID_REQUEST,
                400,
            )

        payload = json_template.copy()
        payload.update(body)

        instances = self.auxiliary_instances(str(body.get("model") or ""))
        if not instances:
            return error_response(
                "No available instance can handle the request", PROXY_ERROR, 503
            )

        url = f"http://{instances[0]}{path}"
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(url, json=payload) as resp,
            ):
                try:
                    content = await resp.json()
                except aiohttp.ContentTypeError:
                    content = {"raw": await resp.text()}
                return JSONResponse(content, status_code=resp.status)
        except Exception:
            logger.exception("Failed to forward request to %s", url)
            return error_response(
                f"Failed to forward request to {url}", SERVER_ERROR, 500
            )

    async def _validated_instance_details(
        self, instance: str
    ) -> Optional[tuple[str, int]]:
        """Return model metadata when a candidate is valid for this proxy."""
        url = f"http://{instance}/v1/models"
        try:
            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as client:
                logger.info("Verifying %s ...", instance)
                async with client.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and len(data["data"]) > 0:
                            model_cur = data["data"][0].get("id", "")
                            expected_models = {self.model} if self.model else set()
                            if self.registry is not None:
                                expected_models.update(
                                    info.model
                                    for info in self.registry.get_all_instances()
                                    if info.model
                                )
                            if model_cur in expected_models:
                                logger.info("Instance: %s could be added.", instance)
                                max_model_len = data["data"][0].get(
                                    "max_model_len", 131072
                                )
                                return model_cur, int(max_model_len)
                            else:
                                logger.warning(
                                    "Mismatch model %s: %s not in %s",
                                    instance,
                                    model_cur,
                                    sorted(expected_models),
                                )
                                return None
                        else:
                            return None
                    else:
                        return None
        except aiohttp.ClientError as e:
            logger.error(str(e))
            return None
        except Exception as e:
            logger.error(str(e))
            return None

    async def validate_instance(self, instance: str) -> bool:
        """Validate that an instance is reachable and serves a known model."""
        return await Proxy._validated_instance_details(self, instance) is not None


def _create_scheduling_policy(
    config: ProxyConfig,
    registry: Optional[InstanceRegistry] = None,
    all_prefill: Optional[list[str]] = None,
    all_decode: Optional[list[str]] = None,
) -> SchedulingPolicy:
    """Instantiate the configured strategy through the common policy factory."""
    prefill = all_prefill if all_prefill is not None else config.prefill
    decode = all_decode if all_decode is not None else config.decode

    strategy = config.scheduling
    strategy_opts = config.scheduling_config.get(strategy, {})

    return default_registry.build(
        strategy,
        prefill_instances=prefill,
        decode_instances=decode,
        workers=list(prefill) + list(decode),
        registry=registry,
        **strategy_opts,
    )


class ProxyServer:

    def __init__(
        self,
        config: ProxyConfig,
        create_completion: Optional[Callable[[Request], StreamingResponse]] = None,
        create_chat_completion: Optional[Callable[[Request], StreamingResponse]] = None,
    ):
        self.config = config
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
        _registered_aggregated: set[str] = set()
        aggregated_instances: dict[str, list[str]] = {}

        if config.instances is not None:
            # Multi-model: register from instances list
            for entry in config.instances:
                addr = entry.address
                if entry.role == "prefill":
                    if addr in _registered_prefill:
                        logger.warning(
                            "Duplicate prefill address %s (model=%s) — "
                            "only the first registration is kept",
                            addr,
                            entry.model,
                        )
                        continue
                    self.registry.add("prefill", addr, model=entry.model)
                    _registered_prefill.add(addr)
                elif entry.role == "decode":
                    if addr in _registered_decode:
                        logger.warning(
                            "Duplicate decode address %s (model=%s) — "
                            "only the first registration is kept",
                            addr,
                            entry.model,
                        )
                        continue
                    self.registry.add("decode", addr, model=entry.model)
                    _registered_decode.add(addr)
                elif entry.role == "aggregated":
                    if addr in _registered_aggregated:
                        logger.warning(
                            "Duplicate aggregated address %s (model=%s) — "
                            "only the first registration is kept",
                            addr,
                            entry.model,
                        )
                        continue
                    self.registry.add("aggregated", addr, model=entry.model)
                    _registered_aggregated.add(addr)
                    aggregated_instances.setdefault(entry.model, []).append(addr)
            # Derive de-duplicated prefill/decode membership.
            seen_p: set[str] = set()
            seen_d: set[str] = set()
            all_prefill: list[str] = []
            all_decode: list[str] = []
            for e in config.instances:
                if e.role == "prefill" and e.address not in seen_p:
                    all_prefill.append(e.address)
                    seen_p.add(e.address)
                elif e.role == "decode" and e.address not in seen_d:
                    all_decode.append(e.address)
                    seen_d.add(e.address)
        else:
            # Legacy single-model: register from prefill/decode lists
            model_name = config.model
            for addr in config.prefill:
                if addr not in _registered_prefill:
                    self.registry.add("prefill", addr, model=model_name)
                    _registered_prefill.add(addr)
            for addr in config.decode:
                if addr not in _registered_decode:
                    self.registry.add("decode", addr, model=model_name)
                    _registered_decode.add(addr)
            all_prefill = list(config.prefill)
            all_decode = list(config.decode)

        self._all_prefill = all_prefill
        self._all_decode = all_decode
        self._all_aggregated = [
            a for addrs in aggregated_instances.values() for a in addrs
        ]
        _registered = _registered_prefill | _registered_decode | _registered_aggregated

        # Create health monitor if enabled
        self.health_monitor = None
        hc_cfg = config.health_check
        if hc_cfg.enabled:
            all_instances = all_prefill + all_decode + self._all_aggregated
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

        # Build per-model scheduler config from models shorthand.
        # Per-model strategies are instantiated when their model is discovered.
        # Fallback chain: model-level → global → load_balanced (default).
        model_scheduler_config = getattr(config, "_model_schedulers", {})
        # Validate scheduler names at startup; collect invalid ones first
        invalid_models = [
            model_name
            for model_name, strategy_name in model_scheduler_config.items()
            if not default_registry.has(strategy_name)
        ]
        for model_name in invalid_models:
            logger.warning(
                "Unknown scheduler %r for model %r; available: %s. "
                "Will fall back to global policy at runtime.",
                model_scheduler_config[model_name],
                model_name,
                default_registry.list_policies(),
            )
            del model_scheduler_config[model_name]

        global_policy = _create_scheduling_policy(
            config,
            self.registry,
            all_prefill=all_prefill,
            all_decode=all_decode,
        )

        self.proxy_instance = Proxy(
            prefill_instances=all_prefill,
            decode_instances=all_decode,
            model=config.model,
            scheduling_policy=global_policy,
            custom_create_completion=create_completion,
            custom_create_chat_completion=create_chat_completion,
            first_token_source=config.first_token_source,
            registry=self.registry,
            aggregated_instances=aggregated_instances,
            model_schedulers=model_scheduler_config,
            tokenizer_path=config.tokenizer_path,
            disaggregated_mode=config.disaggregated_mode,
            zmq_config=config.zmq,
            scheduling_options=config.scheduling_config.get(config.scheduling, {}),
            model_policy_options=config.scheduling_config,
        )

    def run_server(self) -> None:
        discovery = NodeDiscovery(
            prefill_instances=self._all_prefill,
            decode_instances=self._all_decode,
            probe_interval=self.config.probe_interval_seconds,
            wait_timeout=self.config.wait_timeout_seconds,
            heartbeat_interval=self.config.heartbeat_interval_seconds,
            registry=self.registry,
            aggregated_instances=self._all_aggregated,
            on_model_discovered=lambda model: asyncio.to_thread(
                self.proxy_instance.ensure_tokenizer, model
            ),
        )
        self.proxy_instance.discovery = discovery
        self.proxy_instance.health_monitor = self.health_monitor

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
            if path in (
                "/health",
                "/ping",
                "/status",
                "/metrics",
                "/instances/add",
                "/instances/remove",
            ) or path.startswith("/status/"):
                return await call_next(request)
            if not discovery.is_ready:
                return error_response("Waiting for backend nodes", PROXY_ERROR, 503)
            return await call_next(request)

        @app.on_event("startup")
        async def _start_discovery():
            if self.config.disaggregated_mode == "zmq":
                from xpyd.zmq_notifications import ZmqNotificationListener

                zmq_config = self.config.zmq
                assert zmq_config is not None
                listener = ZmqNotificationListener(
                    zmq_config.host,
                    zmq_config.port,
                    zmq_config.notification_timeout_seconds,
                )
                await listener.start()
                self.proxy_instance.zmq_notifications = listener
            await discovery.start()
            if self.health_monitor:
                await self.health_monitor.start()

        @app.on_event("shutdown")
        async def _stop_discovery():
            await discovery.stop()
            if self.proxy_instance.zmq_notifications is not None:
                await self.proxy_instance.zmq_notifications.stop()
            if self.health_monitor:
                await self.health_monitor.stop()

        @app.get("/status/instances")
        async def _instance_status():
            """Return per-instance health and circuit breaker state."""
            result: dict[str, list] = {
                "prefill_instances": [],
                "decode_instances": [],
                "aggregated_instances": [],
            }
            for info in self.registry.get_all_instances():
                result[f"{info.role}_instances"].append(
                    {
                        "address": info.address,
                        "status": info.status.value,
                        "circuit": info.circuit_breaker_state.value,
                        "active_requests": info.active_request_count,
                        "last_check": info.last_health_check,
                    }
                )
            return JSONResponse(result)

        app.include_router(self.proxy_instance.router)
        config = uvicorn.Config(
            app,
            host=self.config.host,
            port=self.port,
            log_level=self.config.log_level,
            loop="uvloop",
        )
        server = uvicorn.Server(config)
        server.run()


_VERSION = "1.8.0"


def _valid_port(value: str) -> int:
    """Parse a TCP port for argparse with a concise user-facing error."""
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _build_parser():
    """Build the subcommand argument parser for the proxy CLI."""
    parser = argparse.ArgumentParser(
        prog="xpyd",
        description="xPyD — lightweight disaggregated proxy server",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"%(prog)s {_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command")

    proxy_parser = subparsers.add_parser(
        "proxy",
        prog="xpyd [proxy]",
        help="Start the proxy server (default command)",
    )
    proxy_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    proxy_parser.add_argument(
        "--validate-config",
        type=str,
        default=None,
        metavar="FILE",
        help="Validate YAML config and exit (no server start)",
    )
    proxy_parser.add_argument(
        "--init-config",
        nargs="?",
        const="./xpyd.yaml",
        default=None,
        metavar="PATH",
        help="Generate xpyd.yaml, optionally through an interactive wizard "
        "(default path: ./xpyd.yaml)",
    )
    proxy_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing --init-config destination",
    )
    proxy_parser.add_argument(
        "--port",
        type=_valid_port,
        default=None,
        help="Override the port from config",
    )
    proxy_parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default=None,
        dest="log_level",
        help="Override log level: debug|info|warning|error",
    )
    proxy_parser.add_argument(
        "--disaggregated-mode",
        choices=("direct", "nixl", "zmq"),
        default=None,
        help="disaggregated transfer mode (default: config value or direct)",
    )
    proxy_parser.add_argument(
        "--first-token-source",
        choices=("prefill", "decode"),
        default=None,
        help="Backend that provides the first client-visible token",
    )

    # fix-config subcommand
    fix_parser = subparsers.add_parser(
        "fix-config",
        help="Auto-fix common config mistakes",
    )
    fix_parser.add_argument(
        "config_path",
        type=str,
        help="Path to YAML configuration file to fix",
    )
    fix_parser.add_argument(
        "--write",
        action="store_true",
        default=False,
        help="Write fixes back to the file (creates timestamped .bak backup). "
        "Note: does not preserve YAML comments or formatting.",
    )
    fix_parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Interactively acknowledge ambiguous suggestions one by one",
    )

    return parser


def _normalize_cli_args(argv: list[str]) -> list[str]:
    """Treat omitted subcommands as the default ``proxy`` command."""
    if argv and argv[0] == "proxy":
        if len(argv) > 1 and argv[1] in {"fix-config", "--version", "-V"}:
            return argv[1:]
        return argv
    if argv and argv[0] in {"fix-config", "--version", "-V"}:
        return argv
    return ["proxy", *argv]


def _resolve_config_path(args):
    """Resolve the config file path: --config > XPYD_CONFIG env > ./xpyd.yaml.

    Returns the path string, or ``None`` when no config can be found.
    """
    if args.config:
        return args.config
    env_config = os.environ.get("XPYD_CONFIG")
    if env_config:
        return env_config
    default_path = "./xpyd.yaml"
    if os.path.exists(default_path):
        print(
            "No config specified; found ./xpyd.yaml and using it.",
            flush=True,
        )
        return default_path
    return None


def _print_config_summary(config: ProxyConfig) -> None:
    """Print topology-aware details for a validated configuration."""
    if config.instances is None:
        topology = "disaggregated" if config.prefill else "decode-only"
        print(f"  topology: {topology}")
        if config.prefill:
            print(f"  transfer: {config.disaggregated_mode}")
        print(f"  model: {config.model}")
        print(f"  prefill: {len(config.prefill)} instances")
        print(f"  decode: {len(config.decode)} instances")
    else:
        roles = {instance.role for instance in config.instances}
        topology = "aggregated" if roles == {"aggregated"} else "disaggregated"
        print(f"  topology: {topology}")
        if topology == "disaggregated":
            print(f"  transfer: {config.disaggregated_mode}")
        models = sorted(
            {instance.model for instance in config.instances if instance.model}
        )
        print(f"  models: {', '.join(models) if models else 'auto-detected'}")
        for role in ("aggregated", "prefill", "decode"):
            count = sum(instance.role == role for instance in config.instances)
            if count:
                print(f"  {role}: {count} instances")
    print(f"  port: {config.port}")
    print(f"  log_level: {config.log_level}")


def _apply_cli_overrides(config: ProxyConfig, args) -> ProxyConfig:
    """Apply explicitly supplied CLI values and revalidate the result."""
    overrides = {
        field: value
        for field in (
            "port",
            "log_level",
            "disaggregated_mode",
            "first_token_source",
        )
        if (value := getattr(args, field, None)) is not None
    }
    if not overrides:
        return config

    model_schedulers = config._model_schedulers.copy()
    updated = ProxyConfig.model_validate({**config.model_dump(), **overrides})
    updated._model_schedulers = model_schedulers
    return updated


def main() -> None:
    """Entry point for the ``xpyd`` CLI."""
    from xpyd.init_config import generate_config

    parser = _build_parser()
    args = parser.parse_args(_normalize_cli_args(sys.argv[1:]))

    if args.command == "fix-config":
        from xpyd.config_fixer import run_fix_config

        sys.exit(
            run_fix_config(
                args.config_path,
                write=args.write,
                interactive=args.interactive,
            )
        )
    elif args.command == "proxy":
        if args.force and args.init_config is None:
            parser.error("--force can only be used with --init-config")

        # --init-config: generate template and exit
        if args.init_config is not None:
            output_path = Path(args.init_config)
            if output_path.exists() and not args.force:
                parser.error(
                    f"config already exists: {output_path}; use --force to overwrite"
                )
            try:
                generate_config(args.init_config)
            except OSError as exc:
                parser.exit(1, f"xpyd: failed to write config: {exc}\n")
            return

        # --validate-config: validate and exit
        if args.validate_config:
            config_path = args.validate_config
            try:
                config = ProxyConfig.from_yaml(config_path)
                print(f"Config is valid: {config_path}")
                _print_config_summary(config)
                sys.exit(0)
            except Exception as exc:
                print(f"Config validation failed: {exc}", file=sys.stderr)
                sys.exit(1)

        # Resolve config path with precedence
        config_path = _resolve_config_path(args)
        if config_path is None:
            config_path = "./xpyd.yaml"
            print(
                "No config specified and ./xpyd.yaml was not found; "
                "starting config initialization."
            )
            generate_config(config_path)
            return
        try:
            config = ProxyConfig.from_yaml(config_path)
            config = _apply_cli_overrides(config, args)
        except (OSError, ValueError) as exc:
            parser.exit(1, f"xpyd: failed to load config: {exc}\n")

        proxy_server = ProxyServer(config=config)
        proxy_server.run_server()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
