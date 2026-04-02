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
import itertools
import json
import logging
import os
import sys
from typing import Callable, Optional

import aiohttp
import requests
import uvicorn
from colorlog.escape_codes import escape_codes
from fastapi import (APIRouter, FastAPI, HTTPException,
                     Request)
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
from xpyd.config import ProxyConfig
from xpyd.discovery import NodeDiscovery
from xpyd.health_monitor import HealthMonitor
from xpyd.registry import InstanceRegistry
from xpyd.routes import register_routes
from xpyd.scheduler import (
    LoadBalancedScheduler,
    RoundRobinSchedulingPolicy,
    SchedulingPolicy,
    default_registry,
)

formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s",
                              "%Y-%m-%d %H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Use a fixed logger name so all modules (scheduler, routes, etc.) can
# reference the same configured logger regardless of import path.
_LOGGER_NAME = "xpyd.proxy"
logger = logging.getLogger(_LOGGER_NAME)
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
        self.d_first_token_generator_class = D_first_token_generator
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def on_done(self,
                prefill_instance: str = None,
                decode_instance: str = None,
                req_len: int = None):
        self.schedule_completion(prefill_instance,
                                 decode_instance,
                                 req_len=req_len)

    def setup_routes(self):
        register_routes(self.router, self)

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

    def get_total_token_length(self, prompt):
        """Compute total token length — delegates to :func:`xpyd.utils.get_total_token_length`."""
        from xpyd.utils import get_total_token_length as _get_total_token_length

        return _get_total_token_length(self.tokenizer, prompt)

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

    async def get_from_instance(self, path: str, is_full_instancelist: int = 0):
        """Fetch data from backend instance(s) via GET."""
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
                            "data": data,
                        }
                except Exception as e:
                    results[inst] = {"status": 500, "error": str(e)}
                    logger.warning("Failed to fetch %s from %s: %s", path, inst, e)

        return JSONResponse(content=results, status_code=200)

    async def post_to_instance(self, request: Request, path: str, json_template: dict):
        """Forward a POST request to a backend instance."""
        body = await request.json()

        missing = [k for k in json_template.keys() if k not in body]
        if missing:
            return JSONResponse(
                {"error": f"Missing required fields: {', '.join(missing)}"},
                status_code=400,
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
                status_code=500,
            )

    async def validate_instance(self, instance: str) -> bool:
        """Validate that an instance is reachable and serves the correct model."""
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
                                logger.info("Instance: %s could be added.", instance)
                                return True
                            else:
                                logger.warning(
                                    "Mismatch model %s : %s != %s",
                                    instance, model_cur, self.model,
                                )
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
    """Build the subcommand argument parser for the proxy CLI."""
    parser = argparse.ArgumentParser(
        prog="xpyd",
        description="xPyD — lightweight PD proxy server",
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"%(prog)s {_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command")

    proxy_parser = subparsers.add_parser(
        "proxy", help="Start the proxy server",
    )
    proxy_parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to YAML configuration file",
    )
    proxy_parser.add_argument(
        "--validate-config", type=str, default=None, metavar="FILE",
        help="Validate YAML config and exit (no server start)",
    )
    proxy_parser.add_argument(
        "--init-config", nargs="?", const="./xpyd.yaml", default=None,
        metavar="PATH",
        help="Generate a default xpyd.yaml template and exit "
             "(default: ./xpyd.yaml)",
    )
    proxy_parser.add_argument(
        "--port", type=int, default=None,
        help="Override the port from config",
    )
    proxy_parser.add_argument(
        "--log-level", type=str, default=None, dest="log_level",
        help="Override log level: debug|info|warning|error",
    )

    return parser


def _resolve_config_path(args):
    """Resolve the config file path: --config > XPYD_CONFIG env > ./xpyd.yaml.

    Returns the path string, or raises ``SystemExit`` with a helpful
    error message when no config can be found.
    """
    if args.config:
        return args.config
    env_config = os.environ.get("XPYD_CONFIG")
    if env_config:
        return env_config
    default_path = os.path.join(os.getcwd(), "xpyd.yaml")
    if os.path.exists(default_path):
        return default_path
    print(
        "Error: No config file found.\n\n"
        "Create one with:  xpyd proxy --init-config\n"
        "Or specify one:   xpyd proxy --config /path/to/config.yaml",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    """Entry point for the ``xpyd`` CLI."""
    from xpyd.init_config import generate_config_template

    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "proxy":
        # --init-config: generate template and exit
        if args.init_config is not None:
            generate_config_template(args.init_config)
            return

        # --validate-config: validate and exit
        if args.validate_config:
            config_path = args.validate_config
            try:
                config = ProxyConfig.from_yaml(config_path)
                print(f"Config is valid: {config_path}")
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
        config_path = _resolve_config_path(args)
        config = ProxyConfig.from_yaml(config_path)

        # Apply CLI overrides
        if args.port is not None:
            config = config.model_copy(update={"port": args.port})
        if args.log_level is not None:
            config = config.model_copy(update={"log_level": args.log_level})

        proxy_server = ProxyServer(config=config)
        proxy_server.run_server()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
