# SPDX-License-Identifier: Apache-2.0
"""Cluster and per-instance status endpoints."""

from collections.abc import Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from xpyd.registry import InstanceInfo


def register(
    router: APIRouter,
    *,
    prefill_instances: list[str],
    decode_instances: list[str],
    get_instances: Callable[[], list[InstanceInfo]] | None = None,
) -> None:
    async def get_status():
        return {
            "prefill_node_count": len(prefill_instances),
            "decode_node_count": len(decode_instances),
            "prefill_nodes": prefill_instances,
            "decode_nodes": decode_instances,
        }

    async def _instance_status():
        if get_instances is None:
            raise RuntimeError("Instance registry is unavailable.")
        result: dict[str, list] = {
            "prefill_instances": [],
            "decode_instances": [],
            "aggregated_instances": [],
        }
        for info in get_instances():
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

    router.get("/status", response_class=JSONResponse)(get_status)
    router.get("/status/instances")(_instance_status)
    router.options("/status")(lambda: None)
