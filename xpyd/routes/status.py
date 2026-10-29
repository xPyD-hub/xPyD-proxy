# SPDX-License-Identifier: Apache-2.0
"""Cluster and per-instance status endpoints."""

from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from xpyd.proxy import Proxy


def register(router: APIRouter, server: "Proxy") -> None:
    async def get_status():
        return {
            "prefill_node_count": len(server.prefill_instances),
            "decode_node_count": len(server.decode_instances),
            "prefill_nodes": server.prefill_instances,
            "decode_nodes": server.decode_instances,
        }

    async def _instance_status():
        result: dict[str, list] = {
            "prefill_instances": [],
            "decode_instances": [],
            "aggregated_instances": [],
        }
        for info in server.registry.get_all_instances():
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
