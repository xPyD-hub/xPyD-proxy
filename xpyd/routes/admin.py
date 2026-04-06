# SPDX-License-Identifier: Apache-2.0
"""Admin route handlers."""

import ipaddress
import itertools
import logging
import os

from fastapi import APIRouter, Header, Request

from xpyd.errors import INVALID_REQUEST, SERVER_ERROR, error_response
from fastapi.responses import JSONResponse

logger = logging.getLogger("xpyd.proxy")


def register(router: APIRouter, server) -> None:
    """Register admin routes on *router*."""

    def _authenticate_api_key(x_api_key: str):
        """Validate admin API key. Returns error_response or None."""
        expected_api_key = os.environ.get("ADMIN_API_KEY")
        if not expected_api_key:
            logger.error("ADMIN_API_KEY is not set in the environment.")
            return error_response(
                "Server configuration error", SERVER_ERROR, 500
            )
        if x_api_key != expected_api_key:
            logger.warning("Unauthorized access attempt on admin endpoint")
            return error_response(
                "Forbidden: Invalid API Key", INVALID_REQUEST, 403
            )
        return None

    async def get_status():
        return {
            "prefill_node_count": len(server.prefill_instances),
            "decode_node_count": len(server.decode_instances),
            "prefill_nodes": server.prefill_instances,
            "decode_nodes": server.decode_instances,
        }

    async def add_instance_endpoint(
        request: Request,
        x_api_key: str = Header(...),
    ):
        auth_error = _authenticate_api_key(x_api_key)
        if auth_error:
            return auth_error

        try:
            data = await request.json()
            logger.warning(str(data))
            instance_type = data.get("type")
            instance = data.get("instance")
            if instance_type not in ["prefill", "decode"]:
                return error_response("Invalid instance type", INVALID_REQUEST, 400)
            if not instance or ":" not in instance:
                return error_response("Invalid instance format", INVALID_REQUEST, 400)
            host, port_str = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port_str)
                if not (0 < port < 65536):
                    return error_response("Invalid port number", INVALID_REQUEST, 400)
            except Exception:
                return error_response("Invalid instance address", INVALID_REQUEST, 400)

            # validate_instance lives on the Proxy class (MicroPDProxyServer.py)
            is_valid = await server.validate_instance(instance)
            if not is_valid:
                return error_response("Instance validation failed", INVALID_REQUEST, 400)

            if instance_type == "prefill":
                with server.scheduling_policy.lock:
                    if instance not in server.prefill_instances:
                        server.prefill_instances.append(instance)
                        server.prefill_cycler = itertools.cycle(server.prefill_instances)
                    else:
                        return error_response("Instance already exists", INVALID_REQUEST, 400)
            else:
                with server.scheduling_policy.lock:
                    if instance not in server.decode_instances:
                        server.decode_instances.append(instance)
                        server.decode_cycler = itertools.cycle(server.decode_instances)
                    else:
                        return error_response("Instance already exists", INVALID_REQUEST, 400)

            return JSONResponse(content={
                "message": f"Added {instance} to {instance_type}_instances."
            })
        except Exception as e:
            logger.error("Error in add_instance_endpoint: %s", str(e))
            return error_response(f"Internal error: {e}", SERVER_ERROR, 500)

    def remove_instance_endpoint(instance_type, instance):
        return

    router.get("/status", response_class=JSONResponse)(get_status)
    router.post("/instances/add")(add_instance_endpoint)
    router.options("/status")(lambda: None)
