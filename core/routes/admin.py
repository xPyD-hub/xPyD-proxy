# SPDX-License-Identifier: Apache-2.0
"""Admin route handlers."""

import ipaddress
import itertools
import logging
import os

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("MicroPDProxyServer")


def register(router: APIRouter, server) -> None:
    """Register admin routes on *router*."""

    def api_key_authenticate(x_api_key: str = Header(...)):
        expected_api_key = os.environ.get("ADMIN_API_KEY")
        if not expected_api_key:
            logger.error("ADMIN_API_KEY is not set in the environment.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error.",
            )
        if x_api_key != expected_api_key:
            logger.warning("Unauthorized access attempt on admin endpoint")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid API Key.",
            )

    async def get_status():
        return {
            "prefill_node_count": len(server.prefill_instances),
            "decode_node_count": len(server.decode_instances),
            "prefill_nodes": server.prefill_instances,
            "decode_nodes": server.decode_instances,
        }

    async def add_instance_endpoint(request: Request):
        try:
            data = await request.json()
            logger.warning(str(data))
            instance_type = data.get("type")
            instance = data.get("instance")
            if instance_type not in ["prefill", "decode"]:
                raise HTTPException(status_code=400, detail="Invalid instance type.")
            if not instance or ":" not in instance:
                raise HTTPException(status_code=400, detail="Invalid instance format.")
            host, port_str = instance.split(":")
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port_str)
                if not (0 < port < 65536):
                    raise HTTPException(status_code=400, detail="Invalid port number.")
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid instance address.") from e

            # validate_instance lives on the Proxy class (MicroPDProxyServer.py)
            is_valid = await server.validate_instance(instance)
            if not is_valid:
                raise HTTPException(status_code=400, detail="Instance validation failed.")

            if instance_type == "prefill":
                with server.scheduling_policy.lock:
                    if instance not in server.prefill_instances:
                        server.prefill_instances.append(instance)
                        server.prefill_cycler = itertools.cycle(server.prefill_instances)
                    else:
                        raise HTTPException(status_code=400, detail="Instance already exists.")
            else:
                with server.scheduling_policy.lock:
                    if instance not in server.decode_instances:
                        server.decode_instances.append(instance)
                        server.decode_cycler = itertools.cycle(server.decode_instances)
                    else:
                        raise HTTPException(status_code=400, detail="Instance already exists.")

            return JSONResponse(content={
                "message": f"Added {instance} to {instance_type}_instances."
            })
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error("Error in add_instance_endpoint: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

    def remove_instance_endpoint(instance_type, instance):
        return

    router.get("/status", response_class=JSONResponse)(get_status)
    router.post("/instances/add", dependencies=[Depends(api_key_authenticate)])(add_instance_endpoint)
    router.options("/status")(lambda: None)
