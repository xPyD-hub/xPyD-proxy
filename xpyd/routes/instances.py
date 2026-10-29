# SPDX-License-Identifier: Apache-2.0
"""Instance administration endpoints."""

import ipaddress
import logging
import os
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse

from xpyd.errors import INVALID_REQUEST, SERVER_ERROR, error_response

logger = logging.getLogger("xpyd.proxy")


def register(
    router: APIRouter,
    *,
    add_instance: Callable[[str, str], Awaitable[bool]],
    remove_instance: Callable[[str, str, float], Awaitable[None]],
) -> None:
    """Register instance administration routes on *router*."""

    def _authenticate_api_key(x_api_key: str):
        """Validate admin API key. Returns error_response or None."""
        expected_api_key = os.environ.get("ADMIN_API_KEY")
        if not expected_api_key:
            logger.error("ADMIN_API_KEY is not set in the environment.")
            return error_response("Server configuration error", SERVER_ERROR, 500)
        if x_api_key != expected_api_key:
            logger.warning("Unauthorized access attempt on admin endpoint")
            return error_response("Forbidden: Invalid API Key", INVALID_REQUEST, 403)
        return None

    async def add_instance_endpoint(
        request: Request,
        x_api_key: str = Header(...),  # noqa: B008 - FastAPI dependency idiom
    ):
        auth_error = _authenticate_api_key(x_api_key)
        if auth_error:
            return auth_error

        try:
            data = await request.json()
            logger.debug("Add instance request: %s", data)
            instance_type = data.get("type")
            instance = data.get("instance")
            if instance_type not in ["prefill", "decode", "aggregated"]:
                return error_response("Invalid instance type", INVALID_REQUEST, 400)
            if not instance or ":" not in instance:
                return error_response("Invalid instance format", INVALID_REQUEST, 400)
            parts = instance.split(":")
            if len(parts) != 2:
                # Matches ProxyConfig address validation: IPv4/hostname only.
                return error_response("Invalid instance format", INVALID_REQUEST, 400)
            host, port_str = parts
            try:
                if host != "localhost":
                    ipaddress.ip_address(host)
                port = int(port_str)
                if not (0 < port < 65536):
                    return error_response("Invalid port number", INVALID_REQUEST, 400)
            except Exception:
                return error_response("Invalid instance address", INVALID_REQUEST, 400)

            is_valid = await add_instance(instance_type, instance)
            if not is_valid:
                return error_response(
                    "Instance validation failed", INVALID_REQUEST, 400
                )

            return JSONResponse(
                content={"message": f"Added {instance} to {instance_type}_instances."}
            )
        except ValueError as e:
            return error_response(str(e), INVALID_REQUEST, 400)
        except Exception as e:
            logger.error("Error in add_instance_endpoint: %s", str(e))
            return error_response(f"Internal error: {e}", SERVER_ERROR, 500)

    async def remove_instance_endpoint(
        request: Request,
        x_api_key: str = Header(...),  # noqa: B008 - FastAPI dependency idiom
    ):
        auth_error = _authenticate_api_key(x_api_key)
        if auth_error:
            return auth_error

        try:
            data = await request.json()
            instance_type = data.get("type")
            instance = data.get("instance")
            timeout_seconds = data.get("timeout_seconds", 60)
            if instance_type not in ["prefill", "decode", "aggregated"]:
                return error_response("Invalid instance type", INVALID_REQUEST, 400)
            if not isinstance(instance, str) or not instance:
                return error_response("Invalid instance", INVALID_REQUEST, 400)
            if (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not 0 <= timeout_seconds <= 3600
            ):
                return error_response(
                    "timeout_seconds must be between 0 and 3600",
                    INVALID_REQUEST,
                    400,
                )

            await remove_instance(
                instance_type,
                instance,
                float(timeout_seconds),
            )
            return JSONResponse(
                content={
                    "message": (f"Removed {instance} from {instance_type}_instances.")
                }
            )
        except KeyError:
            return error_response("Instance not found", INVALID_REQUEST, 404)
        except ValueError as exc:
            return error_response(str(exc), INVALID_REQUEST, 400)
        except TimeoutError as exc:
            return error_response(str(exc), SERVER_ERROR, 504)
        except Exception as exc:
            logger.exception("Error in remove_instance_endpoint")
            return error_response(f"Internal error: {exc}", SERVER_ERROR, 500)

    router.post("/instances/add")(add_instance_endpoint)
    router.post("/instances/remove")(remove_instance_endpoint)
    router.options("/instances/add")(lambda: None)
    router.options("/instances/remove")(lambda: None)
