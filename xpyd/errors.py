# SPDX-License-Identifier: Apache-2.0
"""Standardized error response helpers (OpenAI-compatible format)."""

from fastapi.responses import JSONResponse

# Error type constants
INVALID_REQUEST = "invalid_request_error"
SERVER_ERROR = "server_error"
PROXY_ERROR = "proxy_error"


def error_response(
    message: str,
    error_type: str = SERVER_ERROR,
    status_code: int = 500,
) -> JSONResponse:
    """Return an OpenAI-compatible JSON error response.

    Format::

        {
            "error": {
                "message": "...",
                "type": "invalid_request_error|server_error|proxy_error",
                "code": null
            }
        }
    """
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "code": None}},
        status_code=status_code,
    )
