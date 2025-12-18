"""
Request Logging Middleware for FastAPI
Logs HTTP requests/responses with timing and correlation IDs
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger, set_correlation_id, get_correlation_id

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request/response logging.
    - Extracts or generates correlation IDs
    - Logs request start and completion with timing
    - Sets correlation ID in response headers
    """

    # Paths to skip logging (health checks, etc.)
    SKIP_PATHS = {"/health", "/health/live", "/ready", "/"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate correlation ID
        correlation_id = request.headers.get(
            "x-correlation-id", request.headers.get("x-request-id", str(uuid.uuid4()))
        )
        set_correlation_id(correlation_id)

        # Check if we should skip logging for this path
        path = request.url.path
        should_log = path not in self.SKIP_PATHS

        start_time = time.time()

        # Log request start
        if should_log:
            logger.info(
                "request_started",
                method=request.method,
                path=path,
                query=str(request.query_params) if request.query_params else None,
                client_ip=request.client.host if request.client else None,
            )

        try:
            response = await call_next(request)

            # Add correlation ID to response headers
            response.headers["x-correlation-id"] = correlation_id

            # Log response
            if should_log:
                duration_ms = (time.time() - start_time) * 1000
                log_method = (
                    logger.error
                    if response.status_code >= 500
                    else logger.warning
                    if response.status_code >= 400
                    else logger.info
                )
                log_method(
                    "request_completed",
                    method=request.method,
                    path=path,
                    status_code=response.status_code,
                    duration_ms=round(duration_ms, 2),
                )

            return response

        except Exception as e:
            # Log exception
            duration_ms = (time.time() - start_time) * 1000
            logger.exception(
                "request_failed",
                method=request.method,
                path=path,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=round(duration_ms, 2),
            )
            raise
