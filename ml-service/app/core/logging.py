"""
Structured Logging Configuration for ML Service
Uses structlog for high-performance JSON logging with sensitive data redaction
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict

import structlog

# Correlation ID context variable for request tracing
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Get the current correlation ID or generate a new one."""
    return correlation_id_var.get() or str(uuid.uuid4())


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id_var.set(correlation_id)


# Sensitive data keys to redact
SENSITIVE_KEYS = {
    "password",
    "token",
    "secret",
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "jwt",
    "bearer",
    "credential",
}


def redact_sensitive_data(
    _logger: Any, _method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Processor to redact sensitive data from logs."""

    def redact(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: "[REDACTED]" if k.lower() in SENSITIVE_KEYS else redact(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [redact(item) for item in obj]
        return obj

    return redact(event_dict)


def add_service_context(
    _logger: Any, _method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Processor to add service metadata to all logs."""
    event_dict["service"] = "nutri-ml-service"
    event_dict["correlation_id"] = get_correlation_id()
    return event_dict


def configure_logging(
    environment: str = "development", log_level: str = "INFO"
) -> None:
    """Configure structured logging based on environment."""

    # Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # Shared processors for all environments
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_service_context,
        redact_sensitive_data,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if environment == "production":
        # Production: JSON output for log aggregators
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Human-readable colored output
        processors = shared_processors + [
            structlog.processors.ExceptionPrettyPrinter(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a structlog logger instance."""
    return structlog.get_logger(name)
