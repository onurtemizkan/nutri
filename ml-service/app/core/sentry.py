"""
Sentry Error Tracking Configuration for ML Service

Captures errors, performance traces, and provides structured error reporting.
Integrates with FastAPI and SQLAlchemy for comprehensive monitoring.
"""

import os
from typing import Any, Dict, Optional

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration

from app.config import settings


def init_sentry() -> None:
    """
    Initialize Sentry error tracking.

    Must be called early in application startup, before any other
    integrations are initialized.
    """
    sentry_dsn = os.environ.get("SENTRY_DSN", "")

    if not sentry_dsn:
        print("Sentry DSN not configured, skipping initialization")
        return

    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=settings.environment,
        release=os.environ.get("GITHUB_SHA", settings.app_version),
        # Integrations
        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
            AsyncioIntegration(),
        ],
        # Performance Monitoring
        # Sample 10% of transactions in production, 100% in development
        traces_sample_rate=(0.1 if settings.environment == "production" else 1.0),
        # Don't send PII by default
        send_default_pii=False,
        # Error filtering
        ignore_errors=[
            # Expected authentication errors
            "Invalid credentials",
            "Token expired",
            "Unauthorized",
            # Rate limiting
            "Too many requests",
        ],
        # Sensitive data scrubbing
        before_send=scrub_sensitive_data,  # type: ignore[arg-type]
    )

    print(f"Sentry initialized for environment: {settings.environment}")


def scrub_sensitive_data(
    event: Dict[str, Any], hint: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Scrub sensitive data from Sentry events before sending.

    Removes authorization tokens, cookies, passwords, and partial-masks emails.

    Args:
        event: The Sentry event dict
        hint: Additional context about the event

    Returns:
        The scrubbed event, or None to drop the event
    """
    # Scrub request headers
    if "request" in event and "headers" in event["request"]:
        headers = event["request"]["headers"]
        sensitive_headers = ["authorization", "cookie", "x-api-key"]
        for header in sensitive_headers:
            if header in headers:
                headers[header] = "[REDACTED]"

    # Scrub request body for sensitive fields
    if "request" in event and "data" in event["request"]:
        data = event["request"]["data"]
        if isinstance(data, dict):
            sensitive_fields = [
                "password",
                "token",
                "secret",
                "api_key",
                "apiKey",
                "credit_card",
                "creditCard",
                "ssn",
                "access_token",
                "accessToken",
                "refresh_token",
                "refreshToken",
            ]
            for field in sensitive_fields:
                if field in data:
                    data[field] = "[REDACTED]"

    # Partial-mask user email for debugging while preserving privacy
    # 'john.doe@example.com' becomes 'j***@example.com'
    if "user" in event and "email" in event.get("user", {}):
        email = event["user"]["email"]
        if email and "@" in email:
            local, domain = email.split("@", 1)
            event["user"]["email"] = f"{local[0]}***@{domain}"

    return event


def capture_exception(
    error: Exception, context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Capture an exception with Sentry.

    Use this for manual error capture when automatic capture isn't sufficient.

    Args:
        error: The exception to capture
        context: Optional additional context to include
    """
    if not os.environ.get("SENTRY_DSN"):
        return

    if context:
        sentry_sdk.set_context("additional", context)

    sentry_sdk.capture_exception(error)


def set_user(user_id: str, email: Optional[str] = None) -> None:
    """
    Set user context for Sentry events.

    Call this after user authentication to associate errors with users.

    Args:
        user_id: The user's ID
        email: Optional user email (will be partial-masked in events)
    """
    if not os.environ.get("SENTRY_DSN"):
        return

    sentry_sdk.set_user(
        {
            "id": user_id,
            **({"email": email} if email else {}),
        }
    )


def clear_user() -> None:
    """
    Clear user context.

    Call this on logout or when user context is no longer valid.
    """
    if not os.environ.get("SENTRY_DSN"):
        return

    sentry_sdk.set_user(None)
