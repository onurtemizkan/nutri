"""
Circuit Breaker Pattern Implementation

Prevents cascade failures by temporarily stopping requests
when the ML service is experiencing issues.

States:
    CLOSED: Normal operation, requests flow through
    OPEN: Service is failing, all requests immediately rejected
    HALF_OPEN: Testing if service recovered, limited requests allowed
"""

import asyncio
import time
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject all
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for ML inference protection.

    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

        if breaker.is_open:
            raise ServiceUnavailable()

        try:
            result = await do_inference()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            raise
    """

    # Configuration
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 3

    # State
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: Optional[float] = field(default=None)
    last_state_change: float = field(default_factory=time.time)

    # Thread safety
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (should reject requests)."""
        if self.state == CircuitState.CLOSED:
            return False

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time is not None:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    # Transition to half-open (will be done on next call)
                    return False  # Allow one request through
            return True

        # HALF_OPEN: allow requests through for testing
        return False

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    async def record_success(self) -> None:
        """Record a successful request."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                logger.debug(
                    "circuit_breaker_success",
                    success_count=self.success_count,
                    threshold=self.success_threshold,
                )

                if self.success_count >= self.success_threshold:
                    self._close()

            elif self.state == CircuitState.OPEN:
                # Successful request while open = transition to half-open
                self._half_open()
                self.success_count = 1

            # In CLOSED state, just reset failure count
            self.failure_count = 0

    async def record_failure(self) -> None:
        """Record a failed request."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            logger.warning(
                "circuit_breaker_failure",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
                state=self.state.value,
            )

            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open = back to open
                self._open()

            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self._open()

    async def check_and_acquire(self) -> bool:
        """
        Check if request should be allowed through.
        Returns True if allowed, False if circuit is open.

        Also handles state transitions.
        """
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if self.last_failure_time is not None:
                    elapsed = time.time() - self.last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self._half_open()
                        return True  # Allow this request as a test
                return False

            if self.state == CircuitState.HALF_OPEN:
                # In half-open, we allow requests through
                return True

            return False

    def _open(self) -> None:
        """Transition to OPEN state."""
        if self.state != CircuitState.OPEN:
            logger.error(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                recovery_timeout=self.recovery_timeout,
            )
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
            self.success_count = 0

    def _half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        if self.state != CircuitState.HALF_OPEN:
            logger.info("circuit_breaker_half_open", testing_recovery=True)
            self.state = CircuitState.HALF_OPEN
            self.last_state_change = time.time()
            self.success_count = 0
            self.failure_count = 0

    def _close(self) -> None:
        """Transition to CLOSED state."""
        if self.state != CircuitState.CLOSED:
            logger.info(
                "circuit_breaker_closed",
                recovered=True,
                success_count=self.success_count,
            )
            self.state = CircuitState.CLOSED
            self.last_state_change = time.time()
            self.failure_count = 0
            self.success_count = 0

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = time.time()
        logger.info("circuit_breaker_reset")

    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "time_in_state": time.time() - self.last_state_change,
        }
