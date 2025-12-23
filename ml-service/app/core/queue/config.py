"""
Queue Configuration

Environment-based configuration for the ML inference queue system.
Adjust these values based on your Hetzner instance type.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class QueueConfig(BaseSettings):
    """
    Configuration for the ML inference queue.

    Instance recommendations:
        CPX21 (3 vCPU):  max_concurrent=1, max_queue_size=30
        CPX31 (4 vCPU):  max_concurrent=2, max_queue_size=50
        CCX23 (4 dedicated): max_concurrent=2, max_queue_size=50
        GEX44 (GPU):     max_concurrent=4, max_queue_size=100
    """

    # -------------------------------------------------------------------------
    # Concurrency Control
    # -------------------------------------------------------------------------
    # Maximum concurrent ML inferences
    # Rule of thumb: (vCPU / 2) for CPU, (GPU count * 2) for GPU
    max_concurrent_inferences: int = 2

    # Number of background workers processing the queue
    # Usually equals max_concurrent_inferences
    num_workers: int = 2

    # -------------------------------------------------------------------------
    # Queue Limits
    # -------------------------------------------------------------------------
    # Maximum requests that can wait in queue
    max_queue_size: int = 50

    # Maximum time a request can wait in queue (seconds)
    request_timeout_seconds: float = 30.0

    # Maximum time for a single inference (seconds)
    inference_timeout_seconds: float = 60.0

    # -------------------------------------------------------------------------
    # Backpressure / Rejection
    # -------------------------------------------------------------------------
    # Start rejecting when queue reaches this percentage
    rejection_threshold_percent: float = 80.0

    # Enable early rejection (503) when queue is filling up
    enable_early_rejection: bool = True

    # -------------------------------------------------------------------------
    # Circuit Breaker
    # -------------------------------------------------------------------------
    # Enable circuit breaker pattern
    enable_circuit_breaker: bool = True

    # Number of consecutive failures before opening circuit
    circuit_failure_threshold: int = 5

    # Seconds to wait before attempting recovery
    circuit_recovery_timeout: float = 30.0

    # Number of successful requests to close circuit
    circuit_success_threshold: int = 3

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    # Enable Prometheus metrics
    enable_metrics: bool = True

    # Prefix for metric names
    metrics_prefix: str = "nutri_ml"

    class Config:
        env_prefix = "QUEUE_"
        env_file = ".env"

    @property
    def rejection_threshold(self) -> int:
        """Absolute queue size at which to start rejecting."""
        return int(self.max_queue_size * self.rejection_threshold_percent / 100)


@lru_cache()
def get_queue_settings() -> QueueConfig:
    """Get cached queue settings instance."""
    return QueueConfig()


# Convenience accessor
queue_settings = get_queue_settings()
