"""
Prometheus Metrics for ML Inference Queue

Provides observability into queue performance, latency, and health.
"""

from typing import Optional
from dataclasses import dataclass, field
from prometheus_client import Counter, Gauge, Histogram, Info

from app.core.queue.config import queue_settings


@dataclass
class QueueMetrics:
    """
    Prometheus metrics for the inference queue.

    Metrics exposed:
        - nutri_ml_queue_size: Current number of requests in queue
        - nutri_ml_active_inferences: Currently running inferences
        - nutri_ml_requests_total: Total requests by status
        - nutri_ml_inference_duration_seconds: Inference time histogram
        - nutri_ml_queue_wait_seconds: Time spent waiting in queue
        - nutri_ml_rejections_total: Rejected requests (503)
        - nutri_ml_timeouts_total: Timed out requests (504)
        - nutri_ml_circuit_breaker_state: Circuit breaker state
    """

    prefix: str = field(default_factory=lambda: queue_settings.metrics_prefix)
    enabled: bool = field(default_factory=lambda: queue_settings.enable_metrics)

    # Gauges (current values)
    queue_size: Optional[Gauge] = field(default=None)
    active_inferences: Optional[Gauge] = field(default=None)
    circuit_breaker_state: Optional[Gauge] = field(default=None)
    workers_active: Optional[Gauge] = field(default=None)

    # Counters (cumulative)
    requests_total: Optional[Counter] = field(default=None)
    rejections_total: Optional[Counter] = field(default=None)
    timeouts_total: Optional[Counter] = field(default=None)
    circuit_breaker_trips: Optional[Counter] = field(default=None)

    # Histograms (distributions)
    inference_duration: Optional[Histogram] = field(default=None)
    queue_wait_time: Optional[Histogram] = field(default=None)
    total_request_time: Optional[Histogram] = field(default=None)

    # Info
    config_info: Optional[Info] = field(default=None)

    def __post_init__(self) -> None:
        """Initialize Prometheus metrics."""
        if not self.enabled:
            return

        p = self.prefix

        # Gauges
        self.queue_size = Gauge(
            f"{p}_queue_size",
            "Current number of requests waiting in queue",
        )

        self.active_inferences = Gauge(
            f"{p}_active_inferences",
            "Number of ML inferences currently running",
        )

        self.circuit_breaker_state = Gauge(
            f"{p}_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
        )

        self.workers_active = Gauge(
            f"{p}_workers_active",
            "Number of active queue workers",
        )

        # Counters
        self.requests_total = Counter(
            f"{p}_requests_total",
            "Total ML inference requests",
            ["status"],  # success, error, timeout, rejected
        )

        self.rejections_total = Counter(
            f"{p}_rejections_total",
            "Total rejected requests (503)",
            ["reason"],  # queue_full, circuit_open, early_rejection
        )

        self.timeouts_total = Counter(
            f"{p}_timeouts_total",
            "Total timed out requests (504)",
            ["stage"],  # queue_wait, inference
        )

        self.circuit_breaker_trips = Counter(
            f"{p}_circuit_breaker_trips_total",
            "Number of times circuit breaker has opened",
        )

        # Histograms with appropriate buckets for ML inference
        self.inference_duration = Histogram(
            f"{p}_inference_duration_seconds",
            "ML inference duration in seconds",
            buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        self.queue_wait_time = Histogram(
            f"{p}_queue_wait_seconds",
            "Time spent waiting in queue",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )

        self.total_request_time = Histogram(
            f"{p}_total_request_seconds",
            "Total request time (queue + inference)",
            buckets=[0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )

        # Info
        self.config_info = Info(
            f"{p}_config",
            "Queue configuration",
        )
        self.config_info.info(
            {
                "max_concurrent": str(queue_settings.max_concurrent_inferences),
                "max_queue_size": str(queue_settings.max_queue_size),
                "request_timeout": str(queue_settings.request_timeout_seconds),
                "circuit_breaker_enabled": str(queue_settings.enable_circuit_breaker),
            }
        )

    def record_request_received(self) -> None:
        """Record that a request was received."""
        if self.enabled and self.queue_size:
            self.queue_size.inc()

    def record_request_dequeued(self) -> None:
        """Record that a request was dequeued for processing."""
        if self.enabled and self.queue_size:
            self.queue_size.dec()

    def record_inference_start(self) -> None:
        """Record start of inference."""
        if self.enabled and self.active_inferences:
            self.active_inferences.inc()

    def record_inference_end(self, duration: float, success: bool = True) -> None:
        """Record end of inference."""
        if not self.enabled:
            return

        if self.active_inferences:
            self.active_inferences.dec()

        if self.inference_duration:
            self.inference_duration.observe(duration)

        if self.requests_total:
            status = "success" if success else "error"
            self.requests_total.labels(status=status).inc()

    def record_queue_wait(self, wait_time: float) -> None:
        """Record time spent waiting in queue."""
        if self.enabled and self.queue_wait_time:
            self.queue_wait_time.observe(wait_time)

    def record_total_time(self, total_time: float) -> None:
        """Record total request time."""
        if self.enabled and self.total_request_time:
            self.total_request_time.observe(total_time)

    def record_rejection(self, reason: str) -> None:
        """Record a rejected request."""
        if not self.enabled:
            return

        if self.rejections_total:
            self.rejections_total.labels(reason=reason).inc()

        if self.requests_total:
            self.requests_total.labels(status="rejected").inc()

    def record_timeout(self, stage: str) -> None:
        """Record a timeout."""
        if not self.enabled:
            return

        if self.timeouts_total:
            self.timeouts_total.labels(stage=stage).inc()

        if self.requests_total:
            self.requests_total.labels(status="timeout").inc()

    def record_circuit_breaker_trip(self) -> None:
        """Record circuit breaker opening."""
        if self.enabled and self.circuit_breaker_trips:
            self.circuit_breaker_trips.inc()

    def set_circuit_breaker_state(self, state: int) -> None:
        """Set circuit breaker state (0=closed, 1=open, 2=half_open)."""
        if self.enabled and self.circuit_breaker_state:
            self.circuit_breaker_state.set(state)

    def set_queue_size(self, size: int) -> None:
        """Set current queue size."""
        if self.enabled and self.queue_size:
            self.queue_size.set(size)

    def set_workers_active(self, count: int) -> None:
        """Set number of active workers."""
        if self.enabled and self.workers_active:
            self.workers_active.set(count)


# Singleton instance
queue_metrics = QueueMetrics()
