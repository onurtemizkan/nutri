"""
ML Inference Queue System

Provides concurrency control, request queuing, and circuit breaker
for the ML inference service.

Usage:
    from app.core.queue import inference_queue, QueueConfig

    # In endpoint
    result = await inference_queue.submit(analyze_func, image)
"""

from app.core.queue.config import QueueConfig, queue_settings
from app.core.queue.manager import InferenceQueue, inference_queue
from app.core.queue.circuit_breaker import CircuitBreaker, CircuitState
from app.core.queue.metrics import QueueMetrics, queue_metrics

__all__ = [
    "QueueConfig",
    "queue_settings",
    "InferenceQueue",
    "inference_queue",
    "CircuitBreaker",
    "CircuitState",
    "QueueMetrics",
    "queue_metrics",
]
