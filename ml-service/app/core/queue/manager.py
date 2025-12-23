"""
ML Inference Queue Manager

Core queue system that manages request queuing, concurrency control,
and worker coordination for ML inference.

Features:
    - Bounded queue with configurable size
    - Semaphore-based concurrency control
    - Background workers for processing
    - Circuit breaker integration
    - Prometheus metrics
    - Graceful timeout handling
"""

import asyncio
import time
from typing import Any, Callable, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from uuid import uuid4

from app.core.queue.config import queue_settings
from app.core.queue.circuit_breaker import CircuitBreaker, CircuitState
from app.core.queue.metrics import queue_metrics
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class InferenceRequest(Generic[T]):
    """
    Represents a queued inference request.

    Attributes:
        id: Unique request identifier
        created_at: Timestamp when request was created
        future: asyncio.Future to deliver result
        args: Positional arguments for inference function
        kwargs: Keyword arguments for inference function
    """

    id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)
    args: tuple = field(default=())
    kwargs: dict = field(default_factory=dict)

    @property
    def age(self) -> float:
        """Time since request was created (seconds)."""
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        """Check if request has exceeded timeout."""
        return self.age > queue_settings.request_timeout_seconds


class InferenceQueue:
    """
    Manages ML inference request queue with concurrency control.

    Usage:
        queue = InferenceQueue()
        await queue.start()

        # Submit inference request
        result = await queue.submit(inference_func, image_data)

        # Shutdown
        await queue.stop()
    """

    def __init__(
        self,
        max_concurrent: Optional[int] = None,
        max_queue_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ):
        # Configuration
        self.max_concurrent = max_concurrent or queue_settings.max_concurrent_inferences
        self.max_queue_size = max_queue_size or queue_settings.max_queue_size
        self.num_workers = num_workers or queue_settings.num_workers

        # Queue and synchronization
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(
            maxsize=self.max_queue_size
        )
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self.max_concurrent)

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=queue_settings.circuit_failure_threshold,
            recovery_timeout=queue_settings.circuit_recovery_timeout,
            success_threshold=queue_settings.circuit_success_threshold,
        )

        # Worker management
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._inference_func: Optional[Callable] = None

        # Stats
        self._total_processed = 0
        self._total_errors = 0

        logger.info(
            "inference_queue_initialized",
            max_concurrent=self.max_concurrent,
            max_queue_size=self.max_queue_size,
            num_workers=self.num_workers,
        )

    @property
    def queue_size(self) -> int:
        """Current number of requests in queue."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        """Check if queue is at capacity."""
        return self._queue.full()

    @property
    def should_reject(self) -> bool:
        """Check if we should reject new requests (backpressure)."""
        if not queue_settings.enable_early_rejection:
            return self.is_full

        return self.queue_size >= queue_settings.rejection_threshold

    @property
    def circuit_state(self) -> CircuitState:
        """Current circuit breaker state."""
        return self._circuit_breaker.state

    def get_status(self) -> dict:
        """Get comprehensive queue status."""
        return {
            "running": self._running,
            "queue_size": self.queue_size,
            "max_queue_size": self.max_queue_size,
            "queue_utilization_percent": round(
                self.queue_size / self.max_queue_size * 100, 1
            ),
            "active_workers": len([w for w in self._workers if not w.done()]),
            "total_workers": self.num_workers,
            "max_concurrent": self.max_concurrent,
            "circuit_breaker": self._circuit_breaker.get_status(),
            "total_processed": self._total_processed,
            "total_errors": self._total_errors,
            "should_reject": self.should_reject,
        }

    async def start(self, inference_func: Callable) -> None:
        """
        Start the queue workers.

        Args:
            inference_func: Async function to call for each inference request
        """
        if self._running:
            logger.warning("inference_queue_already_running")
            return

        self._inference_func = inference_func
        self._running = True

        # Start background workers
        for i in range(self.num_workers):
            worker = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"inference_worker_{i}",
            )
            self._workers.append(worker)

        queue_metrics.set_workers_active(self.num_workers)

        logger.info(
            "inference_queue_started",
            num_workers=self.num_workers,
            max_concurrent=self.max_concurrent,
        )

    async def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the queue and wait for workers to finish.

        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self._running:
            return

        logger.info("inference_queue_stopping")
        self._running = False

        # Wait for queue to drain (with timeout)
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "inference_queue_drain_timeout",
                remaining=self.queue_size,
            )

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        queue_metrics.set_workers_active(0)

        logger.info("inference_queue_stopped")

    async def submit(
        self,
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Submit an inference request to the queue.

        Args:
            *args: Arguments to pass to inference function
            timeout: Optional timeout override
            **kwargs: Keyword arguments to pass to inference function

        Returns:
            Result from inference function

        Raises:
            asyncio.TimeoutError: If request times out
            RuntimeError: If queue is not running
            Exception: If inference fails
        """
        if not self._running:
            raise RuntimeError("Inference queue is not running")

        request_timeout = timeout or queue_settings.request_timeout_seconds
        request_start = time.time()

        # Check circuit breaker
        if queue_settings.enable_circuit_breaker:
            if not await self._circuit_breaker.check_and_acquire():
                queue_metrics.record_rejection("circuit_open")
                raise CircuitOpenError("Circuit breaker is open")

        # Check backpressure
        if self.should_reject:
            queue_metrics.record_rejection("early_rejection")
            raise QueueFullError(
                f"Queue at {self.queue_size}/{self.max_queue_size} - rejecting"
            )

        # Create request
        request = InferenceRequest(args=args, kwargs=kwargs)

        # Try to enqueue
        try:
            self._queue.put_nowait(request)
            queue_metrics.record_request_received()
            queue_metrics.set_queue_size(self.queue_size)

            logger.debug(
                "request_queued",
                request_id=request.id,
                queue_size=self.queue_size,
            )

        except asyncio.QueueFull:
            queue_metrics.record_rejection("queue_full")
            raise QueueFullError("Queue is full")

        # Wait for result
        try:
            result = await asyncio.wait_for(request.future, timeout=request_timeout)
            total_time = time.time() - request_start
            queue_metrics.record_total_time(total_time)

            logger.debug(
                "request_completed",
                request_id=request.id,
                total_time_ms=round(total_time * 1000, 1),
            )

            return result

        except asyncio.TimeoutError:
            queue_metrics.record_timeout("queue_wait")
            logger.warning(
                "request_timeout",
                request_id=request.id,
                timeout=request_timeout,
            )
            raise

    async def _worker_loop(self, worker_id: int) -> None:
        """
        Background worker that processes queued requests.

        Args:
            worker_id: Unique identifier for this worker
        """
        logger.info("inference_worker_started", worker_id=worker_id)

        while self._running:
            try:
                # Get request from queue (with timeout to check running flag)
                try:
                    request = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                queue_metrics.record_request_dequeued()
                queue_metrics.set_queue_size(self.queue_size)

                # Check if request already expired
                if request.is_expired:
                    logger.warning(
                        "request_expired_in_queue",
                        request_id=request.id,
                        age=request.age,
                    )
                    if not request.future.done():
                        request.future.set_exception(
                            asyncio.TimeoutError("Request expired in queue")
                        )
                    self._queue.task_done()
                    continue

                # Record queue wait time
                queue_wait = time.time() - request.created_at
                queue_metrics.record_queue_wait(queue_wait)

                # Acquire semaphore for inference
                async with self._semaphore:
                    await self._process_request(request, worker_id)

                self._queue.task_done()

            except asyncio.CancelledError:
                logger.info("inference_worker_cancelled", worker_id=worker_id)
                break
            except Exception as e:
                logger.exception(
                    "inference_worker_error",
                    worker_id=worker_id,
                    error=str(e),
                )

        logger.info("inference_worker_stopped", worker_id=worker_id)

    async def _process_request(
        self,
        request: InferenceRequest,
        worker_id: int,
    ) -> None:
        """
        Process a single inference request.

        Args:
            request: The inference request to process
            worker_id: ID of the worker processing this request
        """
        if request.future.done():
            # Request already completed (e.g., cancelled)
            return

        inference_start = time.time()
        queue_metrics.record_inference_start()

        try:
            # Run inference with timeout
            result = await asyncio.wait_for(
                self._inference_func(*request.args, **request.kwargs),
                timeout=queue_settings.inference_timeout_seconds,
            )

            # Set result
            if not request.future.done():
                request.future.set_result(result)

            inference_duration = time.time() - inference_start
            queue_metrics.record_inference_end(inference_duration, success=True)
            await self._circuit_breaker.record_success()
            self._total_processed += 1

            logger.debug(
                "inference_completed",
                request_id=request.id,
                worker_id=worker_id,
                duration_ms=round(inference_duration * 1000, 1),
            )

        except asyncio.TimeoutError:
            inference_duration = time.time() - inference_start
            queue_metrics.record_inference_end(inference_duration, success=False)
            queue_metrics.record_timeout("inference")
            await self._circuit_breaker.record_failure()
            self._total_errors += 1

            if not request.future.done():
                request.future.set_exception(asyncio.TimeoutError("Inference timeout"))

            logger.error(
                "inference_timeout",
                request_id=request.id,
                duration_ms=round(inference_duration * 1000, 1),
            )

        except Exception as e:
            inference_duration = time.time() - inference_start
            queue_metrics.record_inference_end(inference_duration, success=False)
            await self._circuit_breaker.record_failure()
            self._total_errors += 1

            if not request.future.done():
                request.future.set_exception(e)

            logger.error(
                "inference_error",
                request_id=request.id,
                error=str(e),
                duration_ms=round(inference_duration * 1000, 1),
            )


class QueueFullError(Exception):
    """Raised when the queue is full and cannot accept new requests."""

    pass


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open."""

    pass


# Singleton instance
inference_queue = InferenceQueue()
