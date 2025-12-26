"""
Nutri ML Service - FastAPI Application
Main entry point for the ML service
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import init_db, close_db
from app.redis_client import redis_client

# Configure structured logging
from app.core.logging import configure_logging, get_logger
from app.middleware.logging import LoggingMiddleware

# Import inference queue
from app.core.queue import inference_queue

# Import Sentry integration
from app.core.sentry import init_sentry, capture_exception

configure_logging(environment=settings.environment, log_level=settings.log_level)
logger = get_logger(__name__)


def warmup_ml_models():
    """
    Pre-load ML models during startup to avoid timeout on first request.

    Models loaded:
    - CLIP classifier (primary) - ~600MB
    - OWL-ViT detector (multi-food) - ~1.5GB
    - Food-101 ViT (fallback) - ~350MB
    - Coarse food classifier (CLIP-based) - shared with primary

    This prevents the first food analysis request from timing out
    while models are being downloaded/loaded.
    """
    import time

    start_time = time.time()
    logger.info("ml_models_warmup_starting")

    try:
        # Import here to avoid circular imports
        from app.services.food_analysis_service import food_analysis_service
        from app.ml_models.coarse_classifier import get_coarse_classifier

        # 1. Pre-load CLIP classifier (primary)
        logger.info("loading_clip_classifier")
        clip_start = time.time()
        ensemble = food_analysis_service._get_ensemble_classifier()
        # Force CLIP to load by calling _get_clip()
        ensemble._get_clip()
        logger.info(
            "clip_classifier_loaded",
            duration_ms=int((time.time() - clip_start) * 1000),
        )

        # 2. Pre-load OWL-ViT detector (if multi-food is enabled and not in fast mode)
        if food_analysis_service._enable_multi_food and not settings.fast_mode:
            logger.info("loading_owlvit_detector")
            detector_start = time.time()
            detector = food_analysis_service._get_food_detector()
            detector.load_model()
            logger.info(
                "owlvit_detector_loaded",
                duration_ms=int((time.time() - detector_start) * 1000),
            )
        elif settings.fast_mode:
            logger.info(
                "owlvit_detector_skipped",
                reason="FAST_MODE=true - OWL-ViT disabled for faster inference",
            )

        # 3. Optionally pre-load Food-101 (fallback) - lighter weight
        logger.info("loading_food101_fallback")
        f101_start = time.time()
        ensemble._get_food_101()
        logger.info(
            "food101_fallback_loaded",
            duration_ms=int((time.time() - f101_start) * 1000),
        )

        # 4. Pre-load Coarse Food Classifier (used by classify-and-search)
        # This is critical for food scanning - it uses CLIP for zero-shot classification
        logger.info("loading_coarse_classifier")
        coarse_start = time.time()
        coarse_classifier = get_coarse_classifier()
        coarse_classifier.load_model()
        logger.info(
            "coarse_classifier_loaded",
            duration_ms=int((time.time() - coarse_start) * 1000),
        )

        total_time = time.time() - start_time
        logger.info(
            "ml_models_warmup_complete",
            total_duration_ms=int(total_time * 1000),
        )

    except Exception as e:
        logger.warning(
            "ml_models_warmup_failed",
            error=str(e),
            message="Models will load lazily on first request (may cause timeout)",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    Runs on startup and shutdown.
    """
    # Initialize Sentry FIRST (before any other initialization)
    init_sentry()

    # Startup
    logger.info(
        "ml_service_starting",
        environment=settings.environment,
        version=settings.app_version,
    )

    # Initialize database
    try:
        await init_db()
        logger.info("database_initialized")
    except Exception as e:
        logger.error("database_initialization_failed", error=str(e))

    # Initialize Redis
    try:
        await redis_client.connect()
        logger.info("redis_connected")
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))

    # Warm up ML models (pre-load to avoid first-request timeout)
    # Can be skipped with SKIP_MODEL_WARMUP=true for faster dev startup
    if settings.skip_model_warmup:
        logger.info(
            "ml_models_warmup_skipped",
            reason="SKIP_MODEL_WARMUP=true",
            warning="First food analysis request will be slow",
        )
    else:
        # Run in thread pool to not block the event loop
        import asyncio
        import concurrent.futures

        logger.info("ml_models_warmup_queued")
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, warmup_ml_models)

    # Start inference queue with the analyze_food function
    from app.services.food_analysis_service import food_analysis_service

    async def inference_wrapper(pil_image, dimensions_obj, cooking_method):
        """Wrapper for the food analysis service inference."""
        return await food_analysis_service.analyze_food(
            pil_image, dimensions_obj, cooking_method
        )

    await inference_queue.start(inference_wrapper)
    logger.info(
        "inference_queue_started",
        max_concurrent=inference_queue.max_concurrent,
        max_queue_size=inference_queue.max_queue_size,
        num_workers=inference_queue.num_workers,
    )

    logger.info("ml_service_ready")

    yield

    # Shutdown
    logger.info("ml_service_shutting_down")

    # Stop inference queue (wait for pending requests)
    await inference_queue.stop(timeout=10.0)
    logger.info("inference_queue_stopped")

    # Close database connections
    await close_db()
    logger.info("database_closed")

    # Close Redis connection
    await redis_client.close()
    logger.info("redis_closed")

    logger.info("ml_service_shutdown_complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Machine Learning API for personalized nutrition insights",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Request Logging Middleware
app.add_middleware(LoggingMiddleware)


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

import time
from datetime import datetime, timezone
from sqlalchemy import text
from app.database import AsyncSessionLocal


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API information"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """
    Liveness probe - just checks if server is running.
    For container orchestrators to verify the process is alive.
    """
    return {"status": "ok"}


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns service status and dependency health with latency measurements.
    """
    checks = {}
    overall_status = "healthy"

    # Database check with latency measurement
    db_start = time.time()
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = {
            "status": "healthy",
            "latency_ms": int((time.time() - db_start) * 1000),
        }
    except Exception as e:
        checks["database"] = {
            "status": "unhealthy",
            "latency_ms": int((time.time() - db_start) * 1000),
            "error": str(e),
        }
        overall_status = "unhealthy"

    # Redis check with latency measurement
    redis_start = time.time()
    try:
        if redis_client.redis:
            await redis_client.redis.ping()
            checks["redis"] = {
                "status": "healthy",
                "latency_ms": int((time.time() - redis_start) * 1000),
            }
        else:
            checks["redis"] = {
                "status": "unavailable",
                "latency_ms": int((time.time() - redis_start) * 1000),
                "error": "Redis client not connected",
            }
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        checks["redis"] = {
            "status": "unhealthy",
            "latency_ms": int((time.time() - redis_start) * 1000),
            "error": str(e),
        }
        if overall_status == "healthy":
            overall_status = "degraded"

    # Determine HTTP status code
    status_code = 200 if overall_status in ["healthy", "degraded"] else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": overall_status,
            "version": settings.app_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        },
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes/load balancers.
    Returns 200 if service is ready to accept requests.
    """
    # Check database connectivity
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        db_ready = True
    except Exception:
        db_ready = False

    # Check Redis connectivity
    redis_ready = False
    try:
        redis_ready = await redis_client.redis.ping() if redis_client.redis else False
    except Exception:
        pass

    # Both database and redis should be ready for full readiness
    if db_ready:
        return {
            "status": "ready",
            "database": "ok",
            "redis": "ok" if redis_ready else "degraded",
        }
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "database not available"},
        )


@app.get("/queue/status", tags=["Health"])
async def queue_status():
    """
    Get ML inference queue status.

    Returns detailed information about:
    - Queue size and capacity
    - Active workers and inferences
    - Circuit breaker state
    - Processing statistics

    Use this endpoint to monitor queue health and identify bottlenecks.
    """
    status = inference_queue.get_status()

    # Determine overall health
    health = "healthy"
    if status["circuit_breaker"]["state"] == "open":
        health = "degraded"
    elif status["queue_utilization_percent"] > 80:
        health = "warning"
    elif status["should_reject"]:
        health = "critical"

    return {
        "health": health,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **status,
    }


@app.post("/queue/reset-circuit-breaker", tags=["Health"])
async def reset_circuit_breaker():
    """
    Reset the circuit breaker to closed state.

    **Use with caution** - only reset if you're confident the underlying
    issue has been resolved. The circuit breaker is there to protect
    the service from cascading failures.
    """
    inference_queue._circuit_breaker.reset()
    logger.info("circuit_breaker_manually_reset")

    return {
        "status": "reset",
        "circuit_breaker": inference_queue._circuit_breaker.get_status(),
    }


# ============================================================================
# DEBUG ENDPOINTS (Non-Production Only)
# ============================================================================

import os

@app.get("/debug/sentry-test", tags=["Debug"])
async def test_sentry():
    """
    Trigger a test error to verify Sentry integration.
    Only available in non-production environments.
    """
    if settings.environment == "production":
        return JSONResponse(status_code=404, content={"error": "Not found"})

    # Capture a test exception
    test_error = ValueError("Test Sentry integration - ML Service")
    capture_exception(test_error, context={
        "test_type": "manual-trigger",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "success": True,
        "message": "Test error sent to Sentry",
        "environment": settings.environment,
        "sentry_enabled": bool(os.environ.get("SENTRY_DSN")),
    }


@app.get("/debug/sentry-throw", tags=["Debug"])
async def test_sentry_throw():
    """
    Throw an unhandled error to verify automatic Sentry capture.
    Only available in non-production environments.
    """
    if settings.environment == "production":
        return JSONResponse(status_code=404, content={"error": "Not found"})

    # This will be caught by the global error handler and sent to Sentry
    raise ValueError("Test unhandled error for Sentry - ML Service")


@app.get("/debug/sentry-status", tags=["Debug"])
async def sentry_status():
    """
    Check Sentry configuration status.
    Only available in non-production environments.
    """
    if settings.environment == "production":
        return JSONResponse(status_code=404, content={"error": "Not found"})

    import sentry_sdk

    return {
        "enabled": bool(os.environ.get("SENTRY_DSN")),
        "environment": settings.environment,
        "client_initialized": sentry_sdk.is_initialized(),
        "dsn": "[CONFIGURED]" if os.environ.get("SENTRY_DSN") else "[NOT SET]",
    }


# ============================================================================
# API ROUTES
# ============================================================================

from app.api import api_router  # noqa: E402

# Register all API routes
app.include_router(api_router)

logger.info("api_routes_registered", routes=["/api/features", "/api/correlations"])


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Global exception handler.
    Catches unhandled exceptions and returns a JSON error response.
    """
    # Capture exception with Sentry for server errors
    capture_exception(exc, context={
        "path": str(request.url.path),
        "method": request.method,
    })

    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.log_level.lower(),
    )
