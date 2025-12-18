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

configure_logging(environment=settings.environment, log_level=settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    Runs on startup and shutdown.
    """
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

    logger.info("ml_service_ready")

    yield

    # Shutdown
    logger.info("ml_service_shutting_down")

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
        return {"status": "ready", "database": "ok", "redis": "ok" if redis_ready else "degraded"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "database not available"},
        )


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
