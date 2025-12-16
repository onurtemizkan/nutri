"""
Nutri ML Service - FastAPI Application
Main entry point for the ML service
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import init_db, close_db
from app.redis_client import redis_client

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    Runs on startup and shutdown.
    """
    # Startup
    logger.info("🚀 Starting Nutri ML Service...")
    logger.info(f"📝 Environment: {settings.environment}")
    logger.info(f"🔧 Version: {settings.app_version}")

    # Initialize database
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")

    # Initialize Redis
    try:
        await redis_client.connect()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {str(e)}")

    logger.info("🎉 ML Service ready!")

    yield

    # Shutdown
    logger.info("👋 Shutting down Nutri ML Service...")

    # Close database connections
    await close_db()
    logger.info("✅ Database closed")

    # Close Redis connection
    await redis_client.close()
    logger.info("✅ Redis closed")

    logger.info("🏁 Shutdown complete")


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
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================


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


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns service status and dependency health.
    """
    # Check Redis connection
    redis_healthy = False
    try:
        redis_healthy = await redis_client.redis.ping() if redis_client.redis else False
    except Exception:
        pass

    # Check database (simplified check)
    # In production, you'd want to do an actual query
    db_healthy = True  # Assume healthy for now

    overall_status = "healthy" if (redis_healthy and db_healthy) else "degraded"

    return {
        "status": overall_status,
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "dependencies": {
            "database": "healthy" if db_healthy else "unhealthy",
            "redis": "healthy" if redis_healthy else "unhealthy",
        },
    }


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes/load balancers.
    Returns 200 if service is ready to accept requests.
    """
    redis_ready = False
    try:
        redis_ready = await redis_client.redis.ping() if redis_client.redis else False
    except Exception:
        pass

    if redis_ready:
        return {"status": "ready"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "dependencies not available"},
        )


# ============================================================================
# API ROUTES
# ============================================================================

from app.api import api_router  # noqa: E402

# Register all API routes
app.include_router(api_router)

logger.info("✅ API routes registered: /api/features, /api/correlations")


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
