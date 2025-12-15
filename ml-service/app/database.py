"""
Database connection and session management
Uses async SQLAlchemy for non-blocking database operations
"""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,  # Log SQL queries in debug mode
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_pre_ping=True,  # Verify connections before using them
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for SQLAlchemy models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function to get database session.
    Yields a session and ensures it's closed after use.

    Usage in FastAPI:
        @app.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions outside of FastAPI dependency injection.

    Usage:
        async with get_db_context() as db:
            result = await db.execute(select(User))
            users = result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database (create tables if they don't exist).
    Note: In production, use Alembic migrations instead.
    """
    async with engine.begin() as conn:
        # Import all models here to ensure they're registered
        from app.models import health_metric, activity, meal, user  # noqa: F401
        from app.models import food_feedback  # noqa: F401

        # Create all tables (only for development)
        if settings.environment == "development":
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created (development mode)")


async def close_db() -> None:
    """
    Close database connections (call on app shutdown).
    """
    await engine.dispose()
    logger.info("Database connections closed")
