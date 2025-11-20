"""
Pytest Configuration and Shared Fixtures

This file provides:
- Test database setup and teardown
- Shared fixtures used across all test files
- Pytest configuration (markers, async support)
"""

import pytest
import pytest_asyncio
import asyncio
from typing import AsyncGenerator, Generator

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """
    Configure pytest markers.

    Markers:
    - slow: Tests that take significant time (model training, etc.)
    - integration: Integration tests requiring external services
    - unit: Fast unit tests
    """
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


# Configure asyncio for async tests
@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest_asyncio.fixture(scope="function")
async def db_engine():
    """
    Create a test database engine.

    Uses in-memory SQLite for fast, isolated tests.
    Each test gets a fresh database.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,  # Set to True for SQL debugging
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Create a test database session.

    Each test gets a fresh session that is rolled back after the test.
    This ensures test isolation.
    """
    async_session = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="function")
def override_get_db(db: AsyncSession):
    """
    Override the get_db dependency for API tests.

    This injects the test database session into FastAPI endpoints.
    """
    async def _override_get_db():
        yield db

    app.dependency_overrides[get_db] = _override_get_db

    yield

    # Cleanup
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def client(override_get_db) -> AsyncGenerator[AsyncClient, None]:
    """
    Create an async HTTP client for testing API endpoints.

    This client automatically uses the test database through the
    override_get_db fixture.

    Usage:
        async def test_endpoint(client: AsyncClient):
            response = await client.get("/api/endpoint")
            assert response.status_code == 200
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ============================================================================
# Test Data Helpers
# ============================================================================


@pytest.fixture
def sample_user_data():
    """
    Sample user data for testing.

    Returns a dictionary suitable for creating a User model.
    """
    return {
        "id": "test_user_001",
        "email": "test@example.com",
        "name": "Test User",
        "height_cm": 175,
        "weight_kg": 70,
        "age": 28,
        "sex": "male",
        "activity_level": "active",
        "goals": ["improve_fitness", "track_health"],
    }


@pytest.fixture
def sample_meal_data():
    """
    Sample meal data for testing.

    Returns a dictionary suitable for creating a Meal model.
    """
    from datetime import datetime

    return {
        "id": "meal_001",
        "user_id": "test_user_001",
        "name": "Breakfast",
        "consumed_at": datetime.now(),
        "calories": 500.0,
        "protein_g": 30.0,
        "carbs_g": 60.0,
        "fat_g": 15.0,
        "fiber_g": 8.0,
    }


@pytest.fixture
def sample_activity_data():
    """
    Sample activity data for testing.

    Returns a dictionary suitable for creating an Activity model.
    """
    from datetime import datetime

    return {
        "id": "activity_001",
        "user_id": "test_user_001",
        "name": "Morning Run",
        "activity_type": "running",
        "started_at": datetime.now(),
        "duration_minutes": 30,
        "intensity": 0.7,
        "calories_burned": 250.0,
    }


@pytest.fixture
def sample_health_metric_data():
    """
    Sample health metric data for testing.

    Returns a dictionary suitable for creating a HealthMetric model.
    """
    from datetime import datetime

    return {
        "id": "metric_001",
        "user_id": "test_user_001",
        "metric_type": "RESTING_HEART_RATE",
        "value": 60.0,
        "recorded_at": datetime.now(),
    }


# ============================================================================
# Assertion Helpers
# ============================================================================


def assert_valid_rhr(value: float):
    """
    Assert that a Resting Heart Rate value is realistic.

    Args:
        value: RHR value in BPM

    Raises:
        AssertionError: If value is not realistic
    """
    assert 40 <= value <= 100, f"RHR {value} BPM is not realistic (expected 40-100)"


def assert_valid_hrv(value: float):
    """
    Assert that a Heart Rate Variability value is realistic.

    Args:
        value: HRV SDNN value in ms

    Raises:
        AssertionError: If value is not realistic
    """
    assert 20 <= value <= 150, f"HRV {value} ms is not realistic (expected 20-150)"


def assert_good_model_performance(r2: float, mape: float):
    """
    Assert that model performance metrics are acceptable.

    Args:
        r2: R-squared score (0-1)
        mape: Mean Absolute Percentage Error (%)

    Raises:
        AssertionError: If performance is not acceptable
    """
    assert r2 > 0.5, f"R² score {r2:.3f} is too low (expected > 0.5)"
    assert mape < 15.0, f"MAPE {mape:.1f}% is too high (expected < 15%)"
    assert r2 <= 1.0, f"R² score {r2:.3f} is impossible (max 1.0)"
    assert mape >= 0, f"MAPE {mape:.1f}% cannot be negative"


def assert_valid_confidence_interval(lower: float, predicted: float, upper: float):
    """
    Assert that confidence interval is valid.

    Args:
        lower: Lower bound
        predicted: Predicted value
        upper: Upper bound

    Raises:
        AssertionError: If interval is not valid
    """
    assert lower < predicted, f"Lower bound {lower} should be less than predicted {predicted}"
    assert predicted < upper, f"Predicted {predicted} should be less than upper bound {upper}"
    assert lower > 0, f"Lower bound {lower} should be positive"


def assert_valid_shap_values(shap_values: list):
    """
    Assert that SHAP values are valid.

    Args:
        shap_values: List of SHAP feature importance dicts

    Raises:
        AssertionError: If SHAP values are not valid
    """
    assert len(shap_values) > 0, "Should have SHAP values"

    for i, feat in enumerate(shap_values):
        assert feat["rank"] == i + 1, f"Features should be ranked (expected {i+1}, got {feat['rank']})"
        assert feat["importance_score"] >= 0, f"Importance score should be non-negative"
        assert feat["impact_direction"] in ["positive", "negative", "neutral"]
        assert feat["impact_magnitude"] in ["strong", "moderate", "weak"]


# ============================================================================
# Test Markers and Utilities
# ============================================================================


def skip_if_slow(reason: str = "Slow test - run with 'pytest -m slow'"):
    """
    Skip test if not running slow tests.

    Usage:
        @skip_if_slow()
        def test_model_training():
            ...
    """
    return pytest.mark.slow


def requires_trained_model(func):
    """
    Decorator to skip tests that require a trained model.

    Usage:
        @requires_trained_model
        async def test_prediction():
            ...
    """
    return pytest.mark.skipif(
        not has_trained_model(),
        reason="No trained model available - run training tests first",
    )(func)


def has_trained_model() -> bool:
    """
    Check if a trained model exists.

    Returns:
        True if at least one model is available
    """
    from pathlib import Path

    models_dir = Path("models")
    return models_dir.exists() and any(models_dir.iterdir())


# ============================================================================
# Performance Benchmarking
# ============================================================================


@pytest.fixture
def benchmark_timer():
    """
    Simple benchmark timer for performance testing.

    Usage:
        def test_feature_engineering(benchmark_timer):
            with benchmark_timer("Feature Engineering"):
                # ... code to benchmark ...
                pass
    """
    import time
    from contextlib import contextmanager

    @contextmanager
    def timer(name: str):
        start = time.time()
        yield
        elapsed = time.time() - start
        print(f"\n⏱️  {name}: {elapsed:.2f}s")

    return timer


# ============================================================================
# Cleanup Helpers
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """
    Automatically cleanup test artifacts after each test.

    This removes:
    - Test models
    - Temporary files
    - Cache entries
    """
    yield

    # Cleanup after test
    from pathlib import Path
    import shutil

    # Remove test models (models with "test_user" in ID)
    models_dir = Path("models")
    if models_dir.exists():
        for model_dir in models_dir.glob("test_user_*"):
            try:
                shutil.rmtree(model_dir)
            except Exception as e:
                print(f"⚠️  Failed to cleanup {model_dir}: {e}")

    # Remove test cache entries
    # TODO: Clear Redis test cache if needed


# ============================================================================
# Logging Configuration
# ============================================================================


@pytest.fixture(autouse=True)
def configure_test_logging():
    """
    Configure logging for tests.

    Reduces noise from libraries while keeping important logs.
    """
    import logging

    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    # Keep our app logs at INFO level
    logging.getLogger("app").setLevel(logging.INFO)

    yield


# ============================================================================
# Test Data Statistics
# ============================================================================


def print_dataset_stats(meals: list, activities: list, metrics: list):
    """
    Print statistics about test dataset.

    Useful for debugging test data generation.

    Args:
        meals: List of meal dicts
        activities: List of activity dicts
        metrics: List of health metric dicts
    """
    print("\n📊 Dataset Statistics:")
    print(f"   Meals: {len(meals)}")
    print(f"   Activities: {len(activities)}")
    print(f"   Health Metrics: {len(metrics)}")

    if meals:
        import numpy as np

        calories = [m["calories"] for m in meals]
        protein = [m["protein_g"] for m in meals]

        print(f"   Avg calories per meal: {np.mean(calories):.1f} kcal")
        print(f"   Avg protein per meal: {np.mean(protein):.1f}g")

    if metrics:
        rhr_metrics = [m for m in metrics if m["metric_type"] == "RESTING_HEART_RATE"]
        hrv_metrics = [m for m in metrics if m["metric_type"] == "HEART_RATE_VARIABILITY_SDNN"]

        if rhr_metrics:
            rhr_values = [m["value"] for m in rhr_metrics]
            print(f"   RHR range: {min(rhr_values):.1f} - {max(rhr_values):.1f} BPM")

        if hrv_metrics:
            hrv_values = [m["value"] for m in hrv_metrics]
            print(f"   HRV range: {min(hrv_values):.1f} - {max(hrv_values):.1f} ms")
