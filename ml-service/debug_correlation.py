import asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app
from tests.conftest import test_user_with_data, override_get_db
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.database import Base
import pytest

@pytest.mark.asyncio
async def debug_correlation():
    # Create test database
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestingSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async def override_get_db():
        async with TestingSessionLocal() as session:
            yield session

    from app.database import get_db
    app.dependency_overrides[get_db] = override_get_db

    # Get user with test data
    from tests.conftest import _generate_test_user_with_data
    async with TestingSessionLocal() as db:
        user_id = await _generate_test_user_with_data(db)
        await db.commit()

    # Make the request
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/correlations/analyze",
            json={
                "user_id": user_id,
                "target_metric": "HEART_RATE_VARIABILITY_SDNN",
                "methods": ["pearson"],
                "lookback_days": 90,
                "significance_threshold": 0.05,
                "min_correlation": 0.25,
                "top_k": 15,
            },
        )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

    if response.status_code != 200:
        print("ERROR DETAILS:")
        print(response.json())

if __name__ == "__main__":
    asyncio.run(debug_correlation())
