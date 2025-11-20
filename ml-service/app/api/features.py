"""
API routes for feature engineering endpoints.
"""

from datetime import date
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.feature_engineering import FeatureEngineeringService
from app.schemas.features import (
    EngineerFeaturesRequest,
    EngineerFeaturesResponse,
    FeatureCategory,
)

router = APIRouter()


@router.post("/engineer", response_model=EngineerFeaturesResponse)
async def engineer_features(
    request: EngineerFeaturesRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Engineer ML features for a specific user and date.

    This is the primary endpoint for feature engineering. It:
    1. Fetches raw data (meals, activities, health metrics) from database
    2. Computes 50+ features across 5 categories
    3. Returns engineered features with quality metrics
    4. Caches results in Redis for 1 hour

    **Example Request**:
    ```json
    {
      "user_id": "user-123",
      "target_date": "2025-01-17",
      "categories": ["ALL"],
      "lookback_days": 30,
      "force_recompute": false
    }
    ```

    **Example Response**:
    ```json
    {
      "user_id": "user-123",
      "target_date": "2025-01-17",
      "computed_at": "2025-01-17T10:30:00Z",
      "cached": false,
      "nutrition": {
        "calories_daily": 2100,
        "protein_daily": 150,
        "protein_7d_avg": 145,
        ...
      },
      "feature_count": 51,
      "missing_features": 3,
      "data_quality_score": 0.94
    }
    ```
    """
    try:
        service = FeatureEngineeringService(db)
        response = await service.engineer_features(
            user_id=request.user_id,
            target_date=request.target_date,
            categories=request.categories,
            lookback_days=request.lookback_days,
            force_recompute=request.force_recompute,
        )
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error engineering features: {str(e)}"
        )


@router.get("/{user_id}/{target_date}", response_model=EngineerFeaturesResponse)
async def get_features(
    user_id: str,
    target_date: date,
    categories: List[FeatureCategory] = Query(default=[FeatureCategory.ALL]),
    lookback_days: int = Query(default=30, ge=7, le=90),
    force_recompute: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
):
    """
    Get engineered features for a user and date (convenience GET endpoint).

    This is a convenience wrapper around POST /engineer that uses query parameters.
    Useful for quick fetches and testing.

    **Query Parameters**:
    - `categories`: Comma-separated list (nutrition, activity, health, temporal, interaction, all)
    - `lookback_days`: Days of historical data (7-90, default: 30)
    - `force_recompute`: Force recomputation (default: false)

    **Example**:
    ```
    GET /api/features/user-123/2025-01-17?categories=nutrition,activity&lookback_days=30
    ```
    """
    try:
        service = FeatureEngineeringService(db)
        response = await service.engineer_features(
            user_id=user_id,
            target_date=target_date,
            categories=categories,
            lookback_days=lookback_days,
            force_recompute=force_recompute,
        )
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching features: {str(e)}"
        )


@router.get("/{user_id}/{target_date}/summary")
async def get_features_summary(
    user_id: str,
    target_date: date,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a summary of features without full details (lightweight endpoint).

    Returns only feature counts and quality metrics, not the actual feature values.
    Useful for quick checks and dashboards.

    **Example Response**:
    ```json
    {
      "user_id": "user-123",
      "target_date": "2025-01-17",
      "feature_count": 51,
      "missing_features": 3,
      "data_quality_score": 0.94,
      "categories_available": ["nutrition", "activity", "health", "temporal", "interaction"],
      "cached": true
    }
    ```
    """
    try:
        service = FeatureEngineeringService(db)
        response = await service.engineer_features(
            user_id=user_id,
            target_date=target_date,
            categories=[FeatureCategory.ALL],
            lookback_days=30,
            force_recompute=False,
        )

        # Return summary only
        categories_available = []
        if response.nutrition:
            categories_available.append("nutrition")
        if response.activity:
            categories_available.append("activity")
        if response.health:
            categories_available.append("health")
        if response.temporal:
            categories_available.append("temporal")
        if response.interaction:
            categories_available.append("interaction")

        return {
            "user_id": response.user_id,
            "target_date": response.target_date,
            "feature_count": response.feature_count,
            "missing_features": response.missing_features,
            "data_quality_score": response.data_quality_score,
            "categories_available": categories_available,
            "cached": response.cached,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching feature summary: {str(e)}"
        )


@router.delete("/{user_id}/cache")
async def invalidate_feature_cache(
    user_id: str,
):
    """
    Invalidate cached features for a user.

    Useful when user data is updated (new meals, activities, health metrics)
    and you want to force recomputation on next request.

    **Example**:
    ```
    DELETE /api/features/user-123/cache
    ```

    **Response**:
    ```json
    {
      "user_id": "user-123",
      "invalidated": true,
      "message": "Feature cache cleared"
    }
    ```
    """
    from app.redis_client import redis_client

    try:
        # Invalidate all feature caches for this user
        count = await redis_client.invalidate_user_cache(user_id)

        return {
            "user_id": user_id,
            "invalidated": True,
            "keys_deleted": count,
            "message": "Feature cache cleared successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error invalidating cache: {str(e)}"
        )
