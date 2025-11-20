"""
API routes for correlation analysis endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.correlation_engine import CorrelationEngineService
from app.schemas.correlations import (
    CorrelationRequest,
    CorrelationResponse,
    LagAnalysisRequest,
    LagAnalysisResponse,
    HealthMetricTarget,
)

router = APIRouter()


@router.post("/analyze", response_model=CorrelationResponse)
async def analyze_correlations(
    request: CorrelationRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze correlations between features and a target health metric.

    This endpoint finds which nutrition/activity features correlate with
    health metrics like RHR, HRV, sleep quality, etc.

    **Use Case**: "Which factors affect my RHR?"

    **Example Request**:
    ```json
    {
      "user_id": "user-123",
      "target_metric": "RESTING_HEART_RATE",
      "methods": ["pearson", "spearman"],
      "lookback_days": 30,
      "significance_threshold": 0.05,
      "min_correlation": 0.3,
      "top_k": 10
    }
    ```

    **Example Response**:
    ```json
    {
      "user_id": "user-123",
      "target_metric": "RESTING_HEART_RATE",
      "analyzed_at": "2025-01-17T10:30:00Z",
      "lookback_days": 30,
      "correlations": [
        {
          "feature_name": "nutrition_protein_daily",
          "feature_category": "nutrition",
          "correlation": -0.68,
          "p_value": 0.002,
          "method": "pearson",
          "is_significant": true,
          "strength": "moderate",
          "direction": "negative",
          "explained_variance": 0.46
        },
        ...
      ],
      "total_features_analyzed": 51,
      "significant_correlations": 8,
      "strongest_positive": {...},
      "strongest_negative": {...},
      "data_quality_score": 0.94
    }
    ```

    **Interpretation**:
    - Negative correlation: Higher protein → Lower RHR (better)
    - P-value < 0.05: Statistically significant
    - Explained variance: 46% of RHR variance explained by protein
    """
    try:
        service = CorrelationEngineService(db)
        response = await service.analyze_correlations(request)
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing correlations: {str(e)}\n{traceback.format_exc()}"
        )


@router.post("/lag-analysis", response_model=LagAnalysisResponse)
async def analyze_lag(
    request: LagAnalysisRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze time-delayed effects (lag analysis).

    This endpoint tests correlations at different time lags to find when
    a feature has the strongest effect on a health metric.

    **Use Case**: "When does protein intake affect HRV?"

    **Example Request**:
    ```json
    {
      "user_id": "user-123",
      "target_metric": "HEART_RATE_VARIABILITY_RMSSD",
      "feature_name": "nutrition_protein_daily",
      "max_lag_hours": 72,
      "lag_step_hours": 6,
      "lookback_days": 30,
      "method": "pearson"
    }
    ```

    **Example Response**:
    ```json
    {
      "user_id": "user-123",
      "target_metric": "HEART_RATE_VARIABILITY_RMSSD",
      "feature_name": "nutrition_protein_daily",
      "analyzed_at": "2025-01-17T10:30:00Z",
      "lag_results": [
        {"lag_hours": 0, "correlation": 0.12, "p_value": 0.234, "is_significant": false},
        {"lag_hours": 6, "correlation": 0.35, "p_value": 0.045, "is_significant": true},
        {"lag_hours": 12, "correlation": 0.58, "p_value": 0.002, "is_significant": true},
        {"lag_hours": 18, "correlation": 0.42, "p_value": 0.018, "is_significant": true},
        {"lag_hours": 24, "correlation": 0.21, "p_value": 0.187, "is_significant": false}
      ],
      "optimal_lag_hours": 12,
      "optimal_correlation": 0.58,
      "immediate_effect": false,
      "delayed_effect": true,
      "effect_duration_hours": 18,
      "interpretation": "nutrition_protein_daily has a delayed effect on Heart Rate Variability Rmssd. Changes in nutrition_protein_daily increases Heart Rate Variability Rmssd after 12 hours. Correlation strength: moderate. The effect lasts for 18 hours."
    }
    ```

    **Interpretation**:
    - Strongest correlation at 12-hour lag
    - Effect starts at 6 hours and lasts 18 hours
    - This means protein eaten today affects HRV starting ~6h later, peaking at 12h
    """
    try:
        service = CorrelationEngineService(db)
        response = await service.analyze_lag(request)
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing lag: {str(e)}"
        )


@router.get("/{user_id}/{target_metric}/summary")
async def get_correlation_summary(
    user_id: str,
    target_metric: HealthMetricTarget,
    lookback_days: int = 30,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a quick summary of top correlations for a user and metric.

    Returns top 5 positive and top 5 negative correlations without full details.

    **Example**:
    ```
    GET /api/correlations/user-123/RESTING_HEART_RATE/summary?lookback_days=30
    ```

    **Example Response**:
    ```json
    {
      "user_id": "user-123",
      "target_metric": "RESTING_HEART_RATE",
      "top_positive": [
        {"feature": "activity_high_intensity_minutes", "correlation": 0.72},
        {"feature": "nutrition_carbs_daily", "correlation": 0.58},
        ...
      ],
      "top_negative": [
        {"feature": "nutrition_protein_daily", "correlation": -0.68},
        {"feature": "health_hrv_7d_avg", "correlation": -0.55},
        ...
      ],
      "data_quality_score": 0.94
    }
    ```
    """
    try:
        service = CorrelationEngineService(db)

        # Run full analysis
        from app.schemas.correlations import CorrelationRequest, CorrelationMethod

        request = CorrelationRequest(
            user_id=user_id,
            target_metric=target_metric,
            methods=[CorrelationMethod.PEARSON],
            lookback_days=lookback_days,
            significance_threshold=0.05,
            min_correlation=0.3,
            top_k=20,
        )

        response = await service.analyze_correlations(request)

        # Extract top positive and negative
        top_positive = [
            {"feature": r.feature_name, "correlation": r.correlation}
            for r in response.correlations
            if r.direction == "positive"
        ][:5]

        top_negative = [
            {"feature": r.feature_name, "correlation": r.correlation}
            for r in response.correlations
            if r.direction == "negative"
        ][:5]

        return {
            "user_id": user_id,
            "target_metric": target_metric,
            "lookback_days": lookback_days,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "total_significant": response.significant_correlations,
            "data_quality_score": response.data_quality_score,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching correlation summary: {str(e)}"
        )
