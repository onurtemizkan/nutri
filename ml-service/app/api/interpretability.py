"""
Interpretability API Routes (Phase 3)

Endpoints for model interpretability and explainability:
- POST /explain - SHAP feature importance (local explanation)
- POST /global-importance - Global feature importance
- POST /what-if - Test what-if scenarios
- POST /counterfactual - Generate counterfactual explanations
- GET /attention/{user_id}/{metric}/{target_date} - Attention weights (TODO)
"""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.interpretability import (
    FeatureImportanceRequest,
    FeatureImportanceResponse,
    GlobalImportanceRequest,
    GlobalImportanceResponse,
    WhatIfRequest,
    WhatIfResponse,
    CounterfactualRequest,
    CounterfactualResponse,
)
from app.services.shap_explainer import SHAPExplainerService
from app.services.what_if import WhatIfService

router = APIRouter()


# ============================================================================
# Feature Importance (SHAP Explanations)
# ============================================================================


@router.post("/explain", response_model=FeatureImportanceResponse)
async def explain_prediction(
    request: FeatureImportanceRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Explain a prediction using SHAP feature importance.

    **What this does:**
    - Shows which features contributed most to a specific prediction
    - Ranks features by importance (SHAP values)
    - Shows direction of impact (positive/negative)
    - Groups features by category (nutrition, activity, health)

    **Use case:** "Why did my model predict 65 BPM for tomorrow's RHR?"

    **Example Request:**
    ```json
    {
        "user_id": "user_123",
        "metric": "RESTING_HEART_RATE",
        "target_date": "2025-01-16",
        "method": "shap",
        "top_k": 10
    }
    ```

    **Example Response:**
    ```json
    {
        "predicted_value": 65.0,
        "baseline_value": 60.0,
        "feature_importances": [
            {
                "feature_name": "nutrition_protein_daily",
                "importance_score": 0.85,
                "rank": 1,
                "shap_value": -0.85,
                "impact_direction": "negative",
                "impact_magnitude": "strong",
                "feature_value": 120.0
            },
            ...
        ],
        "summary": "The top 3 drivers of your resting heart rate prediction are: protein daily (decreasing), workout intensity (increasing), sleep duration (decreasing)",
        "top_nutrition_features": ["nutrition_protein_daily", "nutrition_late_night_carbs"],
        "top_activity_features": ["activity_workout_intensity_avg"],
        "top_health_features": ["health_hrv_sdnn_lag_1"]
    }
    ```
    """
    try:
        explainer_service = SHAPExplainerService(db)
        result = await explainer_service.explain_prediction(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Explanation failed: {str(e)}"
        )


@router.post("/global-importance", response_model=GlobalImportanceResponse)
async def get_global_importance(
    request: GlobalImportanceRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Get global feature importance for a trained model.

    **What this does:**
    - Shows which features are generally most important across all predictions
    - Provides mean importance and standard deviation
    - Shows overall category importance (nutrition vs activity vs health)

    **Use case:** "Which features matter most for RHR predictions in general?"

    **Example Request:**
    ```json
    {
        "model_id": "user_123_RESTING_HEART_RATE_20250115_103045",
        "method": "shap",
        "top_k": 20
    }
    ```

    **Example Response:**
    ```json
    {
        "model_id": "user_123_RESTING_HEART_RATE_20250115_103045",
        "metric": "RESTING_HEART_RATE",
        "method": "shap",
        "feature_importances": [
            {
                "feature_name": "nutrition_protein_7d_avg",
                "mean_importance": 0.75,
                "std_importance": 0.15,
                "rank": 1,
                "impact_direction": "positive"
            },
            ...
        ],
        "summary": "Across all predictions for RESTING_HEART_RATE, the most important features are: protein 7d avg, workout intensity avg, sleep duration avg, hrv sdnn lag 1, late night calories",
        "nutrition_importance": 0.45,
        "activity_importance": 0.30,
        "health_importance": 0.25
    }
    ```
    """
    try:
        explainer_service = SHAPExplainerService(db)
        result = await explainer_service.get_global_importance(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Global importance calculation failed: {str(e)}"
        )


# ============================================================================
# What-If Scenarios
# ============================================================================


@router.post("/what-if", response_model=WhatIfResponse)
async def test_what_if_scenarios(
    request: WhatIfRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Test what-if scenarios to see how changes affect predictions.

    **What this does:**
    - Tests hypothetical changes to features
    - Shows how predictions would change
    - Identifies best and worst scenarios
    - Provides actionable recommendations

    **Use case:** "What if I ate 50g more protein tomorrow? What if I did a high-intensity workout?"

    **Example Request:**
    ```json
    {
        "user_id": "user_123",
        "metric": "RESTING_HEART_RATE",
        "target_date": "2025-01-16",
        "scenarios": [
            {
                "scenario_name": "High Protein Day",
                "changes": [
                    {
                        "feature_name": "nutrition_protein_daily",
                        "current_value": 100.0,
                        "new_value": 150.0,
                        "change_description": "+50g protein"
                    }
                ]
            },
            {
                "scenario_name": "High Intensity Workout",
                "changes": [
                    {
                        "feature_name": "activity_workout_intensity_avg",
                        "current_value": 0.5,
                        "new_value": 0.9,
                        "change_description": "High intensity workout"
                    }
                ]
            },
            {
                "scenario_name": "Perfect Day",
                "changes": [
                    {
                        "feature_name": "nutrition_protein_daily",
                        "current_value": 100.0,
                        "new_value": 150.0,
                        "change_description": "+50g protein"
                    },
                    {
                        "feature_name": "activity_workout_intensity_avg",
                        "current_value": 0.5,
                        "new_value": 0.7,
                        "change_description": "Moderate workout"
                    },
                    {
                        "feature_name": "nutrition_late_night_carbs",
                        "current_value": 80.0,
                        "new_value": 20.0,
                        "change_description": "-60g late night carbs"
                    }
                ]
            }
        ]
    }
    ```

    **Example Response:**
    ```json
    {
        "baseline_prediction": 65.0,
        "scenarios": [
            {
                "scenario_name": "High Protein Day",
                "predicted_value": 62.5,
                "change_from_baseline": -2.5,
                "percent_change": -3.8,
                "confidence_score": 0.85
            },
            {
                "scenario_name": "High Intensity Workout",
                "predicted_value": 67.0,
                "change_from_baseline": 2.0,
                "percent_change": 3.1,
                "confidence_score": 0.80
            },
            {
                "scenario_name": "Perfect Day",
                "predicted_value": 60.0,
                "change_from_baseline": -5.0,
                "percent_change": -7.7,
                "confidence_score": 0.82
            }
        ],
        "best_scenario": "Perfect Day",
        "best_value": 60.0,
        "worst_scenario": "High Intensity Workout",
        "worst_value": 67.0,
        "summary": "Testing 3 scenarios for resting heart rate, your baseline prediction is 65.0. The best scenario is 'Perfect Day' with a predicted value of 60.0 (-5.0 from baseline). The worst scenario is 'High Intensity Workout' with 67.0.",
        "recommendation": "To achieve the best outcome (60.0 resting heart rate), consider: +50g protein, Moderate workout."
    }
    ```
    """
    try:
        what_if_service = WhatIfService(db)
        result = await what_if_service.test_what_if_scenarios(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"What-if analysis failed: {str(e)}"
        )


# ============================================================================
# Counterfactual Explanations
# ============================================================================


@router.post("/counterfactual", response_model=CounterfactualResponse)
async def generate_counterfactual(
    request: CounterfactualRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate counterfactual explanations.

    **What this does:**
    - Finds minimal changes needed to reach a target value
    - Uses intelligent search to identify high-impact features
    - Provides specific, actionable suggestions
    - Calculates plausibility of suggested changes

    **Use case:** "What's the minimal change I need to make to get my RHR down to 60 BPM?"

    **Example Request:**
    ```json
    {
        "user_id": "user_123",
        "metric": "RESTING_HEART_RATE",
        "target_date": "2025-01-16",
        "target_type": "target_value",
        "target_value": 60.0,
        "max_changes": 3
    }
    ```

    **Example Response:**
    ```json
    {
        "current_prediction": 65.0,
        "target_prediction": 60.0,
        "counterfactual": {
            "current_prediction": 65.0,
            "target_prediction": 60.0,
            "achieved_prediction": 60.2,
            "changes": [
                {
                    "feature_name": "nutrition_protein_daily",
                    "current_value": 100.0,
                    "suggested_value": 130.0,
                    "change_amount": 30.0,
                    "change_description": "+30.0g protein"
                },
                {
                    "feature_name": "nutrition_late_night_carbs",
                    "current_value": 80.0,
                    "suggested_value": 50.0,
                    "change_amount": -30.0,
                    "change_description": "-30.0g late night carbs"
                }
            ],
            "plausibility_score": 0.9,
            "summary": "To move from 65.0 to 60.0 resting heart rate, you should: +30.0g protein, -30.0g late night carbs. This would achieve approximately 60.2."
        }
    }
    ```

    **Target Types:**
    - `improve`: Automatically improve the metric by 5%
    - `target_value`: Reach a specific target value (must provide target_value)
    - `minimize_change`: Find minimal changes for a target
    """
    try:
        what_if_service = WhatIfService(db)
        result = await what_if_service.generate_counterfactual(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Counterfactual generation failed: {str(e)}"
        )
