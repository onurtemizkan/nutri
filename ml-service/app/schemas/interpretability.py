"""
Pydantic schemas for model interpretability and explanations.

Phase 3 features:
- SHAP feature importance
- Attention weights
- What-if scenarios
- Counterfactual explanations
"""

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.predictions import PredictionMetric


# ============================================================================
# Feature Importance
# ============================================================================


class ImportanceMethod(str, Enum):
    """Methods for calculating feature importance."""

    SHAP = "shap"  # SHAP values (Shapley Additive Explanations)
    LIME = "lime"  # Local Interpretable Model-agnostic Explanations
    PERMUTATION = "permutation"  # Permutation importance
    ATTENTION = "attention"  # LSTM attention weights


class FeatureImportance(BaseModel):
    """Single feature importance score."""

    feature_name: str = Field(..., description="Feature name")
    feature_category: str = Field(
        ..., description="Category (nutrition, activity, etc.)"
    )
    importance_score: float = Field(
        ..., description="Importance score (SHAP value or similar)"
    )
    rank: int = Field(..., description="Rank (1 = most important)")

    # For SHAP
    shap_value: Optional[float] = Field(
        None, description="SHAP value (positive = increases prediction)"
    )
    base_value: Optional[float] = Field(None, description="Baseline prediction value")

    # For human understanding
    impact_direction: str = Field(..., description="positive/negative/neutral")
    impact_magnitude: str = Field(..., description="strong/moderate/weak")

    # Actual feature value
    feature_value: Optional[float] = Field(
        None, description="Actual feature value used"
    )


class FeatureImportanceRequest(BaseModel):
    """Request to explain a prediction."""

    user_id: str = Field(..., description="User ID")
    metric: PredictionMetric = Field(..., description="Health metric")
    target_date: date = Field(..., description="Date of prediction to explain")

    method: ImportanceMethod = Field(
        default=ImportanceMethod.SHAP, description="Explanation method"
    )

    top_k: int = Field(
        default=10, ge=1, le=51, description="Number of top features to return (1-51)"
    )


class FeatureImportanceResponse(BaseModel):
    """Response with feature importance explanations."""

    user_id: str
    metric: PredictionMetric
    target_date: date
    method: ImportanceMethod

    # Prediction being explained
    predicted_value: float = Field(..., description="The prediction being explained")
    baseline_value: float = Field(..., description="Baseline/average prediction")

    # Feature importances
    feature_importances: List[FeatureImportance] = Field(
        ..., description="Ranked list of feature importances"
    )

    # Summary
    summary: str = Field(..., description="Natural language summary of top drivers")

    # Top features by category
    top_nutrition_features: List[str] = Field(
        default=[], description="Top nutrition drivers"
    )
    top_activity_features: List[str] = Field(
        default=[], description="Top activity drivers"
    )
    top_health_features: List[str] = Field(default=[], description="Top health drivers")


# ============================================================================
# Attention Weights
# ============================================================================


class AttentionWeight(BaseModel):
    """Attention weight for a specific time step (day)."""

    day_offset: int = Field(
        ..., description="Days before prediction (0 = yesterday, 29 = 30 days ago)"
    )
    day_date: date = Field(..., description="Actual date")
    attention_score: float = Field(
        ..., ge=0, le=1, description="Attention weight (0-1)"
    )
    importance_rank: int = Field(
        ..., description="Rank by importance (1 = most important)"
    )


class AttentionWeightsRequest(BaseModel):
    """Request to get attention weights for a prediction."""

    user_id: str = Field(..., description="User ID")
    metric: PredictionMetric = Field(..., description="Health metric")
    target_date: date = Field(..., description="Date of prediction")


class AttentionWeightsResponse(BaseModel):
    """Response with attention weights showing which days matter most."""

    user_id: str
    metric: PredictionMetric
    target_date: date
    predicted_value: float

    # Attention weights for each day in sequence
    attention_weights: List[AttentionWeight] = Field(
        ..., description="Attention weights for each day (sorted by date)"
    )

    # Summary
    most_important_days: List[date] = Field(
        ..., description="Top 3 most important days"
    )
    summary: str = Field(
        ..., description="Natural language explanation of which days mattered"
    )


# ============================================================================
# What-If Scenarios (Counterfactuals)
# ============================================================================


class WhatIfChange(BaseModel):
    """A hypothetical change to test."""

    feature_name: str = Field(
        ..., description="Feature to modify (e.g., 'nutrition_protein_daily')"
    )
    current_value: float = Field(..., description="Current value")
    new_value: float = Field(..., description="New hypothetical value")
    change_description: str = Field(
        ..., description="Human-readable change (e.g., '+50g protein')"
    )


class WhatIfScenario(BaseModel):
    """A what-if scenario with multiple feature changes."""

    scenario_name: str = Field(
        ..., description="Scenario name (e.g., 'High Protein Day')"
    )
    changes: List[WhatIfChange] = Field(
        ..., min_length=1, max_length=10, description="Features to change (1-10)"
    )


class WhatIfRequest(BaseModel):
    """Request to test what-if scenarios."""

    user_id: str = Field(..., description="User ID")
    metric: PredictionMetric = Field(..., description="Health metric")
    target_date: date = Field(..., description="Date to predict for")

    scenarios: List[WhatIfScenario] = Field(
        ..., min_length=1, max_length=5, description="Scenarios to test (1-5)"
    )

    # Include baseline prediction
    include_baseline: bool = Field(
        default=True, description="Include current prediction as baseline"
    )


class WhatIfResult(BaseModel):
    """Result of a single what-if scenario."""

    scenario_name: str
    predicted_value: float = Field(..., description="Predicted value with changes")

    change_from_baseline: float = Field(
        ..., description="Difference from baseline prediction"
    )
    percent_change: float = Field(..., description="Percentage change from baseline")

    confidence_score: float = Field(
        ..., ge=0, le=1, description="Confidence in this prediction"
    )

    # Which changes had the biggest impact
    biggest_drivers: List[str] = Field(
        default=[], description="Features that drove the change most"
    )


class WhatIfResponse(BaseModel):
    """Response with what-if scenario results."""

    user_id: str
    metric: PredictionMetric
    target_date: date

    # Baseline prediction
    baseline_prediction: float = Field(
        ..., description="Current prediction (no changes)"
    )
    baseline_confidence: float = Field(..., description="Baseline confidence")

    # Scenario results
    scenarios: List[WhatIfResult] = Field(..., description="Results for each scenario")

    # Best and worst scenarios
    best_scenario: str = Field(..., description="Scenario with best outcome")
    best_value: float = Field(..., description="Best predicted value")

    worst_scenario: str = Field(..., description="Scenario with worst outcome")
    worst_value: float = Field(..., description="Worst predicted value")

    # Summary
    summary: str = Field(..., description="Natural language summary of scenarios")
    recommendation: str = Field(..., description="Which scenario to pursue")


# ============================================================================
# Counterfactual Explanations
# ============================================================================


class CounterfactualTarget(str, Enum):
    """Target for counterfactual generation."""

    IMPROVE = "improve"  # Find changes that improve the prediction
    TARGET_VALUE = "target_value"  # Find changes to reach a specific value
    MINIMIZE_CHANGE = "minimize_change"  # Find minimal changes for a target


class CounterfactualRequest(BaseModel):
    """Request to generate counterfactual explanations."""

    user_id: str = Field(..., description="User ID")
    metric: PredictionMetric = Field(..., description="Health metric")
    target_date: date = Field(..., description="Date to predict for")

    target_type: CounterfactualTarget = Field(
        default=CounterfactualTarget.IMPROVE,
        description="What kind of counterfactual to generate",
    )

    target_value: Optional[float] = Field(
        None, description="Target value (required if target_type='target_value')"
    )

    max_changes: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of features to change (1-10)",
    )

    allowed_features: Optional[List[str]] = Field(
        None, description="Restrict changes to these features (None = all features)"
    )


class CounterfactualChange(BaseModel):
    """A suggested change to achieve the target."""

    feature_name: str
    current_value: float
    suggested_value: float
    change_amount: float
    change_description: str = Field(
        ..., description="Human-readable (e.g., '+20g protein')"
    )


class CounterfactualExplanation(BaseModel):
    """A counterfactual explanation."""

    current_prediction: float
    target_prediction: float
    achieved_prediction: float

    changes: List[CounterfactualChange] = Field(..., description="Suggested changes")

    plausibility_score: float = Field(
        ..., ge=0, le=1, description="How realistic these changes are (0-1)"
    )

    summary: str = Field(
        ..., description="Natural language explanation of changes needed"
    )


class CounterfactualResponse(BaseModel):
    """Response with counterfactual explanations."""

    user_id: str
    metric: PredictionMetric
    target_date: date

    current_prediction: float
    target_prediction: float

    # Best counterfactual found
    counterfactual: CounterfactualExplanation

    # Alternative counterfactuals
    alternatives: List[CounterfactualExplanation] = Field(
        default=[], description="Alternative ways to achieve similar results"
    )


# ============================================================================
# Global Feature Importance (Across All Predictions)
# ============================================================================


class GlobalFeatureImportance(BaseModel):
    """Global feature importance across all training data."""

    feature_name: str
    feature_category: str

    mean_importance: float = Field(
        ..., description="Average importance across all samples"
    )
    std_importance: float = Field(..., description="Standard deviation of importance")

    rank: int = Field(..., description="Global rank (1 = most important)")

    impact_direction: str = Field(..., description="positive/negative/mixed")


class GlobalImportanceRequest(BaseModel):
    """Request for global feature importance."""

    model_id: str = Field(..., description="Model to analyze")

    method: ImportanceMethod = Field(
        default=ImportanceMethod.SHAP, description="Method to use"
    )

    top_k: int = Field(
        default=20, ge=1, le=51, description="Number of top features (1-51)"
    )


class GlobalImportanceResponse(BaseModel):
    """Response with global feature importance."""

    model_id: str
    metric: PredictionMetric
    method: ImportanceMethod

    feature_importances: List[GlobalFeatureImportance] = Field(
        ..., description="Ranked global feature importances"
    )

    summary: str = Field(..., description="Natural language summary of key drivers")

    # Category summaries
    nutrition_importance: float = Field(
        ..., description="Overall nutrition importance (0-1)"
    )
    activity_importance: float = Field(
        ..., description="Overall activity importance (0-1)"
    )
    health_importance: float = Field(..., description="Overall health importance (0-1)")
