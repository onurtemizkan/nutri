"""
Pydantic schemas for correlation analysis.
"""

from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class CorrelationMethod(str, Enum):
    """Statistical correlation methods."""

    PEARSON = "pearson"  # Linear correlation
    SPEARMAN = "spearman"  # Rank-based correlation
    KENDALL = "kendall"  # Rank-based correlation (tau)
    GRANGER = "granger"  # Granger causality test


class HealthMetricTarget(str, Enum):
    """Target health metrics for correlation analysis."""

    RHR = "RESTING_HEART_RATE"
    HRV_SDNN = "HEART_RATE_VARIABILITY_SDNN"
    HRV_RMSSD = "HEART_RATE_VARIABILITY_RMSSD"
    SLEEP_DURATION = "SLEEP_DURATION"
    SLEEP_QUALITY = "SLEEP_QUALITY_SCORE"
    RECOVERY_SCORE = "RECOVERY_SCORE"
    VO2_MAX = "VO2_MAX"
    RESPIRATORY_RATE = "RESPIRATORY_RATE_RESTING"


# ============================================================================
# Correlation Request/Response Models
# ============================================================================

class CorrelationRequest(BaseModel):
    """Request to analyze correlations between features and health metrics."""

    user_id: str = Field(..., description="User ID")
    target_metric: HealthMetricTarget = Field(..., description="Health metric to correlate with")
    methods: List[CorrelationMethod] = Field(
        default=[CorrelationMethod.PEARSON, CorrelationMethod.SPEARMAN],
        description="Correlation methods to use"
    )
    lookback_days: int = Field(
        default=30,
        ge=14,
        le=180,
        description="Days of historical data to analyze (14-180)"
    )
    significance_threshold: float = Field(
        default=0.05,
        ge=0.001,
        le=0.1,
        description="P-value threshold for statistical significance"
    )
    min_correlation: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum absolute correlation to report"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Return top K correlations"
    )


class CorrelationResult(BaseModel):
    """A single correlation result."""

    feature_name: str = Field(..., description="Name of the feature")
    feature_category: str = Field(..., description="Category (nutrition, activity, health, etc.)")
    correlation: float = Field(..., ge=-1, le=1, description="Correlation coefficient (-1 to 1)")
    p_value: float = Field(..., description="P-value for statistical significance")
    sample_size: int = Field(..., description="Number of data points used")
    method: CorrelationMethod = Field(..., description="Correlation method used")

    # Interpretation helpers
    is_significant: bool = Field(..., description="True if p_value < significance_threshold")
    strength: str = Field(..., description="weak/moderate/strong")
    direction: str = Field(..., description="positive/negative/none")

    # Effect size
    explained_variance: Optional[float] = Field(
        None, description="R² value (for Pearson)"
    )


class CorrelationResponse(BaseModel):
    """Response containing correlation analysis results."""

    user_id: str
    target_metric: HealthMetricTarget
    analyzed_at: datetime
    lookback_days: int

    correlations: List[CorrelationResult] = Field(
        ..., description="Correlation results sorted by absolute correlation"
    )

    # Summary statistics
    total_features_analyzed: int
    significant_correlations: int
    strongest_positive: Optional[CorrelationResult] = None
    strongest_negative: Optional[CorrelationResult] = None

    # Data quality
    data_quality_score: float = Field(..., ge=0, le=1)
    missing_days: int = Field(..., description="Days with missing data")
    warning: Optional[str] = Field(None, description="Warning if data quality is low")


# ============================================================================
# Lag Analysis Models
# ============================================================================

class LagAnalysisRequest(BaseModel):
    """Request to analyze lagged correlations (time-delayed effects)."""

    user_id: str = Field(..., description="User ID")
    target_metric: HealthMetricTarget = Field(..., description="Health metric to correlate with")
    feature_name: str = Field(..., description="Specific feature to analyze")

    max_lag_hours: int = Field(
        default=72,
        ge=6,
        le=168,
        description="Maximum lag to test in hours (6-168)"
    )
    lag_step_hours: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Step size for lag analysis in hours"
    )

    lookback_days: int = Field(
        default=30,
        ge=14,
        le=180,
        description="Days of historical data to analyze"
    )
    method: CorrelationMethod = Field(
        default=CorrelationMethod.PEARSON,
        description="Correlation method to use"
    )


class LagAnalysisResult(BaseModel):
    """Correlation result at a specific lag."""

    lag_hours: int = Field(..., description="Lag in hours")
    correlation: float = Field(..., ge=-1, le=1, description="Correlation coefficient")
    p_value: float = Field(..., description="P-value")
    is_significant: bool = Field(..., description="True if p_value < 0.05")


class LagAnalysisResponse(BaseModel):
    """Response containing lag analysis results."""

    user_id: str
    target_metric: HealthMetricTarget
    feature_name: str
    analyzed_at: datetime

    lag_results: List[LagAnalysisResult] = Field(
        ..., description="Correlation at each lag, sorted by lag_hours"
    )

    # Summary
    optimal_lag_hours: Optional[int] = Field(
        None, description="Lag with strongest correlation"
    )
    optimal_correlation: Optional[float] = Field(
        None, description="Correlation at optimal lag"
    )

    # Interpretation
    immediate_effect: bool = Field(
        ..., description="True if strongest correlation at lag=0"
    )
    delayed_effect: bool = Field(
        ..., description="True if strongest correlation at lag>0"
    )
    effect_duration_hours: Optional[int] = Field(
        None, description="Duration of significant correlation"
    )

    interpretation: str = Field(
        ..., description="Natural language interpretation of lag analysis"
    )


# ============================================================================
# Helper Functions
# ============================================================================

def interpret_correlation_strength(correlation: float) -> str:
    """Interpret correlation strength."""
    abs_corr = abs(correlation)
    if abs_corr < 0.3:
        return "weak"
    elif abs_corr < 0.7:
        return "moderate"
    else:
        return "strong"


def interpret_correlation_direction(correlation: float) -> str:
    """Interpret correlation direction."""
    if abs(correlation) < 0.1:
        return "none"
    elif correlation > 0:
        return "positive"
    else:
        return "negative"
