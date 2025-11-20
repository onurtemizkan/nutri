"""
Pydantic schemas for feature engineering.
"""

from datetime import date, datetime
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator


class FeatureCategory(str, Enum):
    """Categories of engineered features."""

    NUTRITION = "nutrition"
    ACTIVITY = "activity"
    HEALTH = "health"
    TEMPORAL = "temporal"
    INTERACTION = "interaction"
    ALL = "all"


# ============================================================================
# Feature Data Models
# ============================================================================

class NutritionFeatures(BaseModel):
    """Nutrition-related features extracted from meal data."""

    # Daily totals
    calories_daily: float = Field(..., description="Total calories consumed today")
    protein_daily: float = Field(..., description="Total protein (g) consumed today")
    carbs_daily: float = Field(..., description="Total carbs (g) consumed today")
    fat_daily: float = Field(..., description="Total fat (g) consumed today")
    fiber_daily: float = Field(..., description="Total fiber (g) consumed today")

    # Rolling averages (7 days)
    calories_7d_avg: float = Field(..., description="7-day average calories")
    protein_7d_avg: float = Field(..., description="7-day average protein")
    carbs_7d_avg: float = Field(..., description="7-day average carbs")
    fat_7d_avg: float = Field(..., description="7-day average fat")

    # Macro ratios
    protein_ratio: float = Field(..., ge=0, le=1, description="Protein calories / total calories")
    carbs_ratio: float = Field(..., ge=0, le=1, description="Carbs calories / total calories")
    fat_ratio: float = Field(..., ge=0, le=1, description="Fat calories / total calories")

    # Meal timing
    first_meal_time: Optional[float] = Field(None, description="Time of first meal (hour of day, 0-24)")
    last_meal_time: Optional[float] = Field(None, description="Time of last meal (hour of day, 0-24)")
    eating_window: Optional[float] = Field(None, description="Hours between first and last meal")
    meal_count: int = Field(..., description="Number of meals consumed today")

    # Late night eating
    late_night_carbs: float = Field(..., description="Carbs consumed after 8pm (g)")
    late_night_calories: float = Field(..., description="Calories consumed after 8pm")

    # Regularity
    meal_regularity: float = Field(..., ge=0, le=1, description="Consistency of meal timing (0-1)")

    # Calorie balance (requires user profile with TDEE)
    calorie_deficit: Optional[float] = Field(None, description="Calories below TDEE (negative = surplus)")
    calorie_deficit_7d: Optional[float] = Field(None, description="7-day average calorie deficit")


class ActivityFeatures(BaseModel):
    """Activity-related features extracted from workout and step data."""

    # Daily activity
    steps_daily: int = Field(..., description="Steps taken today")
    active_minutes_daily: float = Field(..., description="Active minutes today")
    calories_burned_daily: float = Field(..., description="Calories burned from activity")

    # Rolling averages (7 days)
    steps_7d_avg: float = Field(..., description="7-day average steps")
    active_minutes_7d_avg: float = Field(..., description="7-day average active minutes")
    calories_burned_7d_avg: float = Field(..., description="7-day average calories burned")

    # Workout intensity
    workout_count_daily: int = Field(..., description="Number of workouts today")
    workout_intensity_avg: Optional[float] = Field(
        None, ge=0, le=1, description="Average workout intensity (0-1)"
    )
    high_intensity_minutes: float = Field(..., description="High-intensity workout minutes today")

    # Recovery
    recovery_time: float = Field(..., description="Hours since last high-intensity workout")
    days_since_rest: int = Field(..., description="Days since last rest day (no workouts)")

    # Activity type distribution (7 days)
    cardio_minutes_7d: float = Field(..., description="Cardio minutes in last 7 days")
    strength_minutes_7d: float = Field(..., description="Strength training minutes in last 7 days")
    flexibility_minutes_7d: float = Field(..., description="Flexibility/yoga minutes in last 7 days")


class HealthFeatures(BaseModel):
    """Health metric features (lagged values and trends)."""

    # RHR (Resting Heart Rate)
    rhr_yesterday: Optional[float] = Field(None, description="RHR from yesterday (bpm)")
    rhr_7d_avg: Optional[float] = Field(None, description="7-day average RHR")
    rhr_7d_std: Optional[float] = Field(None, description="7-day RHR standard deviation")
    rhr_trend: Optional[float] = Field(None, description="RHR trend: slope of last 7 days")
    rhr_baseline: Optional[float] = Field(None, description="User's 30-day RHR baseline")
    rhr_deviation: Optional[float] = Field(None, description="Current RHR deviation from baseline")

    # HRV (Heart Rate Variability)
    hrv_yesterday: Optional[float] = Field(None, description="HRV from yesterday (ms)")
    hrv_7d_avg: Optional[float] = Field(None, description="7-day average HRV")
    hrv_7d_std: Optional[float] = Field(None, description="7-day HRV standard deviation")
    hrv_trend: Optional[float] = Field(None, description="HRV trend: slope of last 7 days")
    hrv_baseline: Optional[float] = Field(None, description="User's 30-day HRV baseline")
    hrv_deviation: Optional[float] = Field(None, description="Current HRV deviation from baseline")

    # Sleep
    sleep_duration_last: Optional[float] = Field(None, description="Sleep duration last night (hours)")
    sleep_quality_last: Optional[float] = Field(
        None, ge=0, le=100, description="Sleep quality last night (0-100)"
    )
    sleep_duration_7d_avg: Optional[float] = Field(None, description="7-day average sleep duration")

    # Recovery score (if available from wearable)
    recovery_score_yesterday: Optional[float] = Field(
        None, ge=0, le=100, description="Recovery score from yesterday (0-100)"
    )
    recovery_score_7d_avg: Optional[float] = Field(None, description="7-day average recovery score")


class TemporalFeatures(BaseModel):
    """Time-based features for temporal patterns."""

    # Basic temporal
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    is_weekend: bool = Field(..., description="True if Saturday or Sunday")
    week_of_year: int = Field(..., ge=1, le=53, description="Week number (1-53)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")

    # Physiological cycles (if available)
    menstrual_cycle_day: Optional[int] = Field(None, description="Day of menstrual cycle (if tracked)")
    menstrual_cycle_phase: Optional[str] = Field(None, description="Cycle phase: follicular/ovulation/luteal")


class InteractionFeatures(BaseModel):
    """Interaction features combining multiple categories."""

    # Nutrition per body weight (requires user profile)
    protein_per_kg: Optional[float] = Field(None, description="Protein (g) per kg body weight")
    calories_per_kg: Optional[float] = Field(None, description="Calories per kg body weight")

    # Nutrition per activity
    carbs_per_active_minute: Optional[float] = Field(None, description="Carbs per active minute")
    protein_per_workout: Optional[float] = Field(None, description="Protein per workout")

    # Recovery-adjusted nutrition
    protein_to_recovery: Optional[float] = Field(None, description="Protein intake / recovery time")
    carbs_to_intensity: Optional[float] = Field(None, description="Carbs / workout intensity")


# ============================================================================
# API Request/Response Models
# ============================================================================

class EngineerFeaturesRequest(BaseModel):
    """Request to engineer features for a specific user and date."""

    user_id: str = Field(..., description="User ID")
    target_date: date = Field(..., description="Date to engineer features for")
    categories: list[FeatureCategory] = Field(
        default=[FeatureCategory.ALL],
        description="Categories of features to engineer"
    )
    lookback_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Days of historical data to use (7-90)"
    )
    force_recompute: bool = Field(
        default=False,
        description="Force recomputation even if cached"
    )


class EngineerFeaturesResponse(BaseModel):
    """Response containing engineered features."""

    user_id: str
    target_date: date
    computed_at: datetime
    cached: bool = Field(..., description="True if returned from cache")

    nutrition: Optional[NutritionFeatures] = None
    activity: Optional[ActivityFeatures] = None
    health: Optional[HealthFeatures] = None
    temporal: Optional[TemporalFeatures] = None
    interaction: Optional[InteractionFeatures] = None

    feature_count: int = Field(..., description="Total number of features computed")
    missing_features: int = Field(..., description="Number of features with missing data")
    data_quality_score: float = Field(
        ..., ge=0, le=1, description="Overall data quality (0-1)"
    )


class FeaturesResponse(BaseModel):
    """Response for retrieving cached features."""

    user_id: str
    target_date: date
    features: Dict[str, float] = Field(..., description="Flat dictionary of all features")
    feature_version: str = Field(..., description="Feature engineering version")
    cached_at: datetime
    expires_at: datetime


# ============================================================================
# Validators
# ============================================================================

@field_validator("target_date")
def validate_target_date(cls, v):
    """Ensure target_date is not in the future."""
    if v > date.today():
        raise ValueError("target_date cannot be in the future")
    return v
