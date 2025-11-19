"""
Pydantic schemas for ML service API request/response validation.
"""

from .features import (
    FeatureCategory,
    NutritionFeatures,
    ActivityFeatures,
    HealthFeatures,
    TemporalFeatures,
    EngineerFeaturesRequest,
    EngineerFeaturesResponse,
    FeaturesResponse,
)
from .correlations import (
    CorrelationMethod,
    CorrelationRequest,
    CorrelationResult,
    CorrelationResponse,
    LagAnalysisRequest,
    LagAnalysisResult,
    LagAnalysisResponse,
)

__all__ = [
    # Features
    "FeatureCategory",
    "NutritionFeatures",
    "ActivityFeatures",
    "HealthFeatures",
    "TemporalFeatures",
    "EngineerFeaturesRequest",
    "EngineerFeaturesResponse",
    "FeaturesResponse",
    # Correlations
    "CorrelationMethod",
    "CorrelationRequest",
    "CorrelationResult",
    "CorrelationResponse",
    "LagAnalysisRequest",
    "LagAnalysisResult",
    "LagAnalysisResponse",
]
