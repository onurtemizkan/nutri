"""SQLAlchemy models for database tables"""
from app.models.user import User
from app.models.meal import Meal
from app.models.health_metric import HealthMetric
from app.models.activity import Activity
from app.models.food_feedback import FoodFeedback, LearnedPrompt
from app.models.sensitivity import (
    # Enums
    AllergenType,
    SensitivityType,
    SensitivitySeverity,
    CompoundLevel,
    FodmapLevel,
    DerivationType,
    ReactionSeverity,
    IngredientCategory,
    # Models
    Ingredient,
    IngredientAllergen,
    MealIngredient,
    UserSensitivity,
    SensitivityExposure,
    SensitivityInsight,
)

__all__ = [
    # Core models
    "User",
    "Meal",
    "HealthMetric",
    "Activity",
    # Food feedback models
    "FoodFeedback",
    "LearnedPrompt",
    # Sensitivity enums
    "AllergenType",
    "SensitivityType",
    "SensitivitySeverity",
    "CompoundLevel",
    "FodmapLevel",
    "DerivationType",
    "ReactionSeverity",
    "IngredientCategory",
    # Sensitivity models
    "Ingredient",
    "IngredientAllergen",
    "MealIngredient",
    "UserSensitivity",
    "SensitivityExposure",
    "SensitivityInsight",
]
