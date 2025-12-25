"""
Sensitivity-related models and enums for food sensitivity tracking.

This module contains:
- Enums for allergen types, sensitivity types, severity levels
- SQLAlchemy models for tracking ingredients, allergens, and user sensitivities
"""

from enum import Enum
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    ForeignKey,
    Boolean,
    Text,
    JSON,
    func,
    Index,
)
from sqlalchemy.orm import relationship
from app.database import Base


# =============================================================================
# ENUMS
# =============================================================================


class AllergenType(str, Enum):
    """FDA Big 9 + EU allergens"""

    # FDA Big 9 (2023)
    MILK = "milk"
    EGGS = "eggs"
    WHEAT = "wheat"
    SOY = "soy"
    PEANUTS = "peanuts"
    TREE_NUTS = "tree_nuts"
    FISH = "fish"
    SHELLFISH_CRUSTACEAN = "shellfish_crustacean"
    SHELLFISH_MOLLUSCAN = "shellfish_molluscan"
    SESAME = "sesame"

    # EU Big 14 additions
    GLUTEN_CEREALS = "gluten_cereals"  # Includes wheat, barley, rye, oats
    CELERY = "celery"
    MUSTARD = "mustard"
    LUPIN = "lupin"
    SULFITES = "sulfites"


class SensitivityType(str, Enum):
    """Types of food sensitivities"""

    ALLERGY = "allergy"  # IgE-mediated, immediate reactions
    INTOLERANCE = "intolerance"  # Non-immune, delayed reactions
    FODMAP = "fodmap"  # Fermentable carbohydrate sensitivity
    HISTAMINE = "histamine"  # Histamine intolerance
    TYRAMINE = "tyramine"  # Tyramine sensitivity


class SensitivitySeverity(str, Enum):
    """Severity level of a sensitivity"""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ReactionSeverity(str, Enum):
    """Severity of a reaction event"""

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EMERGENCY = "emergency"  # Anaphylactic or life-threatening


class CompoundLevel(str, Enum):
    """Level of compound content (histamine, tyramine, oxalate, etc.)"""

    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class FodmapLevel(str, Enum):
    """FODMAP content level based on Monash University guidelines"""

    LOW = "low"  # Safe for most IBS patients
    MEDIUM = "medium"  # May trigger symptoms in some
    HIGH = "high"  # Likely to trigger symptoms


class DerivationType(str, Enum):
    """How an allergen relates to an ingredient"""

    DIRECTLY_CONTAINS = "directly_contains"  # Primary ingredient IS the allergen
    DERIVED_FROM = "derived_from"  # Made from allergen (e.g., lecithin from soy)
    MAY_CONTAIN = "may_contain"  # Cross-contamination risk
    LIKELY_CONTAINS = "likely_contains"  # High probability of containing
    PROCESSED_WITH = "processed_with"  # Processed on shared equipment
    FREE_FROM = "free_from"  # Certified free of allergen


class IngredientCategory(str, Enum):
    """Food ingredient categories"""

    DAIRY = "dairy"
    EGGS = "eggs"
    GRAINS = "grains"
    LEGUMES = "legumes"
    NUTS_SEEDS = "nuts_seeds"
    SEAFOOD = "seafood"
    MEAT = "meat"
    VEGETABLES = "vegetables"
    FRUITS = "fruits"
    FERMENTED = "fermented"
    BEVERAGES = "beverages"
    ADDITIVES = "additives"
    OTHER = "other"


# =============================================================================
# SQLALCHEMY MODELS
# =============================================================================


class Ingredient(Base):
    """Master ingredient database with sensitivity data"""

    __tablename__ = "Ingredient"
    __table_args__ = (
        Index("Ingredient_name_idx", "name"),
        Index("Ingredient_category_idx", "category"),
    )

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    display_name = Column("displayName", String, nullable=False)
    category = Column(String, nullable=False)  # IngredientCategory

    # FODMAP data
    fodmap_level = Column("fodmapLevel", String, nullable=True)  # FodmapLevel
    fodmap_types = Column("fodmapTypes", JSON, nullable=True)  # List of FODMAP types

    # Biogenic amines (mg per 100g)
    histamine_mg = Column("histamineMg", Float, nullable=True)
    histamine_level = Column("histamineLevel", String, nullable=True)  # CompoundLevel
    tyramine_mg = Column("tyramineMg", Float, nullable=True)
    tyramine_level = Column("tyramineLevel", String, nullable=True)  # CompoundLevel

    # Other compounds
    oxalate_mg = Column("oxalateMg", Float, nullable=True)
    oxalate_level = Column("oxalateLevel", String, nullable=True)  # CompoundLevel
    salicylate_mg = Column("salicylateMg", Float, nullable=True)
    salicylate_level = Column("salicylateLevel", String, nullable=True)  # CompoundLevel
    lectin_level = Column("lectinLevel", String, nullable=True)  # CompoundLevel

    # Flags
    is_nightshade = Column("isNightshade", Boolean, default=False)
    is_fermented = Column("isFermented", Boolean, default=False)
    is_aged = Column("isAged", Boolean, default=False)
    is_histamine_liberator = Column("isHistamineLiberator", Boolean, default=False)

    # Metadata
    sources = Column(JSON, nullable=True)  # Data sources
    created_at = Column("createdAt", DateTime, server_default=func.now())
    updated_at = Column(
        "updatedAt", DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    allergens = relationship(
        "IngredientAllergen",
        back_populates="ingredient",
        lazy="select",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Ingredient(id={self.id}, name={self.name})>"


class IngredientAllergen(Base):
    """Mapping between ingredients and allergens"""

    __tablename__ = "IngredientAllergen"
    __table_args__ = (
        Index("IngredientAllergen_ingredientId_idx", "ingredientId"),
        Index("IngredientAllergen_allergenType_idx", "allergenType"),
    )

    id = Column(String, primary_key=True)
    ingredient_id = Column(
        "ingredientId",
        String,
        ForeignKey("Ingredient.id", ondelete="CASCADE"),
        nullable=False,
    )
    allergen_type = Column("allergenType", String, nullable=False)  # AllergenType
    confidence = Column(Float, default=1.0)  # 0.0-1.0 confidence score
    derivation = Column(String, nullable=False)  # DerivationType

    # Relationship
    ingredient = relationship("Ingredient", back_populates="allergens")

    def __repr__(self):
        return f"<IngredientAllergen(ingredient={self.ingredient_id}, allergen={self.allergen_type})>"


class MealIngredient(Base):
    """Ingredients detected in a meal"""

    __tablename__ = "MealIngredient"
    __table_args__ = (
        Index("MealIngredient_mealId_idx", "mealId"),
        Index("MealIngredient_ingredientId_idx", "ingredientId"),
    )

    id = Column(String, primary_key=True)
    meal_id = Column(
        "mealId", String, ForeignKey("Meal.id", ondelete="CASCADE"), nullable=False
    )
    ingredient_id = Column(
        "ingredientId",
        String,
        ForeignKey("Ingredient.id", ondelete="SET NULL"),
        nullable=True,
    )
    ingredient_name = Column(
        "ingredientName", String, nullable=False
    )  # Store name even if no DB match

    # Quantity estimation
    estimated_grams = Column("estimatedGrams", Float, nullable=True)
    confidence = Column(Float, default=0.5)  # Detection confidence

    # Sensitivity data at time of meal (denormalized for historical tracking)
    detected_allergens = Column("detectedAllergens", JSON, nullable=True)  # List
    fodmap_level = Column("fodmapLevel", String, nullable=True)
    histamine_level = Column("histamineLevel", String, nullable=True)
    tyramine_level = Column("tyramineLevel", String, nullable=True)

    created_at = Column("createdAt", DateTime, server_default=func.now())

    def __repr__(self):
        return (
            f"<MealIngredient(meal={self.meal_id}, ingredient={self.ingredient_name})>"
        )


class UserSensitivity(Base):
    """User's known sensitivities/allergies"""

    __tablename__ = "UserSensitivity"
    __table_args__ = (
        Index("UserSensitivity_userId_idx", "userId"),
        Index("UserSensitivity_allergenType_idx", "allergenType"),
    )

    id = Column(String, primary_key=True)
    user_id = Column(
        "userId", String, ForeignKey("User.id", ondelete="CASCADE"), nullable=False
    )

    # What they're sensitive to
    allergen_type = Column(
        "allergenType", String, nullable=True
    )  # AllergenType (if allergen)
    sensitivity_type = Column(
        "sensitivityType", String, nullable=False
    )  # SensitivityType
    severity = Column(String, nullable=False)  # SensitivitySeverity

    # For non-allergen sensitivities (FODMAP, histamine, etc.)
    compound_type = Column("compoundType", String, nullable=True)  # What compound
    threshold_level = Column(
        "thresholdLevel", String, nullable=True
    )  # CompoundLevel they can tolerate

    # Medical info
    diagnosed = Column(Boolean, default=False)  # Medically diagnosed
    diagnosis_date = Column("diagnosisDate", DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    # Auto-detected vs user-reported
    source = Column(
        String, default="user_reported"
    )  # user_reported, ml_detected, medical
    confidence = Column(Float, default=1.0)  # For ML-detected sensitivities

    created_at = Column("createdAt", DateTime, server_default=func.now())
    updated_at = Column(
        "updatedAt", DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationship
    user = relationship("User", back_populates="sensitivities")

    def __repr__(self):
        return f"<UserSensitivity(user={self.user_id}, type={self.sensitivity_type})>"


class SensitivityExposure(Base):
    """Tracks exposure events when user eats something they're sensitive to"""

    __tablename__ = "SensitivityExposure"
    __table_args__ = (
        Index("SensitivityExposure_userId_idx", "userId"),
        Index("SensitivityExposure_mealId_idx", "mealId"),
        Index("SensitivityExposure_timestamp_idx", "timestamp"),
    )

    id = Column(String, primary_key=True)
    user_id = Column(
        "userId", String, ForeignKey("User.id", ondelete="CASCADE"), nullable=False
    )
    meal_id = Column(
        "mealId", String, ForeignKey("Meal.id", ondelete="SET NULL"), nullable=True
    )
    sensitivity_id = Column(
        "sensitivityId",
        String,
        ForeignKey("UserSensitivity.id", ondelete="SET NULL"),
        nullable=True,
    )

    # What triggered it
    trigger_ingredient = Column("triggerIngredient", String, nullable=True)
    trigger_allergen = Column("triggerAllergen", String, nullable=True)  # AllergenType
    trigger_compound = Column(
        "triggerCompound", String, nullable=True
    )  # histamine, fodmap, etc.

    # Exposure details
    estimated_amount = Column("estimatedAmount", Float, nullable=True)  # grams
    timestamp = Column(DateTime, server_default=func.now())

    # Reaction tracking
    reaction_severity = Column(
        "reactionSeverity", String, nullable=True
    )  # ReactionSeverity
    reaction_onset_minutes = Column("reactionOnsetMinutes", Integer, nullable=True)
    reaction_duration_hours = Column("reactionDurationHours", Float, nullable=True)
    symptoms = Column(JSON, nullable=True)  # List of symptom strings

    # HRV data at time of exposure (for ML training)
    pre_exposure_hrv = Column("preExposureHrv", JSON, nullable=True)
    post_exposure_hrv = Column("postExposureHrv", JSON, nullable=True)

    # User feedback
    user_confirmed = Column("userConfirmed", Boolean, nullable=True)
    user_notes = Column("userNotes", Text, nullable=True)

    created_at = Column("createdAt", DateTime, server_default=func.now())

    # Relationship
    user = relationship("User", back_populates="sensitivity_exposures")

    def __repr__(self):
        return f"<SensitivityExposure(user={self.user_id}, trigger={self.trigger_ingredient})>"


class SensitivityInsight(Base):
    """ML-generated insights about user's sensitivities"""

    __tablename__ = "SensitivityInsight"
    __table_args__ = (
        Index("SensitivityInsight_userId_idx", "userId"),
        Index("SensitivityInsight_insightType_idx", "insightType"),
    )

    id = Column(String, primary_key=True)
    user_id = Column(
        "userId", String, ForeignKey("User.id", ondelete="CASCADE"), nullable=False
    )

    # Insight details
    insight_type = Column(
        "insightType", String, nullable=False
    )  # pattern_detected, threshold_identified, cross_reactivity, etc.
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)

    # What sensitivity it relates to
    related_sensitivity_type = Column("relatedSensitivityType", String, nullable=True)
    related_allergen = Column("relatedAllergen", String, nullable=True)
    related_compound = Column("relatedCompound", String, nullable=True)

    # Evidence and confidence
    confidence = Column(Float, nullable=False)  # 0.0-1.0
    evidence_count = Column(
        "evidenceCount", Integer, default=0
    )  # Number of data points
    evidence_summary = Column("evidenceSummary", JSON, nullable=True)

    # Recommendations
    recommendations = Column(JSON, nullable=True)  # List of recommendation strings
    foods_to_avoid = Column("foodsToAvoid", JSON, nullable=True)  # List of foods
    safe_alternatives = Column("safeAlternatives", JSON, nullable=True)  # List of foods

    # Status
    is_active = Column("isActive", Boolean, default=True)
    dismissed_at = Column("dismissedAt", DateTime, nullable=True)

    created_at = Column("createdAt", DateTime, server_default=func.now())
    updated_at = Column(
        "updatedAt", DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationship
    user = relationship("User", back_populates="sensitivity_insights")

    def __repr__(self):
        return f"<SensitivityInsight(user={self.user_id}, type={self.insight_type})>"
