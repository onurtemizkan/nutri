"""
Pydantic schemas for food analysis endpoints.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator


class DimensionsInput(BaseModel):
    """AR measurement dimensions in centimeters"""

    width: float = Field(..., gt=0, description="Width in cm")
    height: float = Field(..., gt=0, description="Height in cm")
    depth: float = Field(..., gt=0, description="Depth in cm")

    @validator("*")
    def validate_dimensions(cls, v):
        """Ensure dimensions are reasonable (< 100cm for food items)"""
        if v > 100:
            raise ValueError("Dimension too large for typical food item")
        return v


class NutritionInfo(BaseModel):
    """Nutrition information per serving"""

    # Macronutrients (required)
    calories: float = Field(..., ge=0)
    protein: float = Field(..., ge=0)  # grams
    carbs: float = Field(..., ge=0)  # grams
    fat: float = Field(..., ge=0)  # grams

    # Extended macros (optional)
    fiber: Optional[float] = Field(None, ge=0)  # grams
    sugar: Optional[float] = Field(None, ge=0)  # grams

    # Fat breakdown (optional)
    saturated_fat: Optional[float] = Field(None, ge=0)  # grams
    trans_fat: Optional[float] = Field(None, ge=0)  # grams
    cholesterol: Optional[float] = Field(None, ge=0)  # mg

    # Minerals (optional) - all in mg
    sodium: Optional[float] = Field(None, ge=0)  # mg
    potassium: Optional[float] = Field(None, ge=0)  # mg
    calcium: Optional[float] = Field(None, ge=0)  # mg
    iron: Optional[float] = Field(None, ge=0)  # mg
    magnesium: Optional[float] = Field(None, ge=0)  # mg
    zinc: Optional[float] = Field(None, ge=0)  # mg
    phosphorus: Optional[float] = Field(None, ge=0)  # mg

    # Vitamins (optional) - various units
    vitamin_a: Optional[float] = Field(None, ge=0)  # mcg RAE
    vitamin_c: Optional[float] = Field(None, ge=0)  # mg
    vitamin_d: Optional[float] = Field(None, ge=0)  # mcg
    vitamin_e: Optional[float] = Field(None, ge=0)  # mg
    vitamin_k: Optional[float] = Field(None, ge=0)  # mcg
    vitamin_b6: Optional[float] = Field(None, ge=0)  # mg
    vitamin_b12: Optional[float] = Field(None, ge=0)  # mcg
    folate: Optional[float] = Field(None, ge=0)  # mcg DFE
    thiamin: Optional[float] = Field(None, ge=0)  # mg (B1)
    riboflavin: Optional[float] = Field(None, ge=0)  # mg (B2)
    niacin: Optional[float] = Field(None, ge=0)  # mg (B3)

    # Amino acids (optional) - mg
    lysine: Optional[float] = Field(None, ge=0)  # mg - essential amino acid
    arginine: Optional[float] = Field(
        None, ge=0
    )  # mg - conditionally essential amino acid


class FoodItemAlternative(BaseModel):
    """Alternative food classification"""

    name: str
    display_name: Optional[str] = None  # Human-readable name
    confidence: float = Field(..., ge=0, le=1)
    boosted: bool = Field(default=False, description="True if boosted by user feedback patterns")
    from_feedback: bool = Field(default=False, description="True if suggested from historical corrections")


class FoodItem(BaseModel):
    """Detected food item with nutrition information"""

    name: str
    display_name: Optional[str] = None  # Human-readable name
    confidence: float = Field(..., ge=0, le=1)
    portion_size: str  # e.g., "1 cup", "150g"
    portion_weight: float = Field(..., gt=0)  # grams
    nutrition: NutritionInfo
    category: Optional[str] = None  # e.g., "fruit", "vegetable", "protein"
    alternatives: Optional[List[FoodItemAlternative]] = None
    needs_confirmation: bool = Field(
        default=False,
        description="True if confidence is low and user should confirm/correct the classification"
    )
    confidence_threshold: float = Field(
        default=0.8,
        description="Threshold used to determine needs_confirmation"
    )


class FoodAnalysisResponse(BaseModel):
    """Response from food analysis"""

    food_items: List[FoodItem]
    measurement_quality: str = Field(..., pattern="^(high|medium|low)$")
    processing_time: float  # milliseconds
    suggestions: Optional[List[str]] = None
    error: Optional[str] = None
    image_hash: Optional[str] = Field(
        None, description="SHA-256 hash of the analyzed image (for feedback submission)"
    )


class ModelInfo(BaseModel):
    """Information about available ML models"""

    name: str
    version: str
    accuracy: Optional[float] = None
    num_classes: int
    description: str
    model_type: Optional[str] = None
    strategy: Optional[str] = None
    multi_food_detection: Optional[bool] = None
    detector: Optional[dict] = None


class ModelsInfoResponse(BaseModel):
    """Response for models info endpoint"""

    available_models: List[ModelInfo]
    active_model: str


class NutritionDBEntry(BaseModel):
    """Nutrition database entry"""

    food_name: str
    fdc_id: Optional[str] = None  # USDA FoodData Central ID
    category: str
    serving_size: str
    serving_weight: float  # grams
    nutrition: NutritionInfo
    common_portions: List[dict] = []  # List of {"description": str, "weight": float}


class NutritionDBSearchResponse(BaseModel):
    """Response for nutrition database search"""

    results: List[NutritionDBEntry]
    query: str
    total_results: int


class MicronutrientEstimationRequest(BaseModel):
    """Request to estimate missing micronutrients for barcode-scanned foods"""

    # Food identification
    food_name: str = Field(..., description="Product or food name")
    categories: Optional[List[str]] = Field(
        None, description="Open Food Facts category tags"
    )

    # Portion info
    portion_weight: float = Field(..., gt=0, description="Serving weight in grams")

    # Ingredients text (for sophisticated ingredient-based estimation)
    ingredients_text: Optional[str] = Field(
        None,
        description="Raw ingredients list from product label. "
        "Used for intelligent ingredient-based micronutrient estimation. "
        "Ingredients are typically listed in descending order by weight.",
    )

    # Macronutrient profile (helps inform estimates)
    protein: Optional[float] = Field(
        None, ge=0, description="Protein per 100g (helps infer B12, zinc for high-protein foods)"
    )
    fiber: Optional[float] = Field(
        None, ge=0, description="Fiber per 100g (helps infer magnesium, folate for high-fiber foods)"
    )
    fat: Optional[float] = Field(
        None, ge=0, description="Fat per 100g"
    )

    # Existing micronutrients (only estimate missing ones)
    existing: Optional[dict] = Field(
        None,
        description="Already known micronutrients (from Open Food Facts). "
        "Keys: potassium, calcium, iron, magnesium, zinc, phosphorus, "
        "vitamin_a, vitamin_c, vitamin_d, vitamin_e, vitamin_k, "
        "vitamin_b6, vitamin_b12, folate, thiamin, riboflavin, niacin",
    )


class MicronutrientEstimationResponse(BaseModel):
    """Response with estimated micronutrients"""

    # Estimated micronutrients (only the ones that were missing)
    estimated: dict = Field(..., description="Estimated micronutrient values")

    # Metadata
    category_used: str = Field(
        ..., description="Food category used for estimation (e.g., fruit, vegetable)"
    )
    confidence: str = Field(
        ...,
        pattern="^(high|medium|low)$",
        description="Estimation confidence based on category match",
    )
    source: str = Field(
        default="category_average",
        description="Source of estimates (category_average, food_database)",
    )


# ==============================================================================
# FEEDBACK SCHEMAS - For user corrections and model improvement
# ==============================================================================


class FoodFeedbackRequest(BaseModel):
    """Request to submit user feedback on a food classification"""

    image_hash: str = Field(
        ...,
        description="SHA-256 hash of the image (from analyze response)",
        min_length=16,
        max_length=64
    )
    original_prediction: str = Field(
        ...,
        description="What the model predicted (food name)",
        min_length=1
    )
    original_confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Model's confidence in original prediction"
    )
    user_selected_food: str = Field(
        ...,
        description="What the user selected/confirmed as correct",
        min_length=1
    )
    alternatives_shown: List[str] = Field(
        default=[],
        description="List of alternative foods that were shown to the user"
    )
    user_description: Optional[str] = Field(
        None,
        description="Optional user description of the food (helps improve prompts)",
        max_length=500
    )
    # Context fields
    user_id: Optional[str] = Field(
        None,
        description="Anonymous user ID for tracking improvement patterns"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for grouping related feedback"
    )
    device_type: Optional[str] = Field(
        None,
        description="Device type (ios, android, web)"
    )


class FoodFeedbackResponse(BaseModel):
    """Response after submitting feedback"""

    success: bool
    feedback_id: int = Field(..., description="ID of the recorded feedback")
    was_correction: bool = Field(
        ...,
        description="True if user selected a different food than predicted"
    )
    message: str
    suggested_prompts: List[str] = Field(
        default=[],
        description="New CLIP prompts generated from user description"
    )


class FeedbackStatsResponse(BaseModel):
    """Statistics about collected feedback"""

    total_feedback: int
    corrections: int
    confirmations: int
    accuracy_rate: float = Field(
        ...,
        ge=0,
        le=1,
        description="Rate of correct predictions (confirmations / total)"
    )
    correction_rate: float = Field(
        ...,
        ge=0,
        le=1,
        description="Rate of corrections needed"
    )
    top_misclassifications: List[dict] = Field(
        default=[],
        description="Most common prediction errors"
    )
    problem_foods: List[dict] = Field(
        default=[],
        description="Foods with highest correction rates"
    )
