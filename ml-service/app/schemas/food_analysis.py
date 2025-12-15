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

    calories: float = Field(..., ge=0)
    protein: float = Field(..., ge=0)  # grams
    carbs: float = Field(..., ge=0)  # grams
    fat: float = Field(..., ge=0)  # grams
    fiber: Optional[float] = Field(None, ge=0)  # grams
    sugar: Optional[float] = Field(None, ge=0)  # grams
    sodium: Optional[float] = Field(None, ge=0)  # mg
    saturated_fat: Optional[float] = Field(None, ge=0)  # grams
    lysine: Optional[float] = Field(None, ge=0)  # mg - essential amino acid
    arginine: Optional[float] = Field(
        None, ge=0
    )  # mg - conditionally essential amino acid


class FoodItemAlternative(BaseModel):
    """Alternative food classification"""

    name: str
    confidence: float = Field(..., ge=0, le=1)


class FoodItem(BaseModel):
    """Detected food item with nutrition information"""

    name: str
    confidence: float = Field(..., ge=0, le=1)
    portion_size: str  # e.g., "1 cup", "150g"
    portion_weight: float = Field(..., gt=0)  # grams
    nutrition: NutritionInfo
    category: Optional[str] = None  # e.g., "fruit", "vegetable", "protein"
    alternatives: Optional[List[FoodItemAlternative]] = None


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
