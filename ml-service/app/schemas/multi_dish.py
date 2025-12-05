"""
Pydantic schemas for multi-dish food analysis.

Supports detecting and classifying multiple food items in a single image,
with bounding boxes for each detected dish.
"""
from typing import List, Optional
from pydantic import BaseModel, Field

# Re-use existing schemas
from app.schemas.food_analysis import NutritionInfo, FoodItemAlternative


class BoundingBox(BaseModel):
    """
    Bounding box coordinates for a detected food item.

    Normalized coordinates (0-1) are relative to image dimensions.
    Pixel coordinates are absolute positions in the original image.
    """
    # Normalized coordinates (0-1 range, relative to image size)
    x1: float = Field(..., ge=0, le=1, description="Left edge (normalized 0-1)")
    y1: float = Field(..., ge=0, le=1, description="Top edge (normalized 0-1)")
    x2: float = Field(..., ge=0, le=1, description="Right edge (normalized 0-1)")
    y2: float = Field(..., ge=0, le=1, description="Bottom edge (normalized 0-1)")

    # Pixel coordinates (absolute positions in original image)
    x1_px: Optional[int] = Field(None, ge=0, description="Left edge in pixels")
    y1_px: Optional[int] = Field(None, ge=0, description="Top edge in pixels")
    x2_px: Optional[int] = Field(None, ge=0, description="Right edge in pixels")
    y2_px: Optional[int] = Field(None, ge=0, description="Bottom edge in pixels")

    @property
    def width(self) -> float:
        """Width of bounding box (normalized)."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height of bounding box (normalized)."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Area of bounding box (normalized, 0-1)."""
        return self.width * self.height

    @property
    def center(self) -> tuple:
        """Center point of bounding box (normalized)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class DetectedDish(BaseModel):
    """
    A single detected dish with classification, location, and nutrition.

    Each dish has:
    - Unique ID within the analysis
    - Classification with confidence
    - Bounding box location
    - Estimated nutrition (based on default portion or LIDAR volume)
    """
    # Identification
    dish_id: int = Field(..., ge=1, description="Unique ID within this analysis (1-based)")
    name: str = Field(..., min_length=1, description="Classified food name")
    confidence: float = Field(..., ge=0, le=1, description="Classification confidence (0-1)")

    # Location in image
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    area_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of image area occupied by this dish"
    )

    # Nutrition (estimated based on portion)
    nutrition: NutritionInfo = Field(..., description="Estimated nutrition values")

    # Classification alternatives
    alternatives: List[FoodItemAlternative] = Field(
        default_factory=list,
        description="Alternative food classifications"
    )

    # Category
    category: Optional[str] = Field(
        None,
        description="Food category (e.g., 'main_course', 'side_dish', 'dessert')"
    )

    # Portion estimation (for future LIDAR integration)
    volume_cm3: Optional[float] = Field(
        None,
        ge=0,
        description="3D volume from LIDAR measurement (cm³)"
    )
    weight_grams: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated weight in grams"
    )
    portion_size: str = Field(
        "estimated",
        description="Portion size description (e.g., '150g', 'estimated', '1 cup')"
    )


class MultiDishAnalysisRequest(BaseModel):
    """
    Request parameters for multi-dish analysis.

    Allows customization of detection behavior.
    """
    min_confidence: float = Field(
        0.3,
        ge=0,
        le=1,
        description="Minimum detection confidence threshold"
    )
    max_dishes: int = Field(
        10,
        ge=1,
        le=20,
        description="Maximum number of dishes to detect"
    )

    # Future LIDAR integration
    depth_map: Optional[List[List[float]]] = Field(
        None,
        description="LIDAR depth map for volume estimation (future feature)"
    )


class MultiDishAnalysisResponse(BaseModel):
    """
    Response from multi-dish food analysis.

    Contains all detected dishes with their classifications and nutrition,
    plus aggregated totals and metadata.
    """
    # Detected dishes (ordered by confidence, highest first)
    dishes: List[DetectedDish] = Field(
        ...,
        description="List of detected dishes, ordered by confidence (highest first)"
    )
    dish_count: int = Field(..., ge=0, description="Number of dishes detected")

    # Aggregated nutrition across all dishes
    total_nutrition: NutritionInfo = Field(
        ...,
        description="Sum of nutrition values across all detected dishes"
    )

    # Image metadata
    image_width: int = Field(..., gt=0, description="Original image width in pixels")
    image_height: int = Field(..., gt=0, description="Original image height in pixels")

    # Processing metadata
    processing_time_ms: float = Field(..., ge=0, description="Total processing time in milliseconds")
    detector_model: str = Field(..., description="Object detection model used")
    classifier_model: str = Field(..., description="Food classification model used")

    # Quality indicators
    detection_quality: str = Field(
        ...,
        pattern="^(high|medium|low)$",
        description="Overall detection quality assessment"
    )
    overlapping_dishes: bool = Field(
        False,
        description="Whether any detected dishes have overlapping bounding boxes"
    )

    # Suggestions for improving results
    suggestions: Optional[List[str]] = Field(
        None,
        description="Suggestions for improving detection/classification accuracy"
    )


class DetectionResult(BaseModel):
    """
    Raw detection result from object detector (internal use).

    Used as intermediate format before classification.
    """
    bbox: tuple = Field(..., description="(x1, y1, x2, y2) normalized coordinates")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    label: str = Field(..., description="Detected label from detector")
