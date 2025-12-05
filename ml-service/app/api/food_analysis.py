"""
Food Analysis API endpoints.
Handles food scanning, classification, and nutrition estimation.

Supports:
- Single-dish analysis (/analyze)
- Multi-dish analysis with bounding boxes (/analyze/multi)
"""
import logging
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, Query, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.schemas.food_analysis import (
    FoodAnalysisResponse,
    ModelsInfoResponse,
    NutritionDBSearchResponse,
    ModelInfo,
    DimensionsInput,
)
from app.schemas.multi_dish import MultiDishAnalysisResponse
from app.services.food_analysis_service import food_analysis_service
from app.services.multi_dish_service import get_multi_dish_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=FoodAnalysisResponse)
async def analyze_food(
    image: UploadFile = File(..., description="Food image (JPEG/PNG, max 10MB)"),
    dimensions: Optional[str] = Form(None, description="JSON string of AR measurements"),
):
    """
    Analyze food image and estimate nutrition.

    **Process:**
    1. Classify food item using computer vision
    2. Estimate portion size (from AR measurements or image analysis)
    3. Calculate nutrition values based on food type and portion size
    4. Return results with confidence scores

    **Parameters:**
    - **image**: Food photo (JPEG or PNG format, max 10MB)
    - **dimensions** (optional): JSON string with AR measurements:
      ```json
      {"width": 10.5, "height": 8.2, "depth": 6.0}
      ```
      All values in centimeters.

    **Returns:**
    - List of detected food items with nutrition information
    - Measurement quality assessment
    - Processing time
    - Suggestions for improving accuracy
    """
    try:
        # Validate file size (10MB limit)
        contents = await image.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image file too large (max 10MB)")

        # Validate file type
        if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Only JPEG and PNG are supported.",
            )

        # Load image
        try:
            pil_image = Image.open(io.BytesIO(contents))
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

        # Parse dimensions if provided
        dimensions_obj = None
        if dimensions:
            try:
                import json

                dims_dict = json.loads(dimensions)
                dimensions_obj = DimensionsInput(**dims_dict)
            except Exception as e:
                logger.warning(f"Error parsing dimensions: {str(e)}")
                # Continue without dimensions

        # Analyze food
        food_items, measurement_quality, processing_time = (
            await food_analysis_service.analyze_food(pil_image, dimensions_obj)
        )

        # Generate suggestions
        suggestions = []
        if measurement_quality == "low":
            suggestions.append(
                "Use AR measurements for better portion size accuracy"
            )
        if len(food_items) > 0 and food_items[0].confidence < 0.8:
            suggestions.append(
                "Take a clearer photo with better lighting for improved classification"
            )
        if dimensions_obj is None:
            suggestions.append(
                "Include a reference object (hand, credit card) for better size estimation"
            )

        return FoodAnalysisResponse(
            food_items=food_items,
            measurement_quality=measurement_quality,
            processing_time=processing_time,
            suggestions=suggestions if suggestions else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Food analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Food analysis failed: {str(e)}",
        )


@router.post("/analyze/multi", response_model=MultiDishAnalysisResponse)
async def analyze_multi_dish(
    image: UploadFile = File(..., description="Food image (JPEG/PNG, max 10MB)"),
    min_confidence: float = Query(
        0.15,
        ge=0,
        le=1,
        description="Minimum detection confidence threshold (0-1)"
    ),
    max_dishes: int = Query(
        10,
        ge=1,
        le=20,
        description="Maximum number of dishes to detect"
    ),
):
    """
    Analyze food image for **multiple dishes** with bounding boxes.

    This endpoint detects and classifies multiple food items in a single image,
    returning bounding boxes for each detected dish along with nutrition estimates.

    **Use Cases:**
    - Plates with multiple food items (main + sides)
    - Food trays or buffet-style meals
    - Meal prep containers with compartments
    - Any image with more than one distinct food item

    **Pipeline:**
    1. Object detection using OWL-ViT (zero-shot, text-prompted)
    2. Each detected region is cropped and classified
    3. Nutrition is estimated for each dish
    4. Results aggregated with total nutrition

    **Parameters:**
    - **image**: Food photo (JPEG or PNG, max 10MB)
    - **min_confidence**: Minimum detection confidence (default: 0.15)
    - **max_dishes**: Maximum dishes to return (default: 10, max: 20)

    **Returns:**
    - List of detected dishes with:
      - Bounding box (normalized 0-1 and pixel coordinates)
      - Classification with confidence
      - Nutrition estimates
      - Alternative classifications
    - Aggregated total nutrition
    - Detection quality assessment
    - Processing time and model info

    **Example Response:**
    ```json
    {
      "dishes": [
        {
          "dish_id": 1,
          "name": "Grilled Chicken",
          "confidence": 0.92,
          "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.7, ...},
          "nutrition": {"calories": 165, "protein": 31, ...},
          ...
        },
        {
          "dish_id": 2,
          "name": "Caesar Salad",
          "confidence": 0.87,
          "bbox": {"x1": 0.55, "y1": 0.1, "x2": 0.95, "y2": 0.5, ...},
          ...
        }
      ],
      "dish_count": 2,
      "total_nutrition": {"calories": 315, "protein": 38, ...},
      ...
    }
    ```

    **Future Features:**
    - LIDAR depth map for accurate volume/portion estimation
    - Fine-tuned object detector for better food detection
    """
    try:
        # Validate file size (10MB limit)
        contents = await image.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image file too large (max 10MB)")

        # Validate file type
        if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Only JPEG and PNG are supported.",
            )

        # Load image
        try:
            pil_image = Image.open(io.BytesIO(contents))
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file")

        # Get multi-dish service and analyze
        service = get_multi_dish_service()
        result = await service.analyze(
            image=pil_image,
            min_confidence=min_confidence,
            max_dishes=max_dishes,
        )

        logger.info(
            f"Multi-dish analysis: {result.dish_count} dishes detected, "
            f"{result.processing_time_ms:.1f}ms"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-dish analysis error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Multi-dish analysis failed: {str(e)}",
        )


@router.get("/models/info", response_model=ModelsInfoResponse)
async def get_models_info():
    """
    Get information about available food classification models.

    **Returns:**
    - List of available models with metadata
    - Currently active model name
    """
    try:
        model_info = food_analysis_service.get_model_info()

        return ModelsInfoResponse(
            available_models=[ModelInfo(**model_info)],
            active_model=model_info["name"],
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model information",
        )


@router.get("/nutrition-db/search", response_model=NutritionDBSearchResponse)
async def search_nutrition_db(q: str):
    """
    Search nutrition database by food name.

    **Parameters:**
    - **q**: Search query (food name)

    **Returns:**
    - List of matching foods with nutrition information
    - Total number of results

    **Example:**
    ```
    GET /api/food/nutrition-db/search?q=chicken
    ```
    """
    try:
        if not q or len(q) < 2:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 2 characters long",
            )

        results = await food_analysis_service.search_nutrition_db(q)

        return NutritionDBSearchResponse(
            results=results,
            query=q,
            total_results=len(results),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Nutrition DB search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Search failed",
        )


@router.get("/health")
async def food_analysis_health():
    """
    Health check for food analysis service.

    **Returns:**
    - Service status
    - Model status
    """
    try:
        model_info = food_analysis_service.get_model_info()

        return {
            "status": "healthy",
            "service": "food-analysis",
            "model": {
                "loaded": True,
                "name": model_info["name"],
                "version": model_info["version"],
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "food-analysis",
                "error": str(e),
            },
        )
