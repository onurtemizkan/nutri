"""
Food Analysis API endpoints.
Handles food scanning, classification, and nutrition estimation.
"""
import logging
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
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
from app.services.food_analysis_service import food_analysis_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=FoodAnalysisResponse)
async def analyze_food(
    image: UploadFile = File(..., description="Food image (JPEG/PNG, max 10MB)"),
    dimensions: Optional[str] = Form(None, description="JSON string of AR measurements"),
    cooking_method: Optional[str] = Form(
        None,
        description="How the food was prepared (raw, cooked, grilled, fried, etc.)"
    ),
):
    """
    Analyze food image and estimate nutrition.

    **Process:**
    1. Classify food item using computer vision
    2. Estimate portion size (from AR measurements or image analysis)
    3. Apply cooking method adjustments for weight/calorie accuracy
    4. Calculate nutrition values based on food type, portion size, and cooking
    5. Return results with confidence scores and validation warnings

    **Parameters:**
    - **image**: Food photo (JPEG or PNG format, max 10MB)
    - **dimensions** (optional): JSON string with AR measurements:
      ```json
      {"width": 10.5, "height": 8.2, "depth": 6.0}
      ```
      All values in centimeters.
    - **cooking_method** (optional): How the food was prepared. Options:
      - raw, cooked, boiled, steamed, grilled, fried, baked, roasted, sauteed, poached

    **Returns:**
    - List of detected food items with nutrition information
    - Measurement quality assessment
    - Processing time
    - Suggestions for improving accuracy (including portion validation warnings)
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
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in dimensions: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid dimensions format: must be valid JSON"
                )

            # Validate dimensions values with Pydantic
            try:
                dimensions_obj = DimensionsInput(**dims_dict)
            except Exception as e:
                logger.error(f"Invalid dimensions values: {str(e)}")
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid dimensions values: {str(e)}"
                )

        # Analyze food with cooking method support
        food_items, measurement_quality, processing_time, service_suggestions = (
            await food_analysis_service.analyze_food(
                pil_image,
                dimensions_obj,
                cooking_method
            )
        )

        # Generate API-level suggestions and merge with service suggestions
        suggestions = list(service_suggestions) if service_suggestions else []

        # Add API-specific suggestions
        if measurement_quality == "low" and dimensions_obj is None:
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
        if cooking_method is None and len(food_items) > 0:
            suggestions.append(
                "Specify cooking method for more accurate nutrition estimates"
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


@router.get("/cooking-methods")
async def get_cooking_methods():
    """
    Get list of supported cooking methods.

    **Returns:**
    - List of cooking method strings
    - Can be used to populate cooking method dropdown in clients
    """
    try:
        methods = food_analysis_service.get_supported_cooking_methods()
        return {
            "cooking_methods": methods,
            "total": len(methods),
        }
    except Exception as e:
        logger.error(f"Error getting cooking methods: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve cooking methods",
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
