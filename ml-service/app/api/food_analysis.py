"""
Food Analysis API endpoints.
Handles food scanning, classification, and nutrition estimation.
"""

import asyncio
import logging
import hashlib
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image
import io

from app.schemas.food_analysis import (
    FoodAnalysisResponse,
    ModelsInfoResponse,
    NutritionDBSearchResponse,
    ModelInfo,
    DimensionsInput,
    MicronutrientEstimationRequest,
    MicronutrientEstimationResponse,
    FoodFeedbackRequest,
    FoodFeedbackResponse,
    FeedbackStatsResponse,
)
from app.services.food_analysis_service import food_analysis_service
from app.services.feedback_service import feedback_service
from app.core.queue import inference_queue
from app.core.queue.manager import QueueFullError, CircuitOpenError
from app.data.food_database import FoodCategory
from app.database import get_db

logger = logging.getLogger(__name__)


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hash of image for feedback deduplication."""
    return hashlib.sha256(image_bytes).hexdigest()


router = APIRouter()


@router.post("/analyze", response_model=FoodAnalysisResponse)
async def analyze_food(
    image: UploadFile = File(..., description="Food image (JPEG/PNG, max 10MB)"),
    dimensions: Optional[str] = Form(
        None, description="JSON string of AR measurements"
    ),
    cooking_method: Optional[str] = Form(
        None,
        description="How the food was prepared (raw, cooked, grilled, fried, etc.)",
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
            raise HTTPException(
                status_code=413, detail="Image file too large (max 10MB)"
            )

        # Compute image hash for feedback support
        image_hash = compute_image_hash(contents)

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
            raise HTTPException(
                status_code=400, detail="Invalid or corrupted image file"
            )

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
                    detail="Invalid dimensions format: must be valid JSON",
                )

            # Validate dimensions values with Pydantic
            try:
                dimensions_obj = DimensionsInput(**dims_dict)
            except Exception as e:
                logger.error(f"Invalid dimensions values: {str(e)}")
                raise HTTPException(
                    status_code=422, detail=f"Invalid dimensions values: {str(e)}"
                )

        # Route through inference queue for concurrency control
        try:
            (
                food_items,
                measurement_quality,
                processing_time,
                service_suggestions,
            ) = await inference_queue.submit(pil_image, dimensions_obj, cooking_method)

        except QueueFullError as e:
            # 503 Service Unavailable - queue is at capacity
            logger.warning(
                f"Queue full, rejecting request: {str(e)}",
                extra={"queue_size": inference_queue.queue_size},
            )
            raise HTTPException(
                status_code=503,
                detail="Service temporarily overloaded. Please try again shortly.",
                headers={"Retry-After": "5"},
            )

        except CircuitOpenError as e:
            # 503 Service Unavailable - circuit breaker is open
            logger.warning(f"Circuit breaker open: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable due to high error rate.",
                headers={"Retry-After": "30"},
            )

        except asyncio.TimeoutError:
            # 504 Gateway Timeout - request took too long
            logger.error("Request timed out waiting for inference")
            raise HTTPException(
                status_code=504,
                detail="Request timed out. The service is under heavy load.",
            )

        # Generate API-level suggestions and merge with service suggestions
        suggestions = list(service_suggestions) if service_suggestions else []

        # Add API-specific suggestions
        if measurement_quality == "low" and dimensions_obj is None:
            suggestions.append("Use AR measurements for better portion size accuracy")
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
            image_hash=image_hash,
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


# ==============================================================================
# MICRONUTRIENT ESTIMATION FOR BARCODE PRODUCTS
# ==============================================================================

# Mapping from Open Food Facts category tags to our FoodCategory enum
# These are common patterns in Open Food Facts category_tags
OFF_CATEGORY_MAPPING: dict[str, FoodCategory] = {
    # Fruits
    "en:fruits": FoodCategory.FRUIT,
    "en:fresh-fruits": FoodCategory.FRUIT,
    "en:dried-fruits": FoodCategory.FRUIT,
    "en:fruit-juices": FoodCategory.BEVERAGE,
    "en:citrus": FoodCategory.FRUIT,
    "en:tropical-fruits": FoodCategory.FRUIT,
    "en:berries": FoodCategory.FRUIT,
    "en:apples": FoodCategory.FRUIT,
    "en:bananas": FoodCategory.FRUIT,
    # Vegetables
    "en:vegetables": FoodCategory.VEGETABLE,
    "en:fresh-vegetables": FoodCategory.VEGETABLE,
    "en:frozen-vegetables": FoodCategory.VEGETABLE,
    "en:canned-vegetables": FoodCategory.VEGETABLE,
    "en:leafy-vegetables": FoodCategory.VEGETABLE,
    "en:root-vegetables": FoodCategory.VEGETABLE,
    "en:salads": FoodCategory.VEGETABLE,
    # Proteins (meat)
    "en:meats": FoodCategory.PROTEIN,
    "en:poultry": FoodCategory.PROTEIN,
    "en:beef": FoodCategory.PROTEIN,
    "en:pork": FoodCategory.PROTEIN,
    "en:chicken": FoodCategory.PROTEIN,
    "en:turkey": FoodCategory.PROTEIN,
    "en:lamb": FoodCategory.PROTEIN,
    "en:sausages": FoodCategory.PROTEIN,
    "en:deli-meats": FoodCategory.PROTEIN,
    "en:eggs": FoodCategory.PROTEIN,
    # Seafood
    "en:seafood": FoodCategory.SEAFOOD,
    "en:fish": FoodCategory.SEAFOOD,
    "en:salmon": FoodCategory.SEAFOOD,
    "en:tuna": FoodCategory.SEAFOOD,
    "en:shrimp": FoodCategory.SEAFOOD,
    "en:shellfish": FoodCategory.SEAFOOD,
    "en:canned-fish": FoodCategory.SEAFOOD,
    # Dairy
    "en:dairies": FoodCategory.DAIRY,
    "en:milks": FoodCategory.DAIRY,
    "en:yogurts": FoodCategory.DAIRY,
    "en:cheeses": FoodCategory.DAIRY,
    "en:butter": FoodCategory.DAIRY,
    "en:cream": FoodCategory.DAIRY,
    "en:ice-cream": FoodCategory.DAIRY,
    # Grains
    "en:cereals-and-potatoes": FoodCategory.GRAIN,
    "en:breads": FoodCategory.GRAIN,
    "en:breakfast-cereals": FoodCategory.GRAIN,
    "en:pasta": FoodCategory.GRAIN,
    "en:rice": FoodCategory.GRAIN,
    "en:noodles": FoodCategory.GRAIN,
    "en:tortillas": FoodCategory.GRAIN,
    "en:oats": FoodCategory.GRAIN,
    # Legumes
    "en:legumes": FoodCategory.LEGUME,
    "en:beans": FoodCategory.LEGUME,
    "en:lentils": FoodCategory.LEGUME,
    "en:chickpeas": FoodCategory.LEGUME,
    "en:tofu": FoodCategory.LEGUME,
    "en:soy": FoodCategory.LEGUME,
    # Nuts & Seeds
    "en:nuts": FoodCategory.NUT,
    "en:seeds": FoodCategory.NUT,
    "en:almonds": FoodCategory.NUT,
    "en:peanuts": FoodCategory.NUT,
    "en:walnuts": FoodCategory.NUT,
    "en:cashews": FoodCategory.NUT,
    "en:nut-butters": FoodCategory.NUT,
    # Baked goods
    "en:biscuits-and-cakes": FoodCategory.BAKED,
    "en:pastries": FoodCategory.BAKED,
    "en:cookies": FoodCategory.BAKED,
    "en:cakes": FoodCategory.BAKED,
    "en:muffins": FoodCategory.BAKED,
    "en:croissants": FoodCategory.BAKED,
    # Snacks
    "en:snacks": FoodCategory.SNACK,
    "en:chips": FoodCategory.SNACK,
    "en:crackers": FoodCategory.SNACK,
    "en:popcorn": FoodCategory.SNACK,
    "en:chocolate": FoodCategory.SNACK,
    "en:candy": FoodCategory.SNACK,
    "en:energy-bars": FoodCategory.SNACK,
    "en:protein-bars": FoodCategory.SNACK,
    # Beverages
    "en:beverages": FoodCategory.BEVERAGE,
    "en:sodas": FoodCategory.BEVERAGE,
    "en:juices": FoodCategory.BEVERAGE,
    "en:waters": FoodCategory.BEVERAGE,
    "en:teas": FoodCategory.BEVERAGE,
    "en:coffees": FoodCategory.BEVERAGE,
    "en:energy-drinks": FoodCategory.BEVERAGE,
    "en:smoothies": FoodCategory.BEVERAGE,
    # Condiments
    "en:condiments": FoodCategory.CONDIMENT,
    "en:sauces": FoodCategory.CONDIMENT,
    "en:dressings": FoodCategory.CONDIMENT,
    "en:ketchup": FoodCategory.CONDIMENT,
    "en:mayonnaise": FoodCategory.CONDIMENT,
    "en:mustard": FoodCategory.CONDIMENT,
    # Mixed/prepared foods
    "en:meals": FoodCategory.MIXED,
    "en:prepared-meals": FoodCategory.MIXED,
    "en:frozen-meals": FoodCategory.MIXED,
    "en:sandwiches": FoodCategory.MIXED,
    "en:pizzas": FoodCategory.MIXED,
    "en:soups": FoodCategory.MIXED,
}


def map_off_categories_to_food_category(
    categories: Optional[list[str]],
) -> tuple[FoodCategory, str]:
    """
    Map Open Food Facts category tags to our FoodCategory enum.

    Args:
        categories: List of Open Food Facts category tags (e.g., ["en:fruits", "en:apples"])

    Returns:
        Tuple of (FoodCategory, confidence level: "high", "medium", "low")
    """
    if not categories:
        return FoodCategory.UNKNOWN, "low"

    # Check each category in order (first match wins, as OFF puts most specific first)
    for cat in categories:
        cat_lower = cat.lower()
        if cat_lower in OFF_CATEGORY_MAPPING:
            return OFF_CATEGORY_MAPPING[cat_lower], "high"

    # Try partial matching for less common categories
    for cat in categories:
        cat_lower = cat.lower()
        # Check if any keyword matches
        for key, food_cat in OFF_CATEGORY_MAPPING.items():
            # Extract the main keyword from the mapping key (e.g., "fruits" from "en:fruits")
            keyword = key.split(":")[-1]
            if keyword in cat_lower:
                return food_cat, "medium"

    return FoodCategory.UNKNOWN, "low"


# All micronutrient keys we track
MICRONUTRIENT_KEYS = [
    "potassium",
    "calcium",
    "iron",
    "magnesium",
    "zinc",
    "phosphorus",
    "vitamin_a",
    "vitamin_c",
    "vitamin_d",
    "vitamin_e",
    "vitamin_k",
    "vitamin_b6",
    "vitamin_b12",
    "folate",
    "thiamin",
    "riboflavin",
    "niacin",
]


@router.post("/estimate-micronutrients", response_model=MicronutrientEstimationResponse)
async def estimate_micronutrients(request: MicronutrientEstimationRequest):
    """
    Sophisticated micronutrient estimation for barcode-scanned products.

    This endpoint uses multiple signals to provide accurate estimates:
    1. **Ingredient parsing** - Analyzes ingredients list to identify nutrient-rich components
    2. **Food name matching** - Matches product name to our food database
    3. **Macronutrient inference** - Uses protein/fiber levels to inform estimates
    4. **Category baseline** - Falls back to category averages when other signals weak

    **Example use case:**
    Product: "Organic Spinach & Kale Smoothie"
    - Ingredient parsing identifies spinach (high iron, folate, vitamin K) and kale (vitamin A, C, K)
    - High fiber → boosts magnesium and folate estimates
    - Category "beverage" provides baseline for any gaps

    **Parameters:**
    - **food_name**: Product name from Open Food Facts
    - **categories**: Open Food Facts category tags (e.g., ["en:beverages", "en:smoothies"])
    - **portion_weight**: Serving size in grams
    - **ingredients_text**: Raw ingredients list (highly recommended for accuracy!)
    - **protein**: Protein per 100g (helps identify B12-rich foods)
    - **fiber**: Fiber per 100g (helps identify magnesium-rich foods)
    - **fat**: Fat per 100g
    - **existing**: Already known micronutrients (won't be overwritten)

    **Returns:**
    - **estimated**: Dictionary of estimated micronutrient values
    - **category_used**: Primary food category identified
    - **confidence**: Overall estimation confidence (high/medium/low)
    - **source**: Signals used (e.g., "ingredients+macros+category")
    """
    try:
        from app.services.ingredient_micronutrient_service import (
            estimate_micronutrients_sophisticated,
        )

        # Use sophisticated multi-signal estimation
        (
            estimated,
            category_used,
            confidence,
            source,
        ) = estimate_micronutrients_sophisticated(
            food_name=request.food_name,
            categories=request.categories,
            portion_weight=request.portion_weight,
            ingredients_text=request.ingredients_text,
            protein=request.protein,
            fiber=request.fiber,
            fat=request.fat,
            existing=request.existing,
        )

        logger.info(
            f"Sophisticated estimation for '{request.food_name}': "
            f"{len(estimated)} nutrients, category={category_used}, "
            f"confidence={confidence}, sources={source}"
        )

        return MicronutrientEstimationResponse(
            estimated=estimated,
            category_used=category_used,
            confidence=confidence,
            source=source,
        )

    except Exception as e:
        logger.error(f"Micronutrient estimation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Micronutrient estimation failed: {str(e)}",
        )


# ==============================================================================
# FEEDBACK ENDPOINTS - For user corrections and model improvement
# ==============================================================================


@router.post("/feedback", response_model=FoodFeedbackResponse)
async def submit_feedback(
    request: FoodFeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit user feedback on a food classification.

    This endpoint records user corrections when the classifier makes mistakes,
    enabling the model to improve over time through:

    1. **Pattern detection** - Identifies common misclassifications
    2. **Prompt generation** - Creates new CLIP prompts from user descriptions
    3. **Confidence boosting** - Suggests likely alternatives for similar images

    **When to call:**
    - When user confirms/corrects a low-confidence classification
    - After user selects an alternative food from the suggestions
    - When user manually enters a different food name

    **Parameters:**
    - **image_hash**: SHA-256 hash of the image (from analyze response)
    - **original_prediction**: What the model predicted
    - **original_confidence**: Model's confidence in original prediction
    - **user_selected_food**: What the user confirmed as correct
    - **alternatives_shown**: List of alternatives that were displayed
    - **user_description** (optional): User's description of the food

    **Returns:**
    - **success**: Whether feedback was recorded
    - **feedback_id**: ID of the recorded feedback (-1 if duplicate)
    - **was_correction**: True if user selected different food than predicted
    - **suggested_prompts**: New prompts generated from user description
    """
    try:
        # Determine if this is a correction or confirmation
        was_correction = (
            request.original_prediction.lower().strip()
            != request.user_selected_food.lower().strip()
        )

        # Submit feedback to the service
        feedback_id, suggested_prompts = await feedback_service.submit_feedback(
            db=db,
            image_hash=request.image_hash,
            original_prediction=request.original_prediction,
            original_confidence=request.original_confidence,
            corrected_label=request.user_selected_food,
            alternatives=[{"name": alt} for alt in request.alternatives_shown],
            user_description=request.user_description,
            user_id=request.user_id,
        )

        await db.commit()

        # Determine response message
        if feedback_id == -1:
            message = "Duplicate feedback - already recorded for this image"
        elif was_correction:
            message = (
                f"Thank you! We've learned that this should be "
                f"'{request.user_selected_food}' instead of "
                f"'{request.original_prediction}'"
            )
        else:
            message = "Thanks for confirming! This helps improve accuracy."

        logger.info(
            f"Feedback recorded: {request.original_prediction} -> "
            f"{request.user_selected_food} "
            f"(correction: {was_correction}, id: {feedback_id})"
        )

        return FoodFeedbackResponse(
            success=feedback_id != -1,
            feedback_id=max(feedback_id, 0),
            was_correction=was_correction,
            message=message,
            suggested_prompts=suggested_prompts,
        )

    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}",
        )


@router.get("/feedback/stats", response_model=FeedbackStatsResponse)
async def get_feedback_stats(db: AsyncSession = Depends(get_db)):
    """
    Get statistics about collected feedback and model accuracy.

    **Returns:**
    - **total_feedback**: Total number of feedback submissions
    - **corrections**: Number of corrections (model was wrong)
    - **confirmations**: Number of confirmations (model was right)
    - **accuracy_rate**: Proportion of correct predictions
    - **correction_rate**: Proportion of corrections needed
    - **top_misclassifications**: Most common prediction errors
    - **problem_foods**: Foods with highest correction rates

    **Use cases:**
    - Dashboard for monitoring model performance
    - Identifying foods that need better training
    - Planning prompt improvements
    """
    try:
        stats = await feedback_service.get_stats(db)

        # Calculate accuracy rate
        total = stats.get("total_feedback", 0)
        # Confirmations = when user selected same as prediction
        # For now we estimate from approved feedback
        confirmations = stats.get("approved_feedback", 0)
        corrections = total - confirmations

        accuracy_rate = confirmations / total if total > 0 else 0.0
        correction_rate = corrections / total if total > 0 else 0.0

        return FeedbackStatsResponse(
            total_feedback=total,
            corrections=corrections,
            confirmations=confirmations,
            accuracy_rate=round(accuracy_rate, 3),
            correction_rate=round(correction_rate, 3),
            top_misclassifications=stats.get("top_misclassifications", []),
            problem_foods=stats.get("problem_foods", []),
        )

    except Exception as e:
        logger.error(f"Feedback stats error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get feedback stats: {str(e)}",
        )


@router.get("/feedback/suggestions/{food_key}")
async def get_food_suggestions(
    food_key: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get prompt suggestions for a specific food based on feedback patterns.

    This endpoint analyzes historical corrections to suggest:
    - New prompts that might improve classification
    - Common foods this is confused with
    - Disambiguation strategies

    **Parameters:**
    - **food_key**: The food identifier (e.g., "apple", "grilled_chicken")

    **Returns:**
    - Current prompts being used
    - Suggested new prompts from user feedback
    - Common confusion patterns
    """
    try:
        suggestions = await feedback_service.get_prompt_suggestions(db, food_key)
        return suggestions

    except Exception as e:
        logger.error(f"Food suggestions error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get suggestions: {str(e)}",
        )


# ==============================================================================
# COARSE CLASSIFICATION - For USDA search integration
# ==============================================================================


@router.post("/coarse-classify")
async def coarse_classify(
    image: UploadFile = File(..., description="Food image (JPEG/PNG, max 10MB)"),
    query: Optional[str] = Form(None, description="Optional user-provided text to enhance classification context and search query"),
):
    """
    Coarse-grained food classification for USDA search integration.

    This endpoint classifies food images into 25-30 high-level categories
    aligned with USDA food groups. It's designed as Tier 1 of the multi-tier
    classification architecture for scaling to 500K+ foods.

    **Use cases:**
    - Pre-filtering USDA search results by food category
    - Determining appropriate USDA data types (Foundation, SR Legacy, Branded, etc.)
    - Providing context hints for enhanced search queries

    **Process:**
    1. Classify image into coarse food category using CLIP zero-shot learning
    2. Return category with confidence and alternatives
    3. Include recommended USDA data types for search filtering
    4. Provide search hints for query enhancement

    **Parameters:**
    - **image**: Food photo (JPEG or PNG format, max 10MB)
    - **query** (optional): User's search query to enhance

    **Returns:**
    - **category**: High-level food category (e.g., "fruits_fresh", "meat_red")
    - **confidence**: Classification confidence (0-1)
    - **usda_datatypes**: Recommended USDA data types for search
    - **alternatives**: Top alternative categories with confidence
    - **search_hints**: Hints for enhancing USDA search

    **Categories include:**
    - Fruits: fresh, processed
    - Vegetables: leafy, root, other, cooked
    - Meat: red, poultry, processed
    - Seafood: fish, shellfish
    - Dairy: milk, cheese, yogurt, other
    - Grains: bread, pasta, rice, cereal, other
    - Legumes, Nuts & Seeds
    - Beverages: hot, cold
    - Snacks: sweet, savory
    - Mixed dishes, Fast food
    - Condiments & Sauces
    - Eggs
    """
    try:
        # Validate file size (10MB limit)
        contents = await image.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413, detail="Image file too large (max 10MB)"
            )

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
            raise HTTPException(
                status_code=400, detail="Invalid or corrupted image file"
            )

        # Import and use coarse classifier
        from app.ml_models.coarse_classifier import get_coarse_classifier

        classifier = get_coarse_classifier()
        result = classifier.classify_with_usda_context(pil_image, query or "")

        logger.info(
            f"Coarse classification: {result['category']} "
            f"(confidence: {result['confidence']:.2%})"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Coarse classification error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Coarse classification failed: {str(e)}",
        )


@router.get("/coarse-classify/categories")
async def get_coarse_categories():
    """
    Get list of coarse food categories and their USDA data type mappings.

    **Returns:**
    - List of all supported food categories
    - Mapping of each category to recommended USDA data types
    - Category descriptions
    """
    try:
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
            get_coarse_classifier,
        )

        classifier = get_coarse_classifier()
        model_info = classifier.get_model_info()

        categories = []
        for cat in FoodCategory:
            if cat != FoodCategory.UNKNOWN:
                categories.append({
                    "category": cat.value,
                    "usda_datatypes": CATEGORY_TO_USDA_DATATYPES.get(cat, []),
                })

        return {
            "categories": categories,
            "total": len(categories),
            "model_info": model_info,
        }

    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get categories: {str(e)}",
        )
