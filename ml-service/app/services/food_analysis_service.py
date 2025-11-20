"""
Food Analysis Service
Handles food classification, portion estimation, and nutrition calculation.
"""
import time
import logging
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

from app.schemas.food_analysis import (
    FoodItem,
    FoodItemAlternative,
    NutritionInfo,
    DimensionsInput,
)

logger = logging.getLogger(__name__)


# Mock nutrition database (in production, use USDA FoodData Central API)
NUTRITION_DATABASE = {
    "apple": {
        "category": "fruit",
        "serving_size": "1 medium (182g)",
        "serving_weight": 182,
        "nutrition": {
            "calories": 95,
            "protein": 0.5,
            "carbs": 25,
            "fat": 0.3,
            "fiber": 4.4,
            "sugar": 19,
        },
    },
    "banana": {
        "category": "fruit",
        "serving_size": "1 medium (118g)",
        "serving_weight": 118,
        "nutrition": {
            "calories": 105,
            "protein": 1.3,
            "carbs": 27,
            "fat": 0.4,
            "fiber": 3.1,
            "sugar": 14,
        },
    },
    "chicken breast": {
        "category": "protein",
        "serving_size": "100g (cooked)",
        "serving_weight": 100,
        "nutrition": {
            "calories": 165,
            "protein": 31,
            "carbs": 0,
            "fat": 3.6,
            "fiber": 0,
        },
    },
    "broccoli": {
        "category": "vegetable",
        "serving_size": "1 cup (91g)",
        "serving_weight": 91,
        "nutrition": {
            "calories": 31,
            "protein": 2.6,
            "carbs": 6,
            "fat": 0.3,
            "fiber": 2.4,
            "sugar": 1.5,
        },
    },
    "rice": {
        "category": "grain",
        "serving_size": "1 cup cooked (158g)",
        "serving_weight": 158,
        "nutrition": {
            "calories": 205,
            "protein": 4.3,
            "carbs": 45,
            "fat": 0.4,
            "fiber": 0.6,
        },
    },
    "salmon": {
        "category": "protein",
        "serving_size": "100g (cooked)",
        "serving_weight": 100,
        "nutrition": {
            "calories": 206,
            "protein": 22,
            "carbs": 0,
            "fat": 13,
            "fiber": 0,
        },
    },
}


class FoodAnalysisService:
    """Service for food classification and nutrition estimation"""

    def __init__(self):
        self.model = None  # Placeholder for ML model
        self.model_name = "mock-food-classifier-v1"
        logger.info(f"Initialized FoodAnalysisService with model: {self.model_name}")

    async def analyze_food(
        self,
        image: Image.Image,
        dimensions: Optional[DimensionsInput] = None,
    ) -> Tuple[List[FoodItem], str, float]:
        """
        Analyze food image and return classification + nutrition estimates.

        Args:
            image: PIL Image of the food
            dimensions: Optional AR measurements (width, height, depth in cm)

        Returns:
            Tuple of (food_items, measurement_quality, processing_time_ms)
        """
        start_time = time.time()

        try:
            # 1. Preprocess image
            processed_image = self._preprocess_image(image)

            # 2. Classify food (mock implementation)
            food_class, confidence, alternatives = await self._classify_food(
                processed_image
            )

            # 3. Estimate portion size
            if dimensions:
                portion_weight = self._estimate_portion_from_dimensions(
                    dimensions, food_class
                )
                measurement_quality = self._assess_measurement_quality(dimensions)
            else:
                # Default portion size if no measurements
                portion_weight = NUTRITION_DATABASE[food_class]["serving_weight"]
                measurement_quality = "low"

            # 4. Calculate nutrition
            nutrition_info = self._calculate_nutrition(food_class, portion_weight)

            # 5. Create food item
            food_item = FoodItem(
                name=food_class.title(),
                confidence=confidence,
                portion_size=f"{int(portion_weight)}g",
                portion_weight=portion_weight,
                nutrition=nutrition_info,
                category=NUTRITION_DATABASE[food_class]["category"],
                alternatives=alternatives,
            )

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            return [food_item], measurement_quality, processing_time

        except Exception as e:
            logger.error(f"Food analysis error: {str(e)}", exc_info=True)
            raise

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model inference.

        Args:
            image: PIL Image

        Returns:
            Preprocessed numpy array
        """
        # Resize to model input size (e.g., 224x224 for ResNet)
        image = image.resize((224, 224))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        return img_array

    async def _classify_food(
        self, image: np.ndarray
    ) -> Tuple[str, float, List[FoodItemAlternative]]:
        """
        Classify food item (mock implementation).

        In production, this would use a trained CNN model (ResNet, EfficientNet, etc.)

        Args:
            image: Preprocessed image array

        Returns:
            Tuple of (food_class, confidence, alternatives)
        """
        # Mock classification - randomly select from database
        # In production: model.predict(image)
        food_classes = list(NUTRITION_DATABASE.keys())
        primary_class = np.random.choice(food_classes)
        confidence = np.random.uniform(0.75, 0.95)

        # Generate mock alternatives
        alternatives = []
        remaining_classes = [c for c in food_classes if c != primary_class]
        for alt_class in remaining_classes[:2]:  # Top 2 alternatives
            alternatives.append(
                FoodItemAlternative(
                    name=alt_class.title(),
                    confidence=round(np.random.uniform(0.4, 0.7), 2),
                )
            )

        logger.info(
            f"Classified as: {primary_class} (confidence: {confidence:.2f})"
        )
        return primary_class, round(confidence, 2), alternatives

    def _estimate_portion_from_dimensions(
        self, dimensions: DimensionsInput, food_class: str
    ) -> float:
        """
        Estimate portion weight from AR measurements.

        Args:
            dimensions: Width, height, depth in cm
            food_class: Food category

        Returns:
            Estimated weight in grams
        """
        # Calculate volume in cm³
        volume_cm3 = dimensions.width * dimensions.height * dimensions.depth

        # Food density estimates (g/cm³)
        # These are rough estimates - in production, use a proper database
        density_map = {
            "apple": 0.7,  # Slightly less dense than water
            "banana": 0.9,
            "chicken breast": 1.0,
            "broccoli": 0.3,  # Very light
            "rice": 0.8,  # Cooked rice
            "salmon": 1.0,
        }

        density = density_map.get(food_class, 0.8)

        # Shape correction factor (food items are not perfect cuboids)
        shape_factor = 0.7  # Assume 70% fill of the bounding box

        # Calculate weight
        weight_grams = volume_cm3 * density * shape_factor

        logger.info(
            f"Estimated portion: {weight_grams:.1f}g "
            f"(volume: {volume_cm3:.1f}cm³, density: {density}g/cm³)"
        )

        return round(weight_grams, 1)

    def _calculate_nutrition(
        self, food_class: str, portion_weight: float
    ) -> NutritionInfo:
        """
        Calculate nutrition info based on food class and portion size.

        Args:
            food_class: Food category
            portion_weight: Weight in grams

        Returns:
            NutritionInfo object with scaled values
        """
        base_data = NUTRITION_DATABASE[food_class]
        base_weight = base_data["serving_weight"]
        base_nutrition = base_data["nutrition"]

        # Scale factor
        scale = portion_weight / base_weight

        # Scale all nutrition values
        scaled_nutrition = NutritionInfo(
            calories=round(base_nutrition["calories"] * scale, 1),
            protein=round(base_nutrition["protein"] * scale, 1),
            carbs=round(base_nutrition["carbs"] * scale, 1),
            fat=round(base_nutrition["fat"] * scale, 1),
            fiber=round(base_nutrition.get("fiber", 0) * scale, 1)
            if "fiber" in base_nutrition
            else None,
            sugar=round(base_nutrition.get("sugar", 0) * scale, 1)
            if "sugar" in base_nutrition
            else None,
        )

        return scaled_nutrition

    def _assess_measurement_quality(self, dimensions: DimensionsInput) -> str:
        """
        Assess quality of AR measurements.

        Args:
            dimensions: AR measurements

        Returns:
            Quality rating: "high", "medium", or "low"
        """
        # Simple heuristic based on dimension ratios
        # In production, use AR confidence scores and plane detection quality

        max_dim = max(dimensions.width, dimensions.height, dimensions.depth)
        min_dim = min(dimensions.width, dimensions.height, dimensions.depth)
        ratio = max_dim / min_dim if min_dim > 0 else 0

        # If dimensions are very disproportionate, quality is lower
        if ratio > 10:
            return "low"
        elif ratio > 5:
            return "medium"
        else:
            return "high"

    async def search_nutrition_db(self, query: str) -> List[dict]:
        """
        Search nutrition database by food name.

        Args:
            query: Search query

        Returns:
            List of matching nutrition database entries
        """
        results = []
        query_lower = query.lower()

        for food_name, data in NUTRITION_DATABASE.items():
            if query_lower in food_name:
                results.append(
                    {
                        "food_name": food_name.title(),
                        "category": data["category"],
                        "serving_size": data["serving_size"],
                        "serving_weight": data["serving_weight"],
                        "nutrition": data["nutrition"],
                    }
                )

        return results

    def get_model_info(self) -> dict:
        """
        Get information about the active food classification model.

        Returns:
            Dict with model metadata
        """
        return {
            "name": self.model_name,
            "version": "1.0.0",
            "accuracy": 0.82,  # Mock accuracy
            "num_classes": len(NUTRITION_DATABASE),
            "description": "Mock food classifier for MVP testing. "
            "In production, replace with trained CNN model (ResNet/EfficientNet).",
        }


# Singleton instance
food_analysis_service = FoodAnalysisService()
