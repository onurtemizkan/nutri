"""
Food Analysis Service
Handles food classification, portion estimation, and nutrition calculation.
"""
import time
import json
import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
from PIL import Image

from app.schemas.food_analysis import (
    FoodItem,
    FoodItemAlternative,
    NutritionInfo,
    DimensionsInput,
)

# Try HuggingFace model first (better accuracy), fallback to torchvision
try:
    from app.ml_models.food_classifier_hf import (
        HuggingFaceFoodClassifier,
        HFClassifierConfig,
        get_hf_food_classifier,
        format_hf_class_name as format_class_name,
        HF_AVAILABLE,
    )
except ImportError:
    HF_AVAILABLE = False
    format_class_name = lambda x: x.replace("_", " ").title()

# Optional fallback to torchvision model (if available)
TORCHVISION_AVAILABLE = False
try:
    from app.ml_models.food_classifier import (
        FoodClassifier,
        FoodClassifierConfig,
        get_food_classifier,
    )
    TORCHVISION_AVAILABLE = True
except ImportError:
    # Torchvision classifier not available, HuggingFace only
    FoodClassifier = None
    FoodClassifierConfig = None
    get_food_classifier = None

logger = logging.getLogger(__name__)

# Path to nutrition database
NUTRITION_DB_PATH = Path(__file__).parent.parent / "data" / "nutrition_database.json"


class NutritionDatabase:
    """
    Singleton class for managing the nutrition database.

    Loads nutrition data from JSON file and provides lookup functionality.
    """

    _instance: Optional["NutritionDatabase"] = None
    _data: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "NutritionDatabase":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_database()
        return cls._instance

    def _load_database(self) -> None:
        """Load the nutrition database from JSON file."""
        try:
            if NUTRITION_DB_PATH.exists():
                with open(NUTRITION_DB_PATH, "r") as f:
                    data = json.load(f)
                    self._data = data.get("foods", {})
                    logger.info(
                        f"Loaded nutrition database with {len(self._data)} foods"
                    )
            else:
                logger.warning(
                    f"Nutrition database not found at {NUTRITION_DB_PATH}, "
                    "using fallback data"
                )
                self._data = self._get_fallback_data()
        except Exception as e:
            logger.error(f"Failed to load nutrition database: {e}")
            self._data = self._get_fallback_data()

    def _get_fallback_data(self) -> Dict[str, Any]:
        """Return minimal fallback nutrition data."""
        return {
            "unknown": {
                "category": "unknown",
                "serving_size": "100g",
                "serving_weight": 100,
                "density": 0.8,
                "nutrition": {
                    "calories": 150,
                    "protein": 5,
                    "carbs": 20,
                    "fat": 5,
                    "fiber": 2,
                    "sugar": 5,
                },
            }
        }

    def get(self, food_class: str) -> Optional[Dict[str, Any]]:
        """
        Get nutrition data for a food class.

        Args:
            food_class: Food class name (snake_case or formatted)

        Returns:
            Nutrition data dict or None if not found
        """
        if self._data is None:
            return None

        # Normalize the food class name
        normalized = food_class.lower().replace(" ", "_")

        return self._data.get(normalized)

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search nutrition database by food name.

        Args:
            query: Search query

        Returns:
            List of matching entries
        """
        if self._data is None:
            return []

        results = []
        query_lower = query.lower()

        for food_name, data in self._data.items():
            if query_lower in food_name.lower():
                results.append({
                    "food_name": format_class_name(food_name),
                    "category": data.get("category", "unknown"),
                    "serving_size": data.get("serving_size", "100g"),
                    "serving_weight": data.get("serving_weight", 100),
                    "nutrition": data.get("nutrition", {}),
                })

        return results

    def get_density(self, food_class: str) -> float:
        """Get density estimate for a food class (g/cm³)."""
        data = self.get(food_class)
        if data:
            return data.get("density", 0.8)
        return 0.8  # Default density

    @property
    def food_count(self) -> int:
        """Return number of foods in database."""
        return len(self._data) if self._data else 0


class FoodAnalysisService:
    """Service for food classification and nutrition estimation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_huggingface: bool = True,
        hf_model_name: Optional[str] = None,
    ):
        """
        Initialize the food analysis service.

        Args:
            model_path: Optional path to fine-tuned model weights (torchvision only)
            use_huggingface: Whether to use HuggingFace model (default: True)
            hf_model_name: HuggingFace model name (default: efficientnet_b1-food101)
        """
        # Initialize nutrition database
        self.nutrition_db = NutritionDatabase()
        self.classifier = None
        self.model_loaded = False
        self.using_huggingface = False

        # Try HuggingFace model first (better accuracy)
        if use_huggingface and HF_AVAILABLE:
            try:
                hf_config = HFClassifierConfig(
                    model_name=hf_model_name or "AventIQ-AI/Food-Classification-AI-Model",
                    version="1.0.0",
                )
                self.classifier = get_hf_food_classifier(hf_config)
                self.model_loaded = True
                self.using_huggingface = True
                logger.info(
                    f"Initialized FoodAnalysisService with HuggingFace model: "
                    f"{hf_config.model_name}"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace model: {e}, falling back")

        # Fallback to torchvision model (if available)
        if TORCHVISION_AVAILABLE and FoodClassifierConfig is not None:
            config = FoodClassifierConfig(
                model_type="efficientnet_b0",
                pretrained=True,
                model_path=model_path,
                version="1.0.0",
            )

            try:
                self.classifier = get_food_classifier(config)
                self.model_loaded = True
                logger.info(
                    f"Initialized FoodAnalysisService with torchvision {config.model_type}, "
                    f"version={config.version}"
                )
            except Exception as e:
                logger.error(f"Failed to load classifier: {e}")
                self.classifier = None
                self.model_loaded = False
        else:
            logger.warning("No classifier available (HuggingFace failed, torchvision not installed)")
            self.classifier = None
            self.model_loaded = False

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
            # 1. Classify food using ML model
            if self.classifier is not None:
                predictions = self.classifier.predict(image, top_k=5)
            else:
                # Fallback to mock if model not loaded
                predictions = self._mock_classify()

            if not predictions:
                raise ValueError("No predictions returned from classifier")

            # Get primary prediction
            food_class, confidence = predictions[0]

            # 2. Get alternatives (skip the primary)
            alternatives = []
            for alt_class, alt_conf in predictions[1:4]:  # Top 3 alternatives
                alternatives.append(
                    FoodItemAlternative(
                        name=format_class_name(alt_class),
                        confidence=round(alt_conf, 2),
                    )
                )

            # 3. Get nutrition data
            nutrition_data = self.nutrition_db.get(food_class)
            if nutrition_data is None:
                logger.warning(f"No nutrition data for {food_class}, using fallback")
                nutrition_data = self.nutrition_db.get("unknown") or {
                    "category": "unknown",
                    "serving_weight": 100,
                    "nutrition": {
                        "calories": 150,
                        "protein": 5,
                        "carbs": 20,
                        "fat": 5,
                    },
                }

            # 4. Estimate portion size
            if dimensions:
                portion_weight = self._estimate_portion_from_dimensions(
                    dimensions, food_class
                )
                measurement_quality = self._assess_measurement_quality(dimensions)
            else:
                # Default portion size if no measurements
                portion_weight = float(nutrition_data.get("serving_weight", 100))
                measurement_quality = "low"

            # 5. Calculate scaled nutrition
            nutrition_info = self._calculate_nutrition(
                nutrition_data, portion_weight
            )

            # 6. Create food item
            food_item = FoodItem(
                name=format_class_name(food_class),
                confidence=round(confidence, 2),
                portion_size=f"{int(portion_weight)}g",
                portion_weight=portion_weight,
                nutrition=nutrition_info,
                category=nutrition_data.get("category", "unknown"),
                alternatives=alternatives,
            )

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            logger.info(
                f"Analyzed food: {food_class} (confidence: {confidence:.2f}), "
                f"processing_time: {processing_time:.1f}ms"
            )

            return [food_item], measurement_quality, processing_time

        except Exception as e:
            logger.error(f"Food analysis error: {str(e)}", exc_info=True)
            raise

    def _mock_classify(self) -> List[Tuple[str, float]]:
        """Fallback mock classification when model is not available."""
        mock_foods = ["pizza", "hamburger", "sushi", "salad", "pasta"]
        return [
            (food, round(np.random.uniform(0.3, 0.9), 2))
            for food in mock_foods
        ]

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

        # Get food-specific density from database
        density = self.nutrition_db.get_density(food_class)

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
        self, nutrition_data: Dict[str, Any], portion_weight: float
    ) -> NutritionInfo:
        """
        Calculate nutrition info based on food data and portion size.

        Args:
            nutrition_data: Nutrition database entry
            portion_weight: Weight in grams

        Returns:
            NutritionInfo object with scaled values
        """
        base_weight = nutrition_data.get("serving_weight", 100)
        base_nutrition = nutrition_data.get("nutrition", {})

        # Scale factor
        scale = portion_weight / base_weight if base_weight > 0 else 1.0

        # Scale all nutrition values
        scaled_nutrition = NutritionInfo(
            calories=round(base_nutrition.get("calories", 0) * scale, 1),
            protein=round(base_nutrition.get("protein", 0) * scale, 1),
            carbs=round(base_nutrition.get("carbs", 0) * scale, 1),
            fat=round(base_nutrition.get("fat", 0) * scale, 1),
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

    async def search_nutrition_db(self, query: str) -> List[Dict[str, Any]]:
        """
        Search nutrition database by food name.

        Args:
            query: Search query

        Returns:
            List of matching nutrition database entries
        """
        return self.nutrition_db.search(query)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the active food classification model.

        Returns:
            Dict with model metadata
        """
        if self.classifier is not None:
            model_info = self.classifier.get_model_info()
            model_info["nutrition_db_foods"] = self.nutrition_db.food_count
            model_info["using_huggingface"] = self.using_huggingface
            return model_info
        else:
            return {
                "name": "fallback-mock",
                "version": "0.0.0",
                "model_loaded": False,
                "using_huggingface": False,
                "nutrition_db_foods": self.nutrition_db.food_count,
                "description": "Mock classifier - real model failed to load",
            }


# Singleton instance
_service_instance: Optional[FoodAnalysisService] = None


def get_food_analysis_service(
    model_path: Optional[str] = None,
    use_huggingface: bool = True,
    hf_model_name: Optional[str] = None,
    force_reload: bool = False
) -> FoodAnalysisService:
    """
    Get or create the singleton FoodAnalysisService instance.

    Args:
        model_path: Optional path to fine-tuned model weights (torchvision only)
        use_huggingface: Whether to use HuggingFace model (default: True)
        hf_model_name: HuggingFace model name (default: efficientnet_b1-food101)
        force_reload: Force recreation of the service

    Returns:
        FoodAnalysisService instance
    """
    global _service_instance

    if _service_instance is None or force_reload:
        _service_instance = FoodAnalysisService(
            model_path=model_path,
            use_huggingface=use_huggingface,
            hf_model_name=hf_model_name,
        )

    return _service_instance


# Lazy initialization - don't load model at import time
def _get_lazy_service() -> FoodAnalysisService:
    """Get the service instance lazily."""
    return get_food_analysis_service()


# Legacy alias for backwards compatibility (lazy loaded)
class _LazyService:
    """Lazy-loading wrapper for backwards compatibility."""
    _instance: Optional[FoodAnalysisService] = None

    def __getattr__(self, name: str) -> Any:
        if self._instance is None:
            self._instance = get_food_analysis_service()
        return getattr(self._instance, name)


food_analysis_service = _LazyService()
