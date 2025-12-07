"""
Food Analysis Service
Handles food classification, portion estimation, and nutrition calculation.
"""
import time
import logging
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image

from app.schemas.food_analysis import (
    FoodItem,
    FoodItemAlternative,
    NutritionInfo,
    DimensionsInput,
)
from app.data.food_database import (
    FOOD_DATABASE,
    CookingMethod,
    FoodEntry,
    get_food_entry,
    get_density,
    get_shape_factor,
    get_cooking_modifier,
    validate_portion,
    get_amino_acids,
    PORTION_VALIDATION,
)

logger = logging.getLogger(__name__)


class FoodAnalysisService:
    """Service for food classification and nutrition estimation"""

    def __init__(self):
        self.model = None  # Placeholder for ML model
        self.model_name = "food-classifier-v2"
        self._food_classes = list(FOOD_DATABASE.keys())
        logger.info(
            f"Initialized FoodAnalysisService with model: {self.model_name}, "
            f"{len(self._food_classes)} food classes"
        )

    async def analyze_food(
        self,
        image: Image.Image,
        dimensions: Optional[DimensionsInput] = None,
        cooking_method: Optional[str] = None,
    ) -> Tuple[List[FoodItem], str, float, List[str]]:
        """
        Analyze food image and return classification + nutrition estimates.

        Args:
            image: PIL Image of the food
            dimensions: Optional AR measurements (width, height, depth in cm)
            cooking_method: Optional cooking method hint

        Returns:
            Tuple of (food_items, measurement_quality, processing_time_ms, suggestions)
        """
        start_time = time.time()
        suggestions = []

        try:
            # 1. Preprocess image
            processed_image = self._preprocess_image(image)

            # 2. Classify food (mock implementation)
            food_class, confidence, alternatives = await self._classify_food(
                processed_image
            )

            # Get food entry for classification
            food_entry = get_food_entry(food_class)

            # 3. Parse cooking method
            cooking_method_enum = self._parse_cooking_method(cooking_method, food_entry)

            # 4. Estimate portion size with improved algorithm
            if dimensions:
                portion_result = self._estimate_portion_from_dimensions(
                    dimensions, food_class, cooking_method_enum
                )
                portion_weight = portion_result["weight"]
                measurement_quality = self._assess_measurement_quality(dimensions)

                # Validate portion and generate warnings
                validation = validate_portion(
                    portion_weight,
                    {
                        "width": dimensions.width,
                        "height": dimensions.height,
                        "depth": dimensions.depth,
                    },
                )
                if validation["warnings"]:
                    suggestions.extend(validation["warnings"])
            else:
                # Default portion size if no measurements
                if food_entry:
                    portion_weight = food_entry.serving_weight
                else:
                    portion_weight = 100  # Default to 100g
                measurement_quality = "low"
                portion_result = {"confidence": 0.4, "method": "default-serving"}
                suggestions.append(
                    "No measurements provided - using standard serving size"
                )

            # 5. Calculate nutrition with improved database
            nutrition_info = self._calculate_nutrition(
                food_class, portion_weight, cooking_method_enum
            )

            # 6. Format portion size string
            portion_size = self._format_portion_size(portion_weight, food_entry)

            # 7. Create food item
            category = food_entry.category.value if food_entry else "unknown"
            food_item = FoodItem(
                name=self._format_food_name(food_class, cooking_method_enum),
                confidence=confidence,
                portion_size=portion_size,
                portion_weight=portion_weight,
                nutrition=nutrition_info,
                category=category,
                alternatives=alternatives,
            )

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            # Add helpful suggestions
            if confidence < 0.7:
                suggestions.append(
                    "Low classification confidence - consider verifying the food type"
                )
            if measurement_quality == "low":
                suggestions.append(
                    "Measurement quality is low - results may be less accurate"
                )

            return [food_item], measurement_quality, processing_time, suggestions

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
        primary_class = np.random.choice(self._food_classes)
        confidence = np.random.uniform(0.75, 0.95)

        # Generate mock alternatives
        alternatives = []
        remaining_classes = [c for c in self._food_classes if c != primary_class]
        selected_alternatives = np.random.choice(
            remaining_classes, size=min(2, len(remaining_classes)), replace=False
        )
        for alt_class in selected_alternatives:
            alt_entry = get_food_entry(alt_class)
            display_name = alt_entry.display_name if alt_entry else alt_class.title()
            alternatives.append(
                FoodItemAlternative(
                    name=display_name,
                    confidence=round(np.random.uniform(0.4, 0.7), 2),
                )
            )

        logger.info(f"Classified as: {primary_class} (confidence: {confidence:.2f})")
        return primary_class, round(confidence, 2), alternatives

    def _parse_cooking_method(
        self, cooking_method_str: Optional[str], food_entry: Optional[FoodEntry]
    ) -> CookingMethod:
        """
        Parse cooking method string to enum.

        Args:
            cooking_method_str: Cooking method string from request
            food_entry: Food entry for default cooking method

        Returns:
            CookingMethod enum
        """
        if cooking_method_str:
            try:
                return CookingMethod(cooking_method_str.lower())
            except ValueError:
                logger.warning(f"Unknown cooking method: {cooking_method_str}")

        # Use food's default cooking method if available
        if food_entry and food_entry.default_cooking_method:
            return food_entry.default_cooking_method

        return CookingMethod.RAW

    def _estimate_portion_from_dimensions(
        self,
        dimensions: DimensionsInput,
        food_class: str,
        cooking_method: CookingMethod,
    ) -> dict:
        """
        Estimate portion weight from AR measurements using food-specific data.

        Args:
            dimensions: Width, height, depth in cm
            food_class: Food category
            cooking_method: How the food was prepared

        Returns:
            Dict with weight estimate and metadata
        """
        # Calculate volume in cm³
        volume_cm3 = dimensions.width * dimensions.height * dimensions.depth

        # Get food entry for specific parameters
        food_entry = get_food_entry(food_class)

        # Get food-specific density and shape factor
        density = get_density(food_class)
        shape_factor = get_shape_factor(food_class)

        # Get cooking modifier
        cooking_mod = get_cooking_modifier(cooking_method)

        # Calculate adjusted volume (accounting for non-cuboid shape)
        adjusted_volume = volume_cm3 * shape_factor

        # Calculate raw weight
        raw_weight = adjusted_volume * density

        # Apply cooking modifier (e.g., moisture loss from grilling)
        final_weight = raw_weight * cooking_mod.weight_multiplier

        # Apply bounds
        final_weight = max(
            PORTION_VALIDATION["min_weight_g"],
            min(PORTION_VALIDATION["max_weight_g"], final_weight),
        )

        # Determine confidence based on method
        if food_entry:
            confidence = 0.8
            method = "density-lookup"
        else:
            confidence = 0.5
            method = "generic-default"

        logger.info(
            f"Estimated portion: {final_weight:.1f}g "
            f"(volume: {volume_cm3:.1f}cm³, density: {density}g/cm³, "
            f"shape_factor: {shape_factor}, cooking: {cooking_method.value})"
        )

        return {
            "weight": round(final_weight, 1),
            "confidence": confidence,
            "method": method,
            "density_used": density,
            "shape_factor_used": shape_factor,
            "cooking_method": cooking_method.value,
            "cooking_modifier": cooking_mod.weight_multiplier,
            "volume_raw": volume_cm3,
            "volume_adjusted": adjusted_volume,
        }

    def _calculate_nutrition(
        self,
        food_class: str,
        portion_weight: float,
        cooking_method: CookingMethod,
    ) -> NutritionInfo:
        """
        Calculate nutrition info based on food class, portion size, and cooking method.

        Args:
            food_class: Food category
            portion_weight: Weight in grams
            cooking_method: How the food was prepared

        Returns:
            NutritionInfo object with scaled values
        """
        food_entry = get_food_entry(food_class)

        if not food_entry:
            # Fallback to generic values
            logger.warning(f"No nutrition data for: {food_class}, using estimates")
            estimated_protein = portion_weight * 0.1
            # Estimate amino acids based on generic protein content
            amino_acids = get_amino_acids(food_class, estimated_protein)
            return NutritionInfo(
                calories=round(portion_weight * 1.5, 1),  # ~1.5 cal/g average
                protein=round(estimated_protein, 1),
                carbs=round(portion_weight * 0.2, 1),
                fat=round(portion_weight * 0.1, 1),
                lysine=amino_acids["lysine"],
                arginine=amino_acids["arginine"],
            )

        # Calculate scale factor based on portion vs serving weight
        scale = portion_weight / food_entry.serving_weight

        # Get cooking modifier for calorie adjustment
        cooking_mod = get_cooking_modifier(cooking_method)

        # Base nutrition values
        calories = food_entry.calories * scale * cooking_mod.calorie_multiplier
        protein = food_entry.protein * scale
        carbs = food_entry.carbs * scale
        fat = food_entry.fat * scale

        # If fried, add fat from oil absorption
        if cooking_method == CookingMethod.FRIED:
            # Estimate ~5g oil absorption per 100g of food
            oil_fat_added = (portion_weight / 100) * 5
            fat += oil_fat_added

        # Optional nutrients
        fiber = food_entry.fiber * scale if food_entry.fiber else None
        sugar = food_entry.sugar * scale if food_entry.sugar else None
        sodium = food_entry.sodium * scale if food_entry.sodium else None
        saturated_fat = (
            food_entry.saturated_fat * scale if food_entry.saturated_fat else None
        )

        # Amino acids - use explicit values if available, otherwise estimate
        amino_acids = get_amino_acids(food_class, protein, food_entry.category)

        return NutritionInfo(
            calories=round(calories, 1),
            protein=round(protein, 1),
            carbs=round(carbs, 1),
            fat=round(fat, 1),
            fiber=round(fiber, 1) if fiber is not None else None,
            sugar=round(sugar, 1) if sugar is not None else None,
            sodium=round(sodium, 1) if sodium is not None else None,
            saturated_fat=round(saturated_fat, 1)
            if saturated_fat is not None
            else None,
            lysine=amino_acids["lysine"],
            arginine=amino_acids["arginine"],
        )

    def _assess_measurement_quality(self, dimensions: DimensionsInput) -> str:
        """
        Assess quality of AR measurements with comprehensive heuristics.

        Args:
            dimensions: AR measurements

        Returns:
            Quality rating: "high", "medium", or "low"
        """
        dims = [dimensions.width, dimensions.height, dimensions.depth]
        max_dim = max(dims)
        min_dim = min(dims)
        ratio = max_dim / min_dim if min_dim > 0 else float("inf")

        # Factor 1: Dimension ratio (proportions)
        if ratio > 10:
            ratio_score = 1  # Very suspicious
        elif ratio > 5:
            ratio_score = 2  # Unusual
        else:
            ratio_score = 3  # Normal

        # Factor 2: Absolute size
        if max_dim > 50:  # > 50cm is unrealistic for food
            size_score = 1
        elif max_dim > 30:  # Unusually large
            size_score = 2
        elif max_dim < 1:  # Very small, hard to measure
            size_score = 1
        else:
            size_score = 3

        # Factor 3: Minimum dimension (very thin items are hard to measure)
        if min_dim < 0.5:
            min_score = 1
        elif min_dim < 1:
            min_score = 2
        else:
            min_score = 3

        # Combined score
        total_score = min(ratio_score, size_score, min_score)

        quality_map = {3: "high", 2: "medium", 1: "low"}
        return quality_map[total_score]

    def _format_portion_size(
        self, weight_g: float, food_entry: Optional[FoodEntry]
    ) -> str:
        """
        Format portion size for display.

        Args:
            weight_g: Weight in grams
            food_entry: Food entry for reference serving

        Returns:
            Formatted portion string
        """
        if food_entry:
            # Calculate how many servings this represents
            servings = weight_g / food_entry.serving_weight
            if 0.9 <= servings <= 1.1:
                return food_entry.serving_size
            elif servings < 1:
                return f"{int(weight_g)}g (~{servings:.1f} serving)"
            else:
                return f"{int(weight_g)}g (~{servings:.1f} servings)"
        else:
            return f"{int(weight_g)}g"

    def _format_food_name(self, food_class: str, cooking_method: CookingMethod) -> str:
        """
        Format food name with cooking method if applicable.

        Args:
            food_class: Food class key
            cooking_method: How it was prepared

        Returns:
            Formatted display name
        """
        food_entry = get_food_entry(food_class)
        base_name = food_entry.display_name if food_entry else food_class.title()

        # Add cooking method if not raw and not already in name
        if cooking_method != CookingMethod.RAW:
            method_str = cooking_method.value.title()
            if method_str.lower() not in base_name.lower():
                return f"{method_str} {base_name}"

        return base_name

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

        for key, entry in FOOD_DATABASE.items():
            # Check name match
            if query_lower in key or query_lower in entry.display_name.lower():
                results.append(self._food_entry_to_dict(entry))
                continue

            # Check aliases
            if entry.aliases:
                for alias in entry.aliases:
                    if query_lower in alias.lower():
                        results.append(self._food_entry_to_dict(entry))
                        break

        return results[:20]  # Limit results

    def _food_entry_to_dict(self, entry: FoodEntry) -> dict:
        """Convert FoodEntry to dict for API response."""
        return {
            "food_name": entry.display_name,
            "fdc_id": entry.fdc_id,
            "category": entry.category.value,
            "serving_size": entry.serving_size,
            "serving_weight": entry.serving_weight,
            "density": entry.density,
            "shape_factor": entry.shape_factor,
            "nutrition": {
                "calories": entry.calories,
                "protein": entry.protein,
                "carbs": entry.carbs,
                "fat": entry.fat,
                "fiber": entry.fiber,
                "sugar": entry.sugar,
                "sodium": entry.sodium,
                "saturated_fat": entry.saturated_fat,
            },
        }

    def get_model_info(self) -> dict:
        """
        Get information about the active food classification model.

        Returns:
            Dict with model metadata
        """
        return {
            "name": self.model_name,
            "version": "2.0.0",
            "accuracy": 0.85,
            "num_classes": len(self._food_classes),
            "categories": list(set(e.category.value for e in FOOD_DATABASE.values())),
            "description": (
                "Food classifier with comprehensive nutrition database. "
                "Supports food-specific density and shape factors for accurate "
                "portion estimation. Includes cooking method adjustments."
            ),
        }

    def get_supported_cooking_methods(self) -> List[str]:
        """Get list of supported cooking methods."""
        return [method.value for method in CookingMethod]


# Singleton instance
food_analysis_service = FoodAnalysisService()
