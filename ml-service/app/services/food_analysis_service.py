"""
Food Analysis Service
Handles food classification, portion estimation, and nutrition calculation.

Uses state-of-the-art Vision Transformer (ViT) model for food classification.
Supports multi-food detection using OWL-ViT for detecting multiple foods on a plate.
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
from app.ml_models.food_classifier import get_food_classifier, FoodClassifier
from app.ml_models.ensemble_classifier import (
    get_ensemble_classifier,
    EnsembleFoodClassifier,
    EnsembleResult,
)
from app.ml_models.food_detector import (
    get_food_detector,
    FoodDetector,
    DetectionResult,
    DetectedRegion,
)

logger = logging.getLogger(__name__)


class FoodAnalysisService:
    """Service for food classification and nutrition estimation"""

    # Multi-food detection settings
    MULTI_FOOD_MIN_REGIONS = 1  # Minimum regions to trigger multi-food mode
    MULTI_FOOD_CONFIDENCE_THRESHOLD = 0.15  # Minimum detection confidence

    def __init__(self, use_ensemble: bool = True, enable_multi_food: bool = True):
        self._food_classifier: Optional[FoodClassifier] = None
        self._ensemble_classifier: Optional[EnsembleFoodClassifier] = None
        self._food_detector: Optional[FoodDetector] = None
        self._use_ensemble = use_ensemble
        self._enable_multi_food = enable_multi_food
        self.model_name = (
            "Ensemble Food Classifier"
            if use_ensemble
            else "nateraw/food (ViT-Food-101)"
        )
        self._food_classes = list(FOOD_DATABASE.keys())
        logger.info(
            f"Initialized FoodAnalysisService with {'ensemble' if use_ensemble else 'single'} model, "
            f"multi-food detection: {enable_multi_food}, "
            f"{len(self._food_classes)} food classes in database"
        )

    def _get_classifier(self) -> FoodClassifier:
        """Get the single food classifier (lazy loading)."""
        if self._food_classifier is None:
            self._food_classifier = get_food_classifier()
        return self._food_classifier

    def _get_ensemble_classifier(self) -> EnsembleFoodClassifier:
        """Get the ensemble classifier (lazy loading)."""
        if self._ensemble_classifier is None:
            self._ensemble_classifier = get_ensemble_classifier()
        return self._ensemble_classifier

    def _get_food_detector(self) -> FoodDetector:
        """Get the food detector (lazy loading)."""
        if self._food_detector is None:
            self._food_detector = get_food_detector()
        return self._food_detector

    async def analyze_food(
        self,
        image: Image.Image,
        dimensions: Optional[DimensionsInput] = None,
        cooking_method: Optional[str] = None,
        enable_multi_food: Optional[bool] = None,
    ) -> Tuple[List[FoodItem], str, float, List[str]]:
        """
        Analyze food image and return classification + nutrition estimates.

        Supports detecting and classifying multiple foods in a single image
        when multi-food detection is enabled.

        Args:
            image: PIL Image of the food
            dimensions: Optional AR measurements (width, height, depth in cm)
            cooking_method: Optional cooking method hint
            enable_multi_food: Override instance-level multi-food setting

        Returns:
            Tuple of (food_items, measurement_quality, processing_time_ms, suggestions)
        """
        start_time = time.time()
        suggestions = []
        use_multi_food = (
            enable_multi_food
            if enable_multi_food is not None
            else self._enable_multi_food
        )

        try:
            # Step 1: Detect food regions if multi-food is enabled
            detected_regions: List[DetectedRegion] = []
            if use_multi_food:
                try:
                    detector = self._get_food_detector()
                    detection_result: DetectionResult = detector.detect(
                        image, confidence_threshold=self.MULTI_FOOD_CONFIDENCE_THRESHOLD
                    )
                    detected_regions = detection_result.regions
                    logger.info(
                        f"Multi-food detection found {len(detected_regions)} food regions"
                    )

                    if len(detected_regions) > 1:
                        suggestions.append(
                            f"Detected {len(detected_regions)} food items in image"
                        )
                except Exception as e:
                    logger.warning(
                        f"Multi-food detection failed, falling back to single: {e}"
                    )
                    detected_regions = []

            # Step 2: Process foods (multi or single)
            food_items = []

            if len(detected_regions) > 1:
                # Multi-food mode: classify each detected region
                food_items = await self._analyze_multi_food(
                    detected_regions, dimensions, cooking_method, suggestions
                )
            elif len(detected_regions) == 1:
                # Single food detected by OWL-ViT: use the hint for better classification
                region = detected_regions[0]
                logger.info(
                    f"Single food mode with OWL-ViT hint: '{region.query_label}'"
                )
                food_item = await self._analyze_single_food(
                    image,
                    dimensions,
                    cooking_method,
                    suggestions,
                    query_hint=region.query_label,
                )
                food_items = [food_item]
            else:
                # No OWL-ViT detection: classify the whole image without hint
                food_item = await self._analyze_single_food(
                    image, dimensions, cooking_method, suggestions
                )
                food_items = [food_item]

            # Assess overall measurement quality
            measurement_quality = (
                self._assess_measurement_quality(dimensions) if dimensions else "low"
            )

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            logger.info(
                f"Food analysis complete: {len(food_items)} items, "
                f"{processing_time:.1f}ms, quality: {measurement_quality}"
            )

            return food_items, measurement_quality, processing_time, suggestions

        except Exception as e:
            logger.error(f"Food analysis error: {str(e)}", exc_info=True)
            raise

    async def _analyze_multi_food(
        self,
        regions: List[DetectedRegion],
        dimensions: Optional[DimensionsInput],
        cooking_method: Optional[str],
        suggestions: List[str],
    ) -> List[FoodItem]:
        """
        Analyze multiple food regions detected in an image.

        Args:
            regions: List of detected food regions with cropped images
            dimensions: Optional AR measurements (applied proportionally)
            cooking_method: Optional cooking method hint
            suggestions: List to append suggestions to

        Returns:
            List of FoodItem objects for each detected food
        """
        food_items = []

        for i, region in enumerate(regions):
            if region.cropped_image is None:
                continue

            try:
                # Classify the cropped region with OWL-ViT query hint
                food_class, confidence, alternatives = await self._classify_food(
                    region.cropped_image, query_hint=region.query_label
                )

                food_entry = get_food_entry(food_class)
                cooking_method_enum = self._parse_cooking_method(
                    cooking_method, food_entry
                )

                # Estimate portion for this region
                if dimensions:
                    # Calculate proportional dimensions based on region size
                    region_dimensions = self._scale_dimensions_for_region(
                        dimensions, region
                    )
                    portion_result = self._estimate_portion_from_dimensions(
                        region_dimensions, food_class, cooking_method_enum
                    )
                    portion_weight = portion_result["weight"]
                else:
                    # Use default serving size
                    portion_weight = food_entry.serving_weight if food_entry else 100

                # Calculate nutrition
                nutrition_info = self._calculate_nutrition(
                    food_class, portion_weight, cooking_method_enum
                )

                # Create food item
                portion_size = self._format_portion_size(portion_weight, food_entry)
                category = food_entry.category.value if food_entry else "unknown"

                food_item = FoodItem(
                    name=self._format_food_name(food_class, cooking_method_enum),
                    confidence=round(
                        confidence * region.confidence, 2
                    ),  # Combine confidences
                    portion_size=portion_size,
                    portion_weight=portion_weight,
                    nutrition=nutrition_info,
                    category=category,
                    alternatives=alternatives,
                )

                food_items.append(food_item)

                logger.info(
                    f"Region {i+1}: {food_class} ({confidence:.2f}), "
                    f"bbox: {region.bbox}, query: {region.query_label}"
                )

            except Exception as e:
                logger.warning(f"Failed to classify region {i+1}: {e}")
                continue

        # Add suggestion if low confidence items
        low_conf_items = [f for f in food_items if f.confidence < 0.5]
        if low_conf_items:
            suggestions.append(
                f"{len(low_conf_items)} item(s) have low confidence - verify manually"
            )

        return food_items

    def _scale_dimensions_for_region(
        self,
        dimensions: DimensionsInput,
        region: DetectedRegion,
    ) -> DimensionsInput:
        """
        Scale dimensions proportionally for a detected region.

        Args:
            dimensions: Original full-image dimensions
            region: Detected region with bounding box

        Returns:
            Scaled dimensions for the region
        """
        # Calculate region size relative to original image
        bbox = region.bbox
        region_width = bbox[2] - bbox[0]
        region_height = bbox[3] - bbox[1]

        # Assume the cropped image came from the original
        # Scale dimensions proportionally
        if region.cropped_image:
            original_width = region.cropped_image.width + (bbox[0] * 2)  # Approximate
            original_height = region.cropped_image.height + (bbox[1] * 2)

            width_scale = region_width / max(original_width, 1)
            height_scale = region_height / max(original_height, 1)
            avg_scale = (width_scale + height_scale) / 2
        else:
            avg_scale = 0.5  # Default to half if we can't calculate

        return DimensionsInput(
            width=dimensions.width * avg_scale,
            height=dimensions.height * avg_scale,
            depth=dimensions.depth * avg_scale,
        )

    async def _analyze_single_food(
        self,
        image: Image.Image,
        dimensions: Optional[DimensionsInput],
        cooking_method: Optional[str],
        suggestions: List[str],
        query_hint: Optional[str] = None,
    ) -> FoodItem:
        """
        Analyze a single food item (original single-food logic).

        Args:
            image: PIL Image of the food
            dimensions: Optional AR measurements
            cooking_method: Optional cooking method hint
            suggestions: List to append suggestions to
            query_hint: Optional OWL-ViT query label for better classification

        Returns:
            Single FoodItem object
        """
        # Classify food using ensemble (with query_hint if available)
        food_class, confidence, alternatives = await self._classify_food(
            image, query_hint=query_hint
        )

        food_entry = get_food_entry(food_class)
        cooking_method_enum = self._parse_cooking_method(cooking_method, food_entry)

        # Estimate portion size
        if dimensions:
            portion_result = self._estimate_portion_from_dimensions(
                dimensions, food_class, cooking_method_enum
            )
            portion_weight = portion_result["weight"]
            _measurement_quality = self._assess_measurement_quality(
                dimensions
            )  # noqa: F841

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
            portion_weight = food_entry.serving_weight if food_entry else 100
            _measurement_quality = "low"  # noqa: F841
            suggestions.append("No measurements provided - using standard serving size")

        # Calculate nutrition
        nutrition_info = self._calculate_nutrition(
            food_class, portion_weight, cooking_method_enum
        )

        # Format outputs
        portion_size = self._format_portion_size(portion_weight, food_entry)
        category = food_entry.category.value if food_entry else "unknown"

        # Add confidence warning
        if confidence < 0.7:
            suggestions.append(
                "Low classification confidence - consider verifying the food type"
            )

        return FoodItem(
            name=self._format_food_name(food_class, cooking_method_enum),
            confidence=confidence,
            portion_size=portion_size,
            portion_weight=portion_weight,
            nutrition=nutrition_info,
            category=category,
            alternatives=alternatives,
        )

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
        self, image: Image.Image, query_hint: Optional[str] = None
    ) -> Tuple[str, float, List[FoodItemAlternative]]:
        """
        Classify food item using ensemble of specialized models.

        Uses:
        - Primary: nateraw/food (ViT fine-tuned on Food-101)
        - Fallback: Fruit/Vegetable classifier when confidence is low
        - Fallback: Ingredient classifier for nuts, grains, etc.

        Args:
            image: PIL Image to classify
            query_hint: Optional OWL-ViT query label for better classification hints

        Returns:
            Tuple of (food_class, confidence, alternatives)
        """
        try:
            if self._use_ensemble:
                # Use ensemble classifier with fallback models
                ensemble = self._get_ensemble_classifier()
                result: EnsembleResult = ensemble.classify(
                    image=image,
                    database_keys=self._food_classes,
                    top_k=3,
                    query_hint=query_hint,
                )

                # Build alternatives list from ensemble result
                alternatives = []
                for alt_class, alt_conf in result.alternatives:
                    alt_entry = get_food_entry(alt_class)
                    display_name = (
                        alt_entry.display_name
                        if alt_entry
                        else alt_class.replace("_", " ").title()
                    )
                    alternatives.append(
                        FoodItemAlternative(
                            name=display_name,
                            confidence=round(alt_conf, 2),
                        )
                    )

                logger.info(
                    f"Ensemble classified as: {result.primary_class} "
                    f"(confidence: {result.confidence:.2f}, "
                    f"models: {', '.join(result.contributing_models)})"
                )
                return result.primary_class, round(result.confidence, 2), alternatives

            else:
                # Single model classification (Food-101 only)
                classifier = self._get_classifier()
                (
                    primary_class,
                    confidence,
                    alt_predictions,
                ) = classifier.classify_with_database_mapping(
                    image=image, database_keys=self._food_classes, top_k=3
                )

                alternatives = []
                for alt_class, alt_conf in alt_predictions:
                    alt_entry = get_food_entry(alt_class)
                    display_name = (
                        alt_entry.display_name
                        if alt_entry
                        else alt_class.replace("_", " ").title()
                    )
                    alternatives.append(
                        FoodItemAlternative(
                            name=display_name,
                            confidence=round(alt_conf, 2),
                        )
                    )

                logger.info(
                    f"ViT classified as: {primary_class} (confidence: {confidence:.2f})"
                )
                return primary_class, round(confidence, 2), alternatives

        except Exception as e:
            logger.error(f"Classification error: {e}, falling back to default")
            # Fallback if all models fail
            fallback_class = self._food_classes[0] if self._food_classes else "unknown"
            return fallback_class, 0.5, []

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
        Get information about the active food classification model(s).

        Returns:
            Dict with model metadata
        """
        # Get detector info if multi-food is enabled
        detector_info = {}
        if self._enable_multi_food:
            try:
                detector = self._get_food_detector()
                detector_info = detector.get_model_info()
            except Exception:
                detector_info = {"detector": "not loaded"}

        if self._use_ensemble:
            # Get ensemble info
            try:
                ensemble = self._get_ensemble_classifier()
                ensemble_info = ensemble.get_model_info()
            except Exception:
                ensemble_info = {}

            return {
                "name": self.model_name,
                "version": "5.0.0",  # Bumped for multi-food support
                "accuracy": 0.92,
                "num_classes": len(self._food_classes),
                "categories": list(
                    set(e.category.value for e in FOOD_DATABASE.values())
                ),
                "model_type": "Ensemble (Multi-Model) + Multi-Food Detection",
                "strategy": "OWL-ViT detection + cascading ensemble classification",
                "multi_food_detection": self._enable_multi_food,
                "description": (
                    "Advanced food analysis pipeline with multi-food detection using OWL-ViT "
                    "and ensemble classification combining Food-101 ViT with specialized classifiers "
                    "for fruits, vegetables, nuts, and raw ingredients. Can detect and classify "
                    "multiple distinct food items in a single image."
                ),
                "detector": detector_info,
                **ensemble_info,
            }
        else:
            # Single model info
            classifier_info = {}
            try:
                classifier = self._get_classifier()
                classifier_info = classifier.get_model_info()
            except Exception:
                pass

            return {
                "name": self.model_name,
                "version": "3.0.0",
                "accuracy": 0.90,
                "num_classes": len(self._food_classes),
                "food_101_classes": 101,
                "categories": list(
                    set(e.category.value for e in FOOD_DATABASE.values())
                ),
                "model_type": "Vision Transformer (ViT)",
                "dataset": "Food-101",
                "multi_food_detection": self._enable_multi_food,
                "description": (
                    "State-of-the-art Vision Transformer (ViT) model fine-tuned on Food-101 dataset. "
                    "Achieves ~90% accuracy on 101 food categories."
                ),
                "detector": detector_info if self._enable_multi_food else None,
                **classifier_info,
            }

    def get_supported_cooking_methods(self) -> List[str]:
        """Get list of supported cooking methods."""
        return [method.value for method in CookingMethod]


# Singleton instance
food_analysis_service = FoodAnalysisService()
