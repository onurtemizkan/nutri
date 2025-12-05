"""
Multi-Dish Analysis Service.

Combines object detection and classification to identify and classify
multiple food items in a single image, with bounding boxes for each.

Pipeline:
1. Detect food regions using object detector (OWL-ViT or YOLO)
2. Crop each detected region
3. Classify each crop using food classifier (HuggingFace)
4. Aggregate nutrition information
5. Return structured results with bounding boxes
"""
import time
import logging
from typing import List, Optional, Tuple

from PIL import Image

from app.ml_models.food_detector import (
    get_food_detector,
    FoodDetectorConfig,
    DetectionResult,
)
from app.ml_models.food_classifier_hf import (
    get_hf_food_classifier,
    HFClassifierConfig,
    format_hf_class_name,
    HF_AVAILABLE,
)
from app.services.food_analysis_service import NutritionDatabase
from app.schemas.multi_dish import (
    DetectedDish,
    BoundingBox,
    MultiDishAnalysisResponse,
)
from app.schemas.food_analysis import NutritionInfo, FoodItemAlternative

logger = logging.getLogger(__name__)


class MultiDishAnalysisService:
    """
    Service for detecting and classifying multiple food items in an image.

    Combines:
    - Object detection (OWL-ViT for zero-shot, or YOLO for speed)
    - Food classification (HuggingFace transformers)
    - Nutrition database lookup

    Each detected food item includes:
    - Bounding box (normalized and pixel coordinates)
    - Classification with confidence
    - Nutrition estimates
    - Alternative classifications
    """

    def __init__(
        self,
        detector_type: str = "owl_vit",
        classifier_model: Optional[str] = None,
        detection_confidence: float = 0.15,
        classification_confidence: float = 0.1,
    ):
        """
        Initialize multi-dish analysis service.

        Args:
            detector_type: "owl_vit" (zero-shot) or "yolov8" (fast)
            classifier_model: HuggingFace model name for classification
            detection_confidence: Minimum confidence for detections
            classification_confidence: Minimum confidence for classifications
        """
        self.detection_confidence = detection_confidence
        self.classification_confidence = classification_confidence

        # Initialize detector
        detector_config = FoodDetectorConfig(
            model_type=detector_type,
            confidence_threshold=detection_confidence,
        )
        self.detector = get_food_detector(detector_config)
        self.detector_type = detector_type

        # Initialize classifier
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers required for food classification")

        classifier_config = HFClassifierConfig(
            model_name=classifier_model or "AventIQ-AI/Food-Classification-AI-Model"
        )
        self.classifier = get_hf_food_classifier(classifier_config)

        # Initialize nutrition database
        self.nutrition_db = NutritionDatabase()

        logger.info(
            f"Initialized MultiDishAnalysisService: "
            f"detector={detector_type}, classifier={classifier_config.model_name}"
        )

    async def analyze(
        self,
        image: Image.Image,
        min_confidence: float = 0.15,
        max_dishes: int = 10,
        depth_map: Optional[List[List[float]]] = None,
    ) -> MultiDishAnalysisResponse:
        """
        Analyze image for multiple food items.

        Args:
            image: PIL Image to analyze
            min_confidence: Minimum confidence for detections
            max_dishes: Maximum number of dishes to return
            depth_map: Optional LIDAR depth data (future feature)

        Returns:
            MultiDishAnalysisResponse with all detected dishes
        """
        start_time = time.time()

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_width, img_height = image.size
        logger.info(f"Analyzing image: {img_width}x{img_height}")

        # Step 1: Detect food regions
        detection_start = time.time()
        detections = self.detector.detect(image)
        detection_time = (time.time() - detection_start) * 1000
        logger.info(f"Detection completed in {detection_time:.1f}ms: {len(detections)} regions")

        # Filter by confidence and limit count
        detections = [d for d in detections if d.confidence >= min_confidence]
        detections = detections[:max_dishes]

        # Step 2: Classify each detected region
        dishes = []
        for idx, detection in enumerate(detections):
            try:
                dish = await self._process_detection(
                    image=image,
                    detection=detection,
                    dish_id=idx + 1,
                    depth_map=depth_map,
                )
                # Only include if classification confidence is reasonable
                if dish.confidence >= self.classification_confidence:
                    dishes.append(dish)
            except Exception as e:
                logger.warning(f"Failed to process detection {idx + 1}: {e}")
                continue

        # Step 3: Sort by confidence (highest first)
        dishes.sort(key=lambda x: x.confidence, reverse=True)

        # Step 4: Aggregate nutrition
        total_nutrition = self._aggregate_nutrition(dishes)

        # Step 5: Assess quality
        detection_quality = self._assess_detection_quality(dishes, detections)
        overlapping = self._check_overlapping(dishes)

        # Generate suggestions
        suggestions = self._generate_suggestions(dishes, detection_quality, img_width, img_height)

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Multi-dish analysis complete: {len(dishes)} dishes, "
            f"{processing_time:.1f}ms total"
        )

        return MultiDishAnalysisResponse(
            dishes=dishes,
            dish_count=len(dishes),
            total_nutrition=total_nutrition,
            image_width=img_width,
            image_height=img_height,
            processing_time_ms=round(processing_time, 1),
            detector_model=self.detector_type,
            classifier_model=self.classifier.get_model_info()["model_name"],
            detection_quality=detection_quality,
            overlapping_dishes=overlapping,
            suggestions=suggestions if suggestions else None,
        )

    async def _process_detection(
        self,
        image: Image.Image,
        detection: DetectionResult,
        dish_id: int,
        depth_map: Optional[List[List[float]]] = None,
    ) -> DetectedDish:
        """
        Process a single detection: crop, classify, get nutrition.

        Args:
            image: Original image
            detection: Detection result with bbox
            dish_id: Unique ID for this dish
            depth_map: Optional LIDAR depth data

        Returns:
            DetectedDish with classification and nutrition
        """
        img_width, img_height = image.size

        # Convert normalized bbox to pixels
        x1_norm, y1_norm, x2_norm, y2_norm = detection.bbox
        x1_px = int(x1_norm * img_width)
        y1_px = int(y1_norm * img_height)
        x2_px = int(x2_norm * img_width)
        y2_px = int(y2_norm * img_height)

        # Add padding (10%) for better classification context
        pad_x = int((x2_px - x1_px) * 0.1)
        pad_y = int((y2_px - y1_px) * 0.1)

        crop_x1 = max(0, x1_px - pad_x)
        crop_y1 = max(0, y1_px - pad_y)
        crop_x2 = min(img_width, x2_px + pad_x)
        crop_y2 = min(img_height, y2_px + pad_y)

        # Crop the detected region
        crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # Classify the cropped region
        predictions = self.classifier.predict(crop, top_k=5)

        if not predictions:
            food_name = "unknown_food"
            confidence = 0.0
            alternatives = []
        else:
            food_name, confidence = predictions[0]
            alternatives = [
                FoodItemAlternative(
                    name=format_hf_class_name(alt_name),
                    confidence=round(alt_conf, 2)
                )
                for alt_name, alt_conf in predictions[1:4]
            ]

        # Get nutrition data
        nutrition_data = self.nutrition_db.get(food_name)
        if nutrition_data is None:
            logger.debug(f"No nutrition data for {food_name}, using fallback")
            nutrition_data = self.nutrition_db.get("unknown") or {
                "category": "unknown",
                "serving_weight": 100,
                "nutrition": {
                    "calories": 150,
                    "protein": 5,
                    "carbs": 20,
                    "fat": 5,
                    "fiber": 2,
                    "sugar": 5,
                }
            }

        # Calculate area percentage
        bbox_area = (x2_norm - x1_norm) * (y2_norm - y1_norm)
        area_percentage = bbox_area * 100

        # Estimate portion weight (default serving, refined with LIDAR later)
        portion_weight = float(nutrition_data.get("serving_weight", 100))

        # Future: Use depth_map for volume estimation
        volume_cm3 = None
        if depth_map is not None:
            volume_cm3 = self._estimate_volume_from_depth(
                depth_map, detection.bbox, img_width, img_height
            )
            if volume_cm3:
                density = nutrition_data.get("density", 0.8)
                portion_weight = volume_cm3 * density

        # Scale nutrition by portion
        base_weight = nutrition_data.get("serving_weight", 100)
        scale = portion_weight / base_weight if base_weight > 0 else 1.0
        base_nutrition = nutrition_data.get("nutrition", {})

        nutrition = NutritionInfo(
            calories=round(base_nutrition.get("calories", 0) * scale, 1),
            protein=round(base_nutrition.get("protein", 0) * scale, 1),
            carbs=round(base_nutrition.get("carbs", 0) * scale, 1),
            fat=round(base_nutrition.get("fat", 0) * scale, 1),
            fiber=round(base_nutrition.get("fiber", 0) * scale, 1) if "fiber" in base_nutrition else None,
            sugar=round(base_nutrition.get("sugar", 0) * scale, 1) if "sugar" in base_nutrition else None,
        )

        return DetectedDish(
            dish_id=dish_id,
            name=format_hf_class_name(food_name),
            confidence=round(confidence, 2),
            bbox=BoundingBox(
                x1=round(x1_norm, 4),
                y1=round(y1_norm, 4),
                x2=round(x2_norm, 4),
                y2=round(y2_norm, 4),
                x1_px=x1_px,
                y1_px=y1_px,
                x2_px=x2_px,
                y2_px=y2_px,
            ),
            area_percentage=round(area_percentage, 2),
            nutrition=nutrition,
            alternatives=alternatives,
            category=nutrition_data.get("category"),
            volume_cm3=volume_cm3,
            weight_grams=round(portion_weight, 1),
            portion_size=f"{int(portion_weight)}g" if volume_cm3 else "estimated",
        )

    def _estimate_volume_from_depth(
        self,
        depth_map: List[List[float]],
        bbox: Tuple[float, float, float, float],
        img_width: int,
        img_height: int,
    ) -> Optional[float]:
        """
        Estimate food volume from LIDAR depth map.

        This is a placeholder for future LIDAR integration.

        The actual implementation would:
        1. Extract depth values within the bounding box
        2. Find the plate/surface baseline
        3. Calculate volume above the baseline
        4. Apply food-specific shape corrections

        Args:
            depth_map: 2D array of depth values (meters)
            bbox: Normalized bounding box
            img_width, img_height: Image dimensions

        Returns:
            Estimated volume in cm³, or None if cannot estimate
        """
        # TODO: Implement LIDAR volume estimation
        # This requires calibrated depth values and surface detection
        logger.debug("LIDAR volume estimation not yet implemented")
        return None

    def _aggregate_nutrition(self, dishes: List[DetectedDish]) -> NutritionInfo:
        """Sum nutrition values across all dishes."""
        total_calories = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0
        total_fiber = 0.0
        total_sugar = 0.0

        for dish in dishes:
            total_calories += dish.nutrition.calories or 0
            total_protein += dish.nutrition.protein or 0
            total_carbs += dish.nutrition.carbs or 0
            total_fat += dish.nutrition.fat or 0
            if dish.nutrition.fiber:
                total_fiber += dish.nutrition.fiber
            if dish.nutrition.sugar:
                total_sugar += dish.nutrition.sugar

        return NutritionInfo(
            calories=round(total_calories, 1),
            protein=round(total_protein, 1),
            carbs=round(total_carbs, 1),
            fat=round(total_fat, 1),
            fiber=round(total_fiber, 1) if total_fiber > 0 else None,
            sugar=round(total_sugar, 1) if total_sugar > 0 else None,
        )

    def _assess_detection_quality(
        self,
        dishes: List[DetectedDish],
        detections: List[DetectionResult],
    ) -> str:
        """Assess overall detection quality."""
        if not dishes:
            return "low"

        # Average classification confidence
        avg_confidence = sum(d.confidence for d in dishes) / len(dishes)

        # Check coverage (total area detected)
        total_area = sum(d.area_percentage for d in dishes)

        if avg_confidence > 0.7 and total_area > 10:
            return "high"
        elif avg_confidence > 0.4:
            return "medium"
        else:
            return "low"

    def _check_overlapping(self, dishes: List[DetectedDish]) -> bool:
        """Check if any dishes have overlapping bounding boxes."""
        for i, d1 in enumerate(dishes):
            for d2 in dishes[i + 1:]:
                iou = self._compute_iou(
                    (d1.bbox.x1, d1.bbox.y1, d1.bbox.x2, d1.bbox.y2),
                    (d2.bbox.x1, d2.bbox.y1, d2.bbox.x2, d2.bbox.y2),
                )
                if iou > 0.3:
                    return True
        return False

    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _generate_suggestions(
        self,
        dishes: List[DetectedDish],
        quality: str,
        img_width: int,
        img_height: int,
    ) -> List[str]:
        """Generate suggestions for improving results."""
        suggestions = []

        if not dishes:
            suggestions.append("No food detected. Try taking a clearer photo with better lighting.")
            return suggestions

        if quality == "low":
            suggestions.append("Low detection confidence. Try taking a photo from directly above the food.")

        # Check for low classification confidence
        low_conf_dishes = [d for d in dishes if d.confidence < 0.5]
        if low_conf_dishes:
            suggestions.append(
                f"{len(low_conf_dishes)} dish(es) have low classification confidence. "
                "Consider manual correction."
            )

        # Check image resolution
        if img_width < 640 or img_height < 640:
            suggestions.append("Higher resolution images may improve detection accuracy.")

        # Suggest LIDAR for portion accuracy
        if any(d.portion_size == "estimated" for d in dishes):
            suggestions.append(
                "Use LIDAR measurements for more accurate portion size estimation."
            )

        return suggestions

    def get_service_info(self) -> dict:
        """Get service metadata."""
        return {
            "detector": self.detector.get_model_info(),
            "classifier": self.classifier.get_model_info(),
            "nutrition_db_foods": self.nutrition_db.food_count,
        }


# Singleton instance
_service_instance: Optional[MultiDishAnalysisService] = None


def get_multi_dish_service(
    detector_type: str = "owl_vit",
    force_reload: bool = False,
) -> MultiDishAnalysisService:
    """
    Get or create singleton MultiDishAnalysisService instance.

    Args:
        detector_type: "owl_vit" or "yolov8"
        force_reload: Force recreation of the service

    Returns:
        MultiDishAnalysisService instance
    """
    global _service_instance

    if _service_instance is None or force_reload:
        _service_instance = MultiDishAnalysisService(detector_type=detector_type)

    return _service_instance
