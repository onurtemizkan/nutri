"""
Multi-Food Object Detector using OWL-ViT

Uses OWL-ViT (Open-World Localization Vision Transformer) for zero-shot
food detection. Can detect multiple food items in a single image without
requiring training on specific food classes.

Model: google/owlvit-base-patch32
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Food-related text queries for zero-shot detection
# These queries help OWL-ViT identify different types of food regions
# EXPANDED for brutal test improvements - covering all major categories
FOOD_QUERIES = [
    # === GENERAL ===
    "a photo of food",
    "a photo of a meal",
    "a photo of a dish",
    "a photo of a plate of food",
    # === PROTEINS (MEAT) ===
    "a photo of meat",
    "a photo of steak",
    "a photo of beef",
    "a photo of chicken",
    "a photo of pork",
    "a photo of bacon",
    "a photo of ham",
    "a photo of lamb",
    "a photo of turkey",
    "a photo of duck",
    "a photo of meatballs",
    "a photo of sausage",
    "a photo of eggs",
    "a photo of fried egg",
    "a photo of scrambled eggs",
    "a photo of omelette",
    # === SEAFOOD (CRITICAL - was 0% accuracy) ===
    "a photo of seafood",
    "a photo of fish",
    "a photo of salmon",
    "a photo of tuna",
    "a photo of shrimp",
    "a photo of prawns",
    "a photo of lobster",
    "a photo of crab",
    "a photo of oysters",
    "a photo of mussels",
    "a photo of clams",
    "a photo of scallops",
    "a photo of calamari",
    "a photo of squid",
    "a photo of octopus",
    "a photo of sashimi",
    "a photo of grilled fish",
    "a photo of fried fish",
    # === CARBS/GRAINS ===
    "a photo of rice",
    "a photo of bread",
    "a photo of pasta",
    "a photo of noodles",
    "a photo of risotto",
    "a photo of couscous",
    "a photo of quinoa",
    "a photo of potatoes",
    "a photo of french fries",
    "a photo of mashed potatoes",
    "a photo of baked potato",
    # === VEGETABLES (EXPANDED) ===
    "a photo of vegetables",
    "a photo of salad",
    "a photo of greens",
    "a photo of broccoli",
    "a photo of carrots",
    "a photo of celery",
    "a photo of cucumber",
    "a photo of tomatoes",
    "a photo of lettuce",
    "a photo of spinach",
    "a photo of cauliflower",
    "a photo of corn",
    "a photo of mushrooms",
    "a photo of asparagus",
    "a photo of zucchini",
    "a photo of bell peppers",
    "a photo of peppers",
    "a photo of eggplant",
    "a photo of aubergine",
    "a photo of kale",
    "a photo of cabbage",
    "a photo of brussels sprouts",
    "a photo of artichoke",
    "a photo of onion",
    "a photo of garlic",
    "a photo of green beans",
    "a photo of peas",
    "a photo of beets",
    "a photo of radish",
    "a photo of turnip",
    # === FRUITS (EXPANDED) ===
    "a photo of fruit",
    "a photo of berries",
    "a photo of banana",
    "a photo of orange",
    "a photo of strawberries",
    "a photo of apple fruit",
    "a photo of grapes",
    "a photo of watermelon",
    "a photo of pineapple",
    "a photo of mango",
    "a photo of kiwi",
    "a photo of kiwi fruit",
    "a photo of peach",
    "a photo of pear",
    "a photo of plum",
    "a photo of cherry",
    "a photo of cherries",
    "a photo of blueberries",
    "a photo of raspberries",
    "a photo of blackberries",
    "a photo of melon",
    "a photo of cantaloupe",
    "a photo of honeydew",
    "a photo of papaya",
    "a photo of passion fruit",
    "a photo of pomegranate",
    "a photo of lemon",
    "a photo of lime",
    "a photo of grapefruit",
    "a photo of avocado",
    "a photo of coconut",
    # === NUTS & SEEDS (EXPANDED) ===
    "a photo of nuts",
    "a photo of almonds",
    "a photo of walnuts",
    "a photo of peanuts",
    "a photo of cashews",
    "a photo of pistachios",
    "a photo of hazelnuts",
    "a photo of pecans",
    "a photo of macadamia",
    "a photo of chestnuts",
    "a photo of seeds",
    "a photo of sunflower seeds",
    "a photo of pumpkin seeds",
    # === DAIRY ===
    "a photo of cheese",
    "a photo of dairy",
    "a photo of milk",
    "a photo of yogurt",
    "a photo of butter",
    "a photo of cream",
    # === PREPARED DISHES (CRITICAL for accuracy) ===
    "a photo of pizza",
    "a photo of hamburger",
    "a photo of burger",
    "a photo of cheeseburger",
    "a photo of sushi",
    "a photo of taco",
    "a photo of tacos",
    "a photo of sandwich",
    "a photo of hot dog",
    "a photo of french fries",
    "a photo of fried chicken",
    "a photo of curry",
    "a photo of ramen",
    "a photo of burrito",
    "a photo of quesadilla",
    "a photo of nachos",
    "a photo of fajitas",
    "a photo of enchiladas",
    "a photo of lasagna",
    "a photo of spaghetti",
    "a photo of macaroni and cheese",
    "a photo of mac and cheese",
    "a photo of risotto",
    "a photo of paella",
    "a photo of stir fry",
    "a photo of kebab",
    "a photo of gyro",
    "a photo of shawarma",
    "a photo of falafel",
    "a photo of hummus",
    "a photo of pita bread",
    "a photo of naan bread",
    "a photo of fish and chips",
    # === ASIAN CUISINE (EXPANDED) ===
    "a photo of dim sum",
    "a photo of dumplings",
    "a photo of gyoza",
    "a photo of spring rolls",
    "a photo of egg rolls",
    "a photo of pad thai",
    "a photo of pho",
    "a photo of vietnamese soup",
    "a photo of bibimbap",
    "a photo of korean food",
    "a photo of fried rice",
    "a photo of lo mein",
    "a photo of chow mein",
    "a photo of teriyaki",
    "a photo of tempura",
    "a photo of miso soup",
    "a photo of edamame",
    "a photo of tofu",
    "a photo of bao bun",
    "a photo of steamed bun",
    # === INDIAN CUISINE ===
    "a photo of indian curry",
    "a photo of butter chicken",
    "a photo of tikka masala",
    "a photo of tandoori",
    "a photo of biryani",
    "a photo of samosa",
    "a photo of pakora",
    "a photo of naan",
    "a photo of dal",
    "a photo of paneer",
    # === BREAKFAST (EXPANDED) ===
    "a photo of pancakes",
    "a photo of waffles",
    "a photo of oatmeal",
    "a photo of cereal",
    "a photo of toast",
    "a photo of croissant",
    "a photo of bagel",
    "a photo of muffin",
    "a photo of english muffin",
    "a photo of french toast",
    "a photo of eggs benedict",
    "a photo of granola",
    "a photo of smoothie",
    "a photo of smoothie bowl",
    "a photo of acai bowl",
    # === SOUP ===
    "a photo of soup",
    "a photo of stew",
    "a photo of chili",
    "a photo of chowder",
    "a photo of bisque",
    "a photo of broth",
    # === DESSERTS (EXPANDED) ===
    "a photo of dessert",
    "a photo of cake",
    "a photo of cheesecake",
    "a photo of chocolate cake",
    "a photo of ice cream",
    "a photo of cookies",
    "a photo of chocolate",
    "a photo of pie",
    "a photo of apple pie",
    "a photo of brownie",
    "a photo of donut",
    "a photo of doughnut",
    "a photo of cupcake",
    "a photo of muffin",
    "a photo of pastry",
    "a photo of croissant",
    "a photo of tiramisu",
    "a photo of pudding",
    "a photo of mousse",
    "a photo of tart",
    "a photo of macaron",
    "a photo of cinnamon roll",
    # === BEVERAGES ===
    "a photo of coffee",
    "a photo of tea",
    "a photo of juice",
    "a photo of smoothie",
    "a photo of milkshake",
    "a photo of latte",
    "a photo of cappuccino",
    # === LEGUMES ===
    "a photo of beans",
    "a photo of lentils",
    "a photo of chickpeas",
    "a photo of black beans",
    "a photo of kidney beans",
    # === OTHER ===
    "a photo of snack",
    "a photo of appetizer",
    "a photo of chips",
    "a photo of popcorn",
    "a photo of pretzel",
    "a photo of guacamole",
    "a photo of salsa",
]


@dataclass
class DetectedRegion:
    """A detected food region in the image."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    query_label: str  # Which query matched
    cropped_image: Optional[Image.Image] = None


@dataclass
class DetectionResult:
    """Results from multi-food detection."""

    regions: List[DetectedRegion]
    num_foods_detected: int
    detection_confidence: float  # Overall confidence


class FoodDetector:
    """
    Zero-shot food object detector using OWL-ViT.

    Detects multiple food items in an image using text-guided detection.
    Each detected region can then be classified by the ensemble classifier.
    """

    # Detection thresholds
    CONFIDENCE_THRESHOLD = 0.10  # Minimum confidence for detection
    NMS_THRESHOLD = 0.3  # Non-maximum suppression IoU threshold
    MAX_DETECTIONS = 10  # Maximum number of food items to detect
    MIN_REGION_SIZE = 50  # Minimum region size in pixels

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the food detector.

        Args:
            device: Device to run inference on ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.device = device or self._detect_device()
        self._model = None
        self._processor = None
        self._loaded = False

        logger.info(f"FoodDetector initialized (device: {self.device})")

    def _detect_device(self) -> str:
        """Detect the best available device.

        Note: MPS is not supported for OWL-ViT due to float64 requirements
        in post-processing. Falls back to CPU on Mac.
        """
        if torch.cuda.is_available():
            return "cuda"
        # MPS doesn't support float64 which OWL-ViT post-processing requires
        # Fall back to CPU on Mac for reliability
        return "cpu"

    def load_model(self) -> None:
        """Load the OWL-ViT model (lazy loading on first inference)."""
        if self._loaded:
            return

        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection  # type: ignore[import-untyped]

            logger.info("Loading OWL-ViT food detector model...")
            model_name = "google/owlvit-base-patch32"

            self._processor = OwlViTProcessor.from_pretrained(model_name)
            self._model = OwlViTForObjectDetection.from_pretrained(model_name)
            self._model = self._model.to(self.device)  # type: ignore[attr-defined]
            self._model.eval()  # type: ignore[attr-defined]

            self._loaded = True
            logger.info(f"OWL-ViT model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load OWL-ViT model: {e}")
            raise

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        confidence_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> DetectionResult:
        """
        Detect multiple food items in an image.

        Args:
            image: PIL Image to analyze
            confidence_threshold: Override default confidence threshold
            max_detections: Override default max detections

        Returns:
            DetectionResult with all detected food regions
        """
        if not self._loaded:
            self.load_model()

        threshold = confidence_threshold or self.CONFIDENCE_THRESHOLD
        max_det = max_detections or self.MAX_DETECTIONS

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process image with all food queries
        inputs = self._processor(text=FOOD_QUERIES, images=image, return_tensors="pt")  # type: ignore[misc]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run detection
        outputs = self._model(**inputs)  # type: ignore[misc]

        # Post-process to get boxes - always on CPU to avoid MPS float64 issues
        # The processor.post_process_object_detection uses float64 internally which MPS doesn't support
        target_sizes = torch.tensor(
            [image.size[::-1]], dtype=torch.float32
        )  # CPU, (height, width)

        # Create a simple namespace with CPU tensors
        class OutputsCPU:
            def __init__(self, logits, pred_boxes):
                self.logits = logits
                self.pred_boxes = pred_boxes

        outputs_cpu = OutputsCPU(
            logits=outputs.logits.detach().cpu(),
            pred_boxes=outputs.pred_boxes.detach().cpu(),
        )

        results = self._processor.post_process_object_detection(  # type: ignore[attr-defined]
            outputs=outputs_cpu, threshold=threshold, target_sizes=target_sizes
        )[
            0
        ]

        # Extract detections
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        # Filter and process detections
        regions = []
        for box, score, label_idx in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)

            # Skip small regions
            width = x2 - x1
            height = y2 - y1
            if width < self.MIN_REGION_SIZE or height < self.MIN_REGION_SIZE:
                continue

            # Ensure bbox is within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)

            # Crop the region
            cropped = image.crop((x1, y1, x2, y2))

            query_label = (
                FOOD_QUERIES[label_idx] if label_idx < len(FOOD_QUERIES) else "food"
            )

            regions.append(
                DetectedRegion(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(score),
                    query_label=query_label,
                    cropped_image=cropped,
                )
            )

        # Apply non-maximum suppression to remove overlapping detections
        regions = self._non_max_suppression(regions)

        # Limit to max detections
        regions = sorted(regions, key=lambda r: r.confidence, reverse=True)[:max_det]

        # Calculate overall confidence
        overall_confidence = max([r.confidence for r in regions]) if regions else 0.0

        logger.info(f"Detected {len(regions)} food regions in image")

        return DetectionResult(
            regions=regions,
            num_foods_detected=len(regions),
            detection_confidence=overall_confidence,
        )

    def _non_max_suppression(
        self, regions: List[DetectedRegion]
    ) -> List[DetectedRegion]:
        """
        Apply non-maximum suppression to remove overlapping detections.

        Args:
            regions: List of detected regions

        Returns:
            Filtered list with overlapping regions removed
        """
        if len(regions) <= 1:
            return regions

        # Sort by confidence (highest first)
        regions = sorted(regions, key=lambda r: r.confidence, reverse=True)

        keep = []
        while regions:
            best = regions.pop(0)
            keep.append(best)

            # Remove regions that overlap too much with the best one
            regions = [
                r
                for r in regions
                if self._calculate_iou(best.bbox, r.bbox) < self.NMS_THRESHOLD
            ]

        return keep

    def _calculate_iou(
        self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1, box2: Bounding boxes as (x1, y1, x2, y2)

        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def detect_single_or_multi(
        self,
        image: Image.Image,
    ) -> Tuple[bool, DetectionResult]:
        """
        Detect if image contains single or multiple food items.

        This is useful to decide whether to use simple classification
        or multi-food pipeline.

        Args:
            image: PIL Image to analyze

        Returns:
            Tuple of (is_multi_food, detection_result)
        """
        result = self.detect(image)
        is_multi = result.num_foods_detected > 1
        return is_multi, result

    def get_model_info(self) -> dict:
        """Get information about the detector model."""
        return {
            "name": "OWL-ViT Food Detector",
            "model": "google/owlvit-base-patch32",
            "type": "Zero-shot Object Detection",
            "num_queries": len(FOOD_QUERIES),
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
            "max_detections": self.MAX_DETECTIONS,
            "device": self.device,
            "loaded": self._loaded,
            "description": (
                "Open-vocabulary object detector that can identify multiple "
                "food items in an image using text-guided detection."
            ),
        }


# Singleton instance for lazy loading
_detector_instance: Optional[FoodDetector] = None


def get_food_detector() -> FoodDetector:
    """Get the singleton food detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FoodDetector()
    return _detector_instance
