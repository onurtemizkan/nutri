"""
Food Detection Module - Detect multiple food items in an image.

Supports multiple detection backends:
- OWL-ViT (zero-shot, text-prompted detection from HuggingFace)
- YOLOv8 (fast, requires fine-tuning for best food detection)

OWL-ViT is recommended for Phase 1 as it works out-of-the-box without training.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Raw detection result from detector."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 normalized (0-1)
    confidence: float
    label: str  # Detected label (e.g., "food", "dish")


@dataclass
class FoodDetectorConfig:
    """Configuration for food detector."""
    model_type: Literal["owl_vit", "yolov8"] = "owl_vit"
    device: str = "auto"  # "auto", "cuda", "mps", or "cpu"
    confidence_threshold: float = 0.15  # Lower threshold for better recall
    nms_threshold: float = 0.5  # Non-max suppression IoU threshold
    # Text prompts for zero-shot detection (OWL-ViT)
    text_prompts: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.text_prompts:
            # Default prompts optimized for food detection
            self.text_prompts = [
                "a photo of food",
                "a photo of a dish",
                "a photo of a meal",
                "a photo of a plate of food",
                "a photo of a bowl of food",
                "a photo of a salad",
                "a photo of soup",
                "a photo of meat",
                "a photo of vegetables",
                "a photo of rice",
                "a photo of pasta",
                "a photo of bread",
                "a photo of dessert",
                "a photo of fruit",
            ]


class BaseFoodDetector(ABC):
    """Abstract base class for food detectors."""

    @abstractmethod
    def detect(self, image: Image.Image) -> List[DetectionResult]:
        """
        Detect food regions in image.

        Args:
            image: PIL Image to analyze

        Returns:
            List of DetectionResult with bounding boxes
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get model metadata."""
        pass


class OWLViTFoodDetector(BaseFoodDetector):
    """
    Zero-shot food detection using OWL-ViT.

    OWL-ViT (Vision Transformer for Open-World Localization) can detect
    objects based on text prompts without fine-tuning.

    Advantages:
    - No training needed
    - Text-prompted detection (flexible)
    - Good accuracy on diverse foods

    Model: google/owlvit-base-patch32 (~340MB)
    """

    def __init__(self, config: Optional[FoodDetectorConfig] = None):
        self.config = config or FoodDetectorConfig(model_type="owl_vit")
        self.device = self._get_device()
        self.model = None
        self.processor = None
        self._load_model()

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if self.config.device != "auto":
            return torch.device(self.config.device)

        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"OWL-ViT: Using CUDA ({torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("OWL-ViT: Using Apple MPS")
        else:
            device = torch.device("cpu")
            logger.info("OWL-ViT: Using CPU")

        return device

    def _load_model(self) -> None:
        """Load OWL-ViT model and processor from HuggingFace."""
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection

            model_name = "google/owlvit-base-patch32"
            logger.info(f"Loading OWL-ViT model: {model_name}")

            self.processor = OwlViTProcessor.from_pretrained(model_name)
            self.model = OwlViTForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"OWL-ViT loaded successfully on {self.device}")

        except ImportError:
            raise ImportError(
                "transformers library required for OWL-ViT. "
                "Install with: pip install transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load OWL-ViT model: {e}")
            raise

    @torch.no_grad()
    def detect(self, image: Image.Image) -> List[DetectionResult]:
        """
        Detect food items using text-prompted zero-shot detection.

        Args:
            image: PIL Image to analyze

        Returns:
            List of DetectionResult sorted by confidence
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_width, img_height = image.size

        # Process image with text prompts
        inputs = self.processor(
            text=self.config.text_prompts,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run detection
        outputs = self.model(**inputs)

        # Post-process: get boxes above threshold
        target_sizes = torch.tensor([[img_height, img_width]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.config.confidence_threshold,
            target_sizes=target_sizes
        )[0]

        # Convert to DetectionResult list
        detections = []
        for score, label_idx, box in zip(
            results["scores"].cpu(),
            results["labels"].cpu(),
            results["boxes"].cpu()
        ):
            # box is in xyxy format (pixels)
            x1, y1, x2, y2 = box.numpy()

            # Normalize to 0-1 range
            bbox_normalized = (
                float(x1 / img_width),
                float(y1 / img_height),
                float(x2 / img_width),
                float(y2 / img_height),
            )

            # Clamp to valid range
            bbox_normalized = tuple(max(0.0, min(1.0, v)) for v in bbox_normalized)

            detections.append(DetectionResult(
                bbox=bbox_normalized,
                confidence=float(score),
                label=self.config.text_prompts[label_idx.item()],
            ))

        # Apply NMS to remove overlapping boxes
        detections = self._apply_nms(detections)

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(f"OWL-ViT detected {len(detections)} food regions")
        return detections

    def _apply_nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Apply non-maximum suppression to remove overlapping boxes.

        Keeps the highest confidence detection when boxes overlap significantly.
        """
        if not detections:
            return []

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        keep = []
        for det in detections:
            should_keep = True
            for kept in keep:
                iou = self._compute_iou(det.bbox, kept.bbox)
                if iou > self.config.nms_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(det)

        return keep

    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_width = max(0, x2 - x1)
        inter_height = max(0, y2 - y1)
        inter_area = inter_width * inter_height

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_type": "owl_vit",
            "model_name": "google/owlvit-base-patch32",
            "device": str(self.device),
            "text_prompts": self.config.text_prompts,
            "confidence_threshold": self.config.confidence_threshold,
            "nms_threshold": self.config.nms_threshold,
        }


class YOLOv8FoodDetector(BaseFoodDetector):
    """
    Food detection using YOLOv8 (Ultralytics).

    Uses COCO pretrained model which has limited food classes.
    Best results require fine-tuning on food-specific datasets.

    Advantages:
    - Very fast inference (~20ms)
    - Easy to fine-tune with custom data
    - Production-ready

    Note: COCO only has these food classes:
    - banana, apple, sandwich, orange, broccoli, carrot
    - hot dog, pizza, donut, cake
    """

    # COCO class IDs that are food-related
    FOOD_CLASS_IDS = {
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot_dog",
        53: "pizza",
        54: "donut",
        55: "cake",
    }

    # Additional classes that might contain food
    CONTAINER_CLASS_IDS = {
        45: "bowl",
        56: "dining_table",
    }

    def __init__(self, config: Optional[FoodDetectorConfig] = None):
        self.config = config or FoodDetectorConfig(model_type="yolov8")
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO

            # Use medium model for balance of speed/accuracy
            model_path = "yolov8m.pt"
            logger.info(f"Loading YOLOv8 model: {model_path}")

            self.model = YOLO(model_path)
            logger.info("YOLOv8 loaded successfully")

        except ImportError:
            raise ImportError(
                "ultralytics library required for YOLOv8. "
                "Install with: pip install ultralytics"
            )
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

    def detect(self, image: Image.Image) -> List[DetectionResult]:
        """
        Detect food items using YOLO object detection.

        Note: Limited to COCO food classes. Fine-tuning recommended.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_width, img_height = image.size

        # Run detection
        results = self.model(
            image,
            conf=self.config.confidence_threshold,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])

                # Filter to food-related classes only
                if class_id not in self.FOOD_CLASS_IDS:
                    continue

                # Get normalized bbox (xyxyn = normalized xyxy)
                x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()

                detections.append(DetectionResult(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(box.conf[0]),
                    label=self.FOOD_CLASS_IDS[class_id],
                ))

        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(f"YOLOv8 detected {len(detections)} food regions")
        return detections

    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_type": "yolov8",
            "model_name": "yolov8m.pt",
            "food_classes": list(self.FOOD_CLASS_IDS.values()),
            "confidence_threshold": self.config.confidence_threshold,
            "note": "Limited to COCO food classes. Fine-tuning recommended for best results.",
        }


# Singleton instances for each detector type
_detector_instances: dict = {}


def get_food_detector(
    config: Optional[FoodDetectorConfig] = None,
    force_reload: bool = False,
) -> BaseFoodDetector:
    """
    Get or create a food detector instance.

    Args:
        config: Detector configuration (defaults to OWL-ViT)
        force_reload: Force recreation of the detector

    Returns:
        BaseFoodDetector instance
    """
    global _detector_instances

    config = config or FoodDetectorConfig()
    detector_type = config.model_type

    if detector_type not in _detector_instances or force_reload:
        logger.info(f"Creating new {detector_type} detector instance")

        if detector_type == "owl_vit":
            _detector_instances[detector_type] = OWLViTFoodDetector(config)
        elif detector_type == "yolov8":
            _detector_instances[detector_type] = YOLOv8FoodDetector(config)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

    return _detector_instances[detector_type]
