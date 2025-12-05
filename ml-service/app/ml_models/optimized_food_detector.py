"""
Optimized Food Detector with MPS/CUDA Acceleration

This module provides optimized inference for food detection using:
- MPS (Apple Silicon) or CUDA GPU acceleration
- Half-precision (FP16) inference where supported
- Optimized NMS settings
- Cached model loading
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

import torch
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

@dataclass
class Detection:
    """Single detection result."""
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized)
    confidence: float
    class_id: int = 0
    class_name: str = "food"


@dataclass
class DetectionResult:
    """Detection results for an image."""
    detections: List[Detection]
    inference_time_ms: float
    image_size: Tuple[int, int]


class OptimizedFoodDetector:
    """
    Optimized food detector with GPU acceleration.

    Usage:
        detector = OptimizedFoodDetector()
        result = detector.detect("path/to/image.jpg")
        for det in result.detections:
            print(f"Found food at {det.box} with confidence {det.confidence:.2f}")
    """

    # Model paths in priority order
    MODEL_PATHS = [
        "results/yolo_training/food_detector_v8s_full/weights/best.pt",
        "results/yolo_training/food_detector_clean/weights/best.pt",
        "results/yolo_training/food_detector_v8n_full/weights/best.pt",
        "results/yolo_training/food_detector/weights/best.pt",
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        use_half: bool = True,
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to YOLO model. If None, uses best available.
            device: Device to use ('mps', 'cuda', 'cpu'). If None, auto-detects.
            confidence_threshold: Minimum confidence for detections.
            iou_threshold: IoU threshold for NMS.
            use_half: Use FP16 inference (faster, slightly less accurate).
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_half = use_half

        # Auto-detect device
        if device is None:
            self.device = self._detect_device()
        else:
            self.device = device

        # Find and load model
        self.model_path = self._find_model(model_path)
        self.model = self._load_model()

        logger.info(f"OptimizedFoodDetector initialized:")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  FP16: {self.use_half and self.device != 'cpu'}")

    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def _find_model(self, model_path: Optional[str]) -> str:
        """Find best available model."""
        if model_path:
            full_path = PROJECT_ROOT / model_path
            if full_path.exists():
                return str(full_path)
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Try models in priority order
        for path in self.MODEL_PATHS:
            full_path = PROJECT_ROOT / path
            if full_path.exists():
                return str(full_path)

        raise FileNotFoundError(
            f"No trained model found. Train a model first using "
            f"scripts/prepare_full_dataset.py and YOLO training."
        )

    def _load_model(self):
        """Load YOLO model with optimizations."""
        from ultralytics import YOLO

        model = YOLO(self.model_path)

        # Set inference parameters
        model.overrides['conf'] = self.confidence_threshold
        model.overrides['iou'] = self.iou_threshold
        model.overrides['device'] = self.device
        model.overrides['verbose'] = False

        # Enable half precision for GPU
        if self.use_half and self.device in ['mps', 'cuda']:
            model.overrides['half'] = True

        # Warmup
        self._warmup(model)

        return model

    def _warmup(self, model, iterations: int = 3):
        """Warmup model for consistent timing."""
        logger.info("Warming up model...")
        dummy_img = Image.new('RGB', (640, 640), color='white')

        for _ in range(iterations):
            _ = model.predict(dummy_img, verbose=False)

    def detect(self, image_source) -> DetectionResult:
        """
        Detect food in image.

        Args:
            image_source: Path to image, PIL Image, or numpy array.

        Returns:
            DetectionResult with detections and timing.
        """
        # Load image if path
        if isinstance(image_source, (str, Path)):
            image = Image.open(image_source).convert('RGB')
        elif isinstance(image_source, Image.Image):
            image = image_source.convert('RGB')
        else:
            image = Image.fromarray(image_source)

        img_width, img_height = image.size

        # Run inference
        start_time = time.perf_counter()
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        inference_time = (time.perf_counter() - start_time) * 1000

        # Parse results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = box
                    # Normalize coordinates
                    detection = Detection(
                        box=(
                            float(x1 / img_width),
                            float(y1 / img_height),
                            float(x2 / img_width),
                            float(y2 / img_height)
                        ),
                        confidence=float(conf),
                        class_id=0,
                        class_name="food"
                    )
                    detections.append(detection)

        return DetectionResult(
            detections=detections,
            inference_time_ms=inference_time,
            image_size=(img_width, img_height)
        )

    def detect_batch(self, images: List) -> List[DetectionResult]:
        """
        Detect food in multiple images.

        Args:
            images: List of image sources.

        Returns:
            List of DetectionResults.
        """
        return [self.detect(img) for img in images]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'use_half': self.use_half and self.device != 'cpu',
        }


# Singleton instance for the service
_detector_instance: Optional[OptimizedFoodDetector] = None


def get_detector(
    confidence_threshold: float = 0.25,
    force_reload: bool = False,
) -> OptimizedFoodDetector:
    """
    Get or create the global detector instance.

    Args:
        confidence_threshold: Minimum detection confidence.
        force_reload: Force reload the model.

    Returns:
        OptimizedFoodDetector instance.
    """
    global _detector_instance

    if _detector_instance is None or force_reload:
        _detector_instance = OptimizedFoodDetector(
            confidence_threshold=confidence_threshold
        )

    return _detector_instance


def detect_food(image_source, confidence: float = 0.25) -> DetectionResult:
    """
    Convenience function for single-image detection.

    Args:
        image_source: Path to image, PIL Image, or numpy array.
        confidence: Minimum detection confidence.

    Returns:
        DetectionResult.
    """
    detector = get_detector(confidence_threshold=confidence)
    return detector.detect(image_source)


if __name__ == '__main__':
    # Quick test
    import sys

    print("="*60)
    print("OPTIMIZED FOOD DETECTOR TEST")
    print("="*60)

    # Initialize detector
    try:
        detector = OptimizedFoodDetector()
        print(f"\nModel info: {detector.get_model_info()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Test on sample images
    test_dir = PROJECT_ROOT / 'datasets/food_detection_full/images/val'
    if not test_dir.exists():
        test_dir = PROJECT_ROOT / 'test_dataset'

    test_images = list(test_dir.glob('*.jpg'))[:10]

    if not test_images:
        print("No test images found!")
        sys.exit(1)

    print(f"\nTesting on {len(test_images)} images...")

    times = []
    det_counts = []

    for img_path in test_images:
        result = detector.detect(img_path)
        times.append(result.inference_time_ms)
        det_counts.append(len(result.detections))

        print(f"  {img_path.name}: {len(result.detections)} detections, {result.inference_time_ms:.1f}ms")

    print(f"\nSummary:")
    print(f"  Average inference: {sum(times)/len(times):.1f}ms")
    print(f"  Average detections: {sum(det_counts)/len(det_counts):.1f}")
    print(f"  Images per second: {1000 / (sum(times)/len(times)):.1f}")
