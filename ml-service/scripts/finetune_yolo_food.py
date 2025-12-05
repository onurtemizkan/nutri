#!/usr/bin/env python3
"""
Fine-tune YOLOv8 for Food Detection.

This script:
1. Downloads food images from free sources
2. Generates pseudo-labels using OWL-ViT (zero-shot detector)
3. Prepares YOLO training dataset
4. Fine-tunes YOLOv8 on food detection
5. Evaluates and compares performance

Usage:
    python scripts/finetune_yolo_food.py
"""
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import httpx
import torch
from PIL import Image
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "food_detection"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
MODELS_DIR = BASE_DIR / "models" / "yolo_food"
RESULTS_DIR = BASE_DIR / "results" / "yolo_training"

# Food image sources (Pexels - free to use)
FOOD_IMAGE_URLS = [
    # Multi-dish plates
    ("https://images.pexels.com/photos/1640777/pexels-photo-1640777.jpeg?w=640", "plate_salad_chicken"),
    ("https://images.pexels.com/photos/1279330/pexels-photo-1279330.jpeg?w=640", "plate_rice_chicken"),
    ("https://images.pexels.com/photos/2116094/pexels-photo-2116094.jpeg?w=640", "breakfast_eggs_bacon"),
    ("https://images.pexels.com/photos/1410235/pexels-photo-1410235.jpeg?w=640", "meat_vegetables"),
    ("https://images.pexels.com/photos/1640774/pexels-photo-1640774.jpeg?w=640", "food_bowls"),
    ("https://images.pexels.com/photos/5966431/pexels-photo-5966431.jpeg?w=640", "asian_dishes"),
    ("https://images.pexels.com/photos/3186654/pexels-photo-3186654.jpeg?w=640", "table_food"),
    ("https://images.pexels.com/photos/8844888/pexels-photo-8844888.jpeg?w=640", "meal_prep"),
    ("https://images.pexels.com/photos/5718025/pexels-photo-5718025.jpeg?w=640", "buffet"),
    ("https://images.pexels.com/photos/4551975/pexels-photo-4551975.jpeg?w=640", "breakfast_spread"),
    # Single dishes (for variety)
    ("https://images.pexels.com/photos/1092730/pexels-photo-1092730.jpeg?w=640", "pizza"),
    ("https://images.pexels.com/photos/1639557/pexels-photo-1639557.jpeg?w=640", "burger_fries"),
    ("https://images.pexels.com/photos/2097090/pexels-photo-2097090.jpeg?w=640", "salad"),
    ("https://images.pexels.com/photos/1351238/pexels-photo-1351238.jpeg?w=640", "pasta"),
    ("https://images.pexels.com/photos/699953/pexels-photo-699953.jpeg?w=640", "sushi"),
    ("https://images.pexels.com/photos/2641886/pexels-photo-2641886.jpeg?w=640", "tacos"),
    ("https://images.pexels.com/photos/1437267/pexels-photo-1437267.jpeg?w=640", "soup"),
    ("https://images.pexels.com/photos/2673353/pexels-photo-2673353.jpeg?w=640", "rice_bowl"),
    ("https://images.pexels.com/photos/1099680/pexels-photo-1099680.jpeg?w=640", "steak"),
    ("https://images.pexels.com/photos/2474661/pexels-photo-2474661.jpeg?w=640", "sandwich"),
    # More variety
    ("https://images.pexels.com/photos/1633525/pexels-photo-1633525.jpeg?w=640", "burger"),
    ("https://images.pexels.com/photos/1583884/pexels-photo-1583884.jpeg?w=640", "pasta_dish"),
    ("https://images.pexels.com/photos/2089712/pexels-photo-2089712.jpeg?w=640", "asian_food"),
    ("https://images.pexels.com/photos/376464/pexels-photo-376464.jpeg?w=640", "pancakes"),
    ("https://images.pexels.com/photos/2092906/pexels-photo-2092906.jpeg?w=640", "noodles"),
    ("https://images.pexels.com/photos/2092897/pexels-photo-2092897.jpeg?w=640", "dim_sum"),
    ("https://images.pexels.com/photos/1640772/pexels-photo-1640772.jpeg?w=640", "healthy_bowl"),
    ("https://images.pexels.com/photos/3026808/pexels-photo-3026808.jpeg?w=640", "curry"),
    ("https://images.pexels.com/photos/2233729/pexels-photo-2233729.jpeg?w=640", "fried_rice"),
    ("https://images.pexels.com/photos/3590401/pexels-photo-3590401.jpeg?w=640", "bbq"),
    # Additional images for better training
    ("https://images.pexels.com/photos/5718071/pexels-photo-5718071.jpeg?w=640", "brunch"),
    ("https://images.pexels.com/photos/1410236/pexels-photo-1410236.jpeg?w=640", "grilled_food"),
    ("https://images.pexels.com/photos/2544829/pexels-photo-2544829.jpeg?w=640", "seafood"),
    ("https://images.pexels.com/photos/1435904/pexels-photo-1435904.jpeg?w=640", "appetizers"),
    ("https://images.pexels.com/photos/1653877/pexels-photo-1653877.jpeg?w=640", "breakfast_plate"),
    ("https://images.pexels.com/photos/2097091/pexels-photo-2097091.jpeg?w=640", "vegetarian"),
    ("https://images.pexels.com/photos/2449665/pexels-photo-2449665.jpeg?w=640", "dinner"),
    ("https://images.pexels.com/photos/1058714/pexels-photo-1058714.jpeg?w=640", "mexican"),
    ("https://images.pexels.com/photos/3026802/pexels-photo-3026802.jpeg?w=640", "indian"),
    ("https://images.pexels.com/photos/2347311/pexels-photo-2347311.jpeg?w=640", "mediterranean"),
]


@dataclass
class DetectionBox:
    """Bounding box detection."""
    x1: float  # normalized 0-1
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str


def download_image(url: str, save_path: Path, timeout: int = 30) -> bool:
    """Download image from URL."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def download_training_images() -> List[Path]:
    """Download food images for training."""
    logger.info("Downloading training images...")

    train_dir = IMAGES_DIR / "train"
    val_dir = IMAGES_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    for i, (url, name) in enumerate(FOOD_IMAGE_URLS):
        # 80% train, 20% val
        split = "train" if i < len(FOOD_IMAGE_URLS) * 0.8 else "val"
        save_dir = train_dir if split == "train" else val_dir

        filename = f"{name}_{hashlib.md5(url.encode()).hexdigest()[:8]}.jpg"
        save_path = save_dir / filename

        if save_path.exists():
            logger.debug(f"Already exists: {filename}")
            downloaded.append(save_path)
            continue

        if download_image(url, save_path):
            downloaded.append(save_path)
            logger.info(f"Downloaded [{split}]: {filename}")

        time.sleep(0.2)  # Rate limiting

    logger.info(f"Total images: {len(downloaded)}")
    return downloaded


def generate_pseudo_labels(image_paths: List[Path]) -> Dict[Path, List[DetectionBox]]:
    """Generate pseudo-labels using OWL-ViT detector."""
    logger.info("Generating pseudo-labels with OWL-ViT...")

    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    # Load OWL-ViT
    model_name = "google/owlvit-base-patch32"
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Text prompts for food detection
    text_prompts = [
        "a photo of food",
        "a photo of a dish",
        "a photo of a meal",
        "a photo of a plate of food",
        "a photo of a bowl of food",
    ]

    all_labels = {}

    with torch.no_grad():
        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                img_width, img_height = image.size

                # Process
                inputs = processor(text=text_prompts, images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Detect
                outputs = model(**inputs)

                # Post-process
                target_sizes = torch.tensor([[img_height, img_width]], device=device)
                results = processor.post_process_object_detection(
                    outputs,
                    threshold=0.1,  # Lower threshold for more labels
                    target_sizes=target_sizes
                )[0]

                # Convert to boxes
                boxes = []
                for score, label_idx, box in zip(
                    results["scores"].cpu(),
                    results["labels"].cpu(),
                    results["boxes"].cpu()
                ):
                    x1, y1, x2, y2 = box.numpy()

                    # Normalize to 0-1
                    x1_norm = float(x1 / img_width)
                    y1_norm = float(y1 / img_height)
                    x2_norm = float(x2 / img_width)
                    y2_norm = float(y2 / img_height)

                    # Clamp
                    x1_norm = max(0, min(1, x1_norm))
                    y1_norm = max(0, min(1, y1_norm))
                    x2_norm = max(0, min(1, x2_norm))
                    y2_norm = max(0, min(1, y2_norm))

                    # Skip invalid boxes
                    if x2_norm <= x1_norm or y2_norm <= y1_norm:
                        continue

                    boxes.append(DetectionBox(
                        x1=x1_norm,
                        y1=y1_norm,
                        x2=x2_norm,
                        y2=y2_norm,
                        confidence=float(score),
                        label="food"  # Single class for food detection
                    ))

                # Apply NMS
                boxes = apply_nms(boxes, iou_threshold=0.5)

                all_labels[img_path] = boxes
                logger.debug(f"{img_path.name}: {len(boxes)} detections")

            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                all_labels[img_path] = []

    total_boxes = sum(len(b) for b in all_labels.values())
    logger.info(f"Generated {total_boxes} pseudo-labels for {len(all_labels)} images")

    return all_labels


def apply_nms(boxes: List[DetectionBox], iou_threshold: float = 0.5) -> List[DetectionBox]:
    """Apply non-maximum suppression."""
    if not boxes:
        return []

    # Sort by confidence
    boxes = sorted(boxes, key=lambda x: x.confidence, reverse=True)

    keep = []
    for box in boxes:
        should_keep = True
        for kept in keep:
            iou = compute_iou(
                (box.x1, box.y1, box.x2, box.y2),
                (kept.x1, kept.y1, kept.x2, kept.y2)
            )
            if iou > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(box)

    return keep


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def save_yolo_labels(labels: Dict[Path, List[DetectionBox]]):
    """Save labels in YOLO format."""
    logger.info("Saving YOLO format labels...")

    train_labels = LABELS_DIR / "train"
    val_labels = LABELS_DIR / "val"
    train_labels.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    for img_path, boxes in labels.items():
        # Determine split from image path
        split = "train" if "train" in str(img_path) else "val"
        labels_dir = train_labels if split == "train" else val_labels

        # YOLO label file
        label_path = labels_dir / f"{img_path.stem}.txt"

        with open(label_path, "w") as f:
            for box in boxes:
                # YOLO format: class_id center_x center_y width height
                center_x = (box.x1 + box.x2) / 2
                center_y = (box.y1 + box.y2) / 2
                width = box.x2 - box.x1
                height = box.y2 - box.y1

                # Class 0 = food
                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    logger.info(f"Saved labels to {LABELS_DIR}")


def create_dataset_yaml() -> Path:
    """Create YOLO dataset configuration."""
    config = {
        "path": str(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "food"
        },
        "nc": 1  # Number of classes
    }

    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Created dataset config: {yaml_path}")
    return yaml_path


def train_yolo(dataset_yaml: Path, epochs: int = 50) -> Path:
    """Fine-tune YOLOv8 on food detection."""
    logger.info(f"Training YOLOv8 for {epochs} epochs...")

    from ultralytics import YOLO

    # Create output directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load pretrained YOLOv8n (nano - fastest)
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=640,
        batch=8,  # Smaller batch for MPS
        device="mps",  # Use Apple Metal
        project=str(RESULTS_DIR),
        name="food_detector",
        exist_ok=True,
        patience=10,  # Early stopping
        save=True,
        plots=True,
        verbose=True,
    )

    # Get best model path
    best_model = RESULTS_DIR / "food_detector" / "weights" / "best.pt"

    # Copy to models directory
    final_model = MODELS_DIR / "yolov8n_food.pt"
    if best_model.exists():
        shutil.copy(best_model, final_model)
        logger.info(f"Saved fine-tuned model: {final_model}")

    return final_model


def evaluate_models(yolo_model_path: Path, test_images: List[Path]) -> Dict:
    """Evaluate and compare OWL-ViT vs fine-tuned YOLO."""
    logger.info("Evaluating models...")

    from ultralytics import YOLO
    from transformers import OwlViTProcessor, OwlViTForObjectDetection

    results = {
        "owl_vit": {"detections": [], "times": []},
        "yolo_finetuned": {"detections": [], "times": []},
    }

    # Load models
    yolo = YOLO(str(yolo_model_path))

    owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    owl_model.to(device)
    owl_model.eval()

    text_prompts = ["a photo of food", "a photo of a dish", "a photo of a meal"]

    for img_path in test_images[:10]:  # Evaluate on subset
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        # YOLO evaluation
        start = time.time()
        yolo_results = yolo(image, conf=0.25, verbose=False)
        yolo_time = (time.time() - start) * 1000

        yolo_detections = len(yolo_results[0].boxes) if yolo_results else 0
        results["yolo_finetuned"]["detections"].append(yolo_detections)
        results["yolo_finetuned"]["times"].append(yolo_time)

        # OWL-ViT evaluation
        with torch.no_grad():
            start = time.time()
            inputs = owl_processor(text=text_prompts, images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = owl_model(**inputs)
            target_sizes = torch.tensor([[img_height, img_width]], device=device)
            owl_results = owl_processor.post_process_object_detection(
                outputs, threshold=0.15, target_sizes=target_sizes
            )[0]
            owl_time = (time.time() - start) * 1000

        owl_detections = len(owl_results["boxes"])
        results["owl_vit"]["detections"].append(owl_detections)
        results["owl_vit"]["times"].append(owl_time)

        logger.debug(f"{img_path.name}: YOLO={yolo_detections}, OWL-ViT={owl_detections}")

    return results


def print_evaluation_report(results: Dict, yolo_model_path: Path):
    """Print evaluation report."""
    print("\n" + "="*70)
    print("FOOD DETECTION MODEL EVALUATION REPORT")
    print("="*70)

    for model_name, data in results.items():
        detections = data["detections"]
        times = data["times"]

        avg_detections = sum(detections) / len(detections) if detections else 0
        avg_time = sum(times) / len(times) if times else 0

        print(f"\n{model_name.upper()}")
        print("-" * 40)
        print(f"  Avg detections per image: {avg_detections:.1f}")
        print(f"  Avg inference time: {avg_time:.1f}ms")
        print(f"  Min time: {min(times):.1f}ms")
        print(f"  Max time: {max(times):.1f}ms")

    # Speed comparison
    owl_avg = sum(results["owl_vit"]["times"]) / len(results["owl_vit"]["times"])
    yolo_avg = sum(results["yolo_finetuned"]["times"]) / len(results["yolo_finetuned"]["times"])
    speedup = owl_avg / yolo_avg if yolo_avg > 0 else 0

    print(f"\nSPEED COMPARISON")
    print("-" * 40)
    print(f"  YOLO is {speedup:.1f}x faster than OWL-ViT")

    print(f"\nMODEL FILES")
    print("-" * 40)
    print(f"  Fine-tuned YOLO: {yolo_model_path}")
    if yolo_model_path.exists():
        size_mb = yolo_model_path.stat().st_size / (1024 * 1024)
        print(f"  Model size: {size_mb:.1f} MB")

    print("\n" + "="*70)


def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("YOLO FOOD DETECTOR FINE-TUNING")
    print("="*70 + "\n")

    # Step 1: Download images
    print("[1/5] Downloading training images...")
    image_paths = download_training_images()

    if len(image_paths) < 10:
        logger.error("Not enough images downloaded. Aborting.")
        return

    # Step 2: Generate pseudo-labels
    print("\n[2/5] Generating pseudo-labels with OWL-ViT...")
    labels = generate_pseudo_labels(image_paths)

    # Step 3: Save in YOLO format
    print("\n[3/5] Preparing YOLO dataset...")
    save_yolo_labels(labels)
    dataset_yaml = create_dataset_yaml()

    # Step 4: Train YOLO
    print("\n[4/5] Fine-tuning YOLOv8...")
    yolo_model_path = train_yolo(dataset_yaml, epochs=30)

    # Step 5: Evaluate
    print("\n[5/5] Evaluating models...")
    val_images = list((IMAGES_DIR / "val").glob("*.jpg"))
    if val_images:
        eval_results = evaluate_models(yolo_model_path, val_images)
        print_evaluation_report(eval_results, yolo_model_path)

    # Save results
    results_file = RESULTS_DIR / "training_summary.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_images": len(image_paths),
        "total_labels": sum(len(b) for b in labels.values()),
        "model_path": str(yolo_model_path),
        "dataset_yaml": str(dataset_yaml),
    }

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete! Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
