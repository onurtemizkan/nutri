#!/usr/bin/env python3
"""
Food-101 Model Evaluation Script

Evaluates the food classification model on the Food-101 test dataset.
Downloads the dataset from HuggingFace if not already cached.

Usage:
    python scripts/evaluate_food_classifier.py [--samples N] [--batch-size B]

Options:
    --samples N      Number of samples to evaluate (default: all 25,250)
    --batch-size B   Batch size for inference (default: 32)
    --quick          Quick test with 1000 samples
"""

import argparse
import logging
import time
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_food101_dataset(num_samples: int = None, streaming: bool = True):
    """
    Load Food-101 test dataset from HuggingFace.

    Args:
        num_samples: Number of samples to load (None for all)
        streaming: Use streaming mode to avoid downloading full dataset

    Returns:
        HuggingFace dataset object or iterator
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install datasets: pip install datasets")
        sys.exit(1)

    logger.info("Loading Food-101 test dataset from HuggingFace...")

    if streaming:
        logger.info("Using streaming mode (no full download required)")
        # Load in streaming mode - no disk space needed
        dataset = load_dataset(
            "ethz/food101",
            split="validation",
            streaming=True
        )

        if num_samples:
            logger.info(f"Will evaluate {num_samples} samples (first {num_samples}, no shuffle)")
            # Don't shuffle - take first N to get diverse class coverage
            dataset = dataset.take(num_samples)

        return dataset, streaming
    else:
        logger.info("This may take a few minutes on first run (downloading ~5GB)")

        # Load the test split
        dataset = load_dataset(
            "ethz/food101",
            split="validation",
        )

        logger.info(f"Loaded {len(dataset)} test images")

        if num_samples and num_samples < len(dataset):
            # Shuffle and take subset for faster evaluation
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
            logger.info(f"Using subset of {num_samples} samples")

        return dataset, streaming


def get_classifier():
    """Load the HuggingFace food classifier model."""
    from app.ml_models.food_classifier_hf import (
        HuggingFaceFoodClassifier,
        HFClassifierConfig,
    )

    logger.info("Loading HuggingFace food classifier...")
    config = HFClassifierConfig(
        model_name="AventIQ-AI/Food-Classification-AI-Model",
        version="1.0.0",
    )
    classifier = HuggingFaceFoodClassifier(config)
    logger.info("✓ Using HuggingFace model (AventIQ-AI/Food-Classification-AI-Model)")
    return classifier, "huggingface"


def normalize_label(label: str) -> str:
    """Normalize label for comparison (lowercase, underscore separated)."""
    return label.lower().replace(" ", "_").replace("-", "_")


def evaluate_model(
    classifier,
    dataset,
    batch_size: int = 32,
    streaming: bool = False,
    num_samples: int = None
) -> Dict[str, float]:
    """
    Evaluate model on dataset.

    Args:
        classifier: Food classifier model
        dataset: HuggingFace dataset with images and labels
        batch_size: Batch size for inference
        streaming: Whether dataset is in streaming mode
        num_samples: Expected number of samples (for progress bar in streaming)

    Returns:
        Dict with evaluation metrics
    """
    # Get class names from model (normalized)
    model_classes = [normalize_label(c) for c in classifier.classes]

    # Get dataset class names dynamically from the dataset builder
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder('ethz/food101')
        dataset_classes = builder.info.features['label'].names
        logger.info(f"Loaded {len(dataset_classes)} class names from dataset")
    except Exception as e:
        logger.warning(f"Failed to load dataset classes dynamically: {e}")
        # Fallback to standard Food-101 order
        dataset_classes = [
            'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
            'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
            'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
            'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
            'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
            'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
            'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
            'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
            'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
            'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
            'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
            'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
            'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
            'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
            'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
            'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
            'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
            'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
            'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
            'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
            'waffles'
        ]

    logger.info(f"Model has {len(model_classes)} classes")
    logger.info(f"Dataset has {len(dataset_classes)} classes")

    # Show first few class mappings for debugging
    logger.info("First 5 model classes: " + ", ".join(model_classes[:5]))
    logger.info("First 5 dataset classes: " + ", ".join(dataset_classes[:5]))

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    # Per-class metrics
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # Confusion tracking
    misclassified_examples = []

    start_time = time.time()

    if streaming:
        # Streaming mode - iterate directly
        iterator = tqdm(dataset, desc="Evaluating", total=num_samples)
        for sample in iterator:
            img = sample["image"]
            label_idx = sample["label"]

            # Get ground truth label
            true_label = normalize_label(dataset_classes[label_idx])

            # Make prediction
            try:
                predictions = classifier.predict(img, top_k=5)
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                continue

            if not predictions:
                continue

            # Normalize predictions
            pred_labels = [normalize_label(p[0]) for p in predictions]
            pred_confidences = [p[1] for p in predictions]

            # Top-1 accuracy
            top1_correct = pred_labels[0] == true_label
            if top1_correct:
                correct_top1 += 1
                class_correct[true_label] += 1
            else:
                # Track misclassified examples (limit to first 100)
                if len(misclassified_examples) < 100:
                    misclassified_examples.append({
                        "true": true_label,
                        "predicted": pred_labels[0],
                        "confidence": pred_confidences[0],
                        "top5": pred_labels[:5]
                    })

            # Top-5 accuracy
            if true_label in pred_labels[:5]:
                correct_top5 += 1

            class_total[true_label] += 1
            total += 1

            # Update progress with running accuracy
            if total % 100 == 0:
                running_acc = correct_top1 / total if total > 0 else 0
                iterator.set_postfix({"acc": f"{running_acc:.1%}"})
    else:
        # Batch mode - process in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            batch = dataset[i:i + batch_size]
            images = batch["image"]
            labels = batch["label"]

            for img, label_idx in zip(images, labels):
                # Get ground truth label
                true_label = normalize_label(dataset_classes[label_idx])

                # Make prediction
                try:
                    predictions = classifier.predict(img, top_k=5)
                except Exception as e:
                    logger.warning(f"Prediction failed: {e}")
                    continue

                if not predictions:
                    continue

                # Normalize predictions
                pred_labels = [normalize_label(p[0]) for p in predictions]
                pred_confidences = [p[1] for p in predictions]

                # Top-1 accuracy
                top1_correct = pred_labels[0] == true_label
                if top1_correct:
                    correct_top1 += 1
                    class_correct[true_label] += 1
                else:
                    # Track misclassified examples (limit to first 100)
                    if len(misclassified_examples) < 100:
                        misclassified_examples.append({
                            "true": true_label,
                            "predicted": pred_labels[0],
                            "confidence": pred_confidences[0],
                            "top5": pred_labels[:5]
                        })

                # Top-5 accuracy
                if true_label in pred_labels[:5]:
                    correct_top5 += 1

                class_total[true_label] += 1
                total += 1

    elapsed_time = time.time() - start_time

    # Calculate metrics
    metrics = {
        "total_samples": total,
        "top1_accuracy": correct_top1 / total if total > 0 else 0,
        "top5_accuracy": correct_top5 / total if total > 0 else 0,
        "correct_top1": correct_top1,
        "correct_top5": correct_top5,
        "evaluation_time_seconds": elapsed_time,
        "samples_per_second": total / elapsed_time if elapsed_time > 0 else 0,
    }

    # Per-class accuracy
    per_class_acc = {}
    for cls in class_total:
        acc = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        per_class_acc[cls] = {
            "accuracy": acc,
            "correct": class_correct[cls],
            "total": class_total[cls]
        }

    metrics["per_class"] = per_class_acc
    metrics["misclassified_examples"] = misclassified_examples[:20]  # First 20

    return metrics


def print_results(metrics: Dict, model_type: str):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print("FOOD-101 MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nModel Type: {model_type}")
    print(f"Total Samples: {metrics['total_samples']:,}")
    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 42)
    print(f"{'Top-1 Accuracy:':<25} {metrics['top1_accuracy']*100:>14.2f}%")
    print(f"{'Top-5 Accuracy:':<25} {metrics['top5_accuracy']*100:>14.2f}%")
    print(f"{'Correct (Top-1):':<25} {metrics['correct_top1']:>15,}")
    print(f"{'Correct (Top-5):':<25} {metrics['correct_top5']:>15,}")
    print(f"{'Evaluation Time:':<25} {metrics['evaluation_time_seconds']:>12.1f}s")
    print(f"{'Throughput:':<25} {metrics['samples_per_second']:>11.1f}/s")

    # Best and worst classes
    per_class = metrics.get("per_class", {})
    if per_class:
        sorted_classes = sorted(
            per_class.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True
        )

        print("\n" + "-" * 42)
        print("TOP 5 BEST CLASSES:")
        for cls, data in sorted_classes[:5]:
            print(f"  {cls:<25} {data['accuracy']*100:>6.1f}% ({data['correct']}/{data['total']})")

        print("\nTOP 5 WORST CLASSES:")
        for cls, data in sorted_classes[-5:]:
            print(f"  {cls:<25} {data['accuracy']*100:>6.1f}% ({data['correct']}/{data['total']})")

    # Example misclassifications
    misclassified = metrics.get("misclassified_examples", [])
    if misclassified:
        print("\n" + "-" * 42)
        print("SAMPLE MISCLASSIFICATIONS:")
        for ex in misclassified[:10]:
            print(f"  True: {ex['true']:<20} -> Pred: {ex['predicted']:<20} ({ex['confidence']:.2%})")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Food-101 classifier")
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick test with 1000 samples"
    )
    args = parser.parse_args()

    num_samples = args.samples
    if args.quick:
        num_samples = 1000
        logger.info("Quick mode: evaluating 1000 samples")

    # Load classifier
    classifier, model_type = get_classifier()
    logger.info(f"Using {model_type} classifier")
    logger.info(f"Model: {classifier.config.model_name if hasattr(classifier.config, 'model_name') else classifier.config.model_type}")

    # Load dataset (streaming mode for disk space efficiency)
    dataset, streaming = load_food101_dataset(num_samples)

    # Evaluate
    logger.info("Starting evaluation...")
    metrics = evaluate_model(
        classifier,
        dataset,
        batch_size=args.batch_size,
        streaming=streaming,
        num_samples=num_samples
    )

    # Print results
    print_results(metrics, model_type)

    # Summary
    print(f"\nSummary: {metrics['top1_accuracy']*100:.2f}% Top-1 Accuracy on {metrics['total_samples']:,} samples")

    return metrics


if __name__ == "__main__":
    main()
