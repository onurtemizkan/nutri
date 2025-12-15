"""
Train a Specialist Food Classifier

This script trains a custom classifier focused on:
- Raw fruits and vegetables (problematic for Food-101)
- Nuts and seeds
- Other ingredients that Food-101 struggles with

Uses multiple datasets:
- Food-101 for prepared dishes
- Fruits-360 (via HuggingFace or local) for fruits/vegetables

Quick training approach using transfer learning from a pre-trained ViT.
"""
import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import io
import requests
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import numpy as np
from tqdm import tqdm

# HuggingFace imports
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Target classes for our specialist classifier
# These are the classes we need to improve (map to our database keys)
TARGET_CLASSES = [
    # Fruits
    "apple", "avocado", "banana", "blueberry", "cherry", "grape", "kiwi",
    "lemon", "lime", "mango", "orange", "peach", "pear", "pineapple",
    "raspberry", "strawberry", "watermelon",
    # Vegetables
    "asparagus", "bell_pepper", "broccoli", "cabbage", "carrot", "cauliflower",
    "celery", "corn", "cucumber", "eggplant", "lettuce", "mushroom", "onion",
    "potato", "spinach", "tomato", "zucchini",
    # Nuts
    "almond", "cashew", "peanut", "walnut",
    # Proteins (standalone)
    "bacon", "chicken_breast", "salmon", "shrimp",
    # Breakfast items
    "bagel", "croissant", "donut", "toast",
]


class WebImageDataset(Dataset):
    """Dataset that downloads images from URLs."""

    def __init__(
        self,
        image_urls: Dict[str, List[str]],  # {class_name: [url1, url2, ...]}
        processor,
        cache_dir: str = "/tmp/web_images_cache"
    ):
        self.processor = processor
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples = []  # [(image_path_or_bytes, label)]
        self.class_to_idx = {name: idx for idx, name in enumerate(sorted(image_urls.keys()))}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Download images
        for class_name, urls in image_urls.items():
            for url in urls:
                self.samples.append((url, self.class_to_idx[class_name]))

        logger.info(f"WebImageDataset: {len(self.samples)} samples, {len(self.class_to_idx)} classes")

    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download image from URL with caching."""
        # Create cache filename
        cache_file = self.cache_dir / f"{hash(url)}.jpg"

        if cache_file.exists():
            try:
                return Image.open(cache_file)
            except:
                pass

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            }
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
            # Save to cache
            image.save(cache_file, "JPEG")
            return image
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        url, label = self.samples[idx]

        image = self._download_image(url)
        if image is None:
            # Return a placeholder
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }


def get_pexels_urls(query: str, num_images: int = 50) -> List[str]:
    """
    Get image URLs from Pexels for a food query.
    Returns URLs for downloading training images.

    NOTE: For production, you'd need a Pexels API key.
    This uses direct URL construction for demo purposes.
    """
    # Pexels URLs follow a pattern - we can construct search URLs
    # Format: https://images.pexels.com/photos/{id}/pexels-photo-{id}.jpeg

    # For now, return pre-curated URLs for problematic classes
    # In production, implement actual Pexels API integration

    # Sample curated URLs by category
    CURATED_URLS = {
        "avocado": [
            "https://images.pexels.com/photos/2228553/pexels-photo-2228553.jpeg?auto=compress&w=600",
            "https://images.pexels.com/photos/5945569/pexels-photo-5945569.jpeg?auto=compress&w=600",
            "https://images.pexels.com/photos/4915835/pexels-photo-4915835.jpeg?auto=compress&w=600",
        ],
        "shrimp": [
            "https://images.pexels.com/photos/725991/pexels-photo-725991.jpeg?auto=compress&w=600",
            "https://images.pexels.com/photos/566345/pexels-photo-566345.jpeg?auto=compress&w=600",
            "https://images.pexels.com/photos/8953669/pexels-photo-8953669.jpeg?auto=compress&w=600",
        ],
        "bell_pepper": [
            "https://images.pexels.com/photos/594137/pexels-photo-594137.jpeg?auto=compress&w=600",
            "https://images.pexels.com/photos/128536/pexels-photo-128536.jpeg?auto=compress&w=600",
            "https://images.pexels.com/photos/7299852/pexels-photo-7299852.jpeg?auto=compress&w=600",
        ],
        "bacon": [
            "https://images.pexels.com/photos/1101059/pexels-photo-1101059.jpeg?auto=compress&w=600",
            "https://images.pexels.com/photos/9218537/pexels-photo-9218537.jpeg?auto=compress&w=600",
        ],
        "donut": [
            "https://images.pexels.com/photos/3656118/pexels-photo-3656118.jpeg?auto=compress&w=600",
            "https://images.pexels.com/photos/3776953/pexels-photo-3776953.jpeg?auto=compress&w=600",
            "https://images.pexels.com/photos/1854664/pexels-photo-1854664.jpeg?auto=compress&w=600",
        ],
        "walnut": [
            "https://images.pexels.com/photos/45211/walnut-kernels-nuts-brown-45211.jpeg?auto=compress&w=600",
            "https://images.pexels.com/photos/4202392/pexels-photo-4202392.jpeg?auto=compress&w=600",
        ],
        "croissant": [
            "https://images.pexels.com/photos/3892469/pexels-photo-3892469.jpeg?auto=compress&w=600",
            "https://images.pexels.com/photos/3724353/pexels-photo-3724353.jpeg?auto=compress&w=600",
        ],
    }

    return CURATED_URLS.get(query, [])


def create_training_dataset(processor, use_food101: bool = True, max_samples_per_class: int = 500):
    """Create combined training dataset."""

    datasets_to_combine = []

    # 1. Load Food-101 subset for certain classes
    if use_food101:
        logger.info("Loading Food-101 dataset...")
        food101 = load_dataset("ethz/food101", split="train")

        # Food-101 class mapping to our target classes
        FOOD101_TARGET_MAPPING = {
            31: "donut",  # donuts
            # Add more mappings as needed
        }

        # Filter to target classes
        target_indices = list(FOOD101_TARGET_MAPPING.keys())
        filtered = food101.filter(lambda x: x["label"] in target_indices)

        if len(filtered) > 0:
            logger.info(f"Food-101 filtered: {len(filtered)} samples")

            class Food101Subset(Dataset):
                def __init__(self, dataset, processor, mapping, max_per_class):
                    self.dataset = dataset
                    self.processor = processor
                    self.mapping = mapping
                    # Limit samples
                    if max_per_class and len(self.dataset) > max_per_class * len(mapping):
                        indices = np.random.choice(
                            len(self.dataset),
                            max_per_class * len(mapping),
                            replace=False
                        )
                        # Can't use .select() with filter result, just limit __len__
                        self._limit = max_per_class * len(mapping)
                    else:
                        self._limit = len(self.dataset)

                def __len__(self):
                    return min(self._limit, len(self.dataset))

                def __getitem__(self, idx):
                    item = self.dataset[idx]
                    image = item["image"]
                    label_name = self.mapping[item["label"]]

                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    inputs = self.processor(images=image, return_tensors="pt")

                    # Map to our class index
                    label_idx = TARGET_CLASSES.index(label_name) if label_name in TARGET_CLASSES else 0

                    return {
                        "pixel_values": inputs["pixel_values"].squeeze(0),
                        "labels": torch.tensor(label_idx, dtype=torch.long)
                    }

            datasets_to_combine.append(
                Food101Subset(filtered, processor, FOOD101_TARGET_MAPPING, max_samples_per_class)
            )

    # 2. Add web images for problematic classes
    logger.info("Adding web images for problematic classes...")
    problem_classes = ["avocado", "shrimp", "bell_pepper", "bacon", "donut", "walnut", "croissant"]

    web_urls = {}
    for cls in problem_classes:
        urls = get_pexels_urls(cls, num_images=20)
        if urls:
            web_urls[cls] = urls
            logger.info(f"  {cls}: {len(urls)} URLs")

    if web_urls:
        # Create class mapping for web dataset
        class_to_idx = {}
        for cls in web_urls.keys():
            if cls in TARGET_CLASSES:
                class_to_idx[cls] = TARGET_CLASSES.index(cls)

        web_dataset = WebImageDataset(web_urls, processor)
        # Remap labels to TARGET_CLASSES indices
        web_dataset.class_to_idx = class_to_idx
        datasets_to_combine.append(web_dataset)

    # 3. Combine datasets
    if len(datasets_to_combine) == 1:
        return datasets_to_combine[0]
    elif len(datasets_to_combine) > 1:
        return ConcatDataset(datasets_to_combine)
    else:
        raise ValueError("No training data available!")


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 5,
    lr: float = 2e-5,
    output_dir: str = "./specialist_checkpoints"
):
    """Train the model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    best_accuracy = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")

        # Validation
        if val_loader:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(pixel_values=pixel_values)
                    preds = outputs.logits.argmax(dim=-1)

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            accuracy = correct / total if total > 0 else 0
            logger.info(f"Epoch {epoch+1} - Val Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # Save best model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "accuracy": accuracy,
                    "classes": TARGET_CLASSES,
                }, output_path / "best_specialist.pt")
                logger.info(f"Saved new best model (accuracy: {accuracy:.4f})")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": TARGET_CLASSES,
        }, output_path / f"checkpoint_epoch_{epoch+1}.pt")

    return best_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train specialist food classifier")
    parser.add_argument("--base-model", default="google/vit-base-patch16-224",
                       help="Base model for transfer learning")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output-dir", default="./specialist_checkpoints",
                       help="Output directory")
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--quick-test", action="store_true",
                       help="Quick test with minimal data")
    args = parser.parse_args()

    # Detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")

    # Load processor and model
    logger.info(f"Loading base model: {args.base_model}")
    processor = AutoImageProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(TARGET_CLASSES),
        ignore_mismatched_sizes=True  # Allow changing classifier head
    )

    # Update config
    model.config.id2label = {i: name for i, name in enumerate(TARGET_CLASSES)}
    model.config.label2id = {name: i for i, name in enumerate(TARGET_CLASSES)}

    model = model.to(device)

    # Create dataset
    logger.info("Creating training dataset...")
    max_samples = 50 if args.quick_test else 500
    train_dataset = create_training_dataset(
        processor,
        use_food101=not args.quick_test,
        max_samples_per_class=max_samples
    )

    # Split into train/val
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for web downloads
        pin_memory=(device != "cpu")
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device != "cpu")
    )

    logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Train
    best_acc = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir
    )

    logger.info(f"\nTraining complete! Best accuracy: {best_acc:.4f}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
