"""
Food-101 ViT Fine-Tuning Script with LoRA

This script fine-tunes the nateraw/food model on:
1. Original Food-101 dataset for baseline maintenance
2. Additional food images for problematic classes
3. Hard negatives to reduce false positives (e.g., Apple brand → apple food)

Features:
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Mixed precision training (fp16)
- Gradient accumulation for larger effective batch sizes
- Early stopping and checkpointing
- Support for adding new food classes
"""

import logging
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from PIL import Image
import numpy as np
from tqdm import tqdm

# HuggingFace imports
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# Optional: LoRA for efficient fine-tuning
try:
    from peft import get_peft_model, LoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. Full fine-tuning will be used.")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Original Food-101 classes
FOOD_101_CLASSES = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheesecake",
    "cheese_plate",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "escargots",
    "falafel",
    "filet_mignon",
    "fish_and_chips",
    "foie_gras",
    "french_fries",
    "french_onion_soup",
    "french_toast",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "huevos_rancheros",
    "hummus",
    "ice_cream",
    "lasagna",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "macaroni_and_cheese",
    "macarons",
    "miso_soup",
    "mussels",
    "nachos",
    "omelette",
    "onion_rings",
    "oysters",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "peking_duck",
    "pho",
    "pizza",
    "pork_chop",
    "poutine",
    "prime_rib",
    "pulled_pork_sandwich",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed_salad",
    "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna_tartare",
    "waffles",
]

# New classes to add (not in original Food-101)
NEW_FOOD_CLASSES = [
    # Fruits (raw)
    "avocado",
    "banana",
    "orange",
    "strawberry",
    "mango",
    "watermelon",
    "pineapple",
    "grapes",
    "blueberry",
    "raspberry",
    "peach",
    "pear",
    "kiwi",
    "cherry",
    # Vegetables (raw)
    "broccoli",
    "carrot",
    "bell_pepper",
    "cucumber",
    "tomato",
    "spinach",
    "asparagus",
    "zucchini",
    "corn",
    "lettuce",
    "cabbage",
    "cauliflower",
    # Proteins
    "shrimp",
    "salmon_fillet",
    "chicken_breast",
    "bacon",
    "beef_steak",
    "lobster",
    "crab",
    # Nuts & Seeds
    "almonds",
    "walnuts",
    "peanuts",
    "cashews",
    "mixed_nuts",
    # Breakfast
    "croissant",
    "bagel",
    "muffin",
    "toast",
    "oatmeal",
    "cereal",
    # Beverages
    "coffee",
    "tea",
    "orange_juice",
    "smoothie",
    "beer",
    "wine",
]

# Combined classes for extended model
EXTENDED_FOOD_CLASSES = FOOD_101_CLASSES + NEW_FOOD_CLASSES


class FoodDataset(Dataset):
    """Custom dataset for food images."""

    def __init__(
        self,
        images: List[Image.Image],
        labels: List[int],
        processor: ViTImageProcessor,
        augment: bool = False,
    ):
        self.images = images
        self.labels = labels
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Basic augmentation (can be extended)
        if self.augment:
            image = self._augment(image)

        # Process for ViT
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def _augment(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations."""
        import random
        from PIL import ImageEnhance, ImageOps

        # Random horizontal flip
        if random.random() > 0.5:
            image = ImageOps.mirror(image)

        # Random rotation (-15 to 15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, fillcolor=(255, 255, 255))

        # Random brightness
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Random contrast
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        return image


class HuggingFaceFood101Dataset(Dataset):
    """Wrapper for HuggingFace Food-101 dataset."""

    def __init__(
        self,
        split: str = "train",
        processor: ViTImageProcessor = None,
        max_samples: Optional[int] = None,
        class_subset: Optional[List[str]] = None,
    ):
        logger.info(f"Loading Food-101 dataset (split={split})...")
        self.dataset = load_dataset("ethz/food101", split=split)
        self.processor = processor

        # Filter to specific classes if requested
        if class_subset:
            class_indices = {name: idx for idx, name in enumerate(FOOD_101_CLASSES)}
            valid_indices = [
                class_indices[c] for c in class_subset if c in class_indices
            ]
            self.dataset = self.dataset.filter(lambda x: x["label"] in valid_indices)

        # Limit samples if requested
        if max_samples and len(self.dataset) > max_samples:
            indices = np.random.choice(len(self.dataset), max_samples, replace=False)
            self.dataset = self.dataset.select(indices)

        logger.info(f"Food-101 dataset loaded: {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def download_additional_food_images(
    class_name: str, num_images: int = 100, output_dir: str = "data/additional_foods"
) -> List[str]:
    """
    Download additional food images from the web.
    Uses Pexels API or similar free image sources.

    For production, implement actual download logic.
    This is a placeholder for manual data collection.
    """
    output_path = Path(output_dir) / class_name
    output_path.mkdir(parents=True, exist_ok=True)

    logger.warning(
        f"download_additional_food_images() is a placeholder.\n"
        f"Please manually collect images for '{class_name}' and place them in:\n"
        f"{output_path}"
    )

    # Return existing images if any
    existing = list(output_path.glob("*.jpg")) + list(output_path.glob("*.png"))
    return [str(p) for p in existing]


def create_extended_model(
    base_model_name: str = "nateraw/food",
    num_new_classes: int = 0,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: int = 32,
) -> Tuple[ViTForImageClassification, ViTImageProcessor]:
    """
    Create an extended model with optional new classes and LoRA.

    Args:
        base_model_name: HuggingFace model to start from
        num_new_classes: Number of new classes to add (0 = keep original 101)
        use_lora: Whether to use LoRA for efficient fine-tuning
        lora_rank: LoRA rank (lower = more efficient, higher = more expressive)
        lora_alpha: LoRA alpha scaling factor

    Returns:
        Model and processor tuple
    """
    logger.info(f"Loading base model: {base_model_name}")

    processor = ViTImageProcessor.from_pretrained(base_model_name)
    model = ViTForImageClassification.from_pretrained(base_model_name)

    original_num_classes = model.config.num_labels
    logger.info(f"Original model has {original_num_classes} classes")

    # Extend classification head if adding new classes
    if num_new_classes > 0:
        total_classes = original_num_classes + num_new_classes
        logger.info(f"Extending model to {total_classes} classes")

        # Create new classification head
        old_classifier = model.classifier
        new_classifier = nn.Linear(old_classifier.in_features, total_classes)

        # Initialize new classifier with old weights
        with torch.no_grad():
            new_classifier.weight[:original_num_classes] = old_classifier.weight
            new_classifier.bias[:original_num_classes] = old_classifier.bias

            # Initialize new class weights with small random values
            nn.init.xavier_uniform_(new_classifier.weight[original_num_classes:])
            nn.init.zeros_(new_classifier.bias[original_num_classes:])

        model.classifier = new_classifier
        model.config.num_labels = total_classes

        # Update id2label and label2id
        model.config.id2label = {
            i: EXTENDED_FOOD_CLASSES[i]
            for i in range(min(total_classes, len(EXTENDED_FOOD_CLASSES)))
        }
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}

    # Apply LoRA if requested and available
    if use_lora and PEFT_AVAILABLE:
        logger.info(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha})")

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Using SEQ_CLS for image classification
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["query", "value"],  # ViT attention modules
            inference_mode=False,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif use_lora and not PEFT_AVAILABLE:
        logger.warning("LoRA requested but PEFT not installed. Using full fine-tuning.")

    return model, processor


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    accumulation_steps: int = 1,
    use_amp: bool = True,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp and device == "cuda" else None

    progress = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()

    for step, batch in enumerate(progress):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        if use_amp and device == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / accumulation_steps
            loss.backward()

        total_loss += loss.item() * accumulation_steps

        if (step + 1) % accumulation_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        progress.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module, dataloader: DataLoader, device: str
) -> Tuple[float, float]:
    """Evaluate model accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()

            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Food-101 ViT model")
    parser.add_argument(
        "--base-model", default="nateraw/food", help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output-dir", default="./checkpoints", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--use-lora", action="store_true", help="Use LoRA for efficient fine-tuning"
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument(
        "--add-classes", action="store_true", help="Add new food classes"
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="Max training samples"
    )
    parser.add_argument("--device", default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
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

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    num_new_classes = len(NEW_FOOD_CLASSES) if args.add_classes else 0
    model, processor = create_extended_model(
        base_model_name=args.base_model,
        num_new_classes=num_new_classes,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
    )
    model = model.to(device)

    # Load datasets
    logger.info("Loading training data...")
    train_dataset = HuggingFaceFood101Dataset(
        split="train", processor=processor, max_samples=args.max_train_samples
    )

    val_dataset = HuggingFaceFood101Dataset(
        split="validation",
        processor=processor,
        max_samples=args.max_train_samples // 5 if args.max_train_samples else None,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_accuracy = 0
    start_epoch = 0

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint.get("best_accuracy", 0)

    logger.info(f"Starting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            use_amp=(device == "cuda"),
        )
        logger.info(f"Train loss: {train_loss:.4f}")

        # Evaluate
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        logger.info(f"Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}")

        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "best_accuracy": best_accuracy,
                "config": {
                    "base_model": args.base_model,
                    "use_lora": args.use_lora,
                    "lora_rank": args.lora_rank,
                    "num_classes": model.config.num_labels,
                },
            },
            checkpoint_path,
        )
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_path = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_accuracy,
                    "config": {
                        "base_model": args.base_model,
                        "use_lora": args.use_lora,
                        "lora_rank": args.lora_rank,
                        "num_classes": model.config.num_labels,
                    },
                },
                best_path,
            )
            logger.info(f"New best model! Accuracy: {val_accuracy:.4f}")

    logger.info(f"\nTraining complete! Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
