#!/usr/bin/env python3
"""
Prepare full dataset for YOLO training:
1. Use all 344 images from test_dataset
2. Generate bounding box labels using OWL-ViT
3. Split into train/val (80/20)
4. Save in YOLO format
"""

import os
import shutil
import random
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Configuration
SOURCE_DIR = Path('./test_dataset')
OUTPUT_DIR = Path('./datasets/food_detection_full')
TRAIN_RATIO = 0.8
MIN_CONFIDENCE = 0.08  # Lower threshold to catch more food items
MAX_DETECTIONS = 25

# Comprehensive food prompts for detection
FOOD_PROMPTS = [
    'food', 'dish', 'meal', 'plate of food', 'bowl of food',
    'rice', 'pasta', 'meat', 'vegetables', 'salad',
    'soup', 'sandwich', 'burger', 'hamburger', 'pizza', 'sushi',
    'curry', 'noodles', 'bread', 'dessert', 'cake', 'pie',
    'steak', 'chicken', 'fish', 'seafood', 'shrimp',
    'tacos', 'burrito', 'nachos', 'fries', 'french fries',
    'pancakes', 'waffles', 'eggs', 'bacon', 'breakfast',
    'ice cream', 'chocolate', 'cookie', 'donut', 'pastry',
    'dumpling', 'spring roll', 'sashimi', 'ramen', 'pho',
    'croissant', 'samosa', 'hummus', 'falafel', 'kebab'
]

def setup_directories():
    """Create YOLO dataset directory structure."""
    for split in ['train', 'val']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure at {OUTPUT_DIR}")

def load_owl_vit():
    """Load OWL-ViT model."""
    print("Loading OWL-ViT model...")
    processor = OwlViTProcessor.from_pretrained('google/owlvit-base-patch32')
    model = OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch32')
    device = torch.device('cpu')  # Use CPU for stability
    model = model.to(device)
    model.eval()
    return processor, model, device

def generate_labels(image_path, processor, model, device):
    """Generate YOLO-format labels using OWL-ViT."""
    try:
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # Prepare inputs
        inputs = processor(text=[FOOD_PROMPTS], images=image, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        target_sizes = torch.tensor([[height, width]])
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=MIN_CONFIDENCE
        )[0]

        # Convert to YOLO format
        labels = []
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()

        # Sort by confidence and take top detections
        sorted_indices = scores.argsort()[::-1][:MAX_DETECTIONS]

        for idx in sorted_indices:
            score = scores[idx]
            if score < MIN_CONFIDENCE:
                continue

            box = boxes[idx]
            x1, y1, x2, y2 = box

            # Convert to YOLO format (class x_center y_center width height)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            # Validate bounds
            if 0 <= x_center <= 1 and 0 <= y_center <= 1 and w > 0.01 and h > 0.01:
                labels.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        return labels

    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return []

def process_dataset(processor, model, device):
    """Process all images and generate labels."""
    # Get all images
    image_files = list(SOURCE_DIR.glob('*.jpg'))
    random.shuffle(image_files)

    print(f"\nFound {len(image_files)} images")

    # Split into train/val
    split_idx = int(len(image_files) * TRAIN_RATIO)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    stats = {'train': {'images': 0, 'labels': 0}, 'val': {'images': 0, 'labels': 0}}

    for split, files in [('train', train_files), ('val', val_files)]:
        print(f"\nProcessing {split} split ({len(files)} images)...")

        for i, img_path in enumerate(files):
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(files)}] Processing...")

            # Generate labels
            labels = generate_labels(img_path, processor, model, device)

            if labels:
                # Copy image
                dest_img = OUTPUT_DIR / 'images' / split / img_path.name
                shutil.copy(img_path, dest_img)

                # Save labels
                label_name = img_path.stem + '.txt'
                label_path = OUTPUT_DIR / 'labels' / split / label_name
                with open(label_path, 'w') as f:
                    f.write('\n'.join(labels))

                stats[split]['images'] += 1
                stats[split]['labels'] += len(labels)

        avg_labels = stats[split]['labels'] / max(stats[split]['images'], 1)
        print(f"  {split}: {stats[split]['images']} images, {stats[split]['labels']} total detections")
        print(f"  Average detections per image: {avg_labels:.1f}")

    return stats

def create_dataset_yaml():
    """Create YOLO dataset configuration."""
    yaml_content = f"""# Food Detection Dataset (Full - 344 images)
path: {OUTPUT_DIR.absolute()}
train: images/train
val: images/val

# Class
nc: 1
names: ['food']
"""
    yaml_path = OUTPUT_DIR / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nCreated dataset config: {yaml_path}")

def main():
    print("=" * 60)
    print("PREPARING FULL DATASET FOR YOLO TRAINING")
    print("=" * 60)

    # Setup
    setup_directories()
    processor, model, device = load_owl_vit()

    # Process
    stats = process_dataset(processor, model, device)

    # Create config
    create_dataset_yaml()

    # Summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nTrain: {stats['train']['images']} images, {stats['train']['labels']} detections")
    print(f"Val: {stats['val']['images']} images, {stats['val']['labels']} detections")
    print(f"\nDataset saved to: {OUTPUT_DIR}")
    print(f"Config: {OUTPUT_DIR / 'dataset.yaml'}")

if __name__ == '__main__':
    main()
