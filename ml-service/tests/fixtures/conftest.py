"""
Pytest fixtures and configuration for food analysis tests.
"""
import pytest
import asyncio
from typing import Generator
from pathlib import Path
from PIL import Image
import numpy as np
import io

from app.schemas.food_analysis import (
    DimensionsInput,
    NutritionInfo,
    FoodItem,
    FoodItemAlternative,
)


# ============================================================================
# Test Images
# ============================================================================

@pytest.fixture
def sample_food_image() -> Image.Image:
    """Create a sample food image for testing."""
    # Create a 224x224 RGB image with random pixels
    img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array, 'RGB')


@pytest.fixture
def large_food_image() -> Image.Image:
    """Create a large food image to test compression."""
    # Create a 4000x3000 RGB image
    img_array = np.random.randint(0, 256, (3000, 4000, 3), dtype=np.uint8)
    return Image.fromarray(img_array, 'RGB')


@pytest.fixture
def blurry_food_image() -> Image.Image:
    """Create a blurry/low-quality food image."""
    # Create a small image and scale up (simulates blur)
    img_array = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    return img.resize((224, 224))


@pytest.fixture
def grayscale_food_image() -> Image.Image:
    """Create a grayscale food image."""
    img_array = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    return Image.fromarray(img_array, 'L')


@pytest.fixture
def image_bytes(sample_food_image: Image.Image) -> bytes:
    """Convert sample image to bytes."""
    img_io = io.BytesIO()
    sample_food_image.save(img_io, format='JPEG')
    img_io.seek(0)
    return img_io.getvalue()


# ============================================================================
# AR Measurements
# ============================================================================

@pytest.fixture
def good_ar_measurements() -> DimensionsInput:
    """AR measurements with good quality (realistic proportions)."""
    return DimensionsInput(
        width=10.5,
        height=8.2,
        depth=6.0
    )


@pytest.fixture
def poor_ar_measurements() -> DimensionsInput:
    """AR measurements with poor quality (unrealistic proportions)."""
    return DimensionsInput(
        width=5.0,
        height=50.0,  # Unrealistic ratio
        depth=2.0
    )


@pytest.fixture
def small_portion_measurements() -> DimensionsInput:
    """AR measurements for small portion."""
    return DimensionsInput(
        width=5.0,
        height=4.0,
        depth=3.0
    )


@pytest.fixture
def large_portion_measurements() -> DimensionsInput:
    """AR measurements for large portion."""
    return DimensionsInput(
        width=20.0,
        height=15.0,
        depth=10.0
    )


# ============================================================================
# Nutrition Data
# ============================================================================

@pytest.fixture
def apple_nutrition() -> NutritionInfo:
    """Nutrition info for a medium apple."""
    return NutritionInfo(
        calories=95,
        protein=0.5,
        carbs=25,
        fat=0.3,
        fiber=4.4,
        sugar=19
    )


@pytest.fixture
def chicken_nutrition() -> NutritionInfo:
    """Nutrition info for 100g chicken breast."""
    return NutritionInfo(
        calories=165,
        protein=31,
        carbs=0,
        fat=3.6,
        fiber=0
    )


@pytest.fixture
def broccoli_nutrition() -> NutritionInfo:
    """Nutrition info for 1 cup broccoli."""
    return NutritionInfo(
        calories=31,
        protein=2.6,
        carbs=6,
        fat=0.3,
        fiber=2.4,
        sugar=1.5
    )


# ============================================================================
# Food Items
# ============================================================================

@pytest.fixture
def apple_food_item(apple_nutrition: NutritionInfo) -> FoodItem:
    """Complete food item for apple."""
    return FoodItem(
        name="Apple",
        confidence=0.92,
        portion_size="1 medium (182g)",
        portion_weight=182,
        nutrition=apple_nutrition,
        category="fruit",
        alternatives=[
            FoodItemAlternative(name="Pear", confidence=0.65),
            FoodItemAlternative(name="Peach", confidence=0.52),
        ]
    )


@pytest.fixture
def chicken_food_item(chicken_nutrition: NutritionInfo) -> FoodItem:
    """Complete food item for chicken breast."""
    return FoodItem(
        name="Chicken Breast",
        confidence=0.88,
        portion_size="100g",
        portion_weight=100,
        nutrition=chicken_nutrition,
        category="protein",
        alternatives=[
            FoodItemAlternative(name="Turkey", confidence=0.72),
        ]
    )


@pytest.fixture
def low_confidence_food_item() -> FoodItem:
    """Food item with low confidence score."""
    return FoodItem(
        name="Unknown Food",
        confidence=0.45,
        portion_size="100g",
        portion_weight=100,
        nutrition=NutritionInfo(
            calories=100,
            protein=5,
            carbs=15,
            fat=2
        ),
        category="unknown"
    )


# ============================================================================
# Test Data Collections
# ============================================================================

@pytest.fixture
def all_food_classes() -> list[str]:
    """List of all supported food classes."""
    return [
        "apple",
        "banana",
        "chicken breast",
        "broccoli",
        "rice",
        "salmon",
    ]


@pytest.fixture
def food_density_map() -> dict[str, float]:
    """Food density estimates for portion calculation."""
    return {
        "apple": 0.7,
        "banana": 0.9,
        "chicken breast": 1.0,
        "broccoli": 0.3,
        "rice": 0.8,
        "salmon": 1.0,
    }


# ============================================================================
# Async Event Loop (for async tests)
# ============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Mock Data Helpers
# ============================================================================

@pytest.fixture
def mock_image_array() -> np.ndarray:
    """Preprocessed image array ready for model inference."""
    # Normalized 224x224x3 array (ImageNet normalization)
    img = np.random.rand(224, 224, 3).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std


@pytest.fixture
def mock_classification_result() -> tuple[str, float, list]:
    """Mock classification result from ML model."""
    return (
        "apple",  # class name
        0.92,  # confidence
        [
            {"name": "pear", "confidence": 0.65},
            {"name": "peach", "confidence": 0.52},
        ]  # alternatives
    )


# ============================================================================
# File Upload Mocks
# ============================================================================

@pytest.fixture
def valid_upload_file(image_bytes: bytes):
    """Mock valid file upload."""
    from fastapi import UploadFile
    from io import BytesIO

    return UploadFile(
        file=BytesIO(image_bytes),
        filename="test_food.jpg",
        headers={"content-type": "image/jpeg"}
    )


@pytest.fixture
def invalid_upload_file():
    """Mock invalid file upload (wrong format)."""
    from fastapi import UploadFile
    from io import BytesIO

    return UploadFile(
        file=BytesIO(b"not an image"),
        filename="test.txt",
        headers={"content-type": "text/plain"}
    )


@pytest.fixture
def oversized_upload_file():
    """Mock oversized file upload (>10MB)."""
    from fastapi import UploadFile
    from io import BytesIO

    # Create 11MB of data
    large_data = b"x" * (11 * 1024 * 1024)

    return UploadFile(
        file=BytesIO(large_data),
        filename="large_image.jpg",
        headers={"content-type": "image/jpeg"}
    )


# ============================================================================
# Expected Results
# ============================================================================

@pytest.fixture
def expected_portion_weights() -> dict[str, dict[str, float]]:
    """Expected portion weights for different foods and measurements."""
    return {
        "apple": {
            "small": 100,
            "medium": 182,
            "large": 250,
        },
        "chicken_breast": {
            "small": 85,
            "medium": 100,
            "large": 150,
        },
        "broccoli": {
            "small": 50,
            "medium": 91,
            "large": 150,
        },
    }


@pytest.fixture
def processing_time_thresholds() -> dict[str, int]:
    """Expected processing time thresholds in milliseconds."""
    return {
        "min": 50,  # Minimum expected time
        "max": 5000,  # Maximum acceptable time (P95 target)
        "timeout": 30000,  # Timeout threshold
    }


# ============================================================================
# Parametrized Test Data
# ============================================================================

@pytest.fixture(params=[
    (10, 10, 10, 0.7, 490),  # cube, apple density
    (20, 15, 5, 1.0, 1050),  # rectangular, chicken density
    (8, 8, 3, 0.3, 40),  # flat, broccoli density
])
def portion_estimation_cases(request):
    """Parametrized test cases for portion estimation."""
    width, height, depth, density, expected_weight = request.param
    return {
        "dimensions": DimensionsInput(width=width, height=height, depth=depth),
        "density": density,
        "expected_weight": expected_weight,
        "tolerance": expected_weight * 0.1,  # 10% tolerance
    }


@pytest.fixture(params=[
    ("apple", 182, 95, 200, 104.4),  # Scale up
    ("apple", 182, 95, 100, 52.2),  # Scale down
    ("chicken_breast", 100, 165, 150, 247.5),  # Scale up
])
def nutrition_scaling_cases(request):
    """Parametrized test cases for nutrition scaling."""
    food, base_weight, base_calories, actual_weight, expected_calories = request.param
    return {
        "food": food,
        "base_weight": base_weight,
        "base_calories": base_calories,
        "actual_weight": actual_weight,
        "expected_calories": expected_calories,
        "tolerance": 0.1,  # Allow small rounding differences
    }
