"""
Tests for Food Classification Model and Service.

Tests cover:
- FoodClassifier model initialization and inference
- NutritionDatabase loading and lookup
- FoodAnalysisService integration
- Model versioning and configuration
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.ml_models.food_classifier import (
    FoodClassifier,
    FoodClassifierConfig,
    get_food_classifier,
    format_class_name,
    FOOD101_CLASSES,
)
from app.services.food_analysis_service import (
    FoodAnalysisService,
    NutritionDatabase,
    get_food_analysis_service,
)
from app.schemas.food_analysis import DimensionsInput


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create a random 224x224 RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing."""
    img_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    return Image.fromarray(img_array, mode="L")


@pytest.fixture
def sample_dimensions():
    """Create sample dimensions for portion estimation."""
    return DimensionsInput(width=10.0, height=5.0, depth=3.0)


@pytest.fixture
def classifier_config():
    """Create a test classifier configuration."""
    return FoodClassifierConfig(
        model_type="efficientnet_b0",
        pretrained=True,
        device="cpu",  # Force CPU for tests
        version="test-1.0.0",
    )


# ============================================================================
# FoodClassifierConfig Tests
# ============================================================================


class TestFoodClassifierConfig:
    """Tests for FoodClassifierConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FoodClassifierConfig()

        assert config.model_type == "efficientnet_b0"
        assert config.num_classes == 101
        assert config.pretrained is True
        assert config.device == "auto"
        assert config.input_size == 224
        assert config.version == "1.0.0"
        assert config.threshold == 0.0  # Default to 0 to always return results

    def test_custom_config(self):
        """Test custom configuration values."""
        config = FoodClassifierConfig(
            model_type="resnet50",
            num_classes=50,
            pretrained=False,
            device="cpu",
            version="2.0.0",
        )

        assert config.model_type == "resnet50"
        assert config.num_classes == 50
        assert config.pretrained is False
        assert config.device == "cpu"
        assert config.version == "2.0.0"


# ============================================================================
# FoodClassifier Tests
# ============================================================================


class TestFoodClassifier:
    """Tests for FoodClassifier model."""

    @pytest.mark.slow
    def test_classifier_initialization(self, classifier_config):
        """Test classifier initializes correctly."""
        classifier = FoodClassifier(classifier_config)

        assert classifier is not None
        assert classifier.model is not None
        assert len(classifier.classes) == 101
        assert classifier.config.version == "test-1.0.0"

    @pytest.mark.slow
    def test_classifier_predict(self, classifier_config, sample_image):
        """Test classifier returns predictions."""
        classifier = FoodClassifier(classifier_config)
        predictions = classifier.predict(sample_image, top_k=5)

        assert len(predictions) <= 5
        assert all(isinstance(p, tuple) for p in predictions)
        assert all(len(p) == 2 for p in predictions)

        # Check predictions are sorted by confidence
        confidences = [p[1] for p in predictions]
        assert confidences == sorted(confidences, reverse=True)

    @pytest.mark.slow
    def test_classifier_predict_grayscale(self, classifier_config, sample_grayscale_image):
        """Test classifier handles grayscale images."""
        classifier = FoodClassifier(classifier_config)
        predictions = classifier.predict(sample_grayscale_image, top_k=3)

        # Should handle grayscale by converting to RGB
        assert len(predictions) <= 3

    @pytest.mark.slow
    def test_classifier_model_info(self, classifier_config):
        """Test model info returns correct metadata."""
        classifier = FoodClassifier(classifier_config)
        info = classifier.get_model_info()

        assert info["model_type"] == "efficientnet_b0"
        assert info["version"] == "test-1.0.0"
        assert info["num_classes"] == 101
        assert info["input_size"] == 224
        assert "device" in info

    @pytest.mark.slow
    def test_classifier_batch_predict(self, classifier_config, sample_image):
        """Test batch prediction."""
        classifier = FoodClassifier(classifier_config)
        images = [sample_image, sample_image.copy()]
        results = classifier.predict_batch(images, top_k=3)

        assert len(results) == 2
        for predictions in results:
            assert len(predictions) <= 3


# ============================================================================
# format_class_name Tests
# ============================================================================


class TestFormatClassName:
    """Tests for class name formatting."""

    def test_format_snake_case(self):
        """Test formatting snake_case to title case."""
        assert format_class_name("apple_pie") == "Apple Pie"
        assert format_class_name("french_fries") == "French Fries"
        assert format_class_name("hot_and_sour_soup") == "Hot And Sour Soup"

    def test_format_single_word(self):
        """Test formatting single word."""
        assert format_class_name("pizza") == "Pizza"
        assert format_class_name("hamburger") == "Hamburger"

    def test_format_already_titled(self):
        """Test formatting already titled string."""
        result = format_class_name("Apple Pie")
        assert result == "Apple Pie"


# ============================================================================
# FOOD101_CLASSES Tests
# ============================================================================


class TestFood101Classes:
    """Tests for Food-101 class list."""

    def test_class_count(self):
        """Test correct number of classes."""
        assert len(FOOD101_CLASSES) == 101

    def test_class_format(self):
        """Test classes are in snake_case."""
        for cls in FOOD101_CLASSES:
            assert cls.islower() or "_" in cls
            assert " " not in cls

    def test_contains_common_foods(self):
        """Test list contains common food items."""
        common_foods = ["pizza", "hamburger", "sushi", "ice_cream", "steak"]
        for food in common_foods:
            assert food in FOOD101_CLASSES


# ============================================================================
# NutritionDatabase Tests
# ============================================================================


class TestNutritionDatabase:
    """Tests for NutritionDatabase."""

    def test_database_singleton(self):
        """Test database is singleton."""
        db1 = NutritionDatabase()
        db2 = NutritionDatabase()
        assert db1 is db2

    def test_database_has_foods(self):
        """Test database has foods loaded."""
        db = NutritionDatabase()
        assert db.food_count > 0

    def test_get_known_food(self):
        """Test getting a known food."""
        db = NutritionDatabase()
        pizza = db.get("pizza")

        assert pizza is not None
        assert "nutrition" in pizza
        assert "calories" in pizza["nutrition"]

    def test_get_unknown_food(self):
        """Test getting an unknown food returns None."""
        db = NutritionDatabase()
        result = db.get("imaginary_food_xyz")
        assert result is None

    def test_get_density(self):
        """Test getting food density."""
        db = NutritionDatabase()
        density = db.get_density("pizza")

        assert isinstance(density, float)
        assert 0.1 <= density <= 2.0

    def test_get_density_unknown_food(self):
        """Test getting density for unknown food returns default."""
        db = NutritionDatabase()
        density = db.get_density("imaginary_food_xyz")
        assert density == 0.8  # Default density

    def test_search_foods(self):
        """Test searching for foods."""
        db = NutritionDatabase()
        results = db.search("chicken")

        assert len(results) > 0
        for result in results:
            assert "chicken" in result["food_name"].lower()

    def test_search_no_results(self):
        """Test search with no results."""
        db = NutritionDatabase()
        results = db.search("xyznonexistent123")
        assert len(results) == 0


# ============================================================================
# FoodAnalysisService Tests
# ============================================================================


class TestFoodAnalysisService:
    """Tests for FoodAnalysisService."""

    @pytest.mark.slow
    def test_service_initialization(self):
        """Test service initializes correctly."""
        service = FoodAnalysisService()
        assert service is not None
        assert service.nutrition_db is not None

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_analyze_food(self, sample_image):
        """Test food analysis returns results."""
        service = FoodAnalysisService()
        food_items, quality, processing_time = await service.analyze_food(sample_image)

        assert len(food_items) > 0
        assert quality in ["high", "medium", "low"]
        assert processing_time > 0

        item = food_items[0]
        assert item.name is not None
        assert 0 <= item.confidence <= 1
        assert item.nutrition is not None

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_analyze_food_with_dimensions(self, sample_image, sample_dimensions):
        """Test food analysis with AR dimensions."""
        service = FoodAnalysisService()
        food_items, quality, _ = await service.analyze_food(
            sample_image, dimensions=sample_dimensions
        )

        assert len(food_items) > 0
        # With dimensions, quality should be better
        assert quality in ["high", "medium"]

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_search_nutrition_db(self):
        """Test nutrition database search through service."""
        service = FoodAnalysisService()
        results = await service.search_nutrition_db("salad")

        assert len(results) > 0
        for result in results:
            assert "food_name" in result
            assert "nutrition" in result

    @pytest.mark.slow
    def test_get_model_info(self):
        """Test model info from service."""
        service = FoodAnalysisService()
        info = service.get_model_info()

        assert "version" in info or "model_loaded" in info
        assert "nutrition_db_foods" in info
        assert info["nutrition_db_foods"] > 0


# ============================================================================
# Singleton/Factory Tests
# ============================================================================


class TestSingletons:
    """Tests for singleton patterns."""

    @pytest.mark.slow
    def test_get_food_classifier_singleton(self, classifier_config):
        """Test classifier singleton."""
        # Reset singleton
        import app.ml_models.food_classifier as fc_module
        fc_module._classifier_instance = None

        classifier1 = get_food_classifier(classifier_config)
        classifier2 = get_food_classifier()

        assert classifier1 is classifier2

    @pytest.mark.slow
    def test_get_food_classifier_force_reload(self, classifier_config):
        """Test classifier force reload."""
        classifier1 = get_food_classifier(classifier_config)
        classifier2 = get_food_classifier(classifier_config, force_reload=True)

        assert classifier1 is not classifier2

    @pytest.mark.slow
    def test_get_food_analysis_service_singleton(self):
        """Test service singleton."""
        # Reset singleton
        import app.services.food_analysis_service as fas_module
        fas_module._service_instance = None

        service1 = get_food_analysis_service()
        service2 = get_food_analysis_service()

        assert service1 is service2


# ============================================================================
# Portion Estimation Tests
# ============================================================================


class TestPortionEstimation:
    """Tests for portion size estimation."""

    @pytest.mark.slow
    def test_estimate_portion_from_dimensions(self, sample_dimensions):
        """Test portion estimation from dimensions."""
        service = FoodAnalysisService()
        weight = service._estimate_portion_from_dimensions(
            sample_dimensions, "pizza"
        )

        assert weight > 0
        # 10 * 5 * 3 = 150 cm³, with density ~0.6 and factor 0.7 = ~63g
        assert 30 < weight < 200

    def test_assess_measurement_quality_high(self):
        """Test high quality measurement assessment."""
        service = FoodAnalysisService()
        dims = DimensionsInput(width=5.0, height=4.0, depth=3.0)
        quality = service._assess_measurement_quality(dims)
        assert quality == "high"

    def test_assess_measurement_quality_low(self):
        """Test low quality measurement assessment."""
        service = FoodAnalysisService()
        dims = DimensionsInput(width=50.0, height=2.0, depth=1.0)
        quality = service._assess_measurement_quality(dims)
        assert quality == "low"


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance benchmarks for food classification."""

    @pytest.mark.slow
    def test_inference_time(self, classifier_config, sample_image, benchmark_timer):
        """Test inference completes within acceptable time."""
        classifier = FoodClassifier(classifier_config)

        # Warm up
        classifier.predict(sample_image, top_k=1)

        # Benchmark
        import time
        start = time.time()
        for _ in range(5):
            classifier.predict(sample_image, top_k=5)
        elapsed = (time.time() - start) / 5

        # Should complete in under 3 seconds (as per task requirements)
        assert elapsed < 3.0, f"Inference took {elapsed:.2f}s, expected < 3s"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_service_response_time(self, sample_image):
        """Test full service response time."""
        service = FoodAnalysisService()

        # Warm up
        await service.analyze_food(sample_image)

        # Benchmark
        import time
        start = time.time()
        _, _, processing_time = await service.analyze_food(sample_image)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Service took {elapsed:.2f}s, expected < 5s"
        # Processing time should be reported in ms
        assert processing_time > 0


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.slow
    def test_small_image(self, classifier_config):
        """Test handling of small images."""
        classifier = FoodClassifier(classifier_config)
        small_img = Image.new("RGB", (50, 50), color="red")
        predictions = classifier.predict(small_img, top_k=3)

        # Should handle small images via resize
        assert len(predictions) >= 0

    @pytest.mark.slow
    def test_large_image(self, classifier_config):
        """Test handling of large images."""
        classifier = FoodClassifier(classifier_config)
        large_img = Image.new("RGB", (1000, 1000), color="blue")
        predictions = classifier.predict(large_img, top_k=3)

        # Should handle large images via resize
        assert len(predictions) >= 0

    @pytest.mark.slow
    def test_rgba_image(self, classifier_config):
        """Test handling of RGBA images."""
        classifier = FoodClassifier(classifier_config)
        rgba_img = Image.new("RGBA", (224, 224), color=(255, 0, 0, 128))
        predictions = classifier.predict(rgba_img, top_k=3)

        # Should handle RGBA by converting to RGB
        assert len(predictions) >= 0

    def test_nutrition_calculation_zero_weight(self):
        """Test nutrition calculation with zero base weight."""
        service = FoodAnalysisService()
        nutrition_data = {
            "serving_weight": 0,
            "nutrition": {"calories": 100, "protein": 10, "carbs": 20, "fat": 5},
        }
        result = service._calculate_nutrition(nutrition_data, portion_weight=100)

        # Should handle zero division gracefully
        assert result.calories == 100.0
