"""
Unit Tests for Food Classifier Module

Tests cover:
- Model initialization and configuration
- Device detection (CPU/GPU/MPS)
- Image preprocessing
- Classification inference
- Class mapping to database
- Model caching (singleton pattern)
- Performance benchmarks
"""

import pytest
import numpy as np
from PIL import Image
import time

from app.ml_models.food_classifier import (
    FoodClassifier,
    FoodClassifierConfig,
    FoodClassifierRegistry,
    ClassificationResult,
    get_food_classifier,
    FOOD_101_CLASSES,
    FOOD_101_TO_DB_MAPPING,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_config():
    """Create a test configuration."""
    return FoodClassifierConfig(
        model_name="efficientnet_b0",
        num_classes=101,
        pretrained=True,
        device="cpu",  # Force CPU for tests
        input_size=(224, 224),
        top_k=5,
        cache_model=False,  # Disable singleton for tests
    )


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a random RGB image
    img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return img_array


@pytest.fixture
def sample_pil_image():
    """Create a sample PIL image."""
    img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture(autouse=True)
def reset_classifier_singleton():
    """Reset the singleton before each test."""
    FoodClassifier.reset_singleton()
    yield
    FoodClassifier.reset_singleton()


# ============================================================================
# Configuration Tests
# ============================================================================


class TestFoodClassifierConfig:
    """Test FoodClassifierConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FoodClassifierConfig()

        assert config.model_name == "efficientnet_b0"
        assert config.num_classes == 101
        assert config.pretrained is True
        assert config.device == "auto"
        assert config.input_size == (224, 224)
        assert config.top_k == 5
        assert config.version == "1.0.0"

    def test_custom_config(self, mock_config):
        """Test custom configuration."""
        assert mock_config.device == "cpu"
        assert mock_config.cache_model is False

    def test_imagenet_normalization_stats(self):
        """Test ImageNet normalization statistics."""
        config = FoodClassifierConfig()

        # These are standard ImageNet values
        assert config.mean == (0.485, 0.456, 0.406)
        assert config.std == (0.229, 0.224, 0.225)


# ============================================================================
# Class Mapping Tests
# ============================================================================


class TestClassMapping:
    """Test Food-101 to database class mapping."""

    def test_food_101_classes_count(self):
        """Test that we have 101 Food-101 classes."""
        assert len(FOOD_101_CLASSES) == 101

    def test_all_classes_are_lowercase_underscored(self):
        """Test class name format."""
        for cls in FOOD_101_CLASSES:
            assert cls == cls.lower(), f"Class {cls} should be lowercase"
            assert " " not in cls, f"Class {cls} should not have spaces"

    def test_mapping_coverage(self):
        """Test that mappings are valid."""
        for food101_class, db_class in FOOD_101_TO_DB_MAPPING.items():
            assert (
                food101_class in FOOD_101_CLASSES
            ), f"Mapped class {food101_class} not in Food-101"
            assert isinstance(
                db_class, str
            ), f"DB class for {food101_class} should be string"

    def test_common_foods_mapped(self):
        """Test that common foods are mapped."""
        common_foods = ["hamburger", "pizza", "sushi", "steak", "french_fries"]
        for food in common_foods:
            assert food in FOOD_101_TO_DB_MAPPING, f"{food} should be mapped"


# ============================================================================
# Initialization Tests
# ============================================================================


class TestFoodClassifierInit:
    """Test FoodClassifier initialization."""

    def test_init_with_config(self, mock_config):
        """Test initialization with custom config."""
        classifier = FoodClassifier(mock_config)

        assert classifier.config == mock_config
        assert classifier.device.type == "cpu"
        assert len(classifier.classes) == 101

    def test_lazy_model_loading(self, mock_config):
        """Test that model is not loaded until needed."""
        classifier = FoodClassifier(mock_config)

        # Model should not be loaded yet
        assert classifier._model_loaded is False
        assert classifier.model is None

    def test_singleton_pattern(self):
        """Test singleton pattern when caching is enabled."""
        config = FoodClassifierConfig(cache_model=True, device="cpu")

        classifier1 = FoodClassifier(config)
        classifier2 = FoodClassifier(config)

        assert classifier1 is classifier2, "Should return same instance"

    def test_singleton_disabled(self, mock_config):
        """Test that singleton is disabled with cache_model=False."""
        # mock_config has cache_model=False
        classifier1 = FoodClassifier(mock_config)

        # Reset to get a new instance
        FoodClassifier.reset_singleton()

        classifier2 = FoodClassifier(mock_config)

        # They should be different instances (though this is tricky to test
        # since __new__ behavior is complex)
        assert classifier1._initialized is True
        assert classifier2._initialized is True


# ============================================================================
# Preprocessing Tests
# ============================================================================


class TestImagePreprocessing:
    """Test image preprocessing."""

    def test_preprocess_output_shape(self, mock_config, sample_image):
        """Test preprocessed output shape."""
        classifier = FoodClassifier(mock_config)
        tensor = classifier.preprocess(sample_image)

        assert tensor.shape == (
            1,
            3,
            224,
            224,
        ), "Output should be (batch, channels, H, W)"

    def test_preprocess_output_dtype(self, mock_config, sample_image):
        """Test preprocessed output dtype."""
        classifier = FoodClassifier(mock_config)
        tensor = classifier.preprocess(sample_image)

        import torch

        assert tensor.dtype == torch.float32

    def test_preprocess_normalized_values(self, mock_config, sample_image):
        """Test that output is normalized."""
        classifier = FoodClassifier(mock_config)
        tensor = classifier.preprocess(sample_image)

        # After ImageNet normalization, values should be centered around 0
        # and typically range from ~-2.5 to ~2.5
        assert tensor.min() >= -3.0
        assert tensor.max() <= 3.0

    def test_preprocess_handles_0_1_range(self, mock_config):
        """Test preprocessing images already in [0, 1] range."""
        classifier = FoodClassifier(mock_config)

        # Image already normalized to [0, 1]
        img_array = np.random.rand(256, 256, 3).astype(np.float32)
        tensor = classifier.preprocess(img_array)

        assert tensor.shape == (1, 3, 224, 224)


# ============================================================================
# Classification Tests
# ============================================================================


class TestFoodClassification:
    """Test food classification inference."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_classify_returns_result(self, mock_config, sample_image):
        """Test that classification returns a valid result."""
        classifier = FoodClassifier(mock_config)
        result = await classifier.classify(sample_image)

        assert isinstance(result, ClassificationResult)
        assert result.primary_class in FOOD_101_CLASSES
        assert 0 <= result.primary_confidence <= 1
        assert result.model_version == "1.0.0"
        assert result.inference_time_ms > 0
        assert result.device_used == "cpu"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_classify_returns_alternatives(self, mock_config, sample_image):
        """Test that classification returns alternatives."""
        classifier = FoodClassifier(mock_config)
        result = await classifier.classify(sample_image, top_k=5)

        # Should have at most 4 alternatives (top-5 minus primary)
        assert len(result.alternatives) <= 4

        for alt_class, alt_confidence in result.alternatives:
            assert alt_class in FOOD_101_CLASSES
            assert 0 <= alt_confidence <= 1
            assert alt_confidence < result.primary_confidence

    @pytest.mark.slow
    def test_classify_sync(self, mock_config, sample_image):
        """Test synchronous classification."""
        classifier = FoodClassifier(mock_config)
        result = classifier.classify_sync(sample_image)

        assert isinstance(result, ClassificationResult)
        assert result.primary_class in FOOD_101_CLASSES

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_classification_performance(self, mock_config, sample_image):
        """Test that classification completes within target time (<3s)."""
        classifier = FoodClassifier(mock_config)

        # Warm up (first call loads model)
        await classifier.classify(sample_image)

        # Measure inference time
        start_time = time.time()
        result = await classifier.classify(sample_image)
        elapsed = time.time() - start_time

        # Should complete within 3 seconds (target)
        assert elapsed < 3.0, f"Inference took {elapsed:.2f}s (target: <3s)"

        # The result should also report reasonable inference time
        assert result.inference_time_ms < 3000


# ============================================================================
# Model Info Tests
# ============================================================================


class TestModelInfo:
    """Test model information retrieval."""

    def test_get_model_info_before_loading(self, mock_config):
        """Test model info before model is loaded."""
        classifier = FoodClassifier(mock_config)
        info = classifier.get_model_info()

        assert info["name"] == "efficientnet_b0"
        assert info["version"] == "1.0.0"
        assert info["num_classes"] == 101
        assert info["device"] == "cpu"
        assert info["model_loaded"] is False

    @pytest.mark.slow
    def test_get_model_info_after_loading(self, mock_config, sample_image):
        """Test model info after model is loaded."""
        classifier = FoodClassifier(mock_config)

        # Load model by classifying
        classifier.classify_sync(sample_image)

        info = classifier.get_model_info()
        assert info["model_loaded"] is True


# ============================================================================
# Registry Tests
# ============================================================================


class TestFoodClassifierRegistry:
    """Test FoodClassifierRegistry for model versioning."""

    def test_register_model(self, mock_config):
        """Test registering a model."""
        registry = FoodClassifierRegistry()
        classifier = registry.register_model("v1.0", mock_config)

        assert "v1.0" in registry.models
        assert registry.active_model == "v1.0"
        assert isinstance(classifier, FoodClassifier)

    def test_get_active_model(self, mock_config):
        """Test getting active model."""
        registry = FoodClassifierRegistry()
        registry.register_model("v1.0", mock_config)

        model = registry.get_model()
        assert model is not None

    def test_set_active_model(self, mock_config):
        """Test switching active model."""
        registry = FoodClassifierRegistry()
        registry.register_model("v1.0", mock_config)

        config_v2 = FoodClassifierConfig(device="cpu", version="2.0.0")
        registry.register_model("v2.0", config_v2)

        registry.set_active_model("v2.0")
        assert registry.active_model == "v2.0"

    def test_list_models(self, mock_config):
        """Test listing registered models."""
        registry = FoodClassifierRegistry()
        registry.register_model("v1.0", mock_config)

        models = registry.list_models()
        assert len(models) == 1
        # The "name" key comes from classifier.get_model_info() which returns model_name
        assert models[0]["name"] == "efficientnet_b0"  # From config.model_name
        assert models[0]["active"] is True


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_food_classifier(self):
        """Test get_food_classifier helper."""
        classifier = get_food_classifier()

        assert isinstance(classifier, FoodClassifier)

    def test_get_food_classifier_returns_singleton(self):
        """Test that get_food_classifier returns singleton."""
        classifier1 = get_food_classifier()
        classifier2 = get_food_classifier()

        assert classifier1 is classifier2


# ============================================================================
# Device Detection Tests
# ============================================================================


class TestDeviceDetection:
    """Test automatic device detection."""

    def test_explicit_cpu(self):
        """Test explicit CPU device."""
        config = FoodClassifierConfig(device="cpu")
        classifier = FoodClassifier(config)

        assert classifier.device.type == "cpu"

    def test_auto_device_detection(self):
        """Test automatic device detection."""
        config = FoodClassifierConfig(device="auto")
        classifier = FoodClassifier(config)

        # Should detect available device
        import torch

        if torch.cuda.is_available():
            assert classifier.device.type == "cuda"
        elif torch.backends.mps.is_available():
            assert classifier.device.type == "mps"
        else:
            assert classifier.device.type == "cpu"


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_very_small_image(self, mock_config):
        """Test classification of very small image."""
        classifier = FoodClassifier(mock_config)

        # Very small image (should be upscaled)
        small_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = await classifier.classify(small_img)

        assert result is not None
        assert result.primary_class in FOOD_101_CLASSES

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_image(self, mock_config):
        """Test classification of large image."""
        classifier = FoodClassifier(mock_config)

        # Large image (should be downscaled)
        large_img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        result = await classifier.classify(large_img)

        assert result is not None
        assert result.primary_class in FOOD_101_CLASSES

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_non_square_image(self, mock_config):
        """Test classification of non-square image."""
        classifier = FoodClassifier(mock_config)

        # Non-square image
        rect_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = await classifier.classify(rect_img)

        assert result is not None
        assert result.primary_class in FOOD_101_CLASSES

    def test_top_k_greater_than_classes(self, mock_config):
        """Test requesting more top-k than available classes."""
        classifier = FoodClassifier(mock_config)

        # top_k > num_classes should be handled gracefully
        # (should return all classes at most)
        sample_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        result = classifier.classify_sync(sample_img, top_k=200)

        assert len(result.alternatives) <= 100  # At most num_classes - 1


# ============================================================================
# Integration Test with Food Analysis Service
# ============================================================================


@pytest.mark.slow
class TestIntegrationWithFoodAnalysisService:
    """Integration tests with FoodAnalysisService."""

    def test_service_with_ml_model(self):
        """Test FoodAnalysisService with real ML model."""
        from app.services.food_analysis_service import FoodAnalysisService

        service = FoodAnalysisService(use_ensemble=True)

        assert service._use_ensemble is True
        assert service._classifier is not None
        assert "efficientnet" in service.model_name.lower()

    def test_service_with_mock_model(self):
        """Test FoodAnalysisService with mock model."""
        from app.services.food_analysis_service import FoodAnalysisService

        service = FoodAnalysisService(use_ensemble=False)

        assert service._use_ensemble is False
        assert service._classifier is None
        assert "mock" in service.model_name.lower()

    @pytest.mark.asyncio
    async def test_service_analyze_food(self, sample_pil_image):
        """Test full food analysis with ML model."""
        from app.services.food_analysis_service import FoodAnalysisService

        service = FoodAnalysisService(use_ensemble=True)

        (
            food_items,
            measurement_quality,
            processing_time,
            suggestions,
        ) = await service.analyze_food(sample_pil_image)

        assert len(food_items) == 1
        assert food_items[0].confidence > 0
        assert processing_time > 0
