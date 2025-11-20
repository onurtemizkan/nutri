"""
Unit tests for Food Analysis Service.
"""
import pytest
import numpy as np
from PIL import Image

from app.services.food_analysis_service import FoodAnalysisService, NUTRITION_DATABASE
from app.schemas.food_analysis import DimensionsInput, NutritionInfo


@pytest.fixture
def service():
    """Create a FoodAnalysisService instance."""
    return FoodAnalysisService()


class TestFoodAnalysisService:
    """Test suite for FoodAnalysisService."""

    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert service.model_name == "mock-food-classifier-v1"
        assert service.model is None  # Mock implementation

    @pytest.mark.asyncio
    async def test_analyze_food_without_measurements(
        self, service, sample_food_image
    ):
        """Test food analysis without AR measurements."""
        # Act
        food_items, quality, processing_time = await service.analyze_food(
            sample_food_image, dimensions=None
        )

        # Assert
        assert len(food_items) == 1
        assert food_items[0].name in [f.title() for f in NUTRITION_DATABASE.keys()]
        assert 0.0 <= food_items[0].confidence <= 1.0
        assert food_items[0].portion_weight > 0
        assert food_items[0].nutrition.calories > 0
        assert quality == "low"  # No measurements = low quality
        assert processing_time > 0

    @pytest.mark.asyncio
    async def test_analyze_food_with_measurements(
        self, service, sample_food_image, good_ar_measurements
    ):
        """Test food analysis with AR measurements."""
        # Act
        food_items, quality, processing_time = await service.analyze_food(
            sample_food_image, dimensions=good_ar_measurements
        )

        # Assert
        assert len(food_items) == 1
        assert food_items[0].portion_weight > 0
        assert quality in ["high", "medium", "low"]
        assert processing_time > 0

    @pytest.mark.asyncio
    async def test_analyze_food_returns_alternatives(
        self, service, sample_food_image
    ):
        """Test that analysis returns alternative food suggestions."""
        # Act
        food_items, _, _ = await service.analyze_food(
            sample_food_image, dimensions=None
        )

        # Assert
        assert food_items[0].alternatives is not None
        assert len(food_items[0].alternatives) >= 1
        for alt in food_items[0].alternatives:
            assert 0.0 < alt.confidence < food_items[0].confidence

    def test_preprocess_image_rgb(self, service, sample_food_image):
        """Test image preprocessing with RGB image."""
        # Act
        processed = service._preprocess_image(sample_food_image)

        # Assert
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.float32
        assert -5 < processed.mean() < 5  # Normalized values

    def test_preprocess_image_grayscale(self, service, grayscale_food_image):
        """Test image preprocessing converts grayscale to RGB."""
        # Act
        processed = service._preprocess_image(grayscale_food_image)

        # Assert
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.float32

    def test_preprocess_image_large(self, service, large_food_image):
        """Test image preprocessing resizes large images."""
        # Act
        processed = service._preprocess_image(large_food_image)

        # Assert
        assert processed.shape == (224, 224, 3)  # Resized to model input size

    @pytest.mark.asyncio
    async def test_classify_food_returns_valid_class(self, service, mock_image_array):
        """Test classification returns a valid food class."""
        # Act
        food_class, confidence, alternatives = await service._classify_food(
            mock_image_array
        )

        # Assert
        assert food_class in NUTRITION_DATABASE.keys()
        assert 0.0 <= confidence <= 1.0
        assert isinstance(alternatives, list)

    @pytest.mark.asyncio
    async def test_classify_food_confidence_range(self, service, mock_image_array):
        """Test classification confidence is in reasonable range."""
        # Act
        _, confidence, _ = await service._classify_food(mock_image_array)

        # Assert - Mock should return confidence between 0.75-0.95
        assert 0.75 <= confidence <= 0.95

    def test_estimate_portion_from_dimensions(
        self, service, good_ar_measurements
    ):
        """Test portion estimation from AR measurements."""
        # Act
        weight = service._estimate_portion_from_dimensions(
            good_ar_measurements, "apple"
        )

        # Assert
        assert weight > 0
        assert 50 < weight < 1000  # Reasonable range for food items (grams)

    def test_estimate_portion_different_foods(self, service, good_ar_measurements):
        """Test portion estimation varies by food type."""
        # Act
        apple_weight = service._estimate_portion_from_dimensions(
            good_ar_measurements, "apple"
        )
        broccoli_weight = service._estimate_portion_from_dimensions(
            good_ar_measurements, "broccoli"
        )

        # Assert - Broccoli is lighter (lower density)
        assert broccoli_weight < apple_weight

    def test_estimate_portion_scales_with_dimensions(self, service):
        """Test portion weight scales with dimensions."""
        # Arrange
        small_dims = DimensionsInput(width=5, height=5, depth=5)
        large_dims = DimensionsInput(width=10, height=10, depth=10)

        # Act
        small_weight = service._estimate_portion_from_dimensions(small_dims, "apple")
        large_weight = service._estimate_portion_from_dimensions(large_dims, "apple")

        # Assert - Larger dimensions = more weight
        # Volume ratio is 8:1, so weight should be approximately 8x
        assert large_weight > small_weight * 5  # Allow some variation

    def test_calculate_nutrition_base_portion(self, service):
        """Test nutrition calculation for base portion."""
        # Arrange
        base_data = NUTRITION_DATABASE["apple"]

        # Act
        nutrition = service._calculate_nutrition(
            "apple", base_data["serving_weight"]
        )

        # Assert
        assert nutrition.calories == base_data["nutrition"]["calories"]
        assert nutrition.protein == base_data["nutrition"]["protein"]
        assert nutrition.carbs == base_data["nutrition"]["carbs"]
        assert nutrition.fat == base_data["nutrition"]["fat"]

    def test_calculate_nutrition_scaled_portion(self, service):
        """Test nutrition calculation scales correctly."""
        # Arrange
        base_data = NUTRITION_DATABASE["apple"]
        base_weight = base_data["serving_weight"]  # 182g
        actual_weight = 364  # Double the base

        # Act
        nutrition = service._calculate_nutrition("apple", actual_weight)

        # Assert - Should be approximately double
        expected_calories = base_data["nutrition"]["calories"] * 2
        assert abs(nutrition.calories - expected_calories) < 1.0

    def test_calculate_nutrition_smaller_portion(self, service):
        """Test nutrition calculation for smaller portion."""
        # Arrange
        base_data = NUTRITION_DATABASE["chicken breast"]
        base_weight = base_data["serving_weight"]  # 100g
        actual_weight = 50  # Half the base

        # Act
        nutrition = service._calculate_nutrition("chicken breast", actual_weight)

        # Assert - Should be approximately half
        expected_protein = base_data["nutrition"]["protein"] / 2
        assert abs(nutrition.protein - expected_protein) < 0.5

    def test_calculate_nutrition_includes_optional_fields(self, service):
        """Test nutrition calculation includes fiber and sugar when available."""
        # Act
        nutrition = service._calculate_nutrition("apple", 182)

        # Assert
        assert nutrition.fiber is not None
        assert nutrition.sugar is not None
        assert nutrition.fiber > 0
        assert nutrition.sugar > 0

    def test_calculate_nutrition_handles_missing_optional_fields(self, service):
        """Test nutrition calculation when optional fields are missing."""
        # Act
        nutrition = service._calculate_nutrition("chicken breast", 100)

        # Assert - Chicken doesn't have sugar in NUTRITION_DATABASE
        assert nutrition.fiber == 0 or nutrition.fiber is None

    def test_assess_measurement_quality_high(self, service, good_ar_measurements):
        """Test measurement quality assessment returns 'high' for good measurements."""
        # Act
        quality = service._assess_measurement_quality(good_ar_measurements)

        # Assert
        assert quality == "high"

    def test_assess_measurement_quality_low(self, service, poor_ar_measurements):
        """Test measurement quality assessment returns 'low' for poor measurements."""
        # Act
        quality = service._assess_measurement_quality(poor_ar_measurements)

        # Assert
        assert quality == "low"

    def test_assess_measurement_quality_proportions(self, service):
        """Test measurement quality based on dimension proportions."""
        # Arrange - Very disproportionate (ratio > 10)
        bad_dims = DimensionsInput(width=50, height=3, depth=2)

        # Act
        quality = service._assess_measurement_quality(bad_dims)

        # Assert
        assert quality == "low"

    @pytest.mark.asyncio
    async def test_search_nutrition_db_exact_match(self, service):
        """Test nutrition database search with exact match."""
        # Act
        results = await service.search_nutrition_db("apple")

        # Assert
        assert len(results) == 1
        assert results[0]["food_name"] == "Apple"

    @pytest.mark.asyncio
    async def test_search_nutrition_db_partial_match(self, service):
        """Test nutrition database search with partial match."""
        # Act
        results = await service.search_nutrition_db("chick")

        # Assert
        assert len(results) >= 1
        assert any("Chicken" in r["food_name"] for r in results)

    @pytest.mark.asyncio
    async def test_search_nutrition_db_no_match(self, service):
        """Test nutrition database search with no matches."""
        # Act
        results = await service.search_nutrition_db("pizza")

        # Assert
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_nutrition_db_case_insensitive(self, service):
        """Test nutrition database search is case insensitive."""
        # Act
        results_lower = await service.search_nutrition_db("apple")
        results_upper = await service.search_nutrition_db("APPLE")
        results_mixed = await service.search_nutrition_db("ApPlE")

        # Assert
        assert len(results_lower) == len(results_upper) == len(results_mixed)

    def test_get_model_info(self, service):
        """Test getting model information."""
        # Act
        info = service.get_model_info()

        # Assert
        assert info["name"] == "mock-food-classifier-v1"
        assert info["version"] == "1.0.0"
        assert "accuracy" in info
        assert info["num_classes"] == len(NUTRITION_DATABASE)
        assert "description" in info

    @pytest.mark.asyncio
    async def test_analyze_food_processing_time_reasonable(
        self, service, sample_food_image
    ):
        """Test that processing time is within reasonable bounds."""
        # Act
        _, _, processing_time = await service.analyze_food(
            sample_food_image, dimensions=None
        )

        # Assert - Should be fast for mock implementation
        assert 0 < processing_time < 5000  # Less than 5 seconds

    @pytest.mark.asyncio
    async def test_analyze_food_with_all_food_types(
        self, service, sample_food_image, all_food_classes
    ):
        """Test analysis works for all supported food types."""
        # This is a probabilistic test since classification is random
        # Run multiple times to get different food classes
        found_classes = set()

        for _ in range(20):  # Run 20 times to likely hit all classes
            food_items, _, _ = await service.analyze_food(
                sample_food_image, dimensions=None
            )
            found_classes.add(food_items[0].name.lower())

        # Assert - Should have found multiple different classes
        assert len(found_classes) >= 3


class TestNutritionDatabase:
    """Test suite for nutrition database structure."""

    def test_nutrition_database_completeness(self):
        """Test all foods have required nutrition fields."""
        for food_name, data in NUTRITION_DATABASE.items():
            assert "category" in data
            assert "serving_size" in data
            assert "serving_weight" in data
            assert "nutrition" in data

            nutrition = data["nutrition"]
            assert "calories" in nutrition
            assert "protein" in nutrition
            assert "carbs" in nutrition
            assert "fat" in nutrition

    def test_nutrition_database_values_positive(self):
        """Test all nutrition values are non-negative."""
        for food_name, data in NUTRITION_DATABASE.items():
            nutrition = data["nutrition"]
            assert nutrition["calories"] >= 0
            assert nutrition["protein"] >= 0
            assert nutrition["carbs"] >= 0
            assert nutrition["fat"] >= 0
            assert data["serving_weight"] > 0

    def test_nutrition_database_categories(self):
        """Test all foods have valid categories."""
        valid_categories = {"fruit", "vegetable", "protein", "grain", "dairy"}

        for food_name, data in NUTRITION_DATABASE.items():
            assert data["category"] in valid_categories


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_analyze_food_with_zero_dimensions(self, service, sample_food_image):
        """Test handling of invalid (zero) dimensions."""
        # This should be caught by Pydantic validation,
        # but test service behavior if it somehow gets through
        invalid_dims = DimensionsInput(width=0.1, height=0.1, depth=0.1)

        # Act
        food_items, quality, _ = await service.analyze_food(
            sample_food_image, dimensions=invalid_dims
        )

        # Assert - Should still return a result
        assert len(food_items) == 1
        assert food_items[0].portion_weight > 0

    @pytest.mark.asyncio
    async def test_analyze_food_with_extreme_dimensions(
        self, service, sample_food_image
    ):
        """Test handling of extreme dimensions."""
        extreme_dims = DimensionsInput(width=99, height=99, depth=99)

        # Act
        food_items, quality, _ = await service.analyze_food(
            sample_food_image, dimensions=extreme_dims
        )

        # Assert
        assert len(food_items) == 1
        assert quality == "low"  # Should be low due to size
        # Weight should be capped or reasonable
        assert food_items[0].portion_weight < 10000  # Less than 10kg
