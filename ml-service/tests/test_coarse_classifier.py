"""
Tests for Coarse Food Category Classifier.

Tests cover:
- Category classification accuracy
- USDA data type mapping
- Model info retrieval
- Edge cases and error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io
import numpy as np


class TestFoodCategory:
    """Tests for FoodCategory enum."""

    def test_all_categories_exist(self):
        """Verify all expected categories are defined."""
        from app.ml_models.coarse_classifier import FoodCategory

        expected_categories = [
            "fruits_fresh", "fruits_processed",
            "vegetables_leafy", "vegetables_root", "vegetables_other", "vegetables_cooked",
            "meat_red", "meat_poultry", "meat_processed",
            "seafood_fish", "seafood_shellfish",
            "dairy_milk", "dairy_cheese", "dairy_yogurt", "dairy_other",
            "grains_bread", "grains_pasta", "grains_rice", "grains_cereal", "grains_other",
            "legumes", "nuts_seeds",
            "beverages_hot", "beverages_cold",
            "snacks_sweet", "snacks_savory",
            "mixed_dishes", "fast_food",
            "condiments_sauces", "eggs", "unknown"
        ]

        for cat_name in expected_categories:
            assert hasattr(FoodCategory, cat_name.upper()), f"Missing category: {cat_name}"

    def test_category_values_are_strings(self):
        """All category values should be strings for JSON serialization."""
        from app.ml_models.coarse_classifier import FoodCategory

        for cat in FoodCategory:
            assert isinstance(cat.value, str)


class TestUSDADataTypeMapping:
    """Tests for USDA data type mappings."""

    def test_all_categories_have_mapping(self):
        """Every category should have USDA data type mapping."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        for cat in FoodCategory:
            assert cat in CATEGORY_TO_USDA_DATATYPES, f"Missing mapping for {cat}"

    def test_mappings_contain_valid_datatypes(self):
        """All mapped data types should be valid USDA types."""
        from app.ml_models.coarse_classifier import CATEGORY_TO_USDA_DATATYPES

        valid_types = {"Foundation", "SR Legacy", "Survey (FNDDS)", "Branded"}

        for category, datatypes in CATEGORY_TO_USDA_DATATYPES.items():
            for dt in datatypes:
                assert dt in valid_types, f"Invalid data type {dt} for {category}"

    def test_fresh_foods_prefer_foundation(self):
        """Fresh/whole foods should prefer Foundation or SR Legacy."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        fresh_categories = [
            FoodCategory.FRUITS_FRESH,
            FoodCategory.VEGETABLES_LEAFY,
            FoodCategory.VEGETABLES_ROOT,
            FoodCategory.MEAT_RED,
            FoodCategory.SEAFOOD_FISH,
            FoodCategory.EGGS,
        ]

        for cat in fresh_categories:
            datatypes = CATEGORY_TO_USDA_DATATYPES[cat]
            assert "Foundation" in datatypes or "SR Legacy" in datatypes, \
                f"Fresh category {cat} should prefer Foundation/SR Legacy"

    def test_branded_categories_include_branded(self):
        """Packaged food categories should include Branded data type."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        branded_categories = [
            FoodCategory.SNACKS_SWEET,
            FoodCategory.SNACKS_SAVORY,
            FoodCategory.FAST_FOOD,
        ]

        for cat in branded_categories:
            datatypes = CATEGORY_TO_USDA_DATATYPES[cat]
            assert "Branded" in datatypes, \
                f"Branded category {cat} should include Branded data type"


class TestCategoryPrompts:
    """Tests for CLIP prompts."""

    def test_all_categories_have_prompts(self):
        """Every category should have associated prompts."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_PROMPTS,
        )

        for cat in FoodCategory:
            assert cat in CATEGORY_PROMPTS, f"Missing prompts for {cat}"
            assert len(CATEGORY_PROMPTS[cat]) > 0, f"Empty prompts for {cat}"

    def test_prompts_are_descriptive(self):
        """Prompts should be descriptive photo descriptions."""
        from app.ml_models.coarse_classifier import CATEGORY_PROMPTS

        for category, prompts in CATEGORY_PROMPTS.items():
            for prompt in prompts:
                assert isinstance(prompt, str)
                assert len(prompt) > 10, f"Prompt too short for {category}: {prompt}"
                # Most prompts should mention "photo" or describe food
                assert "photo" in prompt.lower() or "food" in prompt.lower() or \
                    any(word in prompt.lower() for word in ["fruit", "vegetable", "meat", "milk"]), \
                    f"Prompt doesn't look like a food description: {prompt}"


class TestCoarseClassification:
    """Tests for CoarseClassification dataclass."""

    def test_classification_has_required_fields(self):
        """CoarseClassification should have all required fields."""
        from app.ml_models.coarse_classifier import (
            CoarseClassification,
            FoodCategory,
        )

        result = CoarseClassification(
            category=FoodCategory.FRUITS_FRESH,
            confidence=0.85,
            subcategory_hints=["appears fresh"],
            usda_datatypes=["Foundation", "SR Legacy"],
            alternatives=[(FoodCategory.VEGETABLES_OTHER, 0.1)],
            texture_features={},
        )

        assert result.category == FoodCategory.FRUITS_FRESH
        assert result.confidence == 0.85
        assert result.subcategory_hints == ["appears fresh"]
        assert "Foundation" in result.usda_datatypes


class TestCoarseFoodClassifier:
    """Tests for CoarseFoodClassifier class."""

    def test_singleton_instance(self):
        """get_coarse_classifier should return singleton."""
        from app.ml_models.coarse_classifier import get_coarse_classifier

        classifier1 = get_coarse_classifier()
        classifier2 = get_coarse_classifier()

        assert classifier1 is classifier2

    def test_model_info_structure(self):
        """get_model_info should return expected structure."""
        from app.ml_models.coarse_classifier import CoarseFoodClassifier

        classifier = CoarseFoodClassifier()
        info = classifier.get_model_info()

        assert "name" in info
        assert "version" in info
        assert "model" in info
        assert "num_categories" in info
        assert "categories" in info
        assert "target_accuracy" in info

        # Should have around 30 categories (excluding UNKNOWN)
        assert 25 <= info["num_categories"] <= 35

    def test_classify_with_usda_context_structure(self):
        """classify_with_usda_context should return expected structure."""
        from app.ml_models.coarse_classifier import CoarseFoodClassifier

        classifier = CoarseFoodClassifier()

        # Mock the classify method
        with patch.object(classifier, 'classify') as mock_classify:
            from app.ml_models.coarse_classifier import (
                CoarseClassification,
                FoodCategory,
            )
            mock_classify.return_value = CoarseClassification(
                category=FoodCategory.FRUITS_FRESH,
                confidence=0.85,
                subcategory_hints=["appears fresh"],
                usda_datatypes=["Foundation", "SR Legacy"],
                alternatives=[(FoodCategory.VEGETABLES_OTHER, 0.1)],
                texture_features={},
            )

            # Create test image
            test_image = Image.new('RGB', (224, 224), color='red')

            result = classifier.classify_with_usda_context(test_image, "apple")

            assert "category" in result
            assert "confidence" in result
            assert "usda_datatypes" in result
            assert "search_hints" in result
            assert "alternatives" in result

    def test_device_detection(self):
        """Should detect correct device (cpu, cuda, or mps)."""
        from app.ml_models.coarse_classifier import CoarseFoodClassifier

        classifier = CoarseFoodClassifier()
        assert classifier._device in ["cpu", "cuda", "mps"]


class TestQueryEnhancement:
    """Tests for query enhancement functionality."""

    def test_get_query_enhancement_for_fruits(self):
        """Fresh fruits should get 'raw' enhancement."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()
        result = classifier._get_query_enhancement(FoodCategory.FRUITS_FRESH, "apple")

        assert "raw" in result.lower() or "apple" in result.lower()

    def test_get_query_enhancement_preserves_query(self):
        """Query enhancement should preserve original query."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()
        result = classifier._get_query_enhancement(FoodCategory.GRAINS_PASTA, "spaghetti")

        assert "spaghetti" in result.lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_classify_returns_unknown_on_error(self):
        """Classification errors should return UNKNOWN category."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()

        # Create invalid image
        invalid_image = Image.new('L', (10, 10))  # Grayscale, small

        # Mock model to raise exception
        with patch.object(classifier, '_model', None):
            with patch.object(classifier, '_loaded', False):
                with patch.object(classifier, 'load_model', side_effect=Exception("Model load failed")):
                    result = classifier.classify(invalid_image)

                    assert result.category == FoodCategory.UNKNOWN
                    assert result.confidence == 0.0

    def test_hint_generation_limits_results(self):
        """Hint generation should limit to top 3 hints."""
        from app.ml_models.coarse_classifier import CoarseFoodClassifier
        import torch

        classifier = CoarseFoodClassifier()

        # Create mock data
        prompts = ["a photo of grilled fresh raw sliced whole food"] * 10
        prompt_to_category = {p: None for p in prompts}
        similarities = torch.tensor([0.5] * 10)

        hints = classifier._generate_hints(
            similarities,
            prompts,
            prompt_to_category,
            None  # best_category
        )

        # Should be limited to max 3 hints
        assert len(hints) <= 3
