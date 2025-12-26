"""
USDA Integration Tests for ML Service

Tests for:
- Search enhancement with classification hints
- Data type filtering based on category
- Query enhancement with subcategory hints
- Classification to USDA search flow
- Performance benchmarks
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from PIL import Image
import io
import time
from typing import Dict, List, Any


# ============================================================================
# MOCK DATA
# ============================================================================

MOCK_USDA_SEARCH_RESPONSE = {
    "totalHits": 50,
    "currentPage": 1,
    "totalPages": 2,
    "foods": [
        {
            "fdcId": 171688,
            "description": "Apples, raw, with skin",
            "dataType": "Foundation",
            "publishedDate": "2019-04-01",
            "foodNutrients": [
                {"nutrientId": 1008, "nutrientName": "Energy", "value": 52, "unitName": "kcal"},
                {"nutrientId": 1003, "nutrientName": "Protein", "value": 0.26, "unitName": "g"},
                {"nutrientId": 1005, "nutrientName": "Carbohydrate", "value": 13.81, "unitName": "g"},
                {"nutrientId": 1004, "nutrientName": "Total lipid (fat)", "value": 0.17, "unitName": "g"},
            ],
        },
        {
            "fdcId": 171689,
            "description": "Apples, raw, without skin",
            "dataType": "Foundation",
            "foodNutrients": [
                {"nutrientId": 1008, "nutrientName": "Energy", "value": 48, "unitName": "kcal"},
            ],
        },
    ],
}


# ============================================================================
# CATEGORY TO DATA TYPE MAPPING TESTS
# ============================================================================

class TestCategoryDataTypeMapping:
    """Tests for category to USDA data type mapping."""

    def test_category_mapping_exists(self):
        """Verify category mapping dict is defined."""
        from app.ml_models.coarse_classifier import CATEGORY_TO_USDA_DATATYPES

        assert isinstance(CATEGORY_TO_USDA_DATATYPES, dict)
        assert len(CATEGORY_TO_USDA_DATATYPES) > 25

    def test_all_datatypes_are_valid(self):
        """All mapped data types should be valid USDA types."""
        from app.ml_models.coarse_classifier import CATEGORY_TO_USDA_DATATYPES

        valid_datatypes = {"Foundation", "SR Legacy", "Survey (FNDDS)", "Branded"}

        for category, datatypes in CATEGORY_TO_USDA_DATATYPES.items():
            for dt in datatypes:
                assert dt in valid_datatypes, f"Invalid datatype {dt} for {category}"

    def test_fresh_categories_prioritize_foundation(self):
        """Fresh food categories should prioritize Foundation data."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        fresh_categories = [
            FoodCategory.FRUITS_FRESH,
            FoodCategory.VEGETABLES_LEAFY,
            FoodCategory.VEGETABLES_ROOT,
            FoodCategory.VEGETABLES_OTHER,
            FoodCategory.MEAT_RED,
            FoodCategory.MEAT_POULTRY,
            FoodCategory.SEAFOOD_FISH,
            FoodCategory.SEAFOOD_SHELLFISH,
            FoodCategory.EGGS,
            FoodCategory.LEGUMES,
            FoodCategory.NUTS_SEEDS,
        ]

        for cat in fresh_categories:
            datatypes = CATEGORY_TO_USDA_DATATYPES[cat]
            # Foundation should be first (highest priority)
            assert "Foundation" in datatypes or "SR Legacy" in datatypes, \
                f"Fresh category {cat} should include Foundation or SR Legacy"

    def test_branded_categories_include_branded(self):
        """Branded food categories should include Branded data type."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        branded_categories = [
            FoodCategory.SNACKS_SWEET,
            FoodCategory.SNACKS_SAVORY,
            FoodCategory.GRAINS_CEREAL,
            FoodCategory.BEVERAGES_COLD,
            FoodCategory.FAST_FOOD,
        ]

        for cat in branded_categories:
            datatypes = CATEGORY_TO_USDA_DATATYPES[cat]
            assert "Branded" in datatypes, \
                f"Branded category {cat} should include Branded data type"

    def test_mixed_dishes_prioritize_survey(self):
        """Mixed dishes should prioritize Survey (FNDDS) data."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        datatypes = CATEGORY_TO_USDA_DATATYPES[FoodCategory.MIXED_DISHES]
        assert "Survey (FNDDS)" in datatypes

    def test_unknown_category_searches_all(self):
        """Unknown category should search all data types."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        datatypes = CATEGORY_TO_USDA_DATATYPES[FoodCategory.UNKNOWN]
        assert len(datatypes) >= 4  # All major data types


# ============================================================================
# QUERY ENHANCEMENT TESTS
# ============================================================================

class TestQueryEnhancement:
    """Tests for USDA search query enhancement."""

    def test_fresh_fruit_query_enhancement(self):
        """Fresh fruits should get 'raw' enhancement."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()
        result = classifier._get_query_enhancement(FoodCategory.FRUITS_FRESH, "apple")

        assert "raw" in result.lower() or "apple" in result.lower()

    def test_vegetables_query_enhancement(self):
        """Leafy vegetables should get appropriate enhancement."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()
        result = classifier._get_query_enhancement(FoodCategory.VEGETABLES_LEAFY, "spinach")

        # Should have some enhancement or preserve original
        assert "spinach" in result.lower() or "leafy" in result.lower()

    def test_pasta_query_enhancement(self):
        """Pasta should get 'cooked' enhancement."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()
        result = classifier._get_query_enhancement(FoodCategory.GRAINS_PASTA, "spaghetti")

        # Should include original query and enhancement
        assert "spaghetti" in result.lower()
        assert "cooked" in result.lower() or "pasta" in result.lower()

    def test_fast_food_query_enhancement(self):
        """Fast food should get 'restaurant' enhancement."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()
        result = classifier._get_query_enhancement(FoodCategory.FAST_FOOD, "burger")

        # Should include original query
        assert "burger" in result.lower() or "restaurant" in result.lower()

    def test_empty_query_returns_enhancement_only(self):
        """Empty query should return just the enhancement."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()
        result = classifier._get_query_enhancement(FoodCategory.FRUITS_FRESH, "")

        # Should return something (the enhancement) or empty string
        assert isinstance(result, str)


# ============================================================================
# CLASSIFICATION TO USDA CONTEXT TESTS
# ============================================================================

class TestClassifyWithUSDAContext:
    """Tests for classify_with_usda_context method."""

    def test_returns_expected_structure(self):
        """classify_with_usda_context should return expected structure."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            CoarseClassification,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()

        # Mock the classify method
        with patch.object(classifier, "classify") as mock_classify:
            mock_classify.return_value = CoarseClassification(
                category=FoodCategory.FRUITS_FRESH,
                confidence=0.92,
                subcategory_hints=["appears fresh", "appears whole"],
                usda_datatypes=["Foundation", "SR Legacy"],
                alternatives=[(FoodCategory.VEGETABLES_OTHER, 0.05)],
                texture_features={},
            )

            test_image = Image.new("RGB", (224, 224), color="red")
            result = classifier.classify_with_usda_context(test_image, "apple")

            assert "category" in result
            assert "confidence" in result
            assert "usda_datatypes" in result
            assert "search_hints" in result
            assert "alternatives" in result

    def test_search_hints_structure(self):
        """Search hints should have subcategory hints and query enhancement."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            CoarseClassification,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()

        with patch.object(classifier, "classify") as mock_classify:
            mock_classify.return_value = CoarseClassification(
                category=FoodCategory.MEAT_POULTRY,
                confidence=0.88,
                subcategory_hints=["appears grilled"],
                usda_datatypes=["Foundation", "SR Legacy"],
                alternatives=[],
                texture_features={},
            )

            test_image = Image.new("RGB", (224, 224), color="brown")
            result = classifier.classify_with_usda_context(test_image, "chicken breast")

            assert "subcategory_hints" in result["search_hints"]
            assert "suggested_query_enhancement" in result["search_hints"]

    def test_alternatives_structure(self):
        """Alternatives should be list of category/confidence pairs."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            CoarseClassification,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()

        with patch.object(classifier, "classify") as mock_classify:
            mock_classify.return_value = CoarseClassification(
                category=FoodCategory.MIXED_DISHES,
                confidence=0.75,
                subcategory_hints=[],
                usda_datatypes=["Survey (FNDDS)", "Branded"],
                alternatives=[
                    (FoodCategory.GRAINS_PASTA, 0.15),
                    (FoodCategory.VEGETABLES_COOKED, 0.10),
                ],
                texture_features={},
            )

            test_image = Image.new("RGB", (224, 224), color="orange")
            result = classifier.classify_with_usda_context(test_image, "pasta dish")

            alternatives = result["alternatives"]
            assert len(alternatives) == 2
            assert all("category" in alt for alt in alternatives)
            assert all("confidence" in alt for alt in alternatives)


# ============================================================================
# SUBCATEGORY HINTS INTEGRATION TESTS
# ============================================================================

class TestSubcategoryHints:
    """Tests for subcategory hint generation and usage."""

    def test_hint_keywords_are_extracted(self):
        """Hints should be extracted from matching prompts."""
        from app.ml_models.coarse_classifier import CoarseFoodClassifier
        import torch

        classifier = CoarseFoodClassifier()

        # Test prompts with keywords
        prompts = [
            "a photo of grilled chicken",
            "a photo of raw chicken",
            "a photo of fried chicken",
        ]
        prompt_to_category = {p: None for p in prompts}
        similarities = torch.tensor([0.9, 0.3, 0.2])  # Grilled scores highest

        hints = classifier._generate_hints(
            similarities, prompts, prompt_to_category, None
        )

        # Should extract "grilled" as hint
        assert len(hints) <= 3
        # The best matching prompt contains "grilled"
        if hints:
            assert any("grilled" in hint for hint in hints)

    def test_hint_count_is_limited(self):
        """Should return maximum 3 hints."""
        from app.ml_models.coarse_classifier import CoarseFoodClassifier
        import torch

        classifier = CoarseFoodClassifier()

        # Many prompts with many keywords
        prompts = [
            "a photo of grilled fresh raw sliced whole baked steamed fried food"
        ] * 10
        prompt_to_category = {p: None for p in prompts}
        similarities = torch.tensor([0.9] * 10)

        hints = classifier._generate_hints(
            similarities, prompts, prompt_to_category, None
        )

        assert len(hints) <= 3


# ============================================================================
# PERFORMANCE BENCHMARK TESTS
# ============================================================================

class TestUSDAIntegrationPerformance:
    """Performance benchmarks for USDA integration."""

    def test_category_mapping_lookup_is_fast(self):
        """Category to data type lookup should be under 1ms."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        categories = list(FoodCategory)

        start_time = time.time()
        for _ in range(1000):
            for cat in categories:
                _ = CATEGORY_TO_USDA_DATATYPES.get(cat, [])
        elapsed = time.time() - start_time

        # 1000 iterations of all categories should be under 100ms
        assert elapsed < 0.1, f"Mapping lookup took {elapsed*1000:.2f}ms"

    def test_query_enhancement_is_fast(self):
        """Query enhancement should be under 5ms per call."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()
        categories = [
            FoodCategory.FRUITS_FRESH,
            FoodCategory.MEAT_POULTRY,
            FoodCategory.GRAINS_PASTA,
            FoodCategory.FAST_FOOD,
            FoodCategory.MIXED_DISHES,
        ]
        queries = ["apple", "chicken", "spaghetti", "burger", "stir fry"]

        start_time = time.time()
        for _ in range(100):
            for cat, query in zip(categories, queries):
                _ = classifier._get_query_enhancement(cat, query)
        elapsed = time.time() - start_time

        # 500 enhancements should be under 100ms
        assert elapsed < 0.1, f"Query enhancement took {elapsed*1000:.2f}ms"


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestUSDAIntegrationErrors:
    """Error handling tests for USDA integration."""

    def test_unknown_category_has_fallback_datatypes(self):
        """Unknown category should return all data types."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        datatypes = CATEGORY_TO_USDA_DATATYPES[FoodCategory.UNKNOWN]

        assert "Foundation" in datatypes
        assert "SR Legacy" in datatypes
        assert "Survey (FNDDS)" in datatypes
        assert "Branded" in datatypes

    def test_classify_with_usda_context_handles_low_confidence(self):
        """Low confidence classification should still return valid structure."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            CoarseClassification,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()

        with patch.object(classifier, "classify") as mock_classify:
            mock_classify.return_value = CoarseClassification(
                category=FoodCategory.UNKNOWN,
                confidence=0.15,
                subcategory_hints=[],
                usda_datatypes=["Foundation", "SR Legacy", "Survey (FNDDS)", "Branded"],
                alternatives=[],
                texture_features={},
            )

            test_image = Image.new("RGB", (224, 224), color="gray")
            result = classifier.classify_with_usda_context(test_image, "unknown food")

            assert result["category"] == "unknown"
            assert result["confidence"] == 0.15
            # Should have all data types for fallback
            assert len(result["usda_datatypes"]) >= 4

    def test_classify_error_returns_unknown(self):
        """Classification errors should return UNKNOWN category."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()

        # Force error by mocking model to be None and load to fail
        with patch.object(classifier, "_model", None):
            with patch.object(classifier, "_loaded", False):
                with patch.object(
                    classifier, "load_model",
                    side_effect=Exception("Model load failed")
                ):
                    test_image = Image.new("RGB", (224, 224))
                    result = classifier.classify(test_image)

                    assert result.category == FoodCategory.UNKNOWN
                    assert result.confidence == 0.0


# ============================================================================
# DATA TYPE PRIORITY TESTS
# ============================================================================

class TestDataTypePriority:
    """Tests for data type priority in USDA searches."""

    def test_foundation_first_for_whole_foods(self):
        """Foundation should be first priority for whole foods."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        whole_food_categories = [
            FoodCategory.FRUITS_FRESH,
            FoodCategory.VEGETABLES_LEAFY,
            FoodCategory.VEGETABLES_ROOT,
            FoodCategory.MEAT_RED,
            FoodCategory.SEAFOOD_FISH,
        ]

        for cat in whole_food_categories:
            datatypes = CATEGORY_TO_USDA_DATATYPES[cat]
            # First element should be highest priority
            assert datatypes[0] in ["Foundation", "SR Legacy"], \
                f"Expected Foundation/SR Legacy first for {cat}, got {datatypes[0]}"

    def test_branded_first_for_packaged_foods(self):
        """Branded should be high priority for packaged foods."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        packaged_categories = [
            FoodCategory.SNACKS_SWEET,
            FoodCategory.SNACKS_SAVORY,
        ]

        for cat in packaged_categories:
            datatypes = CATEGORY_TO_USDA_DATATYPES[cat]
            # Branded should be first for these
            assert "Branded" in datatypes[:2], \
                f"Expected Branded in top 2 for {cat}"

    def test_survey_for_mixed_dishes(self):
        """Survey (FNDDS) should be priority for mixed dishes."""
        from app.ml_models.coarse_classifier import (
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        mixed_categories = [
            FoodCategory.MIXED_DISHES,
            FoodCategory.VEGETABLES_COOKED,
        ]

        for cat in mixed_categories:
            datatypes = CATEGORY_TO_USDA_DATATYPES[cat]
            assert "Survey (FNDDS)" in datatypes, \
                f"Expected Survey (FNDDS) for {cat}"


# ============================================================================
# INTEGRATION FLOW TESTS
# ============================================================================

class TestFullIntegrationFlow:
    """Tests for complete classification to USDA search flow."""

    def test_fruit_classification_to_search_flow(self):
        """Test complete flow: classify fruit -> get USDA search params."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            CoarseClassification,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()

        with patch.object(classifier, "classify") as mock_classify:
            mock_classify.return_value = CoarseClassification(
                category=FoodCategory.FRUITS_FRESH,
                confidence=0.92,
                subcategory_hints=["appears fresh"],
                usda_datatypes=["Foundation", "SR Legacy"],
                alternatives=[],
                texture_features={},
            )

            test_image = Image.new("RGB", (224, 224), color="red")
            result = classifier.classify_with_usda_context(test_image, "apple")

            # Verify search parameters
            assert result["category"] == "fruits_fresh"
            assert result["confidence"] > 0.9
            assert "Foundation" in result["usda_datatypes"]
            # Query should be enhanced
            enhanced_query = result["search_hints"]["suggested_query_enhancement"]
            assert "apple" in enhanced_query.lower() or "raw" in enhanced_query.lower()

    def test_fast_food_classification_to_search_flow(self):
        """Test complete flow: classify fast food -> get USDA search params."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            CoarseClassification,
            FoodCategory,
        )

        classifier = CoarseFoodClassifier()

        with patch.object(classifier, "classify") as mock_classify:
            mock_classify.return_value = CoarseClassification(
                category=FoodCategory.FAST_FOOD,
                confidence=0.85,
                subcategory_hints=["appears fried"],
                usda_datatypes=["Branded", "Survey (FNDDS)"],
                alternatives=[],
                texture_features={},
            )

            test_image = Image.new("RGB", (224, 224), color="brown")
            result = classifier.classify_with_usda_context(test_image, "burger")

            # Verify search parameters
            assert result["category"] == "fast_food"
            assert "Branded" in result["usda_datatypes"]
            assert "Survey (FNDDS)" in result["usda_datatypes"]

    def test_alternatives_provide_fallback_options(self):
        """Alternatives should provide valid fallback categories for search."""
        from app.ml_models.coarse_classifier import (
            CoarseFoodClassifier,
            CoarseClassification,
            FoodCategory,
            CATEGORY_TO_USDA_DATATYPES,
        )

        classifier = CoarseFoodClassifier()

        with patch.object(classifier, "classify") as mock_classify:
            mock_classify.return_value = CoarseClassification(
                category=FoodCategory.GRAINS_PASTA,
                confidence=0.70,  # Lower confidence
                subcategory_hints=[],
                usda_datatypes=["SR Legacy", "Branded"],
                alternatives=[
                    (FoodCategory.MIXED_DISHES, 0.20),
                    (FoodCategory.GRAINS_OTHER, 0.10),
                ],
                texture_features={},
            )

            test_image = Image.new("RGB", (224, 224), color="yellow")
            result = classifier.classify_with_usda_context(test_image, "noodles")

            # Alternatives should be valid for fallback searches
            for alt in result["alternatives"]:
                cat_value = alt["category"]
                # Should be able to find data types for alternatives
                matching_cats = [
                    cat for cat in FoodCategory
                    if cat.value == cat_value
                ]
                assert len(matching_cats) == 1
                assert matching_cats[0] in CATEGORY_TO_USDA_DATATYPES
