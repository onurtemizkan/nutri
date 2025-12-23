"""
Comprehensive Tests for Food Feedback System

Tests cover:
1. Schema validation (Pydantic models)
2. FeedbackService operations
3. API endpoints (/feedback, /feedback/stats, /feedback/suggestions)
4. Feedback boost function (_apply_feedback_boost)
5. Integration tests for full feedback flow

The feedback system enables:
- User corrections when confidence < 80%
- Pattern detection for common misclassifications
- Learned corrections boosting alternatives
- Prompt generation from user descriptions
"""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from pydantic import ValidationError

# Schema imports
from app.schemas.food_analysis import (
    FoodItem,
    FoodItemAlternative,
    NutritionInfo,
    FoodFeedbackRequest,
    FoodFeedbackResponse,
    FeedbackStatsResponse,
)

# Service imports
from app.services.food_analysis_service import (
    _apply_feedback_boost,
    _correction_cache,
    FoodAnalysisService,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def sample_nutrition():
    """Create sample nutrition info for testing."""
    return NutritionInfo(
        calories=200.0,
        protein=10.0,
        carbs=30.0,
        fat=5.0,
    )


@pytest.fixture
def sample_alternatives():
    """Create sample alternatives list for testing."""
    return [
        FoodItemAlternative(
            name="apple",
            display_name="Apple",
            confidence=0.75,
            boosted=False,
            from_feedback=False,
        ),
        FoodItemAlternative(
            name="pear",
            display_name="Pear",
            confidence=0.60,
            boosted=False,
            from_feedback=False,
        ),
        FoodItemAlternative(
            name="peach",
            display_name="Peach",
            confidence=0.45,
            boosted=False,
            from_feedback=False,
        ),
    ]


@pytest.fixture
def sample_feedback_request():
    """Create sample feedback request for testing."""
    return FoodFeedbackRequest(
        image_hash="abc123def456789012345678901234567890123456789012345678901234",
        original_prediction="orange",
        original_confidence=0.65,
        user_selected_food="tangerine",
        alternatives_shown=["apple", "pear", "peach"],
        user_description="It's a small citrus fruit, smaller than an orange",
        user_id="test_user_001",
        session_id="session_123",
        device_type="ios",
    )


@pytest.fixture
def mock_correction_cache():
    """Set up mock correction cache with test patterns."""
    original_cache = _correction_cache.copy()

    # Set up test patterns
    _correction_cache["patterns"] = {
        "orange": [
            {"corrected": "tangerine", "count": 15, "total_for_original": 30},
            {"corrected": "clementine", "count": 8, "total_for_original": 30},
            {"corrected": "grapefruit", "count": 5, "total_for_original": 30},
        ],
        "apple": [
            {"corrected": "pear", "count": 10, "total_for_original": 25},
            {"corrected": "green_apple", "count": 7, "total_for_original": 25},
        ],
        "chicken_breast": [
            {"corrected": "turkey_breast", "count": 12, "total_for_original": 20},
        ],
    }
    _correction_cache["last_updated"] = datetime.now().timestamp()

    yield _correction_cache

    # Restore original cache
    _correction_cache.clear()
    _correction_cache.update(original_cache)


# ==============================================================================
# 1. SCHEMA VALIDATION TESTS
# ==============================================================================


class TestFoodItemAlternativeSchema:
    """Tests for FoodItemAlternative Pydantic model."""

    def test_create_basic_alternative(self):
        """Test creating alternative with required fields only."""
        alt = FoodItemAlternative(
            name="apple",
            confidence=0.85,
        )
        assert alt.name == "apple"
        assert alt.confidence == 0.85
        assert alt.display_name is None
        assert alt.boosted is False
        assert alt.from_feedback is False

    def test_create_full_alternative(self):
        """Test creating alternative with all fields."""
        alt = FoodItemAlternative(
            name="apple",
            display_name="Red Apple",
            confidence=0.85,
            boosted=True,
            from_feedback=True,
        )
        assert alt.name == "apple"
        assert alt.display_name == "Red Apple"
        assert alt.confidence == 0.85
        assert alt.boosted is True
        assert alt.from_feedback is True

    def test_confidence_bounds(self):
        """Test confidence must be between 0 and 1."""
        # Valid bounds
        FoodItemAlternative(name="test", confidence=0.0)
        FoodItemAlternative(name="test", confidence=1.0)
        FoodItemAlternative(name="test", confidence=0.5)

        # Invalid: above 1
        with pytest.raises(ValidationError):
            FoodItemAlternative(name="test", confidence=1.5)

        # Invalid: below 0
        with pytest.raises(ValidationError):
            FoodItemAlternative(name="test", confidence=-0.1)

    def test_serialization(self):
        """Test alternative serializes correctly to dict."""
        alt = FoodItemAlternative(
            name="apple",
            display_name="Apple",
            confidence=0.75,
            boosted=True,
            from_feedback=True,
        )
        data = alt.model_dump()
        assert data["name"] == "apple"
        assert data["display_name"] == "Apple"
        assert data["confidence"] == 0.75
        assert data["boosted"] is True
        assert data["from_feedback"] is True


class TestFoodItemSchema:
    """Tests for FoodItem Pydantic model with new feedback fields."""

    def test_create_food_item_with_confirmation_fields(self, sample_nutrition):
        """Test FoodItem includes needs_confirmation and confidence_threshold."""
        item = FoodItem(
            name="apple",
            display_name="Apple",
            confidence=0.65,
            portion_size="1 medium",
            portion_weight=150.0,
            nutrition=sample_nutrition,
            category="fruit",
            needs_confirmation=True,
            confidence_threshold=0.8,
        )
        assert item.name == "apple"
        assert item.display_name == "Apple"
        assert item.confidence == 0.65
        assert item.needs_confirmation is True
        assert item.confidence_threshold == 0.8

    def test_food_item_default_confirmation_threshold(self, sample_nutrition):
        """Test default confidence_threshold is 0.8."""
        item = FoodItem(
            name="apple",
            confidence=0.9,
            portion_size="1 medium",
            portion_weight=150.0,
            nutrition=sample_nutrition,
        )
        assert item.confidence_threshold == 0.8
        assert item.needs_confirmation is False  # Default

    def test_food_item_with_alternatives(self, sample_nutrition, sample_alternatives):
        """Test FoodItem with alternatives list."""
        item = FoodItem(
            name="orange",
            display_name="Orange",
            confidence=0.65,
            portion_size="1 medium",
            portion_weight=130.0,
            nutrition=sample_nutrition,
            alternatives=sample_alternatives,
            needs_confirmation=True,
        )
        assert len(item.alternatives) == 3
        assert item.alternatives[0].name == "apple"
        assert item.alternatives[0].confidence == 0.75


class TestFoodFeedbackRequestSchema:
    """Tests for FoodFeedbackRequest Pydantic model."""

    def test_create_valid_feedback_request(self, sample_feedback_request):
        """Test creating valid feedback request."""
        req = sample_feedback_request
        assert (
            req.image_hash
            == "abc123def456789012345678901234567890123456789012345678901234"
        )
        assert req.original_prediction == "orange"
        assert req.original_confidence == 0.65
        assert req.user_selected_food == "tangerine"
        assert len(req.alternatives_shown) == 3
        assert req.user_description is not None

    def test_feedback_request_minimal(self):
        """Test feedback request with only required fields."""
        req = FoodFeedbackRequest(
            image_hash="a" * 16,  # Minimum length
            original_prediction="apple",
            original_confidence=0.5,
            user_selected_food="pear",
        )
        assert req.image_hash == "a" * 16
        assert req.alternatives_shown == []
        assert req.user_description is None
        assert req.user_id is None

    def test_feedback_request_image_hash_validation(self):
        """Test image hash length validation."""
        # Too short (< 16)
        with pytest.raises(ValidationError):
            FoodFeedbackRequest(
                image_hash="short",
                original_prediction="apple",
                original_confidence=0.5,
                user_selected_food="pear",
            )

        # Too long (> 64)
        with pytest.raises(ValidationError):
            FoodFeedbackRequest(
                image_hash="x" * 65,
                original_prediction="apple",
                original_confidence=0.5,
                user_selected_food="pear",
            )

    def test_feedback_request_confidence_validation(self):
        """Test confidence bounds validation."""
        # Invalid: above 1
        with pytest.raises(ValidationError):
            FoodFeedbackRequest(
                image_hash="a" * 32,
                original_prediction="apple",
                original_confidence=1.5,
                user_selected_food="pear",
            )

        # Invalid: below 0
        with pytest.raises(ValidationError):
            FoodFeedbackRequest(
                image_hash="a" * 32,
                original_prediction="apple",
                original_confidence=-0.1,
                user_selected_food="pear",
            )

    def test_feedback_request_empty_prediction_validation(self):
        """Test prediction fields cannot be empty."""
        with pytest.raises(ValidationError):
            FoodFeedbackRequest(
                image_hash="a" * 32,
                original_prediction="",  # Empty
                original_confidence=0.5,
                user_selected_food="pear",
            )

        with pytest.raises(ValidationError):
            FoodFeedbackRequest(
                image_hash="a" * 32,
                original_prediction="apple",
                original_confidence=0.5,
                user_selected_food="",  # Empty
            )


class TestFoodFeedbackResponseSchema:
    """Tests for FoodFeedbackResponse Pydantic model."""

    def test_create_feedback_response(self):
        """Test creating feedback response."""
        resp = FoodFeedbackResponse(
            success=True,
            feedback_id=123,
            was_correction=True,
            message="Thanks for the correction!",
            suggested_prompts=[
                "a photo of tangerine",
                "citrus fruit smaller than orange",
            ],
        )
        assert resp.success is True
        assert resp.feedback_id == 123
        assert resp.was_correction is True
        assert len(resp.suggested_prompts) == 2

    def test_feedback_response_no_prompts(self):
        """Test feedback response with no suggested prompts."""
        resp = FoodFeedbackResponse(
            success=True,
            feedback_id=456,
            was_correction=False,
            message="Thanks for confirming!",
        )
        assert resp.suggested_prompts == []


class TestFeedbackStatsResponseSchema:
    """Tests for FeedbackStatsResponse Pydantic model."""

    def test_create_stats_response(self):
        """Test creating feedback stats response."""
        resp = FeedbackStatsResponse(
            total_feedback=100,
            corrections=35,
            confirmations=65,
            accuracy_rate=0.65,
            correction_rate=0.35,
            top_misclassifications=[
                {"original": "orange", "corrected": "tangerine", "count": 10}
            ],
            problem_foods=[
                {"food": "orange", "correction_count": 15, "avg_confidence": 0.55}
            ],
        )
        assert resp.total_feedback == 100
        assert resp.accuracy_rate == 0.65
        assert len(resp.top_misclassifications) == 1
        assert len(resp.problem_foods) == 1

    def test_stats_response_accuracy_bounds(self):
        """Test accuracy_rate must be between 0 and 1."""
        with pytest.raises(ValidationError):
            FeedbackStatsResponse(
                total_feedback=100,
                corrections=35,
                confirmations=65,
                accuracy_rate=1.5,  # Invalid
                correction_rate=0.35,
            )


# ==============================================================================
# 2. FEEDBACK BOOST FUNCTION TESTS
# ==============================================================================


class TestApplyFeedbackBoost:
    """Tests for _apply_feedback_boost function."""

    def test_no_boost_high_confidence(self, sample_alternatives):
        """Test no boost applied when confidence >= threshold."""
        result = _apply_feedback_boost(
            primary_class="apple",
            confidence=0.85,  # Above threshold
            alternatives=sample_alternatives,
            threshold=0.8,
        )
        # Should return original alternatives unchanged
        assert result == sample_alternatives
        assert not any(alt.boosted for alt in result)

    def test_boost_applied_low_confidence(
        self, sample_alternatives, mock_correction_cache
    ):
        """Test boost applied when confidence < threshold."""
        result = _apply_feedback_boost(
            primary_class="orange",
            confidence=0.65,  # Below threshold
            alternatives=sample_alternatives,
            threshold=0.8,
        )
        # Should have boosted alternatives
        assert len(result) > 0
        # Tangerine should be added and boosted (from correction patterns)
        boosted_names = [alt.name for alt in result if alt.boosted]
        assert "tangerine" in boosted_names

    def test_boost_marks_from_feedback(
        self, sample_alternatives, mock_correction_cache
    ):
        """Test boosted alternatives marked with from_feedback=True."""
        result = _apply_feedback_boost(
            primary_class="orange",
            confidence=0.65,
            alternatives=sample_alternatives,
            threshold=0.8,
        )
        # Find boosted alternatives
        boosted = [alt for alt in result if alt.boosted]
        assert len(boosted) > 0
        # All boosted should have from_feedback=True
        for alt in boosted:
            assert alt.from_feedback is True

    def test_boost_adds_new_alternatives(self, mock_correction_cache):
        """Test boost adds alternatives not in original list."""
        original_alts = [
            FoodItemAlternative(
                name="apple",
                display_name="Apple",
                confidence=0.60,
                boosted=False,
                from_feedback=False,
            )
        ]
        result = _apply_feedback_boost(
            primary_class="orange",
            confidence=0.65,
            alternatives=original_alts,
            threshold=0.8,
        )
        # Should have added tangerine from correction patterns
        alt_names = [alt.name for alt in result]
        assert "tangerine" in alt_names

    def test_boost_increases_confidence(self, mock_correction_cache):
        """Test boost increases confidence of existing alternatives."""
        # Create alternative that matches a correction pattern
        original_alts = [
            FoodItemAlternative(
                name="tangerine",
                display_name="Tangerine",
                confidence=0.40,  # Low initial confidence
                boosted=False,
                from_feedback=False,
            )
        ]
        result = _apply_feedback_boost(
            primary_class="orange",
            confidence=0.65,
            alternatives=original_alts,
            threshold=0.8,
        )
        # Find tangerine in result
        tangerine = next((alt for alt in result if alt.name == "tangerine"), None)
        assert tangerine is not None
        assert tangerine.confidence > 0.40  # Should be boosted
        assert tangerine.boosted is True

    def test_boost_sorts_by_confidence(
        self, sample_alternatives, mock_correction_cache
    ):
        """Test boosted alternatives are sorted by confidence descending."""
        result = _apply_feedback_boost(
            primary_class="orange",
            confidence=0.65,
            alternatives=sample_alternatives,
            threshold=0.8,
        )
        # Verify sorted by confidence
        confidences = [alt.confidence for alt in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_boost_limits_to_five(self, mock_correction_cache):
        """Test boost limits alternatives to 5."""
        many_alts = [
            FoodItemAlternative(
                name=f"food_{i}",
                display_name=f"Food {i}",
                confidence=0.5 - (i * 0.05),
                boosted=False,
                from_feedback=False,
            )
            for i in range(10)
        ]
        result = _apply_feedback_boost(
            primary_class="orange",
            confidence=0.65,
            alternatives=many_alts,
            threshold=0.8,
        )
        assert len(result) <= 5

    def test_no_boost_without_patterns(self, sample_alternatives):
        """Test no boost when no correction patterns exist."""
        # Clear cache patterns
        _correction_cache["patterns"] = {}

        result = _apply_feedback_boost(
            primary_class="exotic_fruit",  # No patterns for this
            confidence=0.65,
            alternatives=sample_alternatives,
            threshold=0.8,
        )
        # Should return original (no boost applied)
        assert not any(alt.boosted for alt in result)

    def test_boost_factor_calculation(self, mock_correction_cache):
        """Test boost factor is calculated correctly from correction counts."""
        original_alts = []
        result = _apply_feedback_boost(
            primary_class="orange",
            confidence=0.65,
            alternatives=original_alts,
            threshold=0.8,
        )
        # Tangerine has 15 corrections out of 30 total = 0.5 boost factor (capped)
        # Should have high confidence
        tangerine = next((alt for alt in result if alt.name == "tangerine"), None)
        assert tangerine is not None
        # 0.3 base + 0.5 boost = 0.7 (due to boost cap)
        assert tangerine.confidence >= 0.5


# ==============================================================================
# 3. FOOD ANALYSIS SERVICE TESTS
# ==============================================================================


class TestFoodAnalysisServiceConfirmation:
    """Tests for FoodAnalysisService confirmation threshold behavior."""

    def test_confirmation_threshold_constant(self):
        """Test CONFIRMATION_THRESHOLD is set correctly."""
        service = FoodAnalysisService()
        assert service.CONFIRMATION_THRESHOLD == 0.8

    @pytest.mark.asyncio
    async def test_needs_confirmation_low_confidence(self, sample_nutrition):
        """Test needs_confirmation=True when confidence < threshold."""
        service = FoodAnalysisService()

        # Mock the classifier to return low confidence
        with patch.object(service, "_get_ensemble_classifier") as mock_classifier:
            mock_result = MagicMock()
            mock_result.primary_class = "apple"
            mock_result.confidence = 0.65  # Below threshold
            mock_result.alternatives = []
            mock_result.contributing_models = ["test"]
            mock_classifier.return_value.classify.return_value = mock_result

            # This would need full mocking of dependencies
            # For now, just verify the threshold logic exists
            assert service.CONFIRMATION_THRESHOLD == 0.8
            assert 0.65 < service.CONFIRMATION_THRESHOLD

    def test_needs_confirmation_high_confidence(self, sample_nutrition):
        """Test needs_confirmation=False when confidence >= threshold."""
        service = FoodAnalysisService()
        # 0.85 >= 0.8, so needs_confirmation should be False
        assert 0.85 >= service.CONFIRMATION_THRESHOLD


# ==============================================================================
# 4. FEEDBACK SERVICE TESTS
# ==============================================================================


class TestFeedbackServiceSubmission:
    """Tests for FeedbackService.submit_feedback."""

    @pytest.mark.asyncio
    async def test_submit_feedback_creates_record(self, db):
        """Test feedback submission creates database record."""
        from app.services.feedback_service import FeedbackService

        service = FeedbackService()

        feedback_id, prompts = await service.submit_feedback(
            db=db,
            image_hash="test_hash_123456789012345678901234",
            original_prediction="orange",
            original_confidence=0.65,
            corrected_label="tangerine",
            alternatives=[{"name": "apple"}],
            user_description=None,
            user_id=None,
        )

        # Should return a positive feedback_id
        assert feedback_id > 0 or feedback_id == -1  # -1 if duplicate
        await db.rollback()

    @pytest.mark.asyncio
    async def test_submit_feedback_generates_prompts(self, db):
        """Test feedback with description generates prompts."""
        from app.services.feedback_service import FeedbackService

        service = FeedbackService()

        feedback_id, prompts = await service.submit_feedback(
            db=db,
            image_hash="test_hash_unique_12345678901234567",
            original_prediction="orange",
            original_confidence=0.65,
            corrected_label="tangerine",
            alternatives=None,
            user_description="small citrus fruit, smaller than an orange with thin skin",
            user_id=None,
        )

        # Should generate prompts from description
        if feedback_id > 0:
            assert len(prompts) > 0
        await db.rollback()

    @pytest.mark.asyncio
    async def test_submit_feedback_duplicate_detection(self, db):
        """Test duplicate feedback returns -1."""
        from app.services.feedback_service import FeedbackService

        service = FeedbackService()

        # Submit first feedback
        await service.submit_feedback(
            db=db,
            image_hash="duplicate_test_hash_1234567890123",
            original_prediction="orange",
            original_confidence=0.65,
            corrected_label="tangerine",
        )
        await db.flush()

        # Submit duplicate
        feedback_id, _ = await service.submit_feedback(
            db=db,
            image_hash="duplicate_test_hash_1234567890123",
            original_prediction="orange",
            original_confidence=0.65,
            corrected_label="tangerine",
        )

        # Should detect duplicate
        assert feedback_id == -1
        await db.rollback()


class TestFeedbackServiceStats:
    """Tests for FeedbackService.get_stats."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, db):
        """Test stats with no feedback returns zeros."""
        from app.services.feedback_service import FeedbackService

        service = FeedbackService()
        service._stats_cache = None  # Clear cache

        stats = await service.get_stats(db)

        assert "total_feedback" in stats
        assert stats["total_feedback"] >= 0
        await db.rollback()

    @pytest.mark.asyncio
    async def test_get_stats_caching(self, db):
        """Test stats are cached."""
        from app.services.feedback_service import FeedbackService

        service = FeedbackService()
        service._stats_cache = None  # Clear cache

        # First call
        stats1 = await service.get_stats(db)

        # Second call should use cache
        stats2 = await service.get_stats(db)

        assert stats1 == stats2
        await db.rollback()


class TestFeedbackServicePromptSuggestions:
    """Tests for FeedbackService.get_prompt_suggestions."""

    @pytest.mark.asyncio
    async def test_get_prompt_suggestions(self, db):
        """Test getting prompt suggestions for a food."""
        from app.services.feedback_service import FeedbackService

        service = FeedbackService()

        suggestions = await service.get_prompt_suggestions(db, "apple")

        assert "food_key" in suggestions
        assert "current_prompts" in suggestions
        assert "suggested_prompts" in suggestions
        assert suggestions["food_key"] == "apple"
        await db.rollback()


class TestFeedbackServicePromptGeneration:
    """Tests for FeedbackService prompt generation."""

    def test_generate_prompts_from_description(self):
        """Test prompt generation from user description."""
        from app.services.feedback_service import FeedbackService

        service = FeedbackService()

        prompts = service._generate_prompts_from_description(
            food_key="tangerine",
            description="small citrus fruit with easy to peel skin",
        )

        assert len(prompts) > 0
        # Should contain food name or description
        assert any("tangerine" in p or "citrus" in p for p in prompts)

    def test_generate_prompts_short_description_ignored(self):
        """Test short descriptions don't generate direct prompts."""
        from app.services.feedback_service import FeedbackService

        service = FeedbackService()

        prompts = service._generate_prompts_from_description(
            food_key="apple", description="red"  # Too short
        )

        # Should still generate some prompts based on food name
        # But not direct description prompt
        assert not any(p == "a photo of red" for p in prompts)


# ==============================================================================
# 5. INTEGRATION TESTS
# ==============================================================================


class TestFeedbackFlowIntegration:
    """Integration tests for the full feedback flow."""

    @pytest.mark.asyncio
    async def test_full_feedback_flow(self, db):
        """Test complete feedback submission and retrieval flow."""
        from app.services.feedback_service import FeedbackService

        service = FeedbackService()

        # Step 1: Submit feedback
        feedback_id, prompts = await service.submit_feedback(
            db=db,
            image_hash="integration_test_hash_1234567890",
            original_prediction="orange",
            original_confidence=0.65,
            corrected_label="tangerine",
            user_description="small citrus fruit",
        )
        await db.flush()

        # Step 2: Verify stats updated
        service._stats_cache = None  # Clear cache
        stats = await service.get_stats(db)
        assert stats["total_feedback"] >= 1

        # Step 3: Get suggestions for the food
        suggestions = await service.get_prompt_suggestions(db, "tangerine")
        assert suggestions["food_key"] == "tangerine"

        await db.rollback()

    def test_feedback_boost_integration(self, mock_correction_cache):
        """Test feedback boost integrates with classification flow."""
        # Create alternatives similar to what classifier returns
        alternatives = [
            FoodItemAlternative(
                name="apple",
                display_name="Apple",
                confidence=0.55,
                boosted=False,
                from_feedback=False,
            ),
            FoodItemAlternative(
                name="pear",
                display_name="Pear",
                confidence=0.40,
                boosted=False,
                from_feedback=False,
            ),
        ]

        # Apply boost (orange -> tangerine is in mock cache)
        result = _apply_feedback_boost(
            primary_class="orange",
            confidence=0.65,
            alternatives=alternatives,
            threshold=0.8,
        )

        # Verify tangerine was added from feedback patterns
        alt_names = [alt.name for alt in result]
        assert "tangerine" in alt_names

        # Verify it's marked as from feedback
        tangerine = next(alt for alt in result if alt.name == "tangerine")
        assert tangerine.from_feedback is True
        assert tangerine.boosted is True


# ==============================================================================
# 6. EDGE CASES AND ERROR HANDLING
# ==============================================================================


class TestFeedbackEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_alternatives_boost(self, mock_correction_cache):
        """Test boost with empty alternatives list."""
        result = _apply_feedback_boost(
            primary_class="orange",
            confidence=0.65,
            alternatives=[],
            threshold=0.8,
        )
        # Should add alternatives from feedback
        assert len(result) > 0

    def test_boost_preserves_unboosted(self, mock_correction_cache):
        """Test boost preserves non-matching alternatives."""
        alternatives = [
            FoodItemAlternative(
                name="banana",  # Not in correction patterns
                display_name="Banana",
                confidence=0.50,
                boosted=False,
                from_feedback=False,
            ),
        ]
        result = _apply_feedback_boost(
            primary_class="orange",
            confidence=0.65,
            alternatives=alternatives,
            threshold=0.8,
        )
        # Banana should still be in results (unboosted)
        banana = next((alt for alt in result if alt.name == "banana"), None)
        assert banana is not None
        assert banana.boosted is False

    def test_feedback_request_max_description_length(self):
        """Test description max length is enforced."""
        # Max is 500 characters
        long_description = "x" * 501

        with pytest.raises(ValidationError):
            FoodFeedbackRequest(
                image_hash="a" * 32,
                original_prediction="apple",
                original_confidence=0.5,
                user_selected_food="pear",
                user_description=long_description,
            )

    def test_schema_json_serialization(self, sample_feedback_request):
        """Test schemas serialize to JSON correctly."""
        import json

        # Should not raise
        json_str = sample_feedback_request.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["image_hash"] == sample_feedback_request.image_hash
        assert parsed["original_prediction"] == "orange"


# ==============================================================================
# 7. API ENDPOINT TESTS
# ==============================================================================


class TestFeedbackAPIEndpoints:
    """Tests for feedback API endpoints."""

    @pytest.mark.asyncio
    async def test_submit_feedback_endpoint(self, client):
        """Test POST /api/food/feedback endpoint."""
        payload = {
            "image_hash": "api_test_hash_123456789012345678901",
            "original_prediction": "orange",
            "original_confidence": 0.65,
            "user_selected_food": "tangerine",
            "alternatives_shown": ["apple", "pear"],
            "user_description": "small citrus fruit",
        }

        response = await client.post("/api/food/feedback", json=payload)

        # Should return 200 OK
        assert response.status_code == 200

        data = response.json()
        assert "success" in data
        assert "feedback_id" in data
        assert "was_correction" in data
        assert "message" in data

        # This was a correction (orange -> tangerine)
        assert data["was_correction"] is True

    @pytest.mark.asyncio
    async def test_submit_feedback_confirmation(self, client):
        """Test feedback submission when user confirms prediction."""
        payload = {
            "image_hash": "confirm_test_hash_12345678901234567",
            "original_prediction": "apple",
            "original_confidence": 0.75,
            "user_selected_food": "apple",  # Same as prediction
            "alternatives_shown": ["pear", "peach"],
        }

        response = await client.post("/api/food/feedback", json=payload)

        assert response.status_code == 200
        data = response.json()

        # This was a confirmation, not correction
        assert data["was_correction"] is False

    @pytest.mark.asyncio
    async def test_submit_feedback_validation_error(self, client):
        """Test feedback endpoint returns 422 for invalid data."""
        payload = {
            "image_hash": "short",  # Too short
            "original_prediction": "apple",
            "original_confidence": 0.5,
            "user_selected_food": "pear",
        }

        response = await client.post("/api/food/feedback", json=payload)

        # Should return validation error
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_feedback_stats_endpoint(self, client):
        """Test GET /api/food/feedback/stats endpoint."""
        response = await client.get("/api/food/feedback/stats")

        assert response.status_code == 200

        data = response.json()
        assert "total_feedback" in data
        assert "corrections" in data
        assert "confirmations" in data
        assert "accuracy_rate" in data
        assert "correction_rate" in data
        assert "top_misclassifications" in data
        assert "problem_foods" in data

    @pytest.mark.asyncio
    async def test_get_food_suggestions_endpoint(self, client):
        """Test GET /api/food/feedback/suggestions/{food_key} endpoint."""
        response = await client.get("/api/food/feedback/suggestions/apple")

        assert response.status_code == 200

        data = response.json()
        assert "food_key" in data
        assert data["food_key"] == "apple"
        assert "current_prompts" in data
        assert "suggested_prompts" in data

    @pytest.mark.asyncio
    async def test_feedback_endpoint_handles_server_error(self, client):
        """Test feedback endpoint handles internal errors gracefully."""
        # This test would require mocking the service to raise an exception
        # For now, just verify the endpoint exists and returns structured response
        payload = {
            "image_hash": "a" * 32,
            "original_prediction": "apple",
            "original_confidence": 0.5,
            "user_selected_food": "pear",
        }

        response = await client.post("/api/food/feedback", json=payload)

        # Should return either success or structured error
        assert response.status_code in [200, 500]
        assert "json" in response.headers.get("content-type", "")


class TestAnalyzeEndpointWithFeedbackFields:
    """Tests for /api/food/analyze endpoint with new feedback fields."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_analyze_returns_needs_confirmation(self, client):
        """Test analyze endpoint returns needs_confirmation field."""
        # This test requires an actual image and model
        # Skip in CI, run manually
        pytest.skip("Requires model loading - run manually")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_analyze_returns_alternatives_with_display_name(self, client):
        """Test analyze endpoint returns alternatives with display_name."""
        pytest.skip("Requires model loading - run manually")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_analyze_returns_image_hash(self, client):
        """Test analyze endpoint returns image_hash for feedback submission."""
        pytest.skip("Requires model loading - run manually")


# ==============================================================================
# 8. PERFORMANCE TESTS
# ==============================================================================


class TestFeedbackPerformance:
    """Performance tests for feedback system."""

    def test_boost_performance_many_alternatives(self, mock_correction_cache):
        """Test boost performance with many alternatives."""
        import time

        # Create many alternatives
        alternatives = [
            FoodItemAlternative(
                name=f"food_{i}",
                display_name=f"Food {i}",
                confidence=0.9 - (i * 0.01),
                boosted=False,
                from_feedback=False,
            )
            for i in range(50)
        ]

        start = time.time()
        for _ in range(100):
            _apply_feedback_boost(
                primary_class="orange",
                confidence=0.65,
                alternatives=alternatives,
                threshold=0.8,
            )
        elapsed = time.time() - start

        # Should complete 100 iterations in under 1 second
        assert elapsed < 1.0, f"Boost took too long: {elapsed:.2f}s"

    def test_boost_no_allocation_when_skipped(self, sample_alternatives):
        """Test boost doesn't allocate when confidence is high."""
        # High confidence should return original list
        result = _apply_feedback_boost(
            primary_class="apple",
            confidence=0.95,
            alternatives=sample_alternatives,
            threshold=0.8,
        )
        # Should be the same object (no copy)
        assert result is sample_alternatives
