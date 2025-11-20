"""
Integration tests for Food Analysis API endpoints.
"""
import pytest
from httpx import AsyncClient
from fastapi import status
import io
from PIL import Image
import json

from app.main import app
from app.services.food_analysis_service import NUTRITION_DATABASE


@pytest.fixture
def client():
    """Create async HTTP client for testing."""
    return AsyncClient(app=app, base_url="http://test")


class TestFoodAnalysisEndpoint:
    """Test suite for POST /api/food/analyze endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_food_success(self, client, sample_food_image):
        """Test successful food analysis request."""
        # Convert PIL image to bytes
        img_bytes = io.BytesIO()
        sample_food_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        # Make request
        response = await client.post(
            "/api/food/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "food_items" in data
        assert len(data["food_items"]) >= 1
        assert "measurement_quality" in data
        assert data["measurement_quality"] in ["high", "medium", "low"]
        assert "processing_time" in data
        assert data["processing_time"] > 0

        # Validate food item structure
        food_item = data["food_items"][0]
        assert "name" in food_item
        assert "confidence" in food_item
        assert 0 <= food_item["confidence"] <= 1
        assert "portion_size" in food_item
        assert "portion_weight" in food_item
        assert "nutrition" in food_item

        # Validate nutrition structure
        nutrition = food_item["nutrition"]
        assert "calories" in nutrition
        assert "protein" in nutrition
        assert "carbs" in nutrition
        assert "fat" in nutrition

    @pytest.mark.asyncio
    async def test_analyze_food_with_dimensions(
        self, client, sample_food_image, good_ar_measurements
    ):
        """Test food analysis with AR dimensions."""
        img_bytes = io.BytesIO()
        sample_food_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        dimensions_data = {
            "width": good_ar_measurements.width,
            "height": good_ar_measurements.height,
            "depth": good_ar_measurements.depth,
        }

        response = await client.post(
            "/api/food/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"dimensions": json.dumps(dimensions_data)},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # With dimensions, quality should potentially be higher
        assert data["measurement_quality"] in ["high", "medium", "low"]

        # Portion weight should reflect the dimensions
        food_item = data["food_items"][0]
        assert food_item["portion_weight"] > 0

    @pytest.mark.asyncio
    async def test_analyze_food_invalid_file_type(self, client):
        """Test rejection of non-image files."""
        text_content = b"This is not an image"

        response = await client.post(
            "/api/food/analyze",
            files={"image": ("test.txt", text_content, "text/plain")},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid image" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_analyze_food_oversized_file(self, client):
        """Test rejection of files over 10MB."""
        # Create a large fake image (11MB)
        large_data = b"x" * (11 * 1024 * 1024)

        response = await client.post(
            "/api/food/analyze",
            files={"image": ("large.jpg", large_data, "image/jpeg")},
        )

        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert "too large" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_analyze_food_missing_image(self, client):
        """Test error when image is not provided."""
        response = await client.post("/api/food/analyze")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_analyze_food_invalid_dimensions_format(
        self, client, sample_food_image
    ):
        """Test error when dimensions have invalid format."""
        img_bytes = io.BytesIO()
        sample_food_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        response = await client.post(
            "/api/food/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"dimensions": "not-json"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid dimensions format" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_analyze_food_invalid_dimensions_values(
        self, client, sample_food_image
    ):
        """Test error when dimensions have invalid values."""
        img_bytes = io.BytesIO()
        sample_food_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        invalid_dimensions = json.dumps({
            "width": -5,  # Negative value
            "height": 10,
            "depth": 5,
        })

        response = await client.post(
            "/api/food/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"dimensions": invalid_dimensions},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_analyze_food_corrupted_image(self, client):
        """Test handling of corrupted image data."""
        corrupted_data = b"JFIF\x00\x00corrupted"

        response = await client.post(
            "/api/food/analyze",
            files={"image": ("corrupt.jpg", corrupted_data, "image/jpeg")},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid image" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_analyze_food_response_includes_suggestions(
        self, client, sample_food_image
    ):
        """Test that response includes helpful suggestions."""
        img_bytes = io.BytesIO()
        sample_food_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        response = await client.post(
            "/api/food/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        )

        data = response.json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    @pytest.mark.asyncio
    async def test_analyze_food_alternatives(self, client, sample_food_image):
        """Test that food items include alternatives."""
        img_bytes = io.BytesIO()
        sample_food_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        response = await client.post(
            "/api/food/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        )

        data = response.json()
        food_item = data["food_items"][0]

        if "alternatives" in food_item and food_item["alternatives"]:
            for alt in food_item["alternatives"]:
                assert "name" in alt
                assert "confidence" in alt
                # Alternatives should have lower confidence than primary
                assert alt["confidence"] < food_item["confidence"]


class TestNutritionSearchEndpoint:
    """Test suite for GET /api/food/nutrition-db/search endpoint."""

    @pytest.mark.asyncio
    async def test_search_exact_match(self, client):
        """Test search with exact food name match."""
        response = await client.get(
            "/api/food/nutrition-db/search",
            params={"q": "apple"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "results" in data
        assert len(data["results"]) >= 1

        result = data["results"][0]
        assert "food_name" in result
        assert result["food_name"].lower() == "apple"
        assert "category" in result
        assert "serving_size" in result
        assert "serving_weight" in result
        assert "nutrition" in result

    @pytest.mark.asyncio
    async def test_search_partial_match(self, client):
        """Test search with partial food name."""
        response = await client.get(
            "/api/food/nutrition-db/search",
            params={"q": "chick"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert len(data["results"]) >= 1
        # Should match "chicken breast"
        assert any("chicken" in r["food_name"].lower() for r in data["results"])

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, client):
        """Test that search is case insensitive."""
        response_lower = await client.get(
            "/api/food/nutrition-db/search",
            params={"q": "apple"}
        )
        response_upper = await client.get(
            "/api/food/nutrition-db/search",
            params={"q": "APPLE"}
        )
        response_mixed = await client.get(
            "/api/food/nutrition-db/search",
            params={"q": "ApPlE"}
        )

        assert response_lower.status_code == status.HTTP_200_OK
        assert response_upper.status_code == status.HTTP_200_OK
        assert response_mixed.status_code == status.HTTP_200_OK

        # All should return same number of results
        assert (
            len(response_lower.json()["results"]) ==
            len(response_upper.json()["results"]) ==
            len(response_mixed.json()["results"])
        )

    @pytest.mark.asyncio
    async def test_search_no_match(self, client):
        """Test search with no matching results."""
        response = await client.get(
            "/api/food/nutrition-db/search",
            params={"q": "pizza"}  # Not in mock database
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) == 0

    @pytest.mark.asyncio
    async def test_search_missing_query(self, client):
        """Test error when query parameter is missing."""
        response = await client.get("/api/food/nutrition-db/search")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_search_empty_query(self, client):
        """Test search with empty query string."""
        response = await client.get(
            "/api/food/nutrition-db/search",
            params={"q": ""}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_search_nutrition_structure(self, client):
        """Test that returned nutrition data has correct structure."""
        response = await client.get(
            "/api/food/nutrition-db/search",
            params={"q": "apple"}
        )

        result = response.json()["results"][0]
        nutrition = result["nutrition"]

        # Required fields
        assert "calories" in nutrition
        assert "protein" in nutrition
        assert "carbs" in nutrition
        assert "fat" in nutrition

        # All values should be non-negative numbers
        assert nutrition["calories"] >= 0
        assert nutrition["protein"] >= 0
        assert nutrition["carbs"] >= 0
        assert nutrition["fat"] >= 0


class TestModelsInfoEndpoint:
    """Test suite for GET /api/food/models/info endpoint."""

    @pytest.mark.asyncio
    async def test_get_models_info_success(self, client):
        """Test successful retrieval of models info."""
        response = await client.get("/api/food/models/info")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "accuracy" in data
        assert "num_classes" in data
        assert "description" in data

    @pytest.mark.asyncio
    async def test_models_info_structure(self, client):
        """Test models info has correct data types."""
        response = await client.get("/api/food/models/info")
        data = response.json()

        assert isinstance(data["name"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["accuracy"], (int, float))
        assert isinstance(data["num_classes"], int)
        assert isinstance(data["description"], str)

        # Validate accuracy is a percentage
        assert 0 <= data["accuracy"] <= 100

        # Num classes should match nutrition database
        assert data["num_classes"] == len(NUTRITION_DATABASE)


class TestHealthEndpoint:
    """Test suite for GET /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        """Test health check returns OK status."""
        response = await client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"


class TestErrorHandling:
    """Test suite for error handling across endpoints."""

    @pytest.mark.asyncio
    async def test_404_not_found(self, client):
        """Test 404 error for non-existent endpoint."""
        response = await client.get("/api/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_405_method_not_allowed(self, client):
        """Test 405 error for wrong HTTP method."""
        response = await client.get("/api/food/analyze")  # Should be POST

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    @pytest.mark.asyncio
    async def test_error_response_format(self, client):
        """Test that error responses have consistent format."""
        response = await client.post("/api/food/analyze")  # Missing required field

        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]

        data = response.json()
        assert "detail" in data


class TestConcurrentRequests:
    """Test suite for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_analyses(self, client, sample_food_image):
        """Test handling of multiple concurrent analysis requests."""
        img_bytes = io.BytesIO()
        sample_food_image.save(img_bytes, format='JPEG')

        # Create multiple requests
        requests = []
        for _ in range(5):
            img_bytes.seek(0)
            requests.append(
                client.post(
                    "/api/food/analyze",
                    files={"image": ("test.jpg", img_bytes.read(), "image/jpeg")},
                )
            )

        # Execute concurrently (httpx handles this with AsyncClient)
        import asyncio
        responses = await asyncio.gather(*requests)

        # All should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "food_items" in data
            assert len(data["food_items"]) >= 1


class TestResponseTimes:
    """Test suite for response time requirements."""

    @pytest.mark.asyncio
    async def test_analyze_response_time(self, client, sample_food_image):
        """Test that analysis response time is reasonable."""
        import time

        img_bytes = io.BytesIO()
        sample_food_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        start_time = time.time()

        response = await client.post(
            "/api/food/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        )

        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        assert response.status_code == status.HTTP_200_OK

        # Should be reasonably fast for mock implementation
        assert elapsed < 5000  # Less than 5 seconds

        # Reported processing time should be close to actual
        data = response.json()
        reported_time = data["processing_time"]

        # Reported time should be less than total elapsed time
        assert reported_time < elapsed


class TestInputValidation:
    """Test suite for comprehensive input validation."""

    @pytest.mark.asyncio
    async def test_dimensions_boundary_values(self, client, sample_food_image):
        """Test dimensions with boundary values."""
        img_bytes = io.BytesIO()
        sample_food_image.save(img_bytes, format='JPEG')

        test_cases = [
            {"width": 0.1, "height": 0.1, "depth": 0.1},  # Very small
            {"width": 100, "height": 100, "depth": 100},  # Very large
            {"width": 10.5, "height": 8.2, "depth": 6.0}, # Normal
        ]

        for dims in test_cases:
            img_bytes.seek(0)
            response = await client.post(
                "/api/food/analyze",
                files={"image": ("test.jpg", img_bytes, "image/jpeg")},
                data={"dimensions": json.dumps(dims)},
            )

            # Should accept all valid positive values
            assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_dimensions_negative_values(self, client, sample_food_image):
        """Test rejection of negative dimension values."""
        img_bytes = io.BytesIO()
        sample_food_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        invalid_dims = json.dumps({"width": -5, "height": 10, "depth": 5})

        response = await client.post(
            "/api/food/analyze",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
            data={"dimensions": invalid_dims},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
