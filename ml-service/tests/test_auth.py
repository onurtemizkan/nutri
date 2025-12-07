"""
Authentication Unit Tests

Unit tests for JWT token handling and password validation.
Integration tests for auth API routes have been removed as those routes are not yet implemented.
"""

import pytest

from app.auth.jwt import JWTHandler
from app.auth.password import PasswordHandler


@pytest.mark.asyncio
class TestPasswordValidation:
    """Test password strength validation."""

    def test_password_too_short(self):
        """Test password too short fails validation."""
        is_valid, msg = PasswordHandler.validate_password_strength("Short1!")
        assert is_valid is False
        assert "at least" in msg.lower()

    def test_password_no_uppercase(self):
        """Test password without uppercase fails."""
        is_valid, msg = PasswordHandler.validate_password_strength("lowercase123!")
        assert is_valid is False
        assert "uppercase" in msg.lower()

    def test_password_no_lowercase(self):
        """Test password without lowercase fails."""
        is_valid, msg = PasswordHandler.validate_password_strength("UPPERCASE123!")
        assert is_valid is False
        assert "lowercase" in msg.lower()

    def test_password_no_digit(self):
        """Test password without digit fails."""
        is_valid, msg = PasswordHandler.validate_password_strength("NoDigits!")
        assert is_valid is False
        assert "digit" in msg.lower()

    def test_password_no_special(self):
        """Test password without special character fails."""
        is_valid, msg = PasswordHandler.validate_password_strength("NoSpecial123")
        assert is_valid is False
        assert "special" in msg.lower()

    def test_password_valid(self):
        """Test valid password passes."""
        is_valid, msg = PasswordHandler.validate_password_strength("ValidPass123!")
        assert is_valid is True
        assert msg == ""


@pytest.mark.asyncio
class TestJWTTokens:
    """Test JWT token functionality."""

    def test_create_access_token(self):
        """Test access token creation."""
        token = JWTHandler.create_access_token(user_id=123, email="test@example.com")
        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify
        payload = JWTHandler.decode_token(token)
        assert payload["sub"] == "123"
        assert payload["email"] == "test@example.com"
        assert payload["type"] == "access"

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        token = JWTHandler.create_refresh_token(user_id=123, email="test@example.com")
        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify
        payload = JWTHandler.decode_token(token)
        assert payload["sub"] == "123"
        assert payload["email"] == "test@example.com"
        assert payload["type"] == "refresh"

    def test_verify_access_token_type(self):
        """Test access token type verification."""
        access_token = JWTHandler.create_access_token(
            user_id=123, email="test@example.com"
        )
        refresh_token = JWTHandler.create_refresh_token(
            user_id=123, email="test@example.com"
        )

        # Access token should verify
        payload = JWTHandler.verify_access_token(access_token)
        assert payload["type"] == "access"

        # Refresh token should fail access verification
        with pytest.raises(Exception):
            JWTHandler.verify_access_token(refresh_token)

    def test_verify_refresh_token_type(self):
        """Test refresh token type verification."""
        access_token = JWTHandler.create_access_token(
            user_id=123, email="test@example.com"
        )
        refresh_token = JWTHandler.create_refresh_token(
            user_id=123, email="test@example.com"
        )

        # Refresh token should verify
        payload = JWTHandler.verify_refresh_token(refresh_token)
        assert payload["type"] == "refresh"

        # Access token should fail refresh verification
        with pytest.raises(Exception):
            JWTHandler.verify_refresh_token(access_token)
