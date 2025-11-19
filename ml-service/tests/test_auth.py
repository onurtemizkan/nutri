"""
Authentication Tests

Tests for user registration, login, token management, and protected routes.
"""

import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta, UTC

from app.main import app
from app.auth.jwt import JWTHandler
from app.auth.password import PasswordHandler


@pytest.mark.asyncio
class TestUserRegistration:
    """Test user registration."""

    async def test_register_success(self, client: AsyncClient):
        """Test successful user registration."""
        response = await client.post(
            "/api/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "name": "Test User",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["name"] == "Test User"
        assert "id" in data
        assert "password" not in data  # Password should not be returned
        assert data["is_active"] is True

    async def test_register_duplicate_email(self, client: AsyncClient):
        """Test registration with duplicate email fails."""
        # Register first user
        await client.post(
            "/api/auth/register",
            json={
                "email": "duplicate@example.com",
                "password": "SecurePass123!",
                "name": "First User",
            },
        )

        # Try to register with same email
        response = await client.post(
            "/api/auth/register",
            json={
                "email": "duplicate@example.com",
                "password": "DifferentPass123!",
                "name": "Second User",
            },
        )

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    async def test_register_weak_password(self, client: AsyncClient):
        """Test registration with weak password fails."""
        # Too short
        response = await client.post(
            "/api/auth/register",
            json={
                "email": "weak@example.com",
                "password": "short",
                "name": "Weak User",
            },
        )
        assert response.status_code == 400

        # No uppercase
        response = await client.post(
            "/api/auth/register",
            json={
                "email": "weak2@example.com",
                "password": "lowercase123!",
                "name": "Weak User",
            },
        )
        assert response.status_code == 400

        # No special character
        response = await client.post(
            "/api/auth/register",
            json={
                "email": "weak3@example.com",
                "password": "NoSpecial123",
                "name": "Weak User",
            },
        )
        assert response.status_code == 400

    async def test_register_invalid_email(self, client: AsyncClient):
        """Test registration with invalid email fails."""
        response = await client.post(
            "/api/auth/register",
            json={
                "email": "not-an-email",
                "password": "SecurePass123!",
                "name": "Test User",
            },
        )
        assert response.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
class TestUserLogin:
    """Test user login."""

    async def test_login_success(self, client: AsyncClient):
        """Test successful login."""
        # Register user first
        await client.post(
            "/api/auth/register",
            json={
                "email": "login@example.com",
                "password": "SecurePass123!",
                "name": "Login User",
            },
        )

        # Login
        response = await client.post(
            "/api/auth/login",
            json={"email": "login@example.com", "password": "SecurePass123!"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 15 * 60  # 15 minutes in seconds

        # Verify tokens are valid JWT
        access_payload = JWTHandler.decode_token(data["access_token"])
        assert access_payload["email"] == "login@example.com"
        assert access_payload["type"] == "access"

        refresh_payload = JWTHandler.decode_token(data["refresh_token"])
        assert refresh_payload["email"] == "login@example.com"
        assert refresh_payload["type"] == "refresh"

    async def test_login_wrong_password(self, client: AsyncClient):
        """Test login with wrong password fails."""
        # Register user
        await client.post(
            "/api/auth/register",
            json={
                "email": "wrong@example.com",
                "password": "CorrectPass123!",
                "name": "Wrong User",
            },
        )

        # Try login with wrong password
        response = await client.post(
            "/api/auth/login",
            json={"email": "wrong@example.com", "password": "WrongPass123!"},
        )

        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower()

    async def test_login_nonexistent_user(self, client: AsyncClient):
        """Test login with non-existent user fails."""
        response = await client.post(
            "/api/auth/login",
            json={"email": "nonexistent@example.com", "password": "AnyPass123!"},
        )

        assert response.status_code == 401
        assert "invalid" in response.json()["detail"].lower()

    async def test_login_account_lockout(self, client: AsyncClient):
        """Test account lockout after multiple failed attempts."""
        # Register user
        await client.post(
            "/api/auth/register",
            json={
                "email": "lockout@example.com",
                "password": "CorrectPass123!",
                "name": "Lockout User",
            },
        )

        # Make 5 failed login attempts
        for i in range(5):
            response = await client.post(
                "/api/auth/login",
                json={"email": "lockout@example.com", "password": "WrongPass123!"},
            )
            # First 4 should be 401, 5th should be 403 (locked)
            if i < 4:
                assert response.status_code == 401
            else:
                assert response.status_code == 403
                assert "locked" in response.json()["detail"].lower()

        # Even correct password should fail when locked
        response = await client.post(
            "/api/auth/login",
            json={"email": "lockout@example.com", "password": "CorrectPass123!"},
        )
        assert response.status_code == 403


@pytest.mark.asyncio
class TestTokenRefresh:
    """Test token refresh functionality."""

    async def test_refresh_token_success(self, client: AsyncClient):
        """Test successful token refresh."""
        # Register and login
        await client.post(
            "/api/auth/register",
            json={
                "email": "refresh@example.com",
                "password": "SecurePass123!",
                "name": "Refresh User",
            },
        )

        login_response = await client.post(
            "/api/auth/login",
            json={"email": "refresh@example.com", "password": "SecurePass123!"},
        )
        tokens = login_response.json()

        # Refresh token
        response = await client.post(
            "/api/auth/refresh",
            json={"refresh_token": tokens["refresh_token"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["access_token"] != tokens["access_token"]  # New token
        assert data["refresh_token"] == tokens["refresh_token"]  # Same refresh token

    async def test_refresh_with_invalid_token(self, client: AsyncClient):
        """Test token refresh with invalid token fails."""
        response = await client.post(
            "/api/auth/refresh",
            json={"refresh_token": "invalid.token.here"},
        )
        assert response.status_code == 401

    async def test_refresh_with_access_token(self, client: AsyncClient):
        """Test refresh using access token instead of refresh token fails."""
        # Login to get tokens
        await client.post(
            "/api/auth/register",
            json={
                "email": "wrongtype@example.com",
                "password": "SecurePass123!",
                "name": "Wrong Type User",
            },
        )

        login_response = await client.post(
            "/api/auth/login",
            json={"email": "wrongtype@example.com", "password": "SecurePass123!"},
        )
        tokens = login_response.json()

        # Try to refresh using access token
        response = await client.post(
            "/api/auth/refresh",
            json={"refresh_token": tokens["access_token"]},  # Wrong token type
        )
        assert response.status_code == 401


@pytest.mark.asyncio
class TestProtectedRoutes:
    """Test accessing protected routes."""

    async def test_access_protected_route_with_valid_token(self, client: AsyncClient):
        """Test accessing /me with valid token."""
        # Register and login
        await client.post(
            "/api/auth/register",
            json={
                "email": "protected@example.com",
                "password": "SecurePass123!",
                "name": "Protected User",
            },
        )

        login_response = await client.post(
            "/api/auth/login",
            json={"email": "protected@example.com", "password": "SecurePass123!"},
        )
        tokens = login_response.json()

        # Access protected route
        response = await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "protected@example.com"
        assert data["name"] == "Protected User"

    async def test_access_protected_route_without_token(self, client: AsyncClient):
        """Test accessing protected route without token fails."""
        response = await client.get("/api/auth/me")
        assert response.status_code == 403  # No auth header

    async def test_access_protected_route_with_invalid_token(
        self, client: AsyncClient
    ):
        """Test accessing protected route with invalid token fails."""
        response = await client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert response.status_code == 401


@pytest.mark.asyncio
class TestLogout:
    """Test user logout."""

    async def test_logout_success(self, client: AsyncClient):
        """Test successful logout."""
        # Register and login
        await client.post(
            "/api/auth/register",
            json={
                "email": "logout@example.com",
                "password": "SecurePass123!",
                "name": "Logout User",
            },
        )

        login_response = await client.post(
            "/api/auth/login",
            json={"email": "logout@example.com", "password": "SecurePass123!"},
        )
        tokens = login_response.json()

        # Logout
        response = await client.post(
            "/api/auth/logout",
            headers={
                "Authorization": f"Bearer {tokens['access_token']}",
                "X-Refresh-Token": tokens["refresh_token"],
            },
        )
        assert response.status_code == 204

        # Try to access protected route with logged out token
        response = await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert response.status_code == 401  # Token blacklisted


@pytest.mark.asyncio
class TestPasswordChange:
    """Test password change functionality."""

    async def test_change_password_success(self, client: AsyncClient):
        """Test successful password change."""
        # Register and login
        await client.post(
            "/api/auth/register",
            json={
                "email": "changepass@example.com",
                "password": "OldPass123!",
                "name": "Change Pass User",
            },
        )

        login_response = await client.post(
            "/api/auth/login",
            json={"email": "changepass@example.com", "password": "OldPass123!"},
        )
        tokens = login_response.json()

        # Change password
        response = await client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            json={
                "current_password": "OldPass123!",
                "new_password": "NewPass123!",
            },
        )
        assert response.status_code == 204

        # Verify can login with new password
        response = await client.post(
            "/api/auth/login",
            json={"email": "changepass@example.com", "password": "NewPass123!"},
        )
        assert response.status_code == 200

        # Verify cannot login with old password
        response = await client.post(
            "/api/auth/login",
            json={"email": "changepass@example.com", "password": "OldPass123!"},
        )
        assert response.status_code == 401

    async def test_change_password_wrong_current(self, client: AsyncClient):
        """Test password change with wrong current password fails."""
        # Register and login
        await client.post(
            "/api/auth/register",
            json={
                "email": "wrongcurrent@example.com",
                "password": "CorrectPass123!",
                "name": "Wrong Current User",
            },
        )

        login_response = await client.post(
            "/api/auth/login",
            json={
                "email": "wrongcurrent@example.com",
                "password": "CorrectPass123!",
            },
        )
        tokens = login_response.json()

        # Try to change with wrong current password
        response = await client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            json={
                "current_password": "WrongPass123!",
                "new_password": "NewPass123!",
            },
        )
        assert response.status_code == 401


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
