"""
Authentication Schemas

Pydantic models for authentication requests and responses.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator, ConfigDict


class RegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(
        ...,
        description="User's password (validated by service layer)",
    )
    name: str = Field(..., min_length=2, max_length=100, description="User's full name")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name doesn't contain invalid characters."""
        if not v.strip():
            raise ValueError("Name cannot be empty")
        # Remove excessive whitespace
        return " ".join(v.split())


class LoginRequest(BaseModel):
    """User login request."""

    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiry in seconds")


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str = Field(..., description="Refresh token to exchange")


class ChangePasswordRequest(BaseModel):
    """Change password request."""

    current_password: str = Field(..., description="Current password")
    new_password: str = Field(
        ...,
        description="New password (validated by service layer)",
    )


class UserResponse(BaseModel):
    """User information response."""

    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User's email")
    name: str = Field(..., description="User's name")
    created_at: datetime = Field(..., description="Account creation date")
    last_login: Optional[datetime] = Field(None, description="Last login date")
    is_active: bool = Field(default=True, description="Account active status")

    model_config = ConfigDict(from_attributes=True)


class PasswordResetRequest(BaseModel):
    """Request password reset (future implementation)."""

    email: EmailStr = Field(..., description="User's email address")


class PasswordResetConfirm(BaseModel):
    """Confirm password reset with token (future implementation)."""

    token: str = Field(..., description="Password reset token")
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password (8-128 characters)",
    )
