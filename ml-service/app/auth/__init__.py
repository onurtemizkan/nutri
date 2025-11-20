"""
Authentication Module

Provides complete authentication system including:
- JWT token management
- Password hashing and validation
- User registration and login
- Token blacklist for logout
- FastAPI dependencies for route protection
"""

from app.auth.jwt import JWTHandler
from app.auth.password import PasswordHandler
from app.auth.service import AuthService
from app.auth.token_blacklist import TokenBlacklist
from app.auth.dependencies import (
    get_current_user,
    get_current_active_user,
    get_optional_current_user,
    require_role,
)
from app.auth.schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    ChangePasswordRequest,
    UserResponse,
)


__all__ = [
    # Core classes
    "JWTHandler",
    "PasswordHandler",
    "AuthService",
    "TokenBlacklist",
    # Dependencies
    "get_current_user",
    "get_current_active_user",
    "get_optional_current_user",
    "require_role",
    # Schemas
    "RegisterRequest",
    "LoginRequest",
    "TokenResponse",
    "RefreshTokenRequest",
    "ChangePasswordRequest",
    "UserResponse",
]
