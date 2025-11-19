"""
Authentication API Endpoints

Handles user authentication, registration, and token management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.database import get_db
from app.models.user import User
from app.auth import (
    AuthService,
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    ChangePasswordRequest,
    UserResponse,
    get_current_user,
)


router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="""
    Register a new user account.

    **Requirements**:
    - Email must be valid and unique
    - Password must be 8-128 characters
    - Password must contain: uppercase, lowercase, digit, special character
    - Name must be 2-100 characters

    **Returns**:
    - User information (without password)
    """,
)
async def register(
    request: RegisterRequest,
    session: AsyncSession = Depends(get_db),
):
    """Register new user."""
    auth_service = AuthService(session)
    return await auth_service.register(request)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and get tokens",
    description="""
    Authenticate user and receive JWT tokens.

    **Security**:
    - Account locked for 30 minutes after 5 failed attempts
    - Tokens expire: access (15 min), refresh (7 days)

    **Returns**:
    - access_token: Use for API requests
    - refresh_token: Use to get new access token
    - expires_in: Access token lifetime in seconds
    """,
)
async def login(
    request: LoginRequest,
    session: AsyncSession = Depends(get_db),
):
    """Login and get JWT tokens."""
    auth_service = AuthService(session)
    return await auth_service.login(request)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="""
    Get a new access token using refresh token.

    **Usage**:
    - Use when access token expires
    - Refresh token must not be expired or blacklisted
    - Returns new access token (same refresh token)
    """,
)
async def refresh_token(
    request: RefreshTokenRequest,
    session: AsyncSession = Depends(get_db),
):
    """Refresh access token."""
    auth_service = AuthService(session)
    return await auth_service.refresh_access_token(request.refresh_token)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Logout (invalidate tokens)",
    description="""
    Logout by blacklisting current tokens.

    **Headers Required**:
    - Authorization: Bearer <access_token>
    - X-Refresh-Token: <refresh_token>

    **Effect**:
    - Both tokens become invalid immediately
    - Tokens stored in blacklist until natural expiry
    """,
)
async def logout(
    current_user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_refresh_token: Optional[str] = Header(None, alias="X-Refresh-Token"),
    session: AsyncSession = Depends(get_db),
):
    """Logout user by blacklisting tokens."""
    if not x_refresh_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refresh token required in X-Refresh-Token header",
        )

    access_token = credentials.credentials
    auth_service = AuthService(session)

    await auth_service.logout(
        access_token=access_token,
        refresh_token=x_refresh_token,
        user_id=current_user.id,
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user info",
    description="""
    Get authenticated user's information.

    **Requires**: Valid access token in Authorization header

    **Returns**: User profile data
    """,
)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse.model_validate(current_user)


@router.post(
    "/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Change password",
    description="""
    Change user password.

    **Requirements**:
    - Must provide correct current password
    - New password must meet strength requirements
    - New password must be different from current

    **Effect**:
    - Password updated immediately
    - All existing sessions remain valid
    - Consider logging out all devices after password change
    """,
)
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Change user password."""
    auth_service = AuthService(session)
    await auth_service.change_password(current_user.id, request)


@router.get(
    "/health",
    summary="Auth health check",
    description="Check if authentication service is operational",
)
async def health_check():
    """Health check for auth service."""
    return {
        "status": "healthy",
        "service": "authentication",
        "features": [
            "registration",
            "login",
            "token_refresh",
            "logout",
            "password_change",
        ],
    }
