"""
Authentication Dependencies

FastAPI dependencies for protecting routes and getting current user.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError

from app.database import get_db
from app.models.user import User
from app.auth.jwt import JWTHandler
from app.auth.token_blacklist import TokenBlacklist
from app.auth.service import AuthService


# OAuth2 bearer token scheme
security = HTTPBearer(
    scheme_name="Bearer",
    description="JWT access token",
    auto_error=True,
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_db),
) -> User:
    """
    Get current authenticated user from JWT token.

    This dependency should be used on all protected routes.

    Args:
        credentials: HTTP authorization credentials (Bearer token)
        session: Database session

    Returns:
        Authenticated user

    Raises:
        HTTPException 401: If token invalid, expired, or user not found
        HTTPException 403: If user account is disabled
    """
    token = credentials.credentials

    # Verify token format and signature
    try:
        payload = JWTHandler.verify_access_token(token)
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if token is blacklisted (user logged out)
    if await TokenBlacklist.is_blacklisted(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth_service = AuthService(session)
    user = await auth_service.get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user account is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user (alias for clarity).

    Args:
        current_user: Current user from get_current_user

    Returns:
        Active user
    """
    return current_user


async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    session: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.

    Use this for routes that work both authenticated and unauthenticated.

    Args:
        credentials: Optional HTTP authorization credentials
        session: Database session

    Returns:
        User if authenticated, None otherwise
    """
    if not credentials:
        return None

    try:
        token = credentials.credentials
        payload = JWTHandler.verify_access_token(token)

        # Check blacklist
        if await TokenBlacklist.is_blacklisted(token):
            return None

        # Get user
        user_id = payload.get("sub")
        if not user_id:
            return None

        auth_service = AuthService(session)
        user = await auth_service.get_user_by_id(user_id)

        if user and user.is_active:
            return user

        return None

    except (InvalidTokenError, ExpiredSignatureError):
        return None


def require_role(required_role: str):
    """
    Dependency factory for role-based access control.

    Future implementation when roles are added to User model.

    Args:
        required_role: Required role name

    Returns:
        Dependency function

    Example:
        @router.get("/admin")
        async def admin_route(user: User = Depends(require_role("admin"))):
            pass
    """

    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        # TODO: Implement role checking when User model has roles
        # if current_user.role != required_role:
        #     raise HTTPException(
        #         status_code=status.HTTP_403_FORBIDDEN,
        #         detail=f"Required role: {required_role}"
        #     )
        return current_user

    return role_checker
