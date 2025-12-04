"""
Authentication Service

Business logic for user authentication, registration, and token management.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status
import logging
import uuid

from app.models.user import User
from app.auth.jwt import JWTHandler
from app.auth.password import PasswordHandler
from app.auth.token_blacklist import TokenBlacklist
from app.auth.schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
    ChangePasswordRequest,
)


logger = logging.getLogger(__name__)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Ensure datetime is timezone-aware (UTC).

    SQLite stores datetimes as text and loses timezone info.
    This function makes naive datetimes timezone-aware.

    Args:
        dt: Datetime that may be naive or aware

    Returns:
        Timezone-aware datetime in UTC, or None if input is None
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        return dt.replace(tzinfo=UTC)
    return dt


class AuthService:
    """Authentication business logic."""

    # Account lockout settings
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 30

    def __init__(self, session: AsyncSession):
        """
        Initialize auth service.

        Args:
            session: Database session
        """
        self.session = session

    async def register(self, request: RegisterRequest) -> UserResponse:
        """
        Register new user.

        Args:
            request: Registration request data

        Returns:
            Created user data

        Raises:
            HTTPException: If email already exists or validation fails
        """
        # Check if email already exists
        result = await self.session.execute(
            select(User).where(User.email == request.email)
        )
        existing_user = result.scalar_one_or_none()

        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Validate password strength
        is_valid, error_msg = PasswordHandler.validate_password_strength(
            request.password
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )

        # Hash password
        hashed_password = PasswordHandler.hash_password(request.password)

        # Create user
        user = User(
            id=str(uuid.uuid4()),  # Generate UUID for ID
            email=request.email,
            password=hashed_password,
            name=request.name,
            is_active=True,
            failed_login_attempts=0,
        )

        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)

        logger.info(f"User registered: {user.email} (ID: {user.id})")

        return UserResponse.model_validate(user)

    async def login(self, request: LoginRequest) -> TokenResponse:
        """
        Authenticate user and return tokens.

        Args:
            request: Login credentials

        Returns:
            Access and refresh tokens

        Raises:
            HTTPException: If credentials invalid or account locked
        """
        # Get user by email
        result = await self.session.execute(
            select(User).where(User.email == request.email)
        )
        user = result.scalar_one_or_none()

        if not user:
            # Don't reveal that email doesn't exist
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Check if account is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is disabled",
            )

        # Check if account is locked
        locked_until_utc = ensure_utc(user.locked_until)
        if locked_until_utc and locked_until_utc > datetime.now(UTC):
            remaining = int((locked_until_utc - datetime.now(UTC)).total_seconds())
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account locked. Try again in {remaining // 60} minutes",
            )

        # Verify password
        if not PasswordHandler.verify_password(request.password, user.password):
            # Increment failed attempts
            user.failed_login_attempts += 1

            # Lock account if too many failures
            if user.failed_login_attempts >= self.MAX_FAILED_ATTEMPTS:
                user.locked_until = datetime.now(UTC) + timedelta(
                    minutes=self.LOCKOUT_DURATION_MINUTES
                )
                await self.session.commit()

                logger.warning(
                    f"Account locked due to failed attempts: {user.email}"
                )

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Account locked due to too many failed attempts. "
                    f"Try again in {self.LOCKOUT_DURATION_MINUTES} minutes",
                )

            await self.session.commit()

            logger.warning(
                f"Failed login attempt for {user.email} "
                f"({user.failed_login_attempts}/{self.MAX_FAILED_ATTEMPTS})"
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        # Password verified - reset failed attempts and update last login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now(UTC)

        # Check if password hash needs update
        if PasswordHandler.needs_rehash(user.password):
            user.password = PasswordHandler.hash_password(request.password)
            logger.info(f"Password hash updated for user {user.email}")

        await self.session.commit()

        # Generate tokens
        access_token = JWTHandler.create_access_token(
            user_id=user.id,
            email=user.email,
        )

        refresh_token = JWTHandler.create_refresh_token(
            user_id=user.id,
            email=user.email,
        )

        logger.info(f"User logged in: {user.email}")

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=JWTHandler.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    async def refresh_access_token(self, refresh_token: str) -> TokenResponse:
        """
        Generate new access token from refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            New access token

        Raises:
            HTTPException: If refresh token invalid or user not found
        """
        # Verify refresh token
        try:
            payload = JWTHandler.verify_refresh_token(refresh_token)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid refresh token: {str(e)}",
            )

        # Check if token is blacklisted
        if await TokenBlacklist.is_blacklisted(refresh_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
            )

        # Get user
        user_id = payload.get("sub")
        result = await self.session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
            )

        # Generate new access token
        access_token = JWTHandler.create_access_token(
            user_id=user.id,
            email=user.email,
        )

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,  # Keep same refresh token
            token_type="bearer",
            expires_in=JWTHandler.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    async def logout(self, access_token: str, refresh_token: str, user_id: str) -> None:
        """
        Logout user by blacklisting tokens.

        Args:
            access_token: Access token to blacklist
            refresh_token: Refresh token to blacklist
            user_id: User ID (UUID string)

        Raises:
            HTTPException: Never raises (graceful failure)
        """
        # Blacklist both tokens
        await TokenBlacklist.blacklist_token(access_token, user_id)
        await TokenBlacklist.blacklist_token(refresh_token, user_id)

        logger.info(f"User logged out: {user_id}")

    async def change_password(
        self, user_id: str, request: ChangePasswordRequest
    ) -> None:
        """
        Change user password.

        Args:
            user_id: User ID
            request: Password change request

        Raises:
            HTTPException: If current password wrong or validation fails
        """
        # Get user
        result = await self.session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        # Verify current password
        if not PasswordHandler.verify_password(
            request.current_password, user.password
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect",
            )

        # Validate new password
        is_valid, error_msg = PasswordHandler.validate_password_strength(
            request.new_password
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )

        # Check new password is different from current
        if PasswordHandler.verify_password(request.new_password, user.password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be different from current password",
            )

        # Update password
        user.password = PasswordHandler.hash_password(request.new_password)
        await self.session.commit()

        logger.info(f"Password changed for user: {user.email}")

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User object or None
        """
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
