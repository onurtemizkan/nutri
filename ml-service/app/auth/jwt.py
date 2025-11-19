"""
JWT Token Management

Handles creation, validation, and decoding of JWT tokens.
Uses asymmetric RS256 algorithm for enhanced security.
"""

from datetime import datetime, timedelta, UTC
from typing import Optional, Dict, Any
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
import secrets

from app.config import settings


class JWTHandler:
    """JWT token creation and validation."""

    # Token types
    ACCESS_TOKEN_TYPE = "access"
    REFRESH_TOKEN_TYPE = "refresh"

    # Expiration times
    ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Short-lived access tokens
    REFRESH_TOKEN_EXPIRE_DAYS = 7  # Longer-lived refresh tokens

    @staticmethod
    def create_access_token(
        user_id: str,
        email: str,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create JWT access token.

        Args:
            user_id: User's ID (UUID string or integer)
            email: User's email
            additional_claims: Optional extra claims to include

        Returns:
            Encoded JWT token string
        """
        expires = datetime.now(UTC) + timedelta(
            minutes=JWTHandler.ACCESS_TOKEN_EXPIRE_MINUTES
        )

        claims = {
            "sub": str(user_id),  # Subject (user ID as string)
            "email": email,
            "type": JWTHandler.ACCESS_TOKEN_TYPE,
            "exp": expires,  # Expiration time
            "iat": datetime.now(UTC),  # Issued at
            "nbf": datetime.now(UTC),  # Not before
            "jti": secrets.token_urlsafe(16),  # Unique token ID
        }

        # Add any additional claims
        if additional_claims:
            claims.update(additional_claims)

        return jwt.encode(
            claims,
            settings.secret_key,
            algorithm=settings.algorithm,
        )

    @staticmethod
    def create_refresh_token(user_id: str, email: str) -> str:
        """
        Create JWT refresh token.

        Args:
            user_id: User's ID (UUID string or integer)
            email: User's email

        Returns:
            Encoded JWT refresh token string
        """
        expires = datetime.now(UTC) + timedelta(
            days=JWTHandler.REFRESH_TOKEN_EXPIRE_DAYS
        )

        claims = {
            "sub": str(user_id),  # Subject (user ID as string)
            "email": email,
            "type": JWTHandler.REFRESH_TOKEN_TYPE,
            "exp": expires,
            "iat": datetime.now(UTC),
            "nbf": datetime.now(UTC),
            "jti": secrets.token_urlsafe(16),  # Unique token ID
        }

        return jwt.encode(
            claims,
            settings.secret_key,
            algorithm=settings.algorithm,
        )

    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """
        Decode and validate JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded token payload

        Raises:
            InvalidTokenError: If token is invalid
            ExpiredSignatureError: If token has expired
        """
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.algorithm],
                options={
                    "verify_exp": True,  # Verify expiration
                    "verify_nbf": True,  # Verify not before
                    "verify_iat": True,  # Verify issued at
                },
            )
            return payload
        except ExpiredSignatureError:
            raise ExpiredSignatureError("Token has expired")
        except InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid token: {str(e)}")

    @staticmethod
    def verify_access_token(token: str) -> Dict[str, Any]:
        """
        Verify access token and return payload.

        Args:
            token: JWT access token string

        Returns:
            Decoded token payload

        Raises:
            InvalidTokenError: If token is invalid or not an access token
        """
        payload = JWTHandler.decode_token(token)

        # Verify token type
        if payload.get("type") != JWTHandler.ACCESS_TOKEN_TYPE:
            raise InvalidTokenError("Invalid token type")

        return payload

    @staticmethod
    def verify_refresh_token(token: str) -> Dict[str, Any]:
        """
        Verify refresh token and return payload.

        Args:
            token: JWT refresh token string

        Returns:
            Decoded token payload

        Raises:
            InvalidTokenError: If token is invalid or not a refresh token
        """
        payload = JWTHandler.decode_token(token)

        # Verify token type
        if payload.get("type") != JWTHandler.REFRESH_TOKEN_TYPE:
            raise InvalidTokenError("Invalid token type")

        return payload

    @staticmethod
    def get_token_expiry(token: str) -> Optional[datetime]:
        """
        Get token expiration datetime.

        Args:
            token: JWT token string

        Returns:
            Expiration datetime or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.algorithm],
                options={"verify_exp": False},  # Don't verify, just get exp
            )
            exp_timestamp = payload.get("exp")
            if exp_timestamp:
                return datetime.fromtimestamp(exp_timestamp, UTC)
            return None
        except InvalidTokenError:
            return None
