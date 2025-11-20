"""
Password Management

Handles password hashing, verification, and validation.
Uses bcrypt with proper work factor for security.
"""

import re
from typing import Tuple
from passlib.context import CryptContext


# Bcrypt context with proper work factor
# Work factor of 12 = ~300ms on modern CPU (good balance)
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,  # Work factor
)


class PasswordHandler:
    """Password hashing and validation."""

    # Password requirements
    MIN_LENGTH = 8
    MAX_LENGTH = 128
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGIT = True
    REQUIRE_SPECIAL = True

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password string

        Raises:
            ValueError: If password is empty
        """
        if not password:
            raise ValueError("Password cannot be empty")

        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash.

        Args:
            plain_password: Plain text password to verify
            hashed_password: Stored hashed password

        Returns:
            True if password matches, False otherwise
        """
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            # Invalid hash format or verification error
            return False

    @staticmethod
    def validate_password_strength(password: str) -> Tuple[bool, str]:
        """
        Validate password meets complexity requirements.

        Args:
            password: Password to validate

        Returns:
            Tuple of (is_valid, error_message)
            If valid: (True, "")
            If invalid: (False, "error message")
        """
        # Check length
        if len(password) < PasswordHandler.MIN_LENGTH:
            return (
                False,
                f"Password must be at least {PasswordHandler.MIN_LENGTH} characters",
            )

        if len(password) > PasswordHandler.MAX_LENGTH:
            return (
                False,
                f"Password must not exceed {PasswordHandler.MAX_LENGTH} characters",
            )

        # Check uppercase
        if PasswordHandler.REQUIRE_UPPERCASE and not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"

        # Check lowercase
        if PasswordHandler.REQUIRE_LOWERCASE and not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"

        # Check digit
        if PasswordHandler.REQUIRE_DIGIT and not re.search(r"\d", password):
            return False, "Password must contain at least one digit"

        # Check special character
        if PasswordHandler.REQUIRE_SPECIAL and not re.search(
            r"[!@#$%^&*(),.?\":{}|<>]", password
        ):
            return (
                False,
                "Password must contain at least one special character (!@#$%^&*...)",
            )

        # Check for common passwords (basic check)
        common_passwords = {
            "password",
            "password123",
            "12345678",
            "qwerty123",
            "admin123",
        }
        if password.lower() in common_passwords:
            return False, "Password is too common, please choose a stronger password"

        return True, ""

    @staticmethod
    def needs_rehash(hashed_password: str) -> bool:
        """
        Check if password hash needs to be updated.

        This can happen if:
        - Hashing algorithm was changed
        - Work factor was increased
        - Hash is using deprecated scheme

        Args:
            hashed_password: Stored hashed password

        Returns:
            True if hash should be regenerated
        """
        return pwd_context.needs_update(hashed_password)
