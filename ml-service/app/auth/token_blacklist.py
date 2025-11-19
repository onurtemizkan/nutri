"""
Token Blacklist

Manages blacklisted tokens in Redis for logout functionality.
Tokens are blacklisted when users logout to prevent reuse.
"""

from datetime import datetime, UTC
from typing import Optional
import logging

from app.redis_client import redis_client
from app.auth.jwt import JWTHandler


logger = logging.getLogger(__name__)


class TokenBlacklist:
    """Manage blacklisted tokens in Redis."""

    PREFIX = "blacklist:token:"

    @staticmethod
    async def blacklist_token(token: str, user_id: str) -> bool:
        """
        Add token to blacklist.

        Args:
            token: JWT token to blacklist
            user_id: User ID (UUID string) for logging

        Returns:
            True if blacklisted successfully
        """
        try:
            # Get token expiry time
            expiry = JWTHandler.get_token_expiry(token)
            if not expiry:
                logger.warning(f"Could not get expiry for token (user {user_id})")
                # Blacklist for 7 days as fallback
                ttl = 7 * 24 * 60 * 60
            else:
                # Calculate TTL from current time to expiry
                now = datetime.now(UTC)
                if expiry <= now:
                    # Token already expired, no need to blacklist
                    return True

                ttl = int((expiry - now).total_seconds())

            # Store in Redis with expiry
            key = f"{TokenBlacklist.PREFIX}{token}"
            await redis_client.set(
                key,
                user_id,  # Store user_id for logging
                ttl=ttl,
            )

            logger.info(f"Token blacklisted for user {user_id}, TTL: {ttl}s")
            return True

        except Exception as e:
            logger.error(f"Failed to blacklist token for user {user_id}: {e}")
            # Even if Redis fails, we should not block logout
            # The token will still expire naturally
            return True

    @staticmethod
    async def is_blacklisted(token: str) -> bool:
        """
        Check if token is blacklisted.

        Args:
            token: JWT token to check

        Returns:
            True if token is blacklisted, False otherwise
        """
        try:
            key = f"{TokenBlacklist.PREFIX}{token}"
            result = await redis_client.get(key)
            return result is not None
        except Exception as e:
            logger.error(f"Failed to check token blacklist: {e}")
            # If Redis is down, fail open (allow token)
            # Token will still be validated for expiry/signature
            return False

    @staticmethod
    async def clear_user_tokens(user_id: str) -> int:
        """
        Clear all blacklisted tokens for a user.

        This is useful for:
        - Account deletion
        - Password reset (logout all sessions)
        - Admin actions

        Args:
            user_id: User ID (UUID string)

        Returns:
            Number of tokens cleared
        """
        try:
            # This requires scanning Redis keys which can be slow
            # Only use for admin actions or account deletion
            pattern = f"{TokenBlacklist.PREFIX}*"
            cursor = 0
            deleted = 0

            # Scan in batches to avoid blocking
            while True:
                cursor, keys = await redis_client.scan(
                    cursor, match=pattern, count=100
                )

                # Check each key's value to see if it matches user_id
                for key in keys:
                    value = await redis_client.get(key)
                    if value == user_id:
                        await redis_client.delete(key)
                        deleted += 1

                if cursor == 0:
                    break

            logger.info(f"Cleared {deleted} blacklisted tokens for user {user_id}")
            return deleted

        except Exception as e:
            logger.error(f"Failed to clear tokens for user {user_id}: {e}")
            return 0
