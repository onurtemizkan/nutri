/**
 * Apple Identity Token Verification Service
 *
 * Verifies Apple Sign In identity tokens with Apple's servers
 * using their public keys from the JWKS endpoint.
 */

import jwt from 'jsonwebtoken';
import jwksClient from 'jwks-rsa';
import { logger } from '../config/logger';

/**
 * Apple ID Token Payload structure
 */
export interface AppleTokenPayload extends jwt.JwtPayload {
  /** Issuer - Always https://appleid.apple.com */
  iss: string;
  /** Audience - Your app's bundle identifier */
  aud: string;
  /** Expiration time (Unix timestamp) */
  exp: number;
  /** Issued at time (Unix timestamp) */
  iat: number;
  /** Subject - Apple user ID (unique, stable identifier) */
  sub: string;
  /** User's email (only provided on first authentication) */
  email?: string;
  /** Whether the email has been verified by Apple */
  email_verified?: boolean;
  /** Whether this is a private relay email */
  is_private_email?: boolean;
  /** Authentication time (Unix timestamp) */
  auth_time?: number;
}

/**
 * JWKS client for fetching Apple's public keys
 * Keys are cached for 24 hours to improve performance
 */
const appleJwksClient = jwksClient({
  jwksUri: 'https://appleid.apple.com/auth/keys',
  cache: true,
  cacheMaxAge: 86400000, // 24 hours
  rateLimit: true,
  jwksRequestsPerMinute: 10,
});

/**
 * Get signing key from Apple's JWKS endpoint
 */
function getAppleSigningKey(header: jwt.JwtHeader, callback: jwt.SigningKeyCallback): void {
  if (!header.kid) {
    callback(new Error('Token header missing kid (key ID)'));
    return;
  }

  appleJwksClient.getSigningKey(header.kid, (err, key) => {
    if (err) {
      callback(err);
      return;
    }

    if (!key) {
      callback(new Error('Unable to retrieve signing key'));
      return;
    }

    const signingKey = key.getPublicKey();
    callback(null, signingKey);
  });
}

/**
 * Check if Apple token verification is enabled
 * Requires APPLE_APP_ID environment variable
 */
export function isAppleVerificationEnabled(): boolean {
  return !!process.env.APPLE_APP_ID;
}

/**
 * Verify Apple identity token with Apple's servers
 *
 * @param identityToken - JWT token from Apple Sign In
 * @returns Decoded and verified token payload
 * @throws Error if token is invalid, expired, or verification fails
 */
export async function verifyAppleIdentityToken(identityToken: string): Promise<AppleTokenPayload> {
  const appId = process.env.APPLE_APP_ID;

  if (!appId) {
    throw new Error('APPLE_APP_ID environment variable is required for token verification');
  }

  return new Promise((resolve, reject) => {
    const options: jwt.VerifyOptions = {
      algorithms: ['RS256'],
      audience: appId,
      issuer: 'https://appleid.apple.com',
      clockTolerance: 60, // 60 seconds clock tolerance
    };

    jwt.verify(identityToken, getAppleSigningKey, options, (err, decoded) => {
      if (err) {
        if (err.name === 'TokenExpiredError') {
          reject(new Error('Apple identity token has expired'));
        } else if (err.name === 'JsonWebTokenError') {
          reject(new Error('Invalid Apple identity token'));
        } else if (err.name === 'NotBeforeError') {
          reject(new Error('Apple identity token not yet valid'));
        } else {
          reject(new Error(`Token verification failed: ${err.message}`));
        }
        return;
      }

      if (!decoded || typeof decoded === 'string') {
        reject(new Error('Invalid token payload'));
        return;
      }

      resolve(decoded as AppleTokenPayload);
    });
  });
}

/**
 * Decode Apple identity token without verification (development only)
 * WARNING: Only use this in development when APPLE_APP_ID is not configured
 *
 * @param identityToken - JWT token from Apple Sign In
 * @returns Decoded (but NOT verified) token payload
 */
export function decodeAppleTokenUnsafe(identityToken: string): { sub: string; email?: string } {
  const tokenParts = identityToken.split('.');
  if (tokenParts.length !== 3) {
    throw new Error('Invalid identity token format');
  }

  let payload: Record<string, unknown>;

  try {
    payload = JSON.parse(Buffer.from(tokenParts[1], 'base64').toString('utf8'));
  } catch {
    throw new Error('Failed to decode Apple identity token');
  }

  if (!payload.sub) {
    throw new Error('Invalid identity token: missing user ID');
  }

  return {
    sub: payload.sub as string,
    email: payload.email as string | undefined,
  };
}

/**
 * Verify or decode Apple token based on configuration
 *
 * In production (APPLE_APP_ID set): Verifies token with Apple's servers
 * In development (APPLE_APP_ID not set): Decodes without verification (logs warning)
 *
 * @param identityToken - JWT token from Apple Sign In
 * @returns Token payload with sub (Apple user ID) and optional email
 */
export async function processAppleToken(
  identityToken: string
): Promise<{ sub: string; email?: string; verified: boolean }> {
  if (isAppleVerificationEnabled()) {
    // Production: Verify with Apple's servers
    const payload = await verifyAppleIdentityToken(identityToken);
    logger.info(
      {
        appleId: payload.sub.substring(0, 8) + '...',
        hasEmail: !!payload.email,
        isPrivateEmail: payload.is_private_email,
      },
      'Apple token verified successfully'
    );
    return {
      sub: payload.sub,
      email: payload.email,
      verified: true,
    };
  } else {
    // Development: Decode without verification
    logger.warn(
      'Apple token verification DISABLED - APPLE_APP_ID not configured. ' +
        'Set APPLE_APP_ID environment variable for production.'
    );
    const payload = decodeAppleTokenUnsafe(identityToken);
    return {
      sub: payload.sub,
      email: payload.email,
      verified: false,
    };
  }
}
