/**
 * Webhook Authentication Middleware
 *
 * Verifies JWS signatures for App Store Server Notifications V2.
 * Uses Apple's public keys from their JWKS endpoint.
 */

import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import jwksClient from 'jwks-rsa';
import { logger } from '../config/logger';

// Apple's JWKS endpoint for App Store Server Notifications
const APPLE_JWKS_URL = 'https://appleid.apple.com/auth/keys';

// Cache for JWKS client
let jwksClientInstance: jwksClient.JwksClient | null = null;

/**
 * Get or create the JWKS client instance
 */
function getJwksClient(): jwksClient.JwksClient {
  if (!jwksClientInstance) {
    jwksClientInstance = jwksClient({
      jwksUri: APPLE_JWKS_URL,
      cache: true,
      cacheMaxEntries: 5,
      cacheMaxAge: 600000, // 10 minutes
      rateLimit: true,
      jwksRequestsPerMinute: 10,
    });
  }
  return jwksClientInstance;
}

/**
 * Get the signing key from Apple's JWKS endpoint
 */
async function getAppleSigningKey(kid: string): Promise<string> {
  const client = getJwksClient();
  const key = await client.getSigningKey(kid);
  return key.getPublicKey();
}

/**
 * Decode and extract header from JWS token without verification
 */
function decodeJwsHeader(token: string): { kid?: string; alg?: string } {
  const parts = token.split('.');
  if (parts.length !== 3) {
    throw new Error('Invalid JWS format');
  }

  const headerJson = Buffer.from(parts[0], 'base64url').toString('utf-8');
  return JSON.parse(headerJson);
}

/**
 * Verify App Store JWS signature
 */
export async function verifyAppStoreSignature(signedPayload: string): Promise<unknown> {
  try {
    // Decode header to get key ID
    const header = decodeJwsHeader(signedPayload);

    if (!header.kid) {
      throw new Error('Missing key ID in JWS header');
    }

    // Get the public key from Apple's JWKS
    const publicKey = await getAppleSigningKey(header.kid);

    // Verify the signature
    const decoded = jwt.verify(signedPayload, publicKey, {
      algorithms: ['ES256', 'RS256'],
    });

    return decoded;
  } catch (error) {
    logger.error({ error }, 'Failed to verify App Store signature');
    throw error;
  }
}

/**
 * Express middleware to verify App Store webhook signatures
 */
export function appStoreWebhookAuth(req: Request, res: Response, next: NextFunction): void {
  const { signedPayload } = req.body as { signedPayload?: string };

  if (!signedPayload) {
    logger.warn('Missing signedPayload in webhook request');
    res.status(400).json({ error: 'Missing signedPayload' });
    return;
  }

  verifyAppStoreSignature(signedPayload)
    .then((payload) => {
      // Attach decoded payload to request for controller
      req.body.decodedPayload = payload;
      next();
    })
    .catch((error) => {
      logger.error({ error }, 'App Store webhook signature verification failed');
      res.status(401).json({ error: 'Invalid signature' });
    });
}

/**
 * Decode a signed transaction or renewal info JWS
 * Note: For internal App Store JWS (transaction info, renewal info),
 * we decode without JWKS verification since these are already
 * verified as part of the outer signed payload.
 */
export function decodeSignedData<T>(signedData: string): T {
  const parts = signedData.split('.');
  if (parts.length !== 3) {
    throw new Error('Invalid JWS format for signed data');
  }

  const payloadJson = Buffer.from(parts[1], 'base64url').toString('utf-8');
  return JSON.parse(payloadJson) as T;
}

export default appStoreWebhookAuth;
