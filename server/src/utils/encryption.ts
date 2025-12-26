/**
 * Encryption Utility for Secure Token Storage
 *
 * Uses AES-256-GCM for authenticated encryption of OAuth tokens
 * and other sensitive data stored in the database.
 */

import crypto from 'crypto';

// Encryption algorithm configuration
const ALGORITHM = 'aes-256-gcm';
const IV_LENGTH = 16; // 128 bits
const TAG_LENGTH = 16; // 128 bits
const KEY_LENGTH = 32; // 256 bits

/**
 * Get the encryption key from environment variable
 * Falls back to a derived key from JWT_SECRET for development
 */
function getEncryptionKey(): Buffer {
  // Try dedicated encryption key first
  if (process.env.CGM_ENCRYPTION_KEY) {
    const key = Buffer.from(process.env.CGM_ENCRYPTION_KEY, 'hex');
    if (key.length !== KEY_LENGTH) {
      throw new Error('CGM_ENCRYPTION_KEY must be 64 hex characters (256 bits)');
    }
    return key;
  }

  // Fall back to deriving key from JWT_SECRET for development
  const secret = process.env.JWT_SECRET;
  if (!secret) {
    throw new Error('CGM_ENCRYPTION_KEY or JWT_SECRET must be set for token encryption');
  }

  // Derive a key from JWT_SECRET using PBKDF2
  return crypto.pbkdf2Sync(
    secret,
    'cgm-token-encryption-salt', // Static salt for key derivation
    100000, // Iterations
    KEY_LENGTH,
    'sha256'
  );
}

/**
 * Encrypt a string value using AES-256-GCM
 *
 * @param plaintext - The string to encrypt
 * @returns Base64-encoded ciphertext with IV and auth tag prepended
 */
export function encrypt(plaintext: string): string {
  if (!plaintext) {
    throw new Error('Cannot encrypt empty value');
  }

  const key = getEncryptionKey();
  const iv = crypto.randomBytes(IV_LENGTH);

  const cipher = crypto.createCipheriv(ALGORITHM, key, iv);

  let ciphertext = cipher.update(plaintext, 'utf8', 'base64');
  ciphertext += cipher.final('base64');

  const authTag = cipher.getAuthTag();

  // Combine IV + authTag + ciphertext into a single base64 string
  const combined = Buffer.concat([iv, authTag, Buffer.from(ciphertext, 'base64')]);

  return combined.toString('base64');
}

/**
 * Decrypt a value that was encrypted with the encrypt function
 *
 * @param encryptedData - Base64-encoded ciphertext with IV and auth tag
 * @returns The original plaintext string
 */
export function decrypt(encryptedData: string): string {
  if (!encryptedData) {
    throw new Error('Cannot decrypt empty value');
  }

  const key = getEncryptionKey();

  try {
    const combined = Buffer.from(encryptedData, 'base64');

    // Extract IV, authTag, and ciphertext
    const iv = combined.subarray(0, IV_LENGTH);
    const authTag = combined.subarray(IV_LENGTH, IV_LENGTH + TAG_LENGTH);
    const ciphertext = combined.subarray(IV_LENGTH + TAG_LENGTH);

    const decipher = crypto.createDecipheriv(ALGORITHM, key, iv);
    decipher.setAuthTag(authTag);

    let plaintext = decipher.update(ciphertext.toString('base64'), 'base64', 'utf8');
    plaintext += decipher.final('utf8');

    return plaintext;
  } catch (error) {
    // Re-throw with a generic message to avoid leaking encryption details
    throw new Error('Failed to decrypt data - data may be corrupted or key has changed');
  }
}

/**
 * Generate a secure random encryption key
 * Use this to generate a new CGM_ENCRYPTION_KEY for production
 *
 * @returns Hex-encoded 256-bit key
 */
export function generateEncryptionKey(): string {
  return crypto.randomBytes(KEY_LENGTH).toString('hex');
}

/**
 * Hash a value for comparison purposes (e.g., for caching or lookup)
 * Uses SHA-256 to create a fixed-length hash
 *
 * @param value - The value to hash
 * @returns Hex-encoded hash
 */
export function hashValue(value: string): string {
  return crypto.createHash('sha256').update(value).digest('hex');
}

/**
 * Validate that an encryption key is properly formatted
 *
 * @param key - The key to validate
 * @returns true if valid, false otherwise
 */
export function isValidEncryptionKey(key: string): boolean {
  if (!key) return false;

  // Must be 64 hex characters (256 bits)
  if (!/^[0-9a-fA-F]{64}$/.test(key)) return false;

  return true;
}
