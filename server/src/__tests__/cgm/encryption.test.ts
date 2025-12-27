/**
 * Encryption Utility Tests
 *
 * Tests for AES-256-GCM encryption/decryption of OAuth tokens
 * and other sensitive CGM data.
 */

import {
  encrypt,
  decrypt,
  generateEncryptionKey,
  hashValue,
  isValidEncryptionKey,
} from '../../utils/encryption';

describe('Encryption Utility', () => {
  // Store original env values
  const originalEnv = process.env;

  beforeEach(() => {
    // Reset environment for each test
    process.env = { ...originalEnv };
    // Ensure JWT_SECRET is set for tests
    process.env.JWT_SECRET = 'test-jwt-secret-for-encryption-tests-minimum-length';
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  // ============================================================================
  // encrypt() and decrypt()
  // ============================================================================

  describe('encrypt() and decrypt()', () => {
    it('should encrypt and decrypt a simple string', () => {
      const plaintext = 'hello-world-test-token';

      const encrypted = encrypt(plaintext);
      const decrypted = decrypt(encrypted);

      expect(decrypted).toBe(plaintext);
      expect(encrypted).not.toBe(plaintext);
    });

    it('should encrypt and decrypt a long OAuth token', () => {
      const token =
        'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWUsImlhdCI6MTUxNjIzOTAyMn0.POstGetfAytaZS82wHcjoTyoqhMyxXiWdR7Nn7A29DNSl0EiXLdwJ6xC6AfgZWF1bOsS_TuYI3OG85AmiExREkrS6tDfTQ2B3WXlrr-wp5AokiRbz3_oB4OxG-W9KcEEbDRcZc0nH3L7LzYptiy1PtAylQGxHTWZXtGz4ht0bAecBgmpdgXMguEIcoqPJ1n3pIWk_dUZegpqx0Lka21H6XxUTxiy8OcaarA8zdnPUnV6AmNP3ecFawIFYdvJB_cm-GvpCSbr8G8y_Mllj8f4x9nBH8pQux89_6gUY618iYv7tuPWBFfEbLxtF2pZS6YC1aSfLQxeNe8djT9YjpvRZA';

      const encrypted = encrypt(token);
      const decrypted = decrypt(encrypted);

      expect(decrypted).toBe(token);
    });

    it('should produce different ciphertext for the same plaintext (random IV)', () => {
      const plaintext = 'same-input-different-output';

      const encrypted1 = encrypt(plaintext);
      const encrypted2 = encrypt(plaintext);

      // Same plaintext should produce different ciphertext due to random IV
      expect(encrypted1).not.toBe(encrypted2);

      // But both should decrypt to the same plaintext
      expect(decrypt(encrypted1)).toBe(plaintext);
      expect(decrypt(encrypted2)).toBe(plaintext);
    });

    it('should handle special characters', () => {
      const plaintext = 'token!@#$%^&*()_+-=[]{}|;:,.<>?/~`';

      const encrypted = encrypt(plaintext);
      const decrypted = decrypt(encrypted);

      expect(decrypted).toBe(plaintext);
    });

    it('should handle unicode characters', () => {
      const plaintext = 'token-with-unicode-\u00e9\u00e8\u00ea-\u4e2d\u6587-\ud83d\ude00';

      const encrypted = encrypt(plaintext);
      const decrypted = decrypt(encrypted);

      expect(decrypted).toBe(plaintext);
    });

    it('should throw error when encrypting empty string', () => {
      expect(() => encrypt('')).toThrow('Cannot encrypt empty value');
    });

    it('should throw error when decrypting empty string', () => {
      expect(() => decrypt('')).toThrow('Cannot decrypt empty value');
    });

    it('should throw error when decrypting invalid base64', () => {
      expect(() => decrypt('not-valid-base64!!!')).toThrow('Failed to decrypt');
    });

    it('should throw error when decrypting corrupted data', () => {
      const encrypted = encrypt('test-value');
      // Corrupt the auth tag area (bytes 16-32 after base64 decode, which is roughly chars 21-42 in base64)
      // We corrupt multiple characters to ensure the auth tag is definitely invalid
      const corrupted = encrypted.slice(0, 25) + 'XXXX' + encrypted.slice(29);

      expect(() => decrypt(corrupted)).toThrow('Failed to decrypt');
    });

    it('should throw error when JWT_SECRET is not set', () => {
      delete process.env.JWT_SECRET;
      delete process.env.CGM_ENCRYPTION_KEY;

      expect(() => encrypt('test')).toThrow('CGM_ENCRYPTION_KEY or JWT_SECRET must be set');
    });

    it('should use CGM_ENCRYPTION_KEY when set', () => {
      // Generate a valid key
      const key = generateEncryptionKey();
      process.env.CGM_ENCRYPTION_KEY = key;

      const plaintext = 'test-with-dedicated-key';
      const encrypted = encrypt(plaintext);
      const decrypted = decrypt(encrypted);

      expect(decrypted).toBe(plaintext);
    });

    it('should reject invalid CGM_ENCRYPTION_KEY length', () => {
      process.env.CGM_ENCRYPTION_KEY = 'too-short';

      expect(() => encrypt('test')).toThrow('CGM_ENCRYPTION_KEY must be 64 hex characters');
    });
  });

  // ============================================================================
  // generateEncryptionKey()
  // ============================================================================

  describe('generateEncryptionKey()', () => {
    it('should generate a 64-character hex string', () => {
      const key = generateEncryptionKey();

      expect(key).toHaveLength(64);
      expect(/^[0-9a-f]{64}$/.test(key)).toBe(true);
    });

    it('should generate unique keys on each call', () => {
      const keys = new Set<string>();

      for (let i = 0; i < 100; i++) {
        keys.add(generateEncryptionKey());
      }

      // All 100 keys should be unique
      expect(keys.size).toBe(100);
    });

    it('should generate keys that work with encrypt/decrypt', () => {
      const key = generateEncryptionKey();
      process.env.CGM_ENCRYPTION_KEY = key;

      const plaintext = 'test-with-generated-key';
      const encrypted = encrypt(plaintext);
      const decrypted = decrypt(encrypted);

      expect(decrypted).toBe(plaintext);
    });
  });

  // ============================================================================
  // hashValue()
  // ============================================================================

  describe('hashValue()', () => {
    it('should produce consistent hash for same input', () => {
      const value = 'test-value-to-hash';

      const hash1 = hashValue(value);
      const hash2 = hashValue(value);

      expect(hash1).toBe(hash2);
    });

    it('should produce different hash for different input', () => {
      const hash1 = hashValue('value1');
      const hash2 = hashValue('value2');

      expect(hash1).not.toBe(hash2);
    });

    it('should produce a 64-character hex string (SHA-256)', () => {
      const hash = hashValue('test');

      expect(hash).toHaveLength(64);
      expect(/^[0-9a-f]{64}$/.test(hash)).toBe(true);
    });

    it('should handle empty string', () => {
      const hash = hashValue('');

      expect(hash).toHaveLength(64);
      // SHA-256 of empty string is a known value
      expect(hash).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    });
  });

  // ============================================================================
  // isValidEncryptionKey()
  // ============================================================================

  describe('isValidEncryptionKey()', () => {
    it('should return true for valid 64-character hex key', () => {
      const key = generateEncryptionKey();
      expect(isValidEncryptionKey(key)).toBe(true);
    });

    it('should return true for uppercase hex characters', () => {
      const key = generateEncryptionKey().toUpperCase();
      expect(isValidEncryptionKey(key)).toBe(true);
    });

    it('should return false for empty string', () => {
      expect(isValidEncryptionKey('')).toBe(false);
    });

    it('should return false for null/undefined', () => {
      expect(isValidEncryptionKey(null as unknown as string)).toBe(false);
      expect(isValidEncryptionKey(undefined as unknown as string)).toBe(false);
    });

    it('should return false for too short key', () => {
      expect(isValidEncryptionKey('abcd1234')).toBe(false);
    });

    it('should return false for too long key', () => {
      const key = generateEncryptionKey() + 'extra';
      expect(isValidEncryptionKey(key)).toBe(false);
    });

    it('should return false for non-hex characters', () => {
      // 64 characters but with invalid hex chars
      const invalid = 'ghijklmnopqrstuvwxyz1234567890abcdef1234567890abcdef12345678901234';
      expect(isValidEncryptionKey(invalid)).toBe(false);
    });
  });
});
