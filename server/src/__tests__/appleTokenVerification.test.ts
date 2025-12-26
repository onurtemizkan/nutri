/**
 * Apple Token Verification Tests
 *
 * Tests for the Apple Sign-In token verification service
 */

import {
  isAppleVerificationEnabled,
  decodeAppleTokenUnsafe,
  processAppleToken,
  verifyAppleIdentityToken,
} from '../services/appleTokenVerification';

// Helper to create a mock JWT token (base64 encoded parts)
function createMockToken(payload: Record<string, unknown>): string {
  const header = { alg: 'RS256', kid: 'test-key-id' };
  const headerB64 = Buffer.from(JSON.stringify(header)).toString('base64');
  const payloadB64 = Buffer.from(JSON.stringify(payload)).toString('base64');
  const signature = 'mock-signature';
  return `${headerB64}.${payloadB64}.${signature}`;
}

describe('Apple Token Verification', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    // Reset environment before each test
    jest.resetModules();
    process.env = { ...originalEnv };
    delete process.env.APPLE_APP_ID;
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  describe('isAppleVerificationEnabled', () => {
    it('should return false when APPLE_APP_ID is not set', () => {
      delete process.env.APPLE_APP_ID;
      expect(isAppleVerificationEnabled()).toBe(false);
    });

    it('should return true when APPLE_APP_ID is set', () => {
      process.env.APPLE_APP_ID = 'com.example.app';
      expect(isAppleVerificationEnabled()).toBe(true);
    });

    it('should return false when APPLE_APP_ID is empty string', () => {
      process.env.APPLE_APP_ID = '';
      expect(isAppleVerificationEnabled()).toBe(false);
    });
  });

  describe('decodeAppleTokenUnsafe', () => {
    it('should decode a valid token with sub and email', () => {
      const payload = {
        sub: 'apple-user-id-123',
        email: 'user@example.com',
        iss: 'https://appleid.apple.com',
        aud: 'com.example.app',
      };
      const token = createMockToken(payload);

      const result = decodeAppleTokenUnsafe(token);

      expect(result.sub).toBe('apple-user-id-123');
      expect(result.email).toBe('user@example.com');
    });

    it('should decode a valid token without email', () => {
      const payload = {
        sub: 'apple-user-id-456',
        iss: 'https://appleid.apple.com',
      };
      const token = createMockToken(payload);

      const result = decodeAppleTokenUnsafe(token);

      expect(result.sub).toBe('apple-user-id-456');
      expect(result.email).toBeUndefined();
    });

    it('should throw error for token with invalid format (not 3 parts)', () => {
      expect(() => decodeAppleTokenUnsafe('invalid-token')).toThrow(
        'Invalid identity token format'
      );
      expect(() => decodeAppleTokenUnsafe('part1.part2')).toThrow('Invalid identity token format');
      expect(() => decodeAppleTokenUnsafe('')).toThrow('Invalid identity token format');
    });

    it('should throw error for token with invalid base64 payload', () => {
      const token = 'header.!!!invalid-base64!!!.signature';
      expect(() => decodeAppleTokenUnsafe(token)).toThrow('Failed to decode Apple identity token');
    });

    it('should throw error for token missing sub claim', () => {
      const payload = {
        email: 'user@example.com',
        iss: 'https://appleid.apple.com',
        // Missing 'sub' claim
      };
      const token = createMockToken(payload);

      expect(() => decodeAppleTokenUnsafe(token)).toThrow(
        'Invalid identity token: missing user ID'
      );
    });

    it('should handle token with private relay email flag', () => {
      const payload = {
        sub: 'apple-user-id-789',
        email: 'private@privaterelay.appleid.com',
        is_private_email: true,
      };
      const token = createMockToken(payload);

      const result = decodeAppleTokenUnsafe(token);

      expect(result.sub).toBe('apple-user-id-789');
      expect(result.email).toBe('private@privaterelay.appleid.com');
    });
  });

  describe('processAppleToken', () => {
    describe('when APPLE_APP_ID is not set (development mode)', () => {
      beforeEach(() => {
        delete process.env.APPLE_APP_ID;
      });

      it('should decode token without verification and return verified: false', async () => {
        const payload = {
          sub: 'apple-user-id-dev',
          email: 'dev@example.com',
        };
        const token = createMockToken(payload);

        const result = await processAppleToken(token);

        expect(result.sub).toBe('apple-user-id-dev');
        expect(result.email).toBe('dev@example.com');
        expect(result.verified).toBe(false);
      });

      it('should throw error for invalid token format in dev mode', async () => {
        await expect(processAppleToken('invalid-token')).rejects.toThrow(
          'Invalid identity token format'
        );
      });

      it('should throw error for token missing sub in dev mode', async () => {
        const payload = { email: 'test@example.com' };
        const token = createMockToken(payload);

        await expect(processAppleToken(token)).rejects.toThrow(
          'Invalid identity token: missing user ID'
        );
      });
    });
  });

  describe('verifyAppleIdentityToken', () => {
    it('should throw error when APPLE_APP_ID is not set', async () => {
      delete process.env.APPLE_APP_ID;
      const token = createMockToken({ sub: 'test' });

      await expect(verifyAppleIdentityToken(token)).rejects.toThrow(
        'APPLE_APP_ID environment variable is required for token verification'
      );
    });

    it('should attempt verification when APPLE_APP_ID is set', async () => {
      process.env.APPLE_APP_ID = 'com.example.app';
      const token = createMockToken({ sub: 'test' });

      // This will fail because we can't actually verify with Apple's servers in tests,
      // but it proves that the function proceeds past the env var check
      await expect(verifyAppleIdentityToken(token)).rejects.toThrow();
    });
  });

  describe('Token payload validation', () => {
    it('should handle tokens with all optional fields', () => {
      const payload = {
        sub: 'full-payload-user',
        email: 'full@example.com',
        email_verified: true,
        is_private_email: false,
        auth_time: Math.floor(Date.now() / 1000),
        iss: 'https://appleid.apple.com',
        aud: 'com.example.app',
        exp: Math.floor(Date.now() / 1000) + 3600,
        iat: Math.floor(Date.now() / 1000),
      };
      const token = createMockToken(payload);

      const result = decodeAppleTokenUnsafe(token);

      expect(result.sub).toBe('full-payload-user');
      expect(result.email).toBe('full@example.com');
    });

    it('should handle tokens with minimal payload (only sub)', () => {
      const payload = {
        sub: 'minimal-user',
      };
      const token = createMockToken(payload);

      const result = decodeAppleTokenUnsafe(token);

      expect(result.sub).toBe('minimal-user');
      expect(result.email).toBeUndefined();
    });
  });
});
