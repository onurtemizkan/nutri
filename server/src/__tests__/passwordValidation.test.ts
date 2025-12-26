/**
 * Password Validation Tests
 *
 * Tests for password strength requirements (OWASP compliant):
 * - Minimum 12 characters
 * - Maximum 128 characters
 * - At least one uppercase letter
 * - At least one lowercase letter
 * - At least one number
 */

import request from 'supertest';
import app from '../index';
import { MIN_PASSWORD_LENGTH, MAX_PASSWORD_LENGTH, PASSWORD_ERRORS } from '../config/constants';
import { passwordSchema } from '../validation/schemas';

describe('Password Validation', () => {
  /**
   * Unit tests for password schema (no database required)
   * These tests validate the Zod password schema directly
   */
  describe('passwordSchema - Unit Tests', () => {
    describe('Length Requirements', () => {
      it('should reject password shorter than 12 characters', () => {
        const result = passwordSchema.safeParse('Short1Abc');
        expect(result.success).toBe(false);
        if (!result.success) {
          expect(result.error.errors[0].message).toContain(`${MIN_PASSWORD_LENGTH} characters`);
        }
      });

      it('should reject password of exactly 11 characters', () => {
        const result = passwordSchema.safeParse('Abcdefgh1Jk'); // 11 chars
        expect(result.success).toBe(false);
        if (!result.success) {
          expect(result.error.errors[0].message).toContain(`${MIN_PASSWORD_LENGTH} characters`);
        }
      });

      it('should accept password of exactly 12 characters', () => {
        const result = passwordSchema.safeParse('Abcdefgh1Jkl'); // 12 chars
        expect(result.success).toBe(true);
      });

      it('should reject password longer than 128 characters', () => {
        const longPassword = 'A'.repeat(60) + 'a'.repeat(60) + '1'.repeat(10); // 130 chars
        const result = passwordSchema.safeParse(longPassword);
        expect(result.success).toBe(false);
        if (!result.success) {
          expect(result.error.errors[0].message).toContain(`${MAX_PASSWORD_LENGTH} characters`);
        }
      });

      it('should accept password of exactly 128 characters', () => {
        const exactPassword = 'A'.repeat(60) + 'a'.repeat(60) + '1'.repeat(8); // 128 chars
        const result = passwordSchema.safeParse(exactPassword);
        expect(result.success).toBe(true);
      });
    });

    describe('Uppercase Requirement', () => {
      it('should reject password without uppercase letter', () => {
        const result = passwordSchema.safeParse('alllowercase123');
        expect(result.success).toBe(false);
        if (!result.success) {
          const messages = result.error.errors.map((e) => e.message).join(' ');
          expect(messages).toContain('uppercase');
        }
      });

      it('should accept password with at least one uppercase letter', () => {
        const result = passwordSchema.safeParse('onlyOneUpper123');
        expect(result.success).toBe(true);
      });
    });

    describe('Lowercase Requirement', () => {
      it('should reject password without lowercase letter', () => {
        const result = passwordSchema.safeParse('ALLUPPERCASE123');
        expect(result.success).toBe(false);
        if (!result.success) {
          const messages = result.error.errors.map((e) => e.message).join(' ');
          expect(messages).toContain('lowercase');
        }
      });

      it('should accept password with at least one lowercase letter', () => {
        const result = passwordSchema.safeParse('ONLYONElower123');
        expect(result.success).toBe(true);
      });
    });

    describe('Number Requirement', () => {
      it('should reject password without number', () => {
        const result = passwordSchema.safeParse('NoNumbersHereX');
        expect(result.success).toBe(false);
        if (!result.success) {
          const messages = result.error.errors.map((e) => e.message).join(' ');
          expect(messages).toContain('number');
        }
      });

      it('should accept password with at least one number', () => {
        const result = passwordSchema.safeParse('HasOneNumber1x');
        expect(result.success).toBe(true);
      });
    });

    describe('Combined Requirements', () => {
      it('should reject password missing all requirements', () => {
        const result = passwordSchema.safeParse('short');
        expect(result.success).toBe(false);
        // Should have multiple errors
        expect(result.error?.errors.length).toBeGreaterThan(0);
      });

      it('should accept a strong password meeting all requirements', () => {
        const result = passwordSchema.safeParse('SecurePassword123!');
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data).toBe('SecurePassword123!');
        }
      });

      it('should accept common password patterns', () => {
        const validPasswords = [
          'MySecretPass1',
          'Password12345',
          'Test1234567890',
          'Abcdefghijk1',
          '1234567890aB',
        ];

        for (const password of validPasswords) {
          const result = passwordSchema.safeParse(password);
          expect(result.success).toBe(true);
        }
      });

      it('should reject common weak password patterns', () => {
        const invalidPasswords = [
          'password', // no uppercase, no number
          '12345678901', // no uppercase, no lowercase
          'ALLUPPERCASE', // no lowercase, no number
          'alllowercase', // no uppercase, no number
          'Short1A', // too short
        ];

        for (const password of invalidPasswords) {
          const result = passwordSchema.safeParse(password);
          expect(result.success).toBe(false);
        }
      });
    });
  });

  /**
   * Integration tests for password validation via API
   * These tests verify the API returns correct error messages for invalid passwords
   * Note: "Accept" tests that require database are covered by the unit tests above
   */
  describe('POST /api/auth/register - API Integration Tests', () => {
    const validUser = {
      email: 'testpw@example.com',
      name: 'Test User',
    };

    it('should return 400 with error for password shorter than 12 characters', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          ...validUser,
          password: 'Short1Abc', // 9 characters
        })
        .expect(400);

      expect(response.body.error).toContain(`${MIN_PASSWORD_LENGTH} characters`);
    });

    it('should return 400 with error for password without uppercase', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          ...validUser,
          email: 'testpw2@example.com',
          password: 'alllowercase123', // no uppercase
        })
        .expect(400);

      expect(response.body.error).toContain('uppercase');
    });

    it('should return 400 with error for password without lowercase', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          ...validUser,
          email: 'testpw3@example.com',
          password: 'ALLUPPERCASE123', // no lowercase
        })
        .expect(400);

      expect(response.body.error).toContain('lowercase');
    });

    it('should return 400 with error for password without number', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          ...validUser,
          email: 'testpw4@example.com',
          password: 'NoNumbersHereX', // no numbers
        })
        .expect(400);

      expect(response.body.error).toContain('number');
    });

    it('should return 400 with error for password exceeding 128 characters', async () => {
      const longPassword = 'A'.repeat(60) + 'a'.repeat(60) + '1'.repeat(10); // 130 chars
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          ...validUser,
          email: 'testpw5@example.com',
          password: longPassword,
        })
        .expect(400);

      expect(response.body.error).toContain(`${MAX_PASSWORD_LENGTH} characters`);
    });
  });

  describe('POST /api/auth/reset-password - API Integration Tests', () => {
    it('should return 400 for weak new password (validates before token check)', async () => {
      // This test doesn't need a valid token - validation happens first
      const response = await request(app)
        .post('/api/auth/reset-password')
        .send({
          token: 'some-token',
          newPassword: 'weak', // too short, no uppercase
        })
        .expect(400);

      expect(response.body.error).toContain('12 characters');
    });

    it('should return 400 for password without uppercase in reset', async () => {
      const response = await request(app)
        .post('/api/auth/reset-password')
        .send({
          token: 'some-token',
          newPassword: 'alllowercase123',
        })
        .expect(400);

      expect(response.body.error).toContain('uppercase');
    });
  });

  describe('Constants Verification', () => {
    it('should have MIN_PASSWORD_LENGTH set to 12', () => {
      expect(MIN_PASSWORD_LENGTH).toBe(12);
    });

    it('should have MAX_PASSWORD_LENGTH set to 128', () => {
      expect(MAX_PASSWORD_LENGTH).toBe(128);
    });

    it('should have all required password error messages', () => {
      expect(PASSWORD_ERRORS.TOO_SHORT).toContain('12');
      expect(PASSWORD_ERRORS.TOO_LONG).toContain('128');
      expect(PASSWORD_ERRORS.MISSING_UPPERCASE).toContain('uppercase');
      expect(PASSWORD_ERRORS.MISSING_LOWERCASE).toContain('lowercase');
      expect(PASSWORD_ERRORS.MISSING_NUMBER).toContain('number');
    });
  });
});
