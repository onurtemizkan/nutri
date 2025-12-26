/**
 * Auth Endpoints Tests
 *
 * Tests all authentication-related endpoints:
 * - User registration
 * - User login
 * - Get profile (protected)
 * - Update profile (protected)
 * - Password reset flow
 */

import request from 'supertest';
import app from '../index';
import {
  createTestUser,
  createTestToken,
  assertValidToken,
  assertUserStructure,
} from './setup';

describe('Auth API Endpoints', () => {
  // ============================================================================
  // POST /api/auth/register
  // ============================================================================

  describe('POST /api/auth/register', () => {
    it('should register a new user successfully', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'newuser@example.com',
          password: 'securepassword123',
          name: 'New User',
        })
        .expect(201);

      expect(response.body).toHaveProperty('user');
      expect(response.body).toHaveProperty('token');
      assertUserStructure(response.body.user);
      assertValidToken(response.body.token);
      expect(response.body.user.email).toBe('newuser@example.com');
      expect(response.body.user.name).toBe('New User');
    });

    it('should reject registration with invalid email', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'invalid-email',
          password: 'securepassword123',
          name: 'Test User',
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('email');
    });

    it('should reject registration with short password', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'test@example.com',
          password: 'short123',
          name: 'Test User',
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('12 characters');
    });

    it('should reject registration with empty name', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'test@example.com',
          password: 'securepassword123',
          name: '',
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Name');
    });

    it('should reject registration with duplicate email', async () => {
      // Create first user
      await createTestUser({ email: 'duplicate@example.com' });

      // Try to register with same email
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          email: 'duplicate@example.com',
          password: 'securepassword123',
          name: 'Test User',
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('exists');
    });
  });

  // ============================================================================
  // POST /api/auth/login
  // ============================================================================

  describe('POST /api/auth/login', () => {
    it('should login successfully with correct credentials', async () => {
      // Create test user
      const bcrypt = require('bcryptjs');
      await createTestUser({
        email: 'login@example.com',
        password: await bcrypt.hash('securepassword123', 10),
      });

      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'login@example.com',
          password: 'securepassword123',
        })
        .expect(200);

      expect(response.body).toHaveProperty('user');
      expect(response.body).toHaveProperty('token');
      assertUserStructure(response.body.user);
      assertValidToken(response.body.token);
      expect(response.body.user.email).toBe('login@example.com');
    });

    it('should reject login with non-existent email', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'nonexistent@example.com',
          password: 'securepassword123',
        })
        .expect(401);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Invalid');
    });

    it('should reject login with incorrect password', async () => {
      // Create test user
      await createTestUser({ email: 'test@example.com' });

      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'test@example.com',
          password: 'wrongpassword123',
        })
        .expect(401);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Invalid');
    });

    it('should reject login with invalid email format', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'invalid-email',
          password: 'securepassword123',
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('email');
    });

    it('should reject login with empty password', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'test@example.com',
          password: '',
        })
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });
  });

  // ============================================================================
  // GET /api/auth/profile (Protected)
  // ============================================================================

  describe('GET /api/auth/profile', () => {
    it('should get user profile with valid token', async () => {
      // Create test user
      const user = await createTestUser({ email: 'profile@example.com' });
      const token = createTestToken(user.id);

      const response = await request(app)
        .get('/api/auth/profile')
        .set('Authorization', `Bearer ${token}`)
        .expect(200);

      assertUserStructure(response.body);
      expect(response.body.email).toBe('profile@example.com');
      expect(response.body.id).toBe(user.id);
    });

    it('should reject request without token', async () => {
      const response = await request(app)
        .get('/api/auth/profile')
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });

    it('should reject request with invalid token', async () => {
      const response = await request(app)
        .get('/api/auth/profile')
        .set('Authorization', 'Bearer invalid-token')
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });

    it('should reject request with expired token', async () => {
      const jwt = require('jsonwebtoken');
      const expiredToken = jwt.sign(
        { userId: 'user123' },
        process.env.JWT_SECRET,
        { expiresIn: '0s' } // Already expired
      );

      // Wait a tiny bit to ensure expiration
      await new Promise(resolve => setTimeout(resolve, 10));

      const response = await request(app)
        .get('/api/auth/profile')
        .set('Authorization', `Bearer ${expiredToken}`)
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });

    it('should reject request for non-existent user', async () => {
      const token = createTestToken('non-existent-user-id');

      const response = await request(app)
        .get('/api/auth/profile')
        .set('Authorization', `Bearer ${token}`)
        .expect(404);

      expect(response.body).toHaveProperty('error');
    });
  });

  // ============================================================================
  // PUT /api/auth/profile (Protected)
  // ============================================================================

  describe('PUT /api/auth/profile', () => {
    it('should update user profile with valid data', async () => {
      // Create test user
      const user = await createTestUser({ email: 'update@example.com' });
      const token = createTestToken(user.id);

      const response = await request(app)
        .put('/api/auth/profile')
        .set('Authorization', `Bearer ${token}`)
        .send({
          name: 'Updated Name',
          goalCalories: 2500,
          goalProtein: 180,
          currentWeight: 75.5,
          goalWeight: 70.0,
          height: 175,
          activityLevel: 'VERY_ACTIVE',
        })
        .expect(200);

      assertUserStructure(response.body);
      expect(response.body.name).toBe('Updated Name');
      expect(response.body.goalCalories).toBe(2500);
      expect(response.body.goalProtein).toBe(180);
      expect(response.body.currentWeight).toBe(75.5);
      expect(response.body.goalWeight).toBe(70.0);
      expect(response.body.height).toBe(175);
      expect(response.body.activityLevel).toBe('VERY_ACTIVE');
    });

    it('should update only provided fields', async () => {
      // Create test user
      const user = await createTestUser({
        email: 'partial@example.com',
        name: 'Original Name',
      });
      const token = createTestToken(user.id);

      const response = await request(app)
        .put('/api/auth/profile')
        .set('Authorization', `Bearer ${token}`)
        .send({
          goalCalories: 2200,
        })
        .expect(200);

      expect(response.body.name).toBe('Original Name'); // Unchanged
      expect(response.body.goalCalories).toBe(2200); // Updated
    });

    it('should reject update without authentication', async () => {
      const response = await request(app)
        .put('/api/auth/profile')
        .send({
          name: 'Updated Name',
        })
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });

    it('should reject update with invalid token', async () => {
      const response = await request(app)
        .put('/api/auth/profile')
        .set('Authorization', 'Bearer invalid-token')
        .send({
          name: 'Updated Name',
        })
        .expect(401);

      expect(response.body).toHaveProperty('error');
    });
  });

  // ============================================================================
  // Password Reset Flow
  // ============================================================================

  describe('Password Reset Flow', () => {
    describe('POST /api/auth/forgot-password', () => {
      it('should initiate password reset for existing user', async () => {
        // Create test user
        await createTestUser({ email: 'reset@example.com' });

        const response = await request(app)
          .post('/api/auth/forgot-password')
          .send({
            email: 'reset@example.com',
          })
          .expect(200);

        expect(response.body).toHaveProperty('message');
        expect(response.body.message).toContain('password reset');

        // In development, should return token
        if (process.env.NODE_ENV === 'test') {
          expect(response.body).toHaveProperty('resetToken');
        }
      });

      it('should not reveal if user does not exist', async () => {
        const response = await request(app)
          .post('/api/auth/forgot-password')
          .send({
            email: 'nonexistent@example.com',
          })
          .expect(200);

        expect(response.body).toHaveProperty('message');
        expect(response.body.message).toContain('password reset');
      });

      it('should reject invalid email format', async () => {
        const response = await request(app)
          .post('/api/auth/forgot-password')
          .send({
            email: 'invalid-email',
          })
          .expect(400);

        expect(response.body).toHaveProperty('error');
        expect(response.body.error).toContain('email');
      });
    });

    describe('POST /api/auth/verify-reset-token', () => {
      it('should verify valid reset token', async () => {
        // Create user and get reset token
        await createTestUser({ email: 'verify@example.com' });

        const forgotResponse = await request(app)
          .post('/api/auth/forgot-password')
          .send({ email: 'verify@example.com' });

        const resetToken = forgotResponse.body.resetToken;

        const response = await request(app)
          .post('/api/auth/verify-reset-token')
          .send({ token: resetToken })
          .expect(200);

        expect(response.body).toHaveProperty('valid', true);
        expect(response.body).toHaveProperty('email', 'verify@example.com');
      });

      it('should reject invalid reset token', async () => {
        const response = await request(app)
          .post('/api/auth/verify-reset-token')
          .send({ token: 'invalid-token' })
          .expect(400);

        expect(response.body).toHaveProperty('error');
        expect(response.body.error).toContain('Invalid');
      });

      it('should reject empty token', async () => {
        const response = await request(app)
          .post('/api/auth/verify-reset-token')
          .send({ token: '' })
          .expect(400);

        expect(response.body).toHaveProperty('error');
      });
    });

    describe('POST /api/auth/reset-password', () => {
      it('should reset password with valid token', async () => {
        // Create user and get reset token
        await createTestUser({ email: 'resetpw@example.com' });

        const forgotResponse = await request(app)
          .post('/api/auth/forgot-password')
          .send({ email: 'resetpw@example.com' });

        const resetToken = forgotResponse.body.resetToken;

        // Reset password
        const response = await request(app)
          .post('/api/auth/reset-password')
          .send({
            token: resetToken,
            newPassword: 'newsecurepassword123',
          })
          .expect(200);

        expect(response.body).toHaveProperty('message');
        expect(response.body.message).toContain('reset successfully');

        // Verify can login with new password
        const loginResponse = await request(app)
          .post('/api/auth/login')
          .send({
            email: 'resetpw@example.com',
            password: 'newsecurepassword123',
          })
          .expect(200);

        expect(loginResponse.body).toHaveProperty('token');
      });

      it('should reject reset with invalid token', async () => {
        const response = await request(app)
          .post('/api/auth/reset-password')
          .send({
            token: 'invalid-token',
            newPassword: 'newsecurepassword123',
          })
          .expect(400);

        expect(response.body).toHaveProperty('error');
        expect(response.body.error).toContain('Invalid');
      });

      it('should reject reset with short password', async () => {
        // Create user and get reset token
        await createTestUser({ email: 'shortpw@example.com' });

        const forgotResponse = await request(app)
          .post('/api/auth/forgot-password')
          .send({ email: 'shortpw@example.com' });

        const resetToken = forgotResponse.body.resetToken;

        const response = await request(app)
          .post('/api/auth/reset-password')
          .send({
            token: resetToken,
            newPassword: 'short123',
          })
          .expect(400);

        expect(response.body).toHaveProperty('error');
        expect(response.body.error).toContain('12 characters');
      });

      it('should invalidate token after successful reset', async () => {
        // Create user and get reset token
        await createTestUser({ email: 'oneuse@example.com' });

        const forgotResponse = await request(app)
          .post('/api/auth/forgot-password')
          .send({ email: 'oneuse@example.com' });

        const resetToken = forgotResponse.body.resetToken;

        // Use token once
        await request(app)
          .post('/api/auth/reset-password')
          .send({
            token: resetToken,
            newPassword: 'newsecurepassword123',
          })
          .expect(200);

        // Try to use token again
        const response = await request(app)
          .post('/api/auth/reset-password')
          .send({
            token: resetToken,
            newPassword: 'anothersecurepass123',
          })
          .expect(400);

        expect(response.body).toHaveProperty('error');
        expect(response.body.error).toContain('Invalid');
      });
    });

    describe('Complete Password Reset Flow', () => {
      it('should complete full password reset flow successfully', async () => {
        const testEmail = 'fullflow@example.com';
        const oldPassword = 'oldsecurepassword123';
        const newPassword = 'newsecurepassword123';

        // 1. Create user
        const bcrypt = require('bcryptjs');
        await createTestUser({
          email: testEmail,
          password: await bcrypt.hash(oldPassword, 10),
        });

        // 2. Verify can login with old password
        await request(app)
          .post('/api/auth/login')
          .send({ email: testEmail, password: oldPassword })
          .expect(200);

        // 3. Request password reset
        const forgotResponse = await request(app)
          .post('/api/auth/forgot-password')
          .send({ email: testEmail })
          .expect(200);

        const resetToken = forgotResponse.body.resetToken;

        // 4. Verify reset token
        await request(app)
          .post('/api/auth/verify-reset-token')
          .send({ token: resetToken })
          .expect(200);

        // 5. Reset password
        await request(app)
          .post('/api/auth/reset-password')
          .send({ token: resetToken, newPassword })
          .expect(200);

        // 6. Verify old password no longer works
        await request(app)
          .post('/api/auth/login')
          .send({ email: testEmail, password: oldPassword })
          .expect(401);

        // 7. Verify new password works
        const loginResponse = await request(app)
          .post('/api/auth/login')
          .send({ email: testEmail, password: newPassword })
          .expect(200);

        expect(loginResponse.body).toHaveProperty('token');
        expect(loginResponse.body.user.email).toBe(testEmail);
      });
    });
  });
});
