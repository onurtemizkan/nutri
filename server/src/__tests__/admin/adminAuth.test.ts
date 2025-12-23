/**
 * Admin Authentication Tests
 *
 * Tests for admin login, MFA, and session management
 */

import request from 'supertest';
import { AdminRole } from '@prisma/client';
import app from '../../index';
import {
  prisma,
  createTestAdminUser,
  createTestAdminToken,
  assertAdminUserStructure,
} from '../setup';

describe('Admin Authentication', () => {
  describe('POST /api/admin/auth/login', () => {
    it('should return MFA setup required for new admin without MFA', async () => {
      // Create admin without MFA
      await createTestAdminUser({
        email: 'newadmin@test.com',
        mfaEnabled: false,
      });

      const response = await request(app)
        .post('/api/admin/auth/login')
        .send({
          email: 'newadmin@test.com',
          password: 'AdminPass123!',
        });

      expect(response.status).toBe(200);
      expect(response.body.requiresMFA).toBe(true);
      expect(response.body.mfaSetupRequired).toBe(true);
      expect(response.body.pendingToken).toBeDefined();
      expect(response.body.qrCode).toBeDefined();
    });

    it('should reject invalid credentials', async () => {
      await createTestAdminUser({
        email: 'admin@test.com',
      });

      const response = await request(app)
        .post('/api/admin/auth/login')
        .send({
          email: 'admin@test.com',
          password: 'wrongpassword',
        });

      expect(response.status).toBe(401);
      expect(response.body.error).toBe('Invalid credentials');
    });

    it('should reject non-existent admin', async () => {
      const response = await request(app)
        .post('/api/admin/auth/login')
        .send({
          email: 'nonexistent@test.com',
          password: 'AdminPass123!',
        });

      expect(response.status).toBe(401);
      expect(response.body.error).toBe('Invalid credentials');
    });

    it('should validate email format', async () => {
      const response = await request(app)
        .post('/api/admin/auth/login')
        .send({
          email: 'invalid-email',
          password: 'AdminPass123!',
        });

      expect(response.status).toBe(400);
    });
  });

  describe('GET /api/admin/auth/me', () => {
    it('should return current admin user info with valid token', async () => {
      const admin = await createTestAdminUser({
        email: 'admin@test.com',
        name: 'Test Admin',
        role: AdminRole.SUPER_ADMIN,
      });

      const token = createTestAdminToken(admin.id);

      const response = await request(app)
        .get('/api/admin/auth/me')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(200);
      assertAdminUserStructure(response.body);
      expect(response.body.email).toBe('admin@test.com');
      expect(response.body.name).toBe('Test Admin');
      expect(response.body.role).toBe('SUPER_ADMIN');
    });

    it('should reject request without token', async () => {
      const response = await request(app).get('/api/admin/auth/me');

      expect(response.status).toBe(401);
      expect(response.body.error).toBeDefined();
    });

    it('should reject request with invalid token', async () => {
      const response = await request(app)
        .get('/api/admin/auth/me')
        .set('Authorization', 'Bearer invalid-token');

      expect(response.status).toBe(401);
    });
  });

  describe('POST /api/admin/auth/logout', () => {
    it('should log out admin and create audit log', async () => {
      const admin = await createTestAdminUser();
      const token = createTestAdminToken(admin.id);

      const response = await request(app)
        .post('/api/admin/auth/logout')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(200);
      expect(response.body.message).toBe('Logged out successfully');

      // Verify audit log was created
      const auditLog = await prisma.adminAuditLog.findFirst({
        where: {
          adminUserId: admin.id,
          action: 'ADMIN_LOGOUT',
        },
      });

      expect(auditLog).toBeDefined();
    });
  });

  describe('POST /api/admin/auth/setup', () => {
    it('should create initial admin when no admins exist', async () => {
      // Ensure no admins exist (must delete audit logs first due to FK constraint)
      await prisma.adminAuditLog.deleteMany();
      await prisma.adminUser.deleteMany();

      const response = await request(app)
        .post('/api/admin/auth/setup')
        .send({
          email: 'first@admin.com',
          password: 'FirstAdmin123!',
          name: 'First Admin',
        });

      expect(response.status).toBe(201);
      expect(response.body.message).toBeDefined();

      // Verify admin was created
      const admin = await prisma.adminUser.findUnique({
        where: { email: 'first@admin.com' },
      });

      expect(admin).toBeDefined();
      expect(admin?.role).toBe('SUPER_ADMIN');
    });

    it('should reject setup when admins already exist', async () => {
      // Ensure clean state and create an admin
      await prisma.adminAuditLog.deleteMany();
      await prisma.adminUser.deleteMany();
      await createTestAdminUser();

      const response = await request(app)
        .post('/api/admin/auth/setup')
        .send({
          email: 'another@admin.com',
          password: 'AnotherAdmin123!',
          name: 'Another Admin',
        });

      expect(response.status).toBe(403);
      expect(response.body.error).toContain('Admin users already exist');
    });
  });
});

describe('Admin Role-Based Access Control', () => {
  describe('SUPER_ADMIN role', () => {
    it('should have access to all admin endpoints', async () => {
      const admin = await createTestAdminUser({
        role: AdminRole.SUPER_ADMIN,
      });
      const token = createTestAdminToken(admin.id);

      // Test access to user listing (requireAnyAdmin)
      const usersResponse = await request(app)
        .get('/api/admin/users')
        .set('Authorization', `Bearer ${token}`);

      expect(usersResponse.status).not.toBe(403);
    });
  });

  describe('SUPPORT role', () => {
    it('should have access to user management endpoints', async () => {
      const admin = await createTestAdminUser({
        role: AdminRole.SUPPORT,
      });
      const token = createTestAdminToken(admin.id, { role: AdminRole.SUPPORT });

      const response = await request(app)
        .get('/api/admin/users')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).not.toBe(403);
    });

    it('should NOT have access to subscription grant endpoint', async () => {
      const admin = await createTestAdminUser({
        email: 'support@test.com',
        role: AdminRole.SUPPORT,
      });
      const token = createTestAdminToken(admin.id, { email: 'support@test.com', role: AdminRole.SUPPORT });

      const response = await request(app)
        .post('/api/admin/subscriptions/some-user-id/grant')
        .set('Authorization', `Bearer ${token}`)
        .send({ duration: 30 });

      expect(response.status).toBe(403);
    });
  });

  describe('ANALYST role', () => {
    it('should have access to analytics endpoints', async () => {
      const admin = await createTestAdminUser({
        role: AdminRole.ANALYST,
      });
      const token = createTestAdminToken(admin.id, { role: AdminRole.ANALYST });

      const response = await request(app)
        .get('/api/admin/analytics/overview')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).not.toBe(403);
    });
  });

  describe('VIEWER role', () => {
    it('should have read-only access', async () => {
      const admin = await createTestAdminUser({
        role: AdminRole.VIEWER,
      });
      const token = createTestAdminToken(admin.id, { role: AdminRole.VIEWER });

      // Should have access to view users
      const response = await request(app)
        .get('/api/admin/users')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).not.toBe(403);
    });
  });
});
