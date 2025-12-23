/**
 * Admin User Management Tests
 *
 * Tests for user listing, viewing, export, and deletion
 */

import request from 'supertest';
import { AdminRole } from '@prisma/client';
import app from '../../index';
import {
  prisma,
  createTestAdminUser,
  createTestAdminToken,
  createTestUser,
} from '../setup';

describe('Admin User Management', () => {
  let adminToken: string;

  beforeEach(async () => {
    const admin = await createTestAdminUser({
      role: AdminRole.SUPER_ADMIN,
    });
    adminToken = createTestAdminToken(admin.id);
  });

  describe('GET /api/admin/users', () => {
    it('should list users with pagination', async () => {
      // Create test users
      await createTestUser({ email: 'user1@test.com', name: 'User One' });
      await createTestUser({ email: 'user2@test.com', name: 'User Two' });
      await createTestUser({ email: 'user3@test.com', name: 'User Three' });

      const response = await request(app)
        .get('/api/admin/users')
        .set('Authorization', `Bearer ${adminToken}`)
        .query({ page: 1, limit: 2 });

      expect(response.status).toBe(200);
      expect(response.body.users).toBeDefined();
      expect(response.body.users.length).toBeLessThanOrEqual(2);
      expect(response.body.pagination).toBeDefined();
      expect(response.body.pagination.total).toBe(3);
    });

    it('should search users by email', async () => {
      await createTestUser({ email: 'john@example.com', name: 'John Doe' });
      await createTestUser({ email: 'jane@example.com', name: 'Jane Doe' });

      const response = await request(app)
        .get('/api/admin/users')
        .set('Authorization', `Bearer ${adminToken}`)
        .query({ search: 'john' });

      expect(response.status).toBe(200);
      expect(response.body.users.length).toBe(1);
      expect(response.body.users[0].email).toBe('john@example.com');
    });

    it('should search users by name', async () => {
      await createTestUser({ email: 'user1@test.com', name: 'Alice Smith' });
      await createTestUser({ email: 'user2@test.com', name: 'Bob Smith' });

      const response = await request(app)
        .get('/api/admin/users')
        .set('Authorization', `Bearer ${adminToken}`)
        .query({ search: 'alice' });

      expect(response.status).toBe(200);
      expect(response.body.users.length).toBe(1);
      expect(response.body.users[0].name).toBe('Alice Smith');
    });

    it('should create audit log for user list access', async () => {
      const admin = await prisma.adminUser.findFirst();

      await request(app)
        .get('/api/admin/users')
        .set('Authorization', `Bearer ${adminToken}`);

      const auditLog = await prisma.adminAuditLog.findFirst({
        where: {
          adminUserId: admin!.id,
          action: 'USER_LIST',
        },
      });

      expect(auditLog).toBeDefined();
    });
  });

  describe('GET /api/admin/users/:id', () => {
    it('should return detailed user information', async () => {
      const user = await createTestUser({
        email: 'detailed@test.com',
        name: 'Detailed User',
        goalCalories: 2500,
      });

      const response = await request(app)
        .get(`/api/admin/users/${user.id}`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(response.status).toBe(200);
      expect(response.body.id).toBe(user.id);
      expect(response.body.email).toBe('detailed@test.com');
      expect(response.body.name).toBe('Detailed User');
      expect(response.body.goalCalories).toBe(2500);
      // Password should never be returned
      expect(response.body.password).toBeUndefined();
    });

    it('should return 404 for non-existent user', async () => {
      // Use a valid CUID format that doesn't exist
      const response = await request(app)
        .get('/api/admin/users/clxxxxxxxxxxxxxxxxxxxxxxxxx')
        .set('Authorization', `Bearer ${adminToken}`);

      expect(response.status).toBe(404);
    });
  });

  describe('POST /api/admin/users/:id/export', () => {
    it('should export all user data for GDPR compliance', async () => {
      const user = await createTestUser({
        email: 'export@test.com',
        name: 'Export User',
      });

      const response = await request(app)
        .post(`/api/admin/users/${user.id}/export`)
        .set('Authorization', `Bearer ${adminToken}`);

      expect(response.status).toBe(200);
      expect(response.body.user).toBeDefined();
      expect(response.body.user.email).toBe('export@test.com');
      // Should include related data
      expect(response.body.meals).toBeDefined();
      expect(response.body.healthMetrics).toBeDefined();
      expect(response.body.activities).toBeDefined();
    });

    it('should create audit log for user export', async () => {
      const user = await createTestUser();
      const admin = await prisma.adminUser.findFirst();

      await request(app)
        .post(`/api/admin/users/${user.id}/export`)
        .set('Authorization', `Bearer ${adminToken}`);

      const auditLog = await prisma.adminAuditLog.findFirst({
        where: {
          adminUserId: admin!.id,
          action: 'USER_EXPORT',
          targetId: user.id,
        },
      });

      expect(auditLog).toBeDefined();
    });
  });

  describe('DELETE /api/admin/users/:id', () => {
    it('should delete user account (SUPER_ADMIN only)', async () => {
      const user = await createTestUser({
        email: 'delete@test.com',
      });

      const response = await request(app)
        .delete(`/api/admin/users/${user.id}`)
        .set('Authorization', `Bearer ${adminToken}`)
        .send({ reason: 'User requested account deletion for testing purposes' });

      expect(response.status).toBe(200);
      expect(response.body.message).toContain('deleted');

      // Verify user is deleted
      const deletedUser = await prisma.user.findUnique({
        where: { id: user.id },
      });
      expect(deletedUser).toBeNull();
    });

    it('should reject deletion by non-SUPER_ADMIN', async () => {
      const supportAdmin = await createTestAdminUser({
        email: 'support@test.com',
        role: AdminRole.SUPPORT,
      });
      const supportToken = createTestAdminToken(supportAdmin.id, { role: AdminRole.SUPPORT });

      const user = await createTestUser();

      const response = await request(app)
        .delete(`/api/admin/users/${user.id}`)
        .set('Authorization', `Bearer ${supportToken}`)
        .send({ reason: 'Testing role-based access control restrictions' });

      expect(response.status).toBe(403);
    });

    it('should create audit log for user deletion', async () => {
      const user = await createTestUser();
      const admin = await prisma.adminUser.findFirst();

      await request(app)
        .delete(`/api/admin/users/${user.id}`)
        .set('Authorization', `Bearer ${adminToken}`)
        .send({ reason: 'User account deletion for audit log testing' });

      const auditLog = await prisma.adminAuditLog.findFirst({
        where: {
          adminUserId: admin!.id,
          action: 'USER_DELETE',
          targetId: user.id,
        },
      });

      expect(auditLog).toBeDefined();
    });

    it('should return 404 for non-existent user', async () => {
      // Use a valid CUID format that doesn't exist
      const response = await request(app)
        .delete('/api/admin/users/clxxxxxxxxxxxxxxxxxxxxxxxxx')
        .set('Authorization', `Bearer ${adminToken}`)
        .send({ reason: 'Attempting to delete non-existent user for testing' });

      expect(response.status).toBe(404);
    });
  });
});
