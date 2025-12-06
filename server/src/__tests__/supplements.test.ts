/**
 * Supplement API Tests
 *
 * Tests all supplement-related endpoints:
 * - Master supplement list (GET /supplements)
 * - User supplement schedules (CRUD)
 * - Supplement logs (intake tracking)
 * - Summaries and analytics
 */

import request from 'supertest';
import app from '../index';
import {
  createTestUser,
  createTestToken,
  createTestSupplement,
  createTestUserSupplement,
  createTestSupplementLog,
  assertSupplementStructure,
  assertUserSupplementStructure,
  assertSupplementLogStructure,
} from './setup';
import { SupplementCategory, ScheduleType } from '@prisma/client';

describe('Supplement API', () => {
  let authToken: string;
  let userId: string;

  beforeEach(async () => {
    const user = await createTestUser();
    userId = user.id;
    authToken = createTestToken(userId);
  });

  // ==========================================================================
  // MASTER SUPPLEMENT LIST
  // ==========================================================================

  describe('GET /api/supplements', () => {
    it('should return all supplements', async () => {
      await createTestSupplement({ name: 'Lysine', category: SupplementCategory.AMINO_ACID });
      await createTestSupplement({ name: 'Vitamin D', category: SupplementCategory.VITAMIN });

      const response = await request(app)
        .get('/api/supplements')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body.length).toBe(2);
      response.body.forEach(assertSupplementStructure);
    });

    it('should filter supplements by category', async () => {
      await createTestSupplement({ name: 'Lysine', category: SupplementCategory.AMINO_ACID });
      await createTestSupplement({ name: 'Vitamin D', category: SupplementCategory.VITAMIN });

      const response = await request(app)
        .get('/api/supplements?category=AMINO_ACID')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body.length).toBe(1);
      expect(response.body[0].category).toBe('AMINO_ACID');
    });

    it('should search supplements by name', async () => {
      await createTestSupplement({ name: 'Lysine', category: SupplementCategory.AMINO_ACID });
      await createTestSupplement({ name: 'Arginine', category: SupplementCategory.AMINO_ACID });

      const response = await request(app)
        .get('/api/supplements?search=lys')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body.length).toBe(1);
      expect(response.body[0].name).toBe('Lysine');
    });

    it('should return 401 without authentication', async () => {
      const response = await request(app).get('/api/supplements');

      expect(response.status).toBe(401);
    });
  });

  describe('GET /api/supplements/:id', () => {
    it('should return a single supplement', async () => {
      const supplement = await createTestSupplement({ name: 'Creatine' });

      const response = await request(app)
        .get(`/api/supplements/${supplement.id}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      assertSupplementStructure(response.body);
      expect(response.body.name).toBe('Creatine');
    });

    it('should return 404 for non-existent supplement', async () => {
      const response = await request(app)
        .get('/api/supplements/non-existent-id')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(404);
    });
  });

  // ==========================================================================
  // USER SUPPLEMENT SCHEDULES
  // ==========================================================================

  describe('POST /api/user-supplements', () => {
    it('should create a daily supplement schedule', async () => {
      const supplement = await createTestSupplement({ name: 'Vitamin D' });

      const response = await request(app)
        .post('/api/user-supplements')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          supplementId: supplement.id,
          dosage: '5000',
          unit: 'IU',
          scheduleType: 'DAILY',
          startDate: new Date().toISOString(),
        });

      expect(response.status).toBe(201);
      assertUserSupplementStructure(response.body);
      expect(response.body.scheduleType).toBe('DAILY');
    });

    it('should create a daily multiple schedule with times', async () => {
      const supplement = await createTestSupplement({ name: 'BCAA' });

      const response = await request(app)
        .post('/api/user-supplements')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          supplementId: supplement.id,
          dosage: '5',
          unit: 'g',
          scheduleType: 'DAILY_MULTIPLE',
          scheduleTimes: ['08:00', '14:00', '20:00'],
          startDate: new Date().toISOString(),
        });

      expect(response.status).toBe(201);
      expect(response.body.scheduleType).toBe('DAILY_MULTIPLE');
      expect(response.body.scheduleTimes).toEqual(['08:00', '14:00', '20:00']);
    });

    it('should create a weekly schedule', async () => {
      const supplement = await createTestSupplement({ name: 'Vitamin K2' });

      const response = await request(app)
        .post('/api/user-supplements')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          supplementId: supplement.id,
          dosage: '100',
          unit: 'mcg',
          scheduleType: 'WEEKLY',
          weeklySchedule: {
            monday: ['08:00'],
            thursday: ['08:00'],
          },
          startDate: new Date().toISOString(),
        });

      expect(response.status).toBe(201);
      expect(response.body.scheduleType).toBe('WEEKLY');
    });

    it('should create an interval schedule', async () => {
      const supplement = await createTestSupplement({ name: 'Vitamin B12' });

      const response = await request(app)
        .post('/api/user-supplements')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          supplementId: supplement.id,
          dosage: '1000',
          unit: 'mcg',
          scheduleType: 'INTERVAL',
          intervalDays: 3,
          startDate: new Date().toISOString(),
        });

      expect(response.status).toBe(201);
      expect(response.body.scheduleType).toBe('INTERVAL');
      expect(response.body.intervalDays).toBe(3);
    });

    it('should fail if DAILY_MULTIPLE missing scheduleTimes', async () => {
      const supplement = await createTestSupplement({ name: 'Test' });

      const response = await request(app)
        .post('/api/user-supplements')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          supplementId: supplement.id,
          dosage: '5',
          unit: 'g',
          scheduleType: 'DAILY_MULTIPLE',
          startDate: new Date().toISOString(),
        });

      expect(response.status).toBe(400);
    });

    it('should fail if WEEKLY missing weeklySchedule', async () => {
      const supplement = await createTestSupplement({ name: 'Test' });

      const response = await request(app)
        .post('/api/user-supplements')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          supplementId: supplement.id,
          dosage: '5',
          unit: 'g',
          scheduleType: 'WEEKLY',
          startDate: new Date().toISOString(),
        });

      expect(response.status).toBe(400);
    });

    it('should fail if INTERVAL missing intervalDays', async () => {
      const supplement = await createTestSupplement({ name: 'Test' });

      const response = await request(app)
        .post('/api/user-supplements')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          supplementId: supplement.id,
          dosage: '5',
          unit: 'g',
          scheduleType: 'INTERVAL',
          startDate: new Date().toISOString(),
        });

      expect(response.status).toBe(400);
    });
  });

  describe('GET /api/user-supplements', () => {
    it('should return user supplement schedules', async () => {
      const supplement = await createTestSupplement({ name: 'Lysine' });
      await createTestUserSupplement(userId, supplement.id);

      const response = await request(app)
        .get('/api/user-supplements')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body.length).toBe(1);
      assertUserSupplementStructure(response.body[0]);
    });

    it('should include inactive when requested', async () => {
      const supplement = await createTestSupplement({ name: 'Lysine' });
      await createTestUserSupplement(userId, supplement.id, { isActive: false });

      const responseWithoutInactive = await request(app)
        .get('/api/user-supplements')
        .set('Authorization', `Bearer ${authToken}`);

      expect(responseWithoutInactive.body.length).toBe(0);

      const responseWithInactive = await request(app)
        .get('/api/user-supplements?includeInactive=true')
        .set('Authorization', `Bearer ${authToken}`);

      expect(responseWithInactive.body.length).toBe(1);
    });
  });

  describe('GET /api/user-supplements/:id', () => {
    it('should return a single user supplement', async () => {
      const supplement = await createTestSupplement({ name: 'BCAA' });
      const userSupplement = await createTestUserSupplement(userId, supplement.id);

      const response = await request(app)
        .get(`/api/user-supplements/${userSupplement.id}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      assertUserSupplementStructure(response.body);
    });

    it('should return 404 for other user\'s supplement', async () => {
      const otherUser = await createTestUser({ email: 'other@example.com' });
      const supplement = await createTestSupplement({ name: 'BCAA' });
      const userSupplement = await createTestUserSupplement(otherUser.id, supplement.id);

      const response = await request(app)
        .get(`/api/user-supplements/${userSupplement.id}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(404);
    });
  });

  describe('PUT /api/user-supplements/:id', () => {
    it('should update a user supplement', async () => {
      const supplement = await createTestSupplement({ name: 'Creatine' });
      const userSupplement = await createTestUserSupplement(userId, supplement.id, { dosage: '3' });

      const response = await request(app)
        .put(`/api/user-supplements/${userSupplement.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          dosage: '5',
          notes: 'Increased dosage',
        });

      expect(response.status).toBe(200);
      expect(response.body.dosage).toBe('5');
      expect(response.body.notes).toBe('Increased dosage');
    });

    it('should deactivate by setting isActive to false', async () => {
      const supplement = await createTestSupplement({ name: 'Creatine' });
      const userSupplement = await createTestUserSupplement(userId, supplement.id);

      const response = await request(app)
        .put(`/api/user-supplements/${userSupplement.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          isActive: false,
        });

      expect(response.status).toBe(200);
      expect(response.body.isActive).toBe(false);
    });
  });

  describe('DELETE /api/user-supplements/:id', () => {
    it('should soft delete (deactivate) a user supplement', async () => {
      const supplement = await createTestSupplement({ name: 'Lysine' });
      const userSupplement = await createTestUserSupplement(userId, supplement.id);

      const response = await request(app)
        .delete(`/api/user-supplements/${userSupplement.id}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body.message).toContain('deactivated');
    });
  });

  describe('GET /api/user-supplements/scheduled', () => {
    it('should return scheduled supplements for today', async () => {
      const supplement = await createTestSupplement({ name: 'Daily Vitamin' });
      await createTestUserSupplement(userId, supplement.id, {
        scheduleType: ScheduleType.DAILY,
        startDate: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7 days ago
      });

      const response = await request(app)
        .get('/api/user-supplements/scheduled')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(Array.isArray(response.body)).toBe(true);
    });
  });

  // ==========================================================================
  // SUPPLEMENT LOGS
  // ==========================================================================

  describe('POST /api/supplement-logs', () => {
    it('should create a supplement log', async () => {
      const supplement = await createTestSupplement({ name: 'Lysine' });

      const response = await request(app)
        .post('/api/supplement-logs')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          supplementId: supplement.id,
          dosage: '1000',
          unit: 'mg',
          takenAt: new Date().toISOString(),
          source: 'MANUAL',
        });

      expect(response.status).toBe(201);
      assertSupplementLogStructure(response.body);
      expect(response.body.dosage).toBe('1000');
    });

    it('should create a log linked to user supplement', async () => {
      const supplement = await createTestSupplement({ name: 'BCAA' });
      const userSupplement = await createTestUserSupplement(userId, supplement.id);

      const response = await request(app)
        .post('/api/supplement-logs')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          supplementId: supplement.id,
          userSupplementId: userSupplement.id,
          dosage: '5',
          unit: 'g',
          takenAt: new Date().toISOString(),
          source: 'SCHEDULED',
        });

      expect(response.status).toBe(201);
      expect(response.body.userSupplementId).toBe(userSupplement.id);
    });

    it('should fail with missing required fields', async () => {
      const response = await request(app)
        .post('/api/supplement-logs')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          dosage: '5',
        });

      expect(response.status).toBe(400);
    });
  });

  describe('POST /api/supplement-logs/bulk', () => {
    it('should create multiple logs at once', async () => {
      const supplement1 = await createTestSupplement({ name: 'Vitamin D' });
      const supplement2 = await createTestSupplement({ name: 'Vitamin K2' });

      const response = await request(app)
        .post('/api/supplement-logs/bulk')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          logs: [
            {
              supplementId: supplement1.id,
              dosage: '5000',
              unit: 'IU',
              takenAt: new Date().toISOString(),
              source: 'MANUAL',
            },
            {
              supplementId: supplement2.id,
              dosage: '100',
              unit: 'mcg',
              takenAt: new Date().toISOString(),
              source: 'MANUAL',
            },
          ],
        });

      expect(response.status).toBe(201);
      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body.length).toBe(2);
    });
  });

  describe('GET /api/supplement-logs', () => {
    it('should return user supplement logs', async () => {
      const supplement = await createTestSupplement({ name: 'Lysine' });
      await createTestSupplementLog(userId, supplement.id);
      await createTestSupplementLog(userId, supplement.id);

      const response = await request(app)
        .get('/api/supplement-logs')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body.length).toBe(2);
    });

    it('should filter logs by supplementId', async () => {
      const supplement1 = await createTestSupplement({ name: 'Lysine' });
      const supplement2 = await createTestSupplement({ name: 'BCAA' });
      await createTestSupplementLog(userId, supplement1.id);
      await createTestSupplementLog(userId, supplement2.id);

      const response = await request(app)
        .get(`/api/supplement-logs?supplementId=${supplement1.id}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body.length).toBe(1);
      expect(response.body[0].supplementId).toBe(supplement1.id);
    });

    it('should filter logs by date range', async () => {
      const supplement = await createTestSupplement({ name: 'Lysine' });
      const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000);
      const today = new Date();

      await createTestSupplementLog(userId, supplement.id, { takenAt: yesterday });
      await createTestSupplementLog(userId, supplement.id, { takenAt: today });

      const startOfToday = new Date(today);
      startOfToday.setHours(0, 0, 0, 0);

      const response = await request(app)
        .get(`/api/supplement-logs?startDate=${startOfToday.toISOString()}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body.length).toBe(1);
    });
  });

  describe('GET /api/supplement-logs/:id', () => {
    it('should return a single log', async () => {
      const supplement = await createTestSupplement({ name: 'Creatine' });
      const log = await createTestSupplementLog(userId, supplement.id);

      const response = await request(app)
        .get(`/api/supplement-logs/${log.id}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      assertSupplementLogStructure(response.body);
    });
  });

  describe('PUT /api/supplement-logs/:id', () => {
    it('should update a supplement log', async () => {
      const supplement = await createTestSupplement({ name: 'Lysine' });
      const log = await createTestSupplementLog(userId, supplement.id, { dosage: '500' });

      const response = await request(app)
        .put(`/api/supplement-logs/${log.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          dosage: '1000',
          notes: 'Doubled dosage',
        });

      expect(response.status).toBe(200);
      expect(response.body.dosage).toBe('1000');
      expect(response.body.notes).toBe('Doubled dosage');
    });
  });

  describe('DELETE /api/supplement-logs/:id', () => {
    it('should delete a supplement log', async () => {
      const supplement = await createTestSupplement({ name: 'Lysine' });
      const log = await createTestSupplementLog(userId, supplement.id);

      const response = await request(app)
        .delete(`/api/supplement-logs/${log.id}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body.message).toContain('deleted');
    });
  });

  // ==========================================================================
  // SUMMARIES AND ANALYTICS
  // ==========================================================================

  describe('GET /api/supplements/summary/daily', () => {
    it('should return daily supplement summary', async () => {
      const supplement = await createTestSupplement({ name: 'Daily Vitamin' });
      await createTestUserSupplement(userId, supplement.id, {
        scheduleType: ScheduleType.DAILY,
        startDate: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
      });
      await createTestSupplementLog(userId, supplement.id);

      const response = await request(app)
        .get('/api/supplements/summary/daily')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('date');
      expect(response.body).toHaveProperty('scheduledCount');
      expect(response.body).toHaveProperty('takenCount');
      expect(response.body).toHaveProperty('adherencePercentage');
    });
  });

  describe('GET /api/supplements/summary/weekly', () => {
    it('should return weekly supplement summary', async () => {
      const response = await request(app)
        .get('/api/supplements/summary/weekly')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('days');
      expect(response.body).toHaveProperty('totalScheduled');
      expect(response.body).toHaveProperty('totalTaken');
      expect(response.body).toHaveProperty('averageAdherence');
      expect(Array.isArray(response.body.days)).toBe(true);
      expect(response.body.days.length).toBe(7);
    });
  });

  describe('GET /api/supplements/:id/stats', () => {
    it('should return supplement statistics', async () => {
      const supplement = await createTestSupplement({ name: 'Creatine' });
      await createTestSupplementLog(userId, supplement.id);
      await createTestSupplementLog(userId, supplement.id);

      const response = await request(app)
        .get(`/api/supplements/${supplement.id}/stats`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('supplementId');
      expect(response.body).toHaveProperty('totalLogs');
      expect(response.body).toHaveProperty('averagePerDay');
      expect(response.body.totalLogs).toBe(2);
    });

    it('should return 404 when no logs exist', async () => {
      const supplement = await createTestSupplement({ name: 'Unused Supplement' });

      const response = await request(app)
        .get(`/api/supplements/${supplement.id}/stats`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(404);
    });
  });

  // ==========================================================================
  // USER DATA MANAGEMENT
  // ==========================================================================

  describe('DELETE /api/supplements/user-data', () => {
    it('should delete all user supplement data', async () => {
      const supplement = await createTestSupplement({ name: 'Test' });
      await createTestUserSupplement(userId, supplement.id);
      await createTestSupplementLog(userId, supplement.id);

      const response = await request(app)
        .delete('/api/supplements/user-data')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body.message).toContain('Deleted');

      // Verify data is deleted
      const logsResponse = await request(app)
        .get('/api/supplement-logs')
        .set('Authorization', `Bearer ${authToken}`);

      expect(logsResponse.body.length).toBe(0);
    });
  });
});
