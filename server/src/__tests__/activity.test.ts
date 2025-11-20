import request from 'supertest';
import app from '../index';
import {
  createTestUser,
  createTestToken,
  createTestActivity,
  assertActivityStructure,
} from './setup';
import { ActivityType, ActivityIntensity } from '@prisma/client';

describe('Activity API Endpoints', () => {
  let authToken: string;
  let userId: string;

  beforeEach(async () => {
    const user = await createTestUser();
    userId = user.id;
    authToken = createTestToken(userId);
  });

  describe('POST /api/activities', () => {
    it('should create an activity', async () => {
      const now = new Date();
      const activityData = {
        activityType: 'RUNNING',
        intensity: 'MODERATE',
        startedAt: now.toISOString(),
        endedAt: new Date(now.getTime() + 30 * 60 * 1000).toISOString(),
        duration: 30,
        caloriesBurned: 250,
        source: 'manual',
      };

      const response = await request(app)
        .post('/api/activities')
        .set('Authorization', `Bearer ${authToken}`)
        .send(activityData)
        .expect(201);

      assertActivityStructure(response.body);
      expect(response.body.activityType).toBe(activityData.activityType);
      expect(response.body.duration).toBe(activityData.duration);
    });

    it('should reject request without authentication', async () => {
      await request(app)
        .post('/api/activities')
        .send({
          activityType: 'RUNNING',
          intensity: 'MODERATE',
          startedAt: new Date().toISOString(),
          endedAt: new Date().toISOString(),
          duration: 30,
          source: 'manual',
        })
        .expect(401);
    });
  });

  describe('GET /api/activities', () => {
    beforeEach(async () => {
      await createTestActivity(userId, { activityType: ActivityType.RUNNING, intensity: ActivityIntensity.HIGH });
      await createTestActivity(userId, { activityType: ActivityType.CYCLING, intensity: ActivityIntensity.MODERATE });
    });

    it('should get all activities for user', async () => {
      const response = await request(app)
        .get('/api/activities')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body.length).toBeGreaterThanOrEqual(2);
      response.body.forEach((activity: unknown) => assertActivityStructure(activity));
    });

    it('should filter activities by type', async () => {
      const response = await request(app)
        .get('/api/activities?activityType=RUNNING')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);
      response.body.forEach((activity: { activityType: string }) => {
        expect(activity.activityType).toBe('RUNNING');
      });
    });
  });

  describe('PUT /api/activities/:id', () => {
    it('should update an activity', async () => {
      const activity = await createTestActivity(userId);

      const updateData = {
        duration: 45,
        caloriesBurned: 350,
      };

      const response = await request(app)
        .put(`/api/activities/${activity.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send(updateData)
        .expect(200);

      expect(response.body.duration).toBe(updateData.duration);
      expect(response.body.caloriesBurned).toBe(updateData.caloriesBurned);
    });
  });

  describe('DELETE /api/activities/:id', () => {
    it('should delete an activity', async () => {
      const activity = await createTestActivity(userId);

      const response = await request(app)
        .delete(`/api/activities/${activity.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('message');
    });
  });
});
