import request from 'supertest';
import app from '../index';
import {
  createTestUser,
  createTestToken,
  createTestHealthMetric,
  assertHealthMetricStructure,
} from './setup';
import { HealthMetricType } from '@prisma/client';

describe('Health Metric API Endpoints', () => {
  let authToken: string;
  let userId: string;

  beforeEach(async () => {
    const user = await createTestUser();
    userId = user.id;
    authToken = createTestToken(userId);
  });

  describe('POST /api/health-metrics', () => {
    it('should create a health metric', async () => {
      const metricData = {
        metricType: 'RESTING_HEART_RATE',
        value: 65,
        unit: 'bpm',
        recordedAt: new Date().toISOString(),
        source: 'manual',
      };

      const response = await request(app)
        .post('/api/health-metrics')
        .set('Authorization', `Bearer ${authToken}`)
        .send(metricData)
        .expect(201);

      assertHealthMetricStructure(response.body);
      expect(response.body.metricType).toBe(metricData.metricType);
      expect(response.body.value).toBe(metricData.value);
    });

    it('should reject request without authentication', async () => {
      await request(app)
        .post('/api/health-metrics')
        .send({ metricType: 'RESTING_HEART_RATE', value: 65, unit: 'bpm', recordedAt: new Date().toISOString(), source: 'manual' })
        .expect(401);
    });
  });

  describe('GET /api/health-metrics', () => {
    beforeEach(async () => {
      await createTestHealthMetric(userId, { metricType: HealthMetricType.RESTING_HEART_RATE, value: 60 });
      await createTestHealthMetric(userId, { metricType: HealthMetricType.STEPS, value: 10000 });
    });

    it('should get all health metrics for user', async () => {
      const response = await request(app)
        .get('/api/health-metrics')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body.length).toBeGreaterThanOrEqual(2);
      response.body.forEach((metric: unknown) => assertHealthMetricStructure(metric));
    });

    it('should filter metrics by type', async () => {
      const response = await request(app)
        .get('/api/health-metrics?metricType=RESTING_HEART_RATE')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(Array.isArray(response.body)).toBe(true);
      response.body.forEach((metric: { metricType: string }) => {
        expect(metric.metricType).toBe('RESTING_HEART_RATE');
      });
    });
  });

  describe('DELETE /api/health-metrics/:id', () => {
    it('should delete a health metric', async () => {
      const metric = await createTestHealthMetric(userId);

      const response = await request(app)
        .delete(`/api/health-metrics/${metric.id}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('message');
    });
  });
});
