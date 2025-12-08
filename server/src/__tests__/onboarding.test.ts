import request from 'supertest';
import app from '../index';
import { createTestUser, createTestToken } from './setup';
import { HTTP_STATUS } from '../config/constants';

describe('Onboarding API', () => {
  let testUser: Awaited<ReturnType<typeof createTestUser>>;
  let authToken: string;

  // Create user before each test since setup.ts cleans DB before each test
  beforeEach(async () => {
    testUser = await createTestUser();
    authToken = createTestToken(testUser.id);
  });

  // Helper to start onboarding for tests that need it
  const startOnboarding = async (token: string) => {
    await request(app)
      .post('/api/onboarding/start')
      .set('Authorization', `Bearer ${token}`)
      .send({ version: '1.0' });
  };

  describe('POST /api/onboarding/start', () => {
    it('should start onboarding for a new user', async () => {
      const response = await request(app)
        .post('/api/onboarding/start')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ version: '1.0' });

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body).toHaveProperty('id');
      expect(response.body).toHaveProperty('userId', testUser.id);
      expect(response.body).toHaveProperty('currentStep', 1);
      expect(response.body).toHaveProperty('totalSteps', 6);
      expect(response.body).toHaveProperty('isComplete', false);
      expect(response.body).toHaveProperty('progress', 0);
    });

    it('should return existing onboarding if already started', async () => {
      // Start onboarding first
      await startOnboarding(authToken);

      // Try to start again
      const response = await request(app)
        .post('/api/onboarding/start')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ version: '1.0' });

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body).toHaveProperty('userId', testUser.id);
    });

    it('should reject unauthenticated request', async () => {
      const response = await request(app)
        .post('/api/onboarding/start')
        .send({ version: '1.0' });

      expect(response.status).toBe(HTTP_STATUS.UNAUTHORIZED);
    });
  });

  describe('GET /api/onboarding/status', () => {
    it('should return onboarding status', async () => {
      // Start onboarding first
      await startOnboarding(authToken);

      const response = await request(app)
        .get('/api/onboarding/status')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body).toHaveProperty('currentStep');
      expect(response.body).toHaveProperty('totalSteps');
      expect(response.body).toHaveProperty('isComplete');
    });

    it('should return 404 if onboarding not started', async () => {
      const response = await request(app)
        .get('/api/onboarding/status')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(HTTP_STATUS.NOT_FOUND);
    });
  });

  describe('PUT /api/onboarding/step/1', () => {
    const step1Data = {
      name: 'Test User',
      dateOfBirth: '1990-01-15',
      biologicalSex: 'MALE',
      height: 180,
      currentWeight: 75,
      activityLevel: 'moderate',
    };

    it('should save step 1 data', async () => {
      // Start onboarding first
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/1')
        .set('Authorization', `Bearer ${authToken}`)
        .send(step1Data);

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body).toHaveProperty('currentStep', 2);
    });

    it('should validate required fields', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/1')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ name: 'Test' });

      expect(response.status).toBe(HTTP_STATUS.BAD_REQUEST);
    });

    it('should validate height range', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/1')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ ...step1Data, height: 10 });

      expect(response.status).toBe(HTTP_STATUS.BAD_REQUEST);
    });

    it('should validate weight range', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/1')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ ...step1Data, currentWeight: 5 });

      expect(response.status).toBe(HTTP_STATUS.BAD_REQUEST);
    });

    it('should validate biologicalSex enum', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/1')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ ...step1Data, biologicalSex: 'INVALID' });

      expect(response.status).toBe(HTTP_STATUS.BAD_REQUEST);
    });
  });

  describe('PUT /api/onboarding/step/2', () => {
    const step2Data = {
      primaryGoal: 'WEIGHT_LOSS',
      dietaryPreferences: ['vegetarian', 'gluten_free'],
      goalWeight: 70,
    };

    it('should save step 2 data', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/2')
        .set('Authorization', `Bearer ${authToken}`)
        .send(step2Data);

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body).toHaveProperty('currentStep', 3);
    });

    it('should validate primaryGoal enum', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/2')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ ...step2Data, primaryGoal: 'INVALID_GOAL' });

      expect(response.status).toBe(HTTP_STATUS.BAD_REQUEST);
    });

    it('should allow custom macros', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/2')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          ...step2Data,
          customMacros: {
            goalCalories: 2000,
            goalProtein: 150,
            goalCarbs: 200,
            goalFat: 65,
          },
        });

      expect(response.status).toBe(HTTP_STATUS.OK);
    });
  });

  describe('PUT /api/onboarding/step/3', () => {
    const step3Data = {
      notificationsEnabled: true,
      notificationTypes: ['meal_reminders', 'insights'],
      healthKitEnabled: false,
      healthKitScopes: [],
      healthConnectEnabled: false,
      healthConnectScopes: [],
    };

    it('should save step 3 data', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/3')
        .set('Authorization', `Bearer ${authToken}`)
        .send(step3Data);

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body).toHaveProperty('currentStep', 4);
    });

    it('should validate notification types', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/3')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          ...step3Data,
          notificationTypes: ['invalid_type'],
        });

      expect(response.status).toBe(HTTP_STATUS.BAD_REQUEST);
    });
  });

  describe('PUT /api/onboarding/step/4', () => {
    const step4Data = {
      chronicConditions: [{ type: 'diabetes_type2' }],
      medications: [],
      supplements: [{ name: 'vitamin_d' }],
      allergies: ['peanuts', 'shellfish'],
    };

    it('should save step 4 data', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/4')
        .set('Authorization', `Bearer ${authToken}`)
        .send(step4Data);

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body).toHaveProperty('currentStep', 5);
    });

    it('should allow empty health background', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/4')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          chronicConditions: [],
          medications: [],
          supplements: [],
          allergies: [],
        });

      expect(response.status).toBe(HTTP_STATUS.OK);
    });
  });

  describe('PUT /api/onboarding/step/5', () => {
    const step5Data = {
      nicotineUse: 'NONE',
      alcoholUse: 'OCCASIONAL',
      caffeineDaily: 3, // cups per day (0-20)
      typicalBedtime: '22:00',
      typicalWakeTime: '07:00',
      sleepQuality: 7,
      stressLevel: 4,
      workSchedule: 'regular',
    };

    it('should save step 5 data', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/5')
        .set('Authorization', `Bearer ${authToken}`)
        .send(step5Data);

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body).toHaveProperty('currentStep', 6);
    });

    it('should validate sleep quality range', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/5')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ ...step5Data, sleepQuality: 15 });

      expect(response.status).toBe(HTTP_STATUS.BAD_REQUEST);
    });

    it('should validate stress level range', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .put('/api/onboarding/step/5')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ ...step5Data, stressLevel: 0 });

      expect(response.status).toBe(HTTP_STATUS.BAD_REQUEST);
    });
  });

  describe('POST /api/onboarding/skip/:step', () => {
    // Helper to set up onboarding through step 3
    const setupThroughStep3 = async (token: string) => {
      await startOnboarding(token);

      await request(app)
        .put('/api/onboarding/step/1')
        .set('Authorization', `Bearer ${token}`)
        .send({
          name: 'Skip Test User',
          dateOfBirth: '1990-01-15',
          biologicalSex: 'FEMALE',
          height: 165,
          currentWeight: 60,
          activityLevel: 'light',
        });

      await request(app)
        .put('/api/onboarding/step/2')
        .set('Authorization', `Bearer ${token}`)
        .send({
          primaryGoal: 'GENERAL_HEALTH',
          dietaryPreferences: [],
        });

      await request(app)
        .put('/api/onboarding/step/3')
        .set('Authorization', `Bearer ${token}`)
        .send({
          notificationsEnabled: false,
          notificationTypes: [],
          healthKitEnabled: false,
          healthKitScopes: [],
          healthConnectEnabled: false,
          healthConnectScopes: [],
        });
    };

    it('should skip step 4 (health background)', async () => {
      await setupThroughStep3(authToken);

      const response = await request(app)
        .post('/api/onboarding/skip/4')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body.skippedSteps).toContain(4);
      expect(response.body.currentStep).toBe(5);
    });

    it('should skip step 5 (lifestyle)', async () => {
      await setupThroughStep3(authToken);
      // Skip step 4 first
      await request(app)
        .post('/api/onboarding/skip/4')
        .set('Authorization', `Bearer ${authToken}`);

      const response = await request(app)
        .post('/api/onboarding/skip/5')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body.skippedSteps).toContain(5);
    });

    it('should reject skipping non-skippable steps', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .post('/api/onboarding/skip/1')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(HTTP_STATUS.BAD_REQUEST);
    });
  });

  describe('POST /api/onboarding/complete', () => {
    it('should complete onboarding', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .post('/api/onboarding/complete')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ skipRemaining: true });

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body).toHaveProperty('isComplete', true);
      expect(response.body).toHaveProperty('completedAt');
      expect(response.body.completedAt).not.toBeNull();
    });

    it('should allow completing without skipping remaining', async () => {
      // Complete all steps first
      await startOnboarding(authToken);

      await request(app)
        .put('/api/onboarding/step/1')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Test User',
          dateOfBirth: '1990-01-15',
          biologicalSex: 'MALE',
          height: 180,
          currentWeight: 75,
          activityLevel: 'moderate',
        });

      await request(app)
        .put('/api/onboarding/step/2')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          primaryGoal: 'WEIGHT_LOSS',
          dietaryPreferences: [],
        });

      await request(app)
        .put('/api/onboarding/step/3')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          notificationsEnabled: false,
          notificationTypes: [],
          healthKitEnabled: false,
          healthKitScopes: [],
          healthConnectEnabled: false,
          healthConnectScopes: [],
        });

      await request(app)
        .put('/api/onboarding/step/4')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          chronicConditions: [],
          medications: [],
          supplements: [],
          allergies: [],
        });

      await request(app)
        .put('/api/onboarding/step/5')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          nicotineUse: 'NONE',
          alcoholUse: 'NONE',
          caffeineDaily: 0, // cups per day (0-20)
          typicalBedtime: '22:00',
          typicalWakeTime: '07:00',
          sleepQuality: 7,
          stressLevel: 5,
          workSchedule: 'regular',
        });

      const response = await request(app)
        .post('/api/onboarding/complete')
        .set('Authorization', `Bearer ${authToken}`)
        .send({ skipRemaining: false });

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body).toHaveProperty('isComplete', true);
    });
  });

  describe('GET /api/onboarding/data', () => {
    it('should return all onboarding data', async () => {
      await startOnboarding(authToken);

      const response = await request(app)
        .get('/api/onboarding/data')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body).toHaveProperty('user');
      expect(response.body).toHaveProperty('status');
      expect(response.body).toHaveProperty('healthBackground');
      expect(response.body).toHaveProperty('lifestyle');
      expect(response.body).toHaveProperty('permissions');
    });

    it('should return user data even without onboarding started', async () => {
      // getData always returns data, status will just be null
      const response = await request(app)
        .get('/api/onboarding/data')
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(HTTP_STATUS.OK);
      expect(response.body).toHaveProperty('user');
      expect(response.body.status).toBeNull();
    });
  });
});
