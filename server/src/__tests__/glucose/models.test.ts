/**
 * CGM (Continuous Glucose Monitor) Models Unit Tests
 *
 * Tests for GlucoseReading, MealGlucoseResponse, and CGMConnection models.
 * Verifies CRUD operations, data validation, and relationship integrity.
 */

import { GlucoseSource, GlucoseTrend } from '@prisma/client';
import {
  prisma,
  createTestUser,
  createTestMeal,
  createTestGlucoseReading,
  createTestGlucoseReadings,
  createTestCGMConnection,
  createTestMealGlucoseResponse,
  assertGlucoseReadingStructure,
  assertMealGlucoseResponseStructure,
} from '../setup';

describe('CGM Models', () => {
  let testUserId: string;
  let testMealId: string;

  beforeEach(async () => {
    const user = await createTestUser();
    testUserId = user.id;

    const meal = await createTestMeal(testUserId);
    testMealId = meal.id;
  });

  describe('GlucoseReading Model', () => {
    it('should create a glucose reading with required fields', async () => {
      const reading = await createTestGlucoseReading(testUserId);

      assertGlucoseReadingStructure(reading);
      expect(reading.userId).toBe(testUserId);
      expect(reading.value).toBe(105.0);
      expect(reading.unit).toBe('mg/dL');
      expect(reading.source).toBe(GlucoseSource.DEXCOM);
    });

    it('should create a glucose reading with all optional fields', async () => {
      const now = new Date();
      const reading = await createTestGlucoseReading(testUserId, {
        value: 120.5,
        sourceId: 'dexcom-reading-123',
        trendArrow: GlucoseTrend.RISING_SLIGHTLY,
        trendRate: 1.5,
        recordedAt: now,
        metadata: { quality: 'good', transmitterId: 'ABC123' },
      });

      expect(reading.value).toBe(120.5);
      expect(reading.sourceId).toBe('dexcom-reading-123');
      expect(reading.trendArrow).toBe(GlucoseTrend.RISING_SLIGHTLY);
      expect(reading.trendRate).toBe(1.5);
      expect(reading.metadata).toEqual({ quality: 'good', transmitterId: 'ABC123' });
    });

    it('should create multiple glucose readings with correct timestamps', async () => {
      const startTime = new Date('2024-01-15T08:00:00Z');
      const readings = await createTestGlucoseReadings(testUserId, 12, {
        startTime,
        intervalMinutes: 5,
        baseValue: 100,
        variance: 10,
      });

      expect(readings.length).toBe(12);

      // Verify timestamps are 5 minutes apart
      for (let i = 1; i < readings.length; i++) {
        const prevTime = new Date(readings[i - 1].recordedAt).getTime();
        const currTime = new Date(readings[i].recordedAt).getTime();
        expect(currTime - prevTime).toBe(5 * 60 * 1000);
      }
    });

    it('should enforce unique constraint on userId + source + recordedAt', async () => {
      const recordedAt = new Date('2024-01-15T08:00:00Z');

      await createTestGlucoseReading(testUserId, {
        source: GlucoseSource.DEXCOM,
        recordedAt,
      });

      // Attempting to create a duplicate should fail
      await expect(
        createTestGlucoseReading(testUserId, {
          source: GlucoseSource.DEXCOM,
          recordedAt,
        })
      ).rejects.toThrow();
    });

    it('should allow same timestamp from different sources', async () => {
      const recordedAt = new Date('2024-01-15T08:00:00Z');

      const dexcomReading = await createTestGlucoseReading(testUserId, {
        source: GlucoseSource.DEXCOM,
        recordedAt,
      });

      const manualReading = await createTestGlucoseReading(testUserId, {
        source: GlucoseSource.MANUAL,
        recordedAt,
      });

      expect(dexcomReading.id).not.toBe(manualReading.id);
      expect(dexcomReading.source).toBe(GlucoseSource.DEXCOM);
      expect(manualReading.source).toBe(GlucoseSource.MANUAL);
    });

    it('should support all GlucoseSource enum values', async () => {
      const sources = [
        GlucoseSource.DEXCOM,
        GlucoseSource.LIBRE,
        GlucoseSource.LEVELS,
        GlucoseSource.MANUAL,
      ];

      for (const source of sources) {
        const reading = await createTestGlucoseReading(testUserId, {
          source,
          recordedAt: new Date(Date.now() + sources.indexOf(source) * 1000),
        });
        expect(reading.source).toBe(source);
      }
    });

    it('should support all GlucoseTrend enum values', async () => {
      const trends = [
        GlucoseTrend.RISING_RAPIDLY,
        GlucoseTrend.RISING,
        GlucoseTrend.RISING_SLIGHTLY,
        GlucoseTrend.STABLE,
        GlucoseTrend.FALLING_SLIGHTLY,
        GlucoseTrend.FALLING,
        GlucoseTrend.FALLING_RAPIDLY,
        GlucoseTrend.NOT_AVAILABLE,
      ];

      for (let i = 0; i < trends.length; i++) {
        const reading = await createTestGlucoseReading(testUserId, {
          trendArrow: trends[i],
          recordedAt: new Date(Date.now() + i * 1000),
        });
        expect(reading.trendArrow).toBe(trends[i]);
      }
    });

    it('should cascade delete readings when user is deleted', async () => {
      await createTestGlucoseReadings(testUserId, 5);

      let count = await prisma.glucoseReading.count({ where: { userId: testUserId } });
      expect(count).toBe(5);

      await prisma.user.delete({ where: { id: testUserId } });

      count = await prisma.glucoseReading.count({ where: { userId: testUserId } });
      expect(count).toBe(0);
    });

    it('should query readings by date range', async () => {
      const baseTime = new Date('2024-01-15T12:00:00Z');

      // Create readings at different times
      await createTestGlucoseReading(testUserId, {
        recordedAt: new Date(baseTime.getTime() - 2 * 60 * 60 * 1000), // 2 hours before
      });
      await createTestGlucoseReading(testUserId, {
        recordedAt: baseTime,
      });
      await createTestGlucoseReading(testUserId, {
        recordedAt: new Date(baseTime.getTime() + 2 * 60 * 60 * 1000), // 2 hours after
      });

      const readings = await prisma.glucoseReading.findMany({
        where: {
          userId: testUserId,
          recordedAt: {
            gte: new Date(baseTime.getTime() - 1 * 60 * 60 * 1000),
            lte: new Date(baseTime.getTime() + 1 * 60 * 60 * 1000),
          },
        },
      });

      expect(readings.length).toBe(1);
    });
  });

  describe('CGMConnection Model', () => {
    it('should create a CGM connection with required fields', async () => {
      const connection = await createTestCGMConnection(testUserId);

      expect(connection.userId).toBe(testUserId);
      expect(connection.provider).toBe(GlucoseSource.DEXCOM);
      expect(connection.isActive).toBe(true);
      expect(connection.accessToken).toBeDefined();
    });

    it('should create a CGM connection with all optional fields', async () => {
      const now = new Date();
      const connection = await createTestCGMConnection(testUserId, {
        provider: GlucoseSource.LIBRE,
        lastSyncAt: now,
        externalUserId: 'libre-user-123',
      });

      expect(connection.provider).toBe(GlucoseSource.LIBRE);
      expect(connection.lastSyncAt).toEqual(now);
      expect(connection.externalUserId).toBe('libre-user-123');
    });

    it('should enforce unique constraint on userId + provider', async () => {
      await createTestCGMConnection(testUserId, {
        provider: GlucoseSource.DEXCOM,
      });

      // Attempting to create duplicate connection for same provider should fail
      await expect(
        createTestCGMConnection(testUserId, {
          provider: GlucoseSource.DEXCOM,
        })
      ).rejects.toThrow();
    });

    it('should allow multiple connections for different providers', async () => {
      const dexcomConnection = await createTestCGMConnection(testUserId, {
        provider: GlucoseSource.DEXCOM,
      });

      const libreConnection = await createTestCGMConnection(testUserId, {
        provider: GlucoseSource.LIBRE,
      });

      expect(dexcomConnection.provider).toBe(GlucoseSource.DEXCOM);
      expect(libreConnection.provider).toBe(GlucoseSource.LIBRE);
    });

    it('should update connection status', async () => {
      const connection = await createTestCGMConnection(testUserId);
      expect(connection.isActive).toBe(true);

      const updatedConnection = await prisma.cGMConnection.update({
        where: { id: connection.id },
        data: {
          isActive: false,
          disconnectedAt: new Date(),
          lastSyncStatus: 'disconnected',
        },
      });

      expect(updatedConnection.isActive).toBe(false);
      expect(updatedConnection.disconnectedAt).toBeDefined();
    });

    it('should update sync status after successful sync', async () => {
      const connection = await createTestCGMConnection(testUserId);

      const syncTime = new Date();
      const updatedConnection = await prisma.cGMConnection.update({
        where: { id: connection.id },
        data: {
          lastSyncAt: syncTime,
          lastSyncStatus: 'success',
          lastSyncError: null,
        },
      });

      expect(updatedConnection.lastSyncAt).toEqual(syncTime);
      expect(updatedConnection.lastSyncStatus).toBe('success');
      expect(updatedConnection.lastSyncError).toBeNull();
    });

    it('should update sync status after failed sync', async () => {
      const connection = await createTestCGMConnection(testUserId);

      const updatedConnection = await prisma.cGMConnection.update({
        where: { id: connection.id },
        data: {
          lastSyncStatus: 'error',
          lastSyncError: 'Token expired',
        },
      });

      expect(updatedConnection.lastSyncStatus).toBe('error');
      expect(updatedConnection.lastSyncError).toBe('Token expired');
    });

    it('should cascade delete connections when user is deleted', async () => {
      await createTestCGMConnection(testUserId, { provider: GlucoseSource.DEXCOM });
      await createTestCGMConnection(testUserId, { provider: GlucoseSource.LIBRE });

      let count = await prisma.cGMConnection.count({ where: { userId: testUserId } });
      expect(count).toBe(2);

      await prisma.user.delete({ where: { id: testUserId } });

      count = await prisma.cGMConnection.count({ where: { userId: testUserId } });
      expect(count).toBe(0);
    });

    it('should query active connections only', async () => {
      await createTestCGMConnection(testUserId, {
        provider: GlucoseSource.DEXCOM,
        isActive: true,
      });

      await createTestCGMConnection(testUserId, {
        provider: GlucoseSource.LIBRE,
        isActive: false,
      });

      const activeConnections = await prisma.cGMConnection.findMany({
        where: { userId: testUserId, isActive: true },
      });

      expect(activeConnections.length).toBe(1);
      expect(activeConnections[0].provider).toBe(GlucoseSource.DEXCOM);
    });
  });

  describe('MealGlucoseResponse Model', () => {
    it('should create a meal glucose response with all fields', async () => {
      const response = await createTestMealGlucoseResponse(testMealId);

      assertMealGlucoseResponseStructure(response);
      expect(response.mealId).toBe(testMealId);
      expect(response.baselineGlucose).toBe(95.0);
      expect(response.peakGlucose).toBe(140.0);
      expect(response.peakTime).toBe(45);
      expect(response.glucoseRise).toBe(45.0);
      expect(response.glucoseScore).toBe(75.0);
    });

    it('should enforce unique constraint on mealId', async () => {
      await createTestMealGlucoseResponse(testMealId);

      // Attempting to create another response for the same meal should fail
      await expect(createTestMealGlucoseResponse(testMealId)).rejects.toThrow();
    });

    it('should allow updating glucose response for a meal', async () => {
      const response = await createTestMealGlucoseResponse(testMealId);

      const updatedResponse = await prisma.mealGlucoseResponse.update({
        where: { id: response.id },
        data: {
          peakGlucose: 160.0,
          glucoseRise: 65.0,
          glucoseScore: 60.0, // Lower score due to higher peak
        },
      });

      expect(updatedResponse.peakGlucose).toBe(160.0);
      expect(updatedResponse.glucoseRise).toBe(65.0);
      expect(updatedResponse.glucoseScore).toBe(60.0);
    });

    it('should cascade delete response when meal is deleted', async () => {
      await createTestMealGlucoseResponse(testMealId);

      let count = await prisma.mealGlucoseResponse.count({ where: { mealId: testMealId } });
      expect(count).toBe(1);

      await prisma.meal.delete({ where: { id: testMealId } });

      count = await prisma.mealGlucoseResponse.count({ where: { mealId: testMealId } });
      expect(count).toBe(0);
    });

    it('should fetch meal with glucose response via include', async () => {
      await createTestMealGlucoseResponse(testMealId);

      const mealWithResponse = await prisma.meal.findUnique({
        where: { id: testMealId },
        include: { glucoseResponse: true },
      });

      expect(mealWithResponse).toBeDefined();
      expect(mealWithResponse?.glucoseResponse).toBeDefined();
      expect(mealWithResponse?.glucoseResponse?.glucoseScore).toBe(75.0);
    });

    it('should calculate correct glucose rise from peak and baseline', async () => {
      const baselineGlucose = 90.0;
      const peakGlucose = 145.0;
      const expectedRise = peakGlucose - baselineGlucose;

      const response = await createTestMealGlucoseResponse(testMealId, {
        baselineGlucose,
        peakGlucose,
        glucoseRise: expectedRise,
      });

      expect(response.glucoseRise).toBe(55.0);
    });

    it('should store all timing metrics correctly', async () => {
      const now = new Date();
      const windowStart = new Date(now.getTime() - 30 * 60 * 1000);
      const windowEnd = new Date(now.getTime() + 180 * 60 * 1000);

      const response = await createTestMealGlucoseResponse(testMealId, {
        peakTime: 50,
        returnToBaseline: 150,
        twoHourGlucose: 105.0,
        windowStart,
        windowEnd,
      });

      expect(response.peakTime).toBe(50);
      expect(response.returnToBaseline).toBe(150);
      expect(response.twoHourGlucose).toBe(105.0);
      expect(response.windowStart).toEqual(windowStart);
      expect(response.windowEnd).toEqual(windowEnd);
    });

    it('should store confidence and reading count metrics', async () => {
      const response = await createTestMealGlucoseResponse(testMealId, {
        readingCount: 48,
        confidence: 0.88,
      });

      expect(response.readingCount).toBe(48);
      expect(response.confidence).toBe(0.88);
    });
  });

  describe('Model Relationships', () => {
    it('should link glucose readings to user', async () => {
      await createTestGlucoseReadings(testUserId, 10);

      const user = await prisma.user.findUnique({
        where: { id: testUserId },
        include: { glucoseReadings: true },
      });

      expect(user?.glucoseReadings.length).toBe(10);
    });

    it('should link CGM connections to user', async () => {
      await createTestCGMConnection(testUserId, { provider: GlucoseSource.DEXCOM });
      await createTestCGMConnection(testUserId, { provider: GlucoseSource.LIBRE });

      const user = await prisma.user.findUnique({
        where: { id: testUserId },
        include: { cgmConnections: true },
      });

      expect(user?.cgmConnections.length).toBe(2);
    });

    it('should link meal glucose response to meal', async () => {
      await createTestMealGlucoseResponse(testMealId);

      const meal = await prisma.meal.findUnique({
        where: { id: testMealId },
        include: { glucoseResponse: true },
      });

      expect(meal?.glucoseResponse).toBeDefined();
      expect(meal?.glucoseResponse?.mealId).toBe(testMealId);
    });
  });
});
