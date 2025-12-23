/**
 * Activity Sync Tests
 * Tests for @kingstinct/react-native-healthkit integration
 */

import { mockKingstinctHealthKit, resetMocks, setupDefaultMocks, mockPlatform } from './test-utils';
import {
  fetchStepCount,
  fetchActiveCalories,
  syncActivityMetrics,
  getTodayActivitySummary,
} from '@/lib/services/healthkit/activity';

// Helper to create mock samples in @kingstinct/react-native-healthkit format
function createKingstinctSample(quantity: number, startDate: Date, endDate?: Date) {
  return {
    quantity,
    startDate,
    endDate: endDate || startDate,
    uuid: `uuid_${Date.now()}_${Math.random()}`,
    device: { name: 'Apple Watch' },
    sourceRevision: { source: { name: 'Apple Watch', bundleIdentifier: 'com.apple.health' } },
  };
}

describe('Activity Sync', () => {
  beforeEach(() => {
    resetMocks();
    setupDefaultMocks();
    mockPlatform.OS = 'ios';
  });

  afterEach(() => {
    mockPlatform.OS = 'ios';
  });

  describe('fetchStepCount', () => {
    it('should return empty array on non-iOS platforms', async () => {
      mockPlatform.OS = 'android';

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchStepCount(options);
      expect(result).toEqual([]);
    });

    it('should transform daily step count samples correctly', async () => {
      const mockSamples = [
        createKingstinctSample(
          8500,
          new Date('2024-01-01T00:00:00Z'),
          new Date('2024-01-01T23:59:59Z')
        ),
        createKingstinctSample(
          10200,
          new Date('2024-01-02T00:00:00Z'),
          new Date('2024-01-02T23:59:59Z')
        ),
      ];

      mockKingstinctHealthKit.queryQuantitySamples.mockResolvedValue(mockSamples);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchStepCount(options);

      expect(result).toHaveLength(2);
      expect(result[0]).toMatchObject({
        metricType: 'STEPS',
        value: 8500,
        unit: 'steps',
        source: 'apple_health',
      });
      expect(result[1]).toMatchObject({
        metricType: 'STEPS',
        value: 10200,
        unit: 'steps',
        source: 'apple_health',
      });
    });

    it('should return empty array on HealthKit error', async () => {
      mockKingstinctHealthKit.queryQuantitySamples.mockRejectedValue(new Error('HealthKit error'));

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchStepCount(options);
      expect(result).toEqual([]);
    });
  });

  describe('fetchActiveCalories', () => {
    it('should transform active energy samples correctly', async () => {
      const mockSamples = [
        createKingstinctSample(
          450,
          new Date('2024-01-01T00:00:00Z'),
          new Date('2024-01-01T23:59:59Z')
        ),
        createKingstinctSample(
          380,
          new Date('2024-01-02T00:00:00Z'),
          new Date('2024-01-02T23:59:59Z')
        ),
      ];

      mockKingstinctHealthKit.queryQuantitySamples.mockResolvedValue(mockSamples);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchActiveCalories(options);

      expect(result).toHaveLength(2);
      expect(result[0]).toMatchObject({
        metricType: 'ACTIVE_CALORIES',
        value: 450,
        unit: 'kcal',
        source: 'apple_health',
      });
    });

    it('should return empty array on HealthKit error', async () => {
      mockKingstinctHealthKit.queryQuantitySamples.mockRejectedValue(new Error('HealthKit error'));

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchActiveCalories(options);
      expect(result).toEqual([]);
    });
  });

  describe('syncActivityMetrics', () => {
    it('should combine steps and active calories', async () => {
      const stepSample = createKingstinctSample(
        8500,
        new Date('2024-01-01T00:00:00Z'),
        new Date('2024-01-01T23:59:59Z')
      );
      const calorieSample = createKingstinctSample(
        450,
        new Date('2024-01-01T00:00:00Z'),
        new Date('2024-01-01T23:59:59Z')
      );

      // First call for steps, second for calories
      mockKingstinctHealthKit.queryQuantitySamples
        .mockResolvedValueOnce([stepSample])
        .mockResolvedValueOnce([calorieSample]);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await syncActivityMetrics(options);

      expect(result).toHaveLength(2);
      const metricTypes = result.map((m) => m.metricType);
      expect(metricTypes).toContain('STEPS');
      expect(metricTypes).toContain('ACTIVE_CALORIES');
    });

    it('should sort results by recordedAt (newest first)', async () => {
      const stepSample = createKingstinctSample(
        8500,
        new Date('2024-01-01T00:00:00Z'),
        new Date('2024-01-01T23:59:59Z')
      );
      const calorieSample = createKingstinctSample(
        450,
        new Date('2024-01-03T00:00:00Z'), // Newer
        new Date('2024-01-03T23:59:59Z')
      );

      mockKingstinctHealthKit.queryQuantitySamples
        .mockResolvedValueOnce([stepSample])
        .mockResolvedValueOnce([calorieSample]);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await syncActivityMetrics(options);

      // Calories sample is newer, should be first
      expect(result[0].metricType).toBe('ACTIVE_CALORIES');
      expect(result[1].metricType).toBe('STEPS');
    });
  });

  describe('getTodayActivitySummary', () => {
    it('should return today steps and calories', async () => {
      const today = new Date();
      const todayStart = new Date(today);
      todayStart.setHours(0, 0, 0, 0);
      const todayEnd = new Date(today);
      todayEnd.setHours(23, 59, 59, 999);

      const stepSample = createKingstinctSample(5000, todayStart, todayEnd);
      const calorieSample = createKingstinctSample(250, todayStart, todayEnd);

      mockKingstinctHealthKit.queryQuantitySamples
        .mockResolvedValueOnce([stepSample])
        .mockResolvedValueOnce([calorieSample]);

      const result = await getTodayActivitySummary();

      expect(result.steps).toBe(5000);
      expect(result.activeCalories).toBe(250);
    });

    it('should return zeros when no data available', async () => {
      mockKingstinctHealthKit.queryQuantitySamples
        .mockResolvedValueOnce([])
        .mockResolvedValueOnce([]);

      const result = await getTodayActivitySummary();

      expect(result.steps).toBe(0);
      expect(result.activeCalories).toBe(0);
    });

    it('should handle HealthKit errors gracefully', async () => {
      mockKingstinctHealthKit.queryQuantitySamples.mockRejectedValue(new Error('HealthKit error'));

      const result = await getTodayActivitySummary();

      expect(result.steps).toBe(0);
      expect(result.activeCalories).toBe(0);
    });
  });
});
