/**
 * Activity Sync Tests
 */

import { mockHealthKit, resetMocks, setupDefaultMocks, mockPlatform } from './test-utils';
import {
  fetchStepCount,
  fetchActiveCalories,
  syncActivityMetrics,
  getTodayActivitySummary,
} from '@/lib/services/healthkit/activity';

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
        {
          value: 8500,
          startDate: '2024-01-01T00:00:00Z',
          endDate: '2024-01-01T23:59:59Z',
        },
        {
          value: 10200,
          startDate: '2024-01-02T00:00:00Z',
          endDate: '2024-01-02T23:59:59Z',
        },
      ];

      mockHealthKit.getDailyStepCountSamples.mockImplementation((options, callback) =>
        callback(null, mockSamples)
      );

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
      mockHealthKit.getDailyStepCountSamples.mockImplementation((options, callback) =>
        callback('HealthKit error', null)
      );

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
        {
          value: 450,
          startDate: '2024-01-01T00:00:00Z',
          endDate: '2024-01-01T23:59:59Z',
          sourceName: 'Apple Watch',
        },
        {
          value: 380,
          startDate: '2024-01-02T00:00:00Z',
          endDate: '2024-01-02T23:59:59Z',
          sourceName: 'Apple Watch',
        },
      ];

      mockHealthKit.getActiveEnergyBurned.mockImplementation((options, callback) =>
        callback(null, mockSamples)
      );

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
      mockHealthKit.getActiveEnergyBurned.mockImplementation((options, callback) =>
        callback('HealthKit error', null)
      );

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
      const stepSamples = [
        {
          value: 8500,
          startDate: '2024-01-01T00:00:00Z',
          endDate: '2024-01-01T23:59:59Z',
        },
      ];

      const calorieSamples = [
        {
          value: 450,
          startDate: '2024-01-01T00:00:00Z',
          endDate: '2024-01-01T23:59:59Z',
          sourceName: 'Apple Watch',
        },
      ];

      mockHealthKit.getDailyStepCountSamples.mockImplementation((options, callback) =>
        callback(null, stepSamples)
      );
      mockHealthKit.getActiveEnergyBurned.mockImplementation((options, callback) =>
        callback(null, calorieSamples)
      );

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
      const stepSamples = [
        {
          value: 8500,
          startDate: '2024-01-01T00:00:00Z',
          endDate: '2024-01-01T23:59:59Z',
        },
      ];

      const calorieSamples = [
        {
          value: 450,
          startDate: '2024-01-03T00:00:00Z', // Newer
          endDate: '2024-01-03T23:59:59Z',
          sourceName: 'Apple Watch',
        },
      ];

      mockHealthKit.getDailyStepCountSamples.mockImplementation((options, callback) =>
        callback(null, stepSamples)
      );
      mockHealthKit.getActiveEnergyBurned.mockImplementation((options, callback) =>
        callback(null, calorieSamples)
      );

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
      const todayStr = today.toISOString().split('T')[0];

      // getStepCount returns a single value object
      mockHealthKit.getStepCount.mockImplementation(
        (
          options: unknown,
          callback: (err: string | null, result: { value: number } | null) => void
        ) => callback(null, { value: 5000 })
      );

      const calorieSamples = [
        {
          value: 250,
          startDate: `${todayStr}T00:00:00Z`,
          endDate: `${todayStr}T23:59:59Z`,
          sourceName: 'Apple Watch',
        },
      ];

      mockHealthKit.getActiveEnergyBurned.mockImplementation(
        (options: unknown, callback: (err: string | null, data: unknown) => void) =>
          callback(null, calorieSamples)
      );

      const result = await getTodayActivitySummary();

      expect(result.steps).toBe(5000);
      expect(result.activeCalories).toBe(250);
    });

    it('should return zeros when no data available', async () => {
      mockHealthKit.getStepCount.mockImplementation(
        (
          options: unknown,
          callback: (err: string | null, result: { value: number } | null) => void
        ) => callback(null, { value: 0 })
      );
      mockHealthKit.getActiveEnergyBurned.mockImplementation(
        (options: unknown, callback: (err: string | null, data: unknown) => void) =>
          callback(null, [])
      );

      const result = await getTodayActivitySummary();

      expect(result.steps).toBe(0);
      expect(result.activeCalories).toBe(0);
    });

    it('should handle HealthKit errors gracefully', async () => {
      mockHealthKit.getStepCount.mockImplementation(
        (
          options: unknown,
          callback: (err: string | null, result: { value: number } | null) => void
        ) => callback('Error', null)
      );
      mockHealthKit.getActiveEnergyBurned.mockImplementation(
        (options: unknown, callback: (err: string | null, data: unknown) => void) =>
          callback('Error', null)
      );

      const result = await getTodayActivitySummary();

      expect(result.steps).toBe(0);
      expect(result.activeCalories).toBe(0);
    });
  });
});
