/**
 * Cardiovascular Sync Tests
 */

import {
  mockHealthKit,
  createMockRestingHeartRateSample,
  createMockHRVSample,
  resetMocks,
  setupDefaultMocks,
  mockPlatform,
} from './test-utils';
import {
  fetchRestingHeartRate,
  fetchHeartRateVariability,
  syncCardiovascularMetrics,
  calculateDailyAverageHeartRate,
} from '@/lib/services/healthkit/cardiovascular';

describe('Cardiovascular Sync', () => {
  beforeEach(() => {
    resetMocks();
    setupDefaultMocks();
    mockPlatform.OS = 'ios';
  });

  afterEach(() => {
    mockPlatform.OS = 'ios';
  });

  describe('fetchRestingHeartRate', () => {
    it('should return empty array on non-iOS platforms', async () => {
      mockPlatform.OS = 'android';

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchRestingHeartRate(options);
      expect(result).toEqual([]);
    });

    it('should transform HealthKit samples to ProcessedHealthMetric format', async () => {
      const mockSamples = [
        createMockRestingHeartRateSample(58, new Date('2024-01-01T08:00:00Z')),
        createMockRestingHeartRateSample(62, new Date('2024-01-02T08:00:00Z')),
      ];

      mockHealthKit.getRestingHeartRateSamples.mockImplementation((options, callback) =>
        callback(null, mockSamples)
      );

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchRestingHeartRate(options);

      expect(result).toHaveLength(2);
      expect(result[0]).toMatchObject({
        metricType: 'RESTING_HEART_RATE',
        value: 58,
        unit: 'bpm',
        source: 'apple_health',
      });
      expect(result[1]).toMatchObject({
        metricType: 'RESTING_HEART_RATE',
        value: 62,
        unit: 'bpm',
        source: 'apple_health',
      });
    });

    it('should return empty array on HealthKit error', async () => {
      mockHealthKit.getRestingHeartRateSamples.mockImplementation((options, callback) =>
        callback('HealthKit error', null)
      );

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchRestingHeartRate(options);
      expect(result).toEqual([]);
    });
  });

  describe('fetchHeartRateVariability', () => {
    it('should transform HRV samples correctly', async () => {
      const mockSamples = [
        createMockHRVSample(45, new Date('2024-01-01T08:00:00Z')),
        createMockHRVSample(52, new Date('2024-01-02T08:00:00Z')),
      ];

      mockHealthKit.getHeartRateVariabilitySamples.mockImplementation((options, callback) =>
        callback(null, mockSamples)
      );

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchHeartRateVariability(options);

      expect(result).toHaveLength(2);
      expect(result[0]).toMatchObject({
        metricType: 'HEART_RATE_VARIABILITY_SDNN',
        value: 45,
        unit: 'ms',
        source: 'apple_health',
      });
    });
  });

  describe('syncCardiovascularMetrics', () => {
    it('should combine RHR and HRV samples', async () => {
      const rhrSamples = [
        createMockRestingHeartRateSample(58, new Date('2024-01-01T08:00:00Z')),
      ];
      const hrvSamples = [createMockHRVSample(45, new Date('2024-01-01T09:00:00Z'))];

      mockHealthKit.getRestingHeartRateSamples.mockImplementation((options, callback) =>
        callback(null, rhrSamples)
      );
      mockHealthKit.getHeartRateVariabilitySamples.mockImplementation((options, callback) =>
        callback(null, hrvSamples)
      );

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await syncCardiovascularMetrics(options);

      expect(result).toHaveLength(2);
      // Should include both RHR and HRV
      const metricTypes = result.map((m) => m.metricType);
      expect(metricTypes).toContain('RESTING_HEART_RATE');
      expect(metricTypes).toContain('HEART_RATE_VARIABILITY_SDNN');
    });

    it('should sort results by recordedAt (newest first)', async () => {
      const rhrSamples = [
        createMockRestingHeartRateSample(58, new Date('2024-01-01T08:00:00Z')),
      ];
      const hrvSamples = [createMockHRVSample(45, new Date('2024-01-02T09:00:00Z'))];

      mockHealthKit.getRestingHeartRateSamples.mockImplementation((options, callback) =>
        callback(null, rhrSamples)
      );
      mockHealthKit.getHeartRateVariabilitySamples.mockImplementation((options, callback) =>
        callback(null, hrvSamples)
      );

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await syncCardiovascularMetrics(options);

      // HRV sample is newer, should be first
      expect(result[0].metricType).toBe('HEART_RATE_VARIABILITY_SDNN');
      expect(result[1].metricType).toBe('RESTING_HEART_RATE');
    });
  });

  describe('calculateDailyAverageHeartRate', () => {
    it('should calculate daily averages correctly', () => {
      const samples = [
        { value: 70, startDate: '2024-01-01T08:00:00Z', endDate: '2024-01-01T08:01:00Z' },
        { value: 80, startDate: '2024-01-01T12:00:00Z', endDate: '2024-01-01T12:01:00Z' },
        { value: 60, startDate: '2024-01-02T08:00:00Z', endDate: '2024-01-02T08:01:00Z' },
      ];

      const result = calculateDailyAverageHeartRate(samples);

      expect(result.get('2024-01-01')).toBe(75); // (70 + 80) / 2
      expect(result.get('2024-01-02')).toBe(60);
    });
  });
});
