/**
 * Respiratory Sync Tests
 * Tests for @kingstinct/react-native-healthkit integration
 */

import { mockKingstinctHealthKit, resetMocks, setupDefaultMocks, mockPlatform } from './test-utils';
import {
  fetchRespiratoryRate,
  fetchOxygenSaturation,
  fetchVo2Max,
  syncRespiratoryMetrics,
} from '@/lib/services/healthkit/respiratory';

// Helper to create mock samples in @kingstinct/react-native-healthkit format
function createKingstinctSample(quantity: number, date: Date) {
  return {
    quantity,
    startDate: date,
    endDate: date,
    uuid: `uuid_${Date.now()}_${Math.random()}`,
    device: { name: 'Apple Watch' },
    sourceRevision: { source: { name: 'Apple Watch', bundleIdentifier: 'com.apple.health' } },
  };
}

describe('Respiratory Sync', () => {
  beforeEach(() => {
    resetMocks();
    setupDefaultMocks();
    mockPlatform.OS = 'ios';
  });

  afterEach(() => {
    mockPlatform.OS = 'ios';
  });

  describe('fetchRespiratoryRate', () => {
    it('should return empty array on non-iOS platforms', async () => {
      mockPlatform.OS = 'android';

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchRespiratoryRate(options);
      expect(result).toEqual([]);
    });

    it('should transform respiratory rate samples correctly', async () => {
      const mockSamples = [
        createKingstinctSample(14, new Date('2024-01-01T08:00:00Z')),
        createKingstinctSample(16, new Date('2024-01-02T08:00:00Z')),
      ];

      mockKingstinctHealthKit.queryQuantitySamples.mockResolvedValue(mockSamples);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchRespiratoryRate(options);

      expect(result).toHaveLength(2);
      expect(result[0]).toMatchObject({
        metricType: 'RESPIRATORY_RATE',
        value: 14,
        unit: 'breaths/min',
        source: 'apple_health',
      });
    });

    it('should return empty array on HealthKit error', async () => {
      mockKingstinctHealthKit.queryQuantitySamples.mockRejectedValue(new Error('HealthKit error'));

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchRespiratoryRate(options);
      expect(result).toEqual([]);
    });
  });

  describe('fetchOxygenSaturation', () => {
    it('should convert SpO2 from decimal to percentage', async () => {
      const mockSamples = [
        createKingstinctSample(0.98, new Date('2024-01-01T08:00:00Z')), // 98% as decimal
        createKingstinctSample(0.96, new Date('2024-01-02T08:00:00Z')), // 96% as decimal
      ];

      mockKingstinctHealthKit.queryQuantitySamples.mockResolvedValue(mockSamples);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchOxygenSaturation(options);

      expect(result).toHaveLength(2);
      expect(result[0]).toMatchObject({
        metricType: 'OXYGEN_SATURATION',
        value: 98, // Converted to percentage
        unit: '%',
        source: 'apple_health',
      });
      expect(result[1].value).toBe(96);
    });

    it('should handle already percentage values', async () => {
      const mockSamples = [
        createKingstinctSample(97, new Date('2024-01-01T08:00:00Z')), // Already a percentage
      ];

      mockKingstinctHealthKit.queryQuantitySamples.mockResolvedValue(mockSamples);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchOxygenSaturation(options);

      // If value > 1, it should be kept as is (already percentage)
      expect(result[0].value).toBe(97);
    });
  });

  describe('fetchVo2Max', () => {
    it('should transform VO2 max samples correctly', async () => {
      const mockSamples = [createKingstinctSample(42.5, new Date('2024-01-01T08:00:00Z'))];

      mockKingstinctHealthKit.queryQuantitySamples.mockResolvedValue(mockSamples);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await fetchVo2Max(options);

      expect(result).toHaveLength(1);
      expect(result[0]).toMatchObject({
        metricType: 'VO2_MAX',
        value: 42.5,
        unit: 'mL/kg/min',
        source: 'apple_health',
      });
    });
  });

  describe('syncRespiratoryMetrics', () => {
    it('should combine all respiratory metrics', async () => {
      const respRateSample = createKingstinctSample(14, new Date('2024-01-01T08:00:00Z'));
      const spo2Sample = createKingstinctSample(0.98, new Date('2024-01-01T09:00:00Z'));
      const vo2MaxSample = createKingstinctSample(42.5, new Date('2024-01-01T10:00:00Z'));

      // Three calls: respRate, spo2, vo2Max
      mockKingstinctHealthKit.queryQuantitySamples
        .mockResolvedValueOnce([respRateSample])
        .mockResolvedValueOnce([spo2Sample])
        .mockResolvedValueOnce([vo2MaxSample]);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await syncRespiratoryMetrics(options);

      expect(result).toHaveLength(3);
      const metricTypes = result.map((m) => m.metricType);
      expect(metricTypes).toContain('RESPIRATORY_RATE');
      expect(metricTypes).toContain('OXYGEN_SATURATION');
      expect(metricTypes).toContain('VO2_MAX');
    });

    it('should sort results by recordedAt (newest first)', async () => {
      const respRateSample = createKingstinctSample(14, new Date('2024-01-01T08:00:00Z'));
      const vo2MaxSample = createKingstinctSample(42.5, new Date('2024-01-03T10:00:00Z')); // Newer

      mockKingstinctHealthKit.queryQuantitySamples
        .mockResolvedValueOnce([respRateSample])
        .mockResolvedValueOnce([])
        .mockResolvedValueOnce([vo2MaxSample]);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-07'),
      };

      const result = await syncRespiratoryMetrics(options);

      // VO2 Max is newer, should be first
      expect(result[0].metricType).toBe('VO2_MAX');
      expect(result[1].metricType).toBe('RESPIRATORY_RATE');
    });
  });
});
