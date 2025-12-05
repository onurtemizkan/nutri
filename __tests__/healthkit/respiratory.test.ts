/**
 * Respiratory Sync Tests
 */

import { mockHealthKit, resetMocks, setupDefaultMocks, mockPlatform } from './test-utils';
import {
  fetchRespiratoryRate,
  fetchOxygenSaturation,
  fetchVo2Max,
  syncRespiratoryMetrics,
} from '@/lib/services/healthkit/respiratory';

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
        {
          value: 14,
          startDate: '2024-01-01T08:00:00Z',
          endDate: '2024-01-01T08:00:00Z',
          sourceName: 'Apple Watch',
        },
        {
          value: 16,
          startDate: '2024-01-02T08:00:00Z',
          endDate: '2024-01-02T08:00:00Z',
          sourceName: 'Apple Watch',
        },
      ];

      mockHealthKit.getRespiratoryRateSamples.mockImplementation((options, callback) =>
        callback(null, mockSamples)
      );

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
      mockHealthKit.getRespiratoryRateSamples.mockImplementation((options, callback) =>
        callback('HealthKit error', null)
      );

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
        {
          value: 0.98, // 98% as decimal
          startDate: '2024-01-01T08:00:00Z',
          endDate: '2024-01-01T08:00:00Z',
          sourceName: 'Apple Watch',
        },
        {
          value: 0.96, // 96% as decimal
          startDate: '2024-01-02T08:00:00Z',
          endDate: '2024-01-02T08:00:00Z',
          sourceName: 'Apple Watch',
        },
      ];

      mockHealthKit.getOxygenSaturationSamples.mockImplementation((options, callback) =>
        callback(null, mockSamples)
      );

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
        {
          value: 97, // Already a percentage (some sources provide this)
          startDate: '2024-01-01T08:00:00Z',
          endDate: '2024-01-01T08:00:00Z',
          sourceName: 'Apple Watch',
        },
      ];

      mockHealthKit.getOxygenSaturationSamples.mockImplementation((options, callback) =>
        callback(null, mockSamples)
      );

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
      const mockSamples = [
        {
          value: 42.5,
          startDate: '2024-01-01T08:00:00Z',
          endDate: '2024-01-01T08:00:00Z',
          sourceName: 'Apple Watch',
        },
      ];

      mockHealthKit.getVo2MaxSamples.mockImplementation((options, callback) =>
        callback(null, mockSamples)
      );

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
      const respRateSamples = [
        {
          value: 14,
          startDate: '2024-01-01T08:00:00Z',
          endDate: '2024-01-01T08:00:00Z',
          sourceName: 'Apple Watch',
        },
      ];

      const spo2Samples = [
        {
          value: 0.98,
          startDate: '2024-01-01T09:00:00Z',
          endDate: '2024-01-01T09:00:00Z',
          sourceName: 'Apple Watch',
        },
      ];

      const vo2MaxSamples = [
        {
          value: 42.5,
          startDate: '2024-01-01T10:00:00Z',
          endDate: '2024-01-01T10:00:00Z',
          sourceName: 'Apple Watch',
        },
      ];

      mockHealthKit.getRespiratoryRateSamples.mockImplementation((options, callback) =>
        callback(null, respRateSamples)
      );
      mockHealthKit.getOxygenSaturationSamples.mockImplementation((options, callback) =>
        callback(null, spo2Samples)
      );
      mockHealthKit.getVo2MaxSamples.mockImplementation((options, callback) =>
        callback(null, vo2MaxSamples)
      );

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
      const respRateSamples = [
        {
          value: 14,
          startDate: '2024-01-01T08:00:00Z',
          endDate: '2024-01-01T08:00:00Z',
          sourceName: 'Apple Watch',
        },
      ];

      const vo2MaxSamples = [
        {
          value: 42.5,
          startDate: '2024-01-03T10:00:00Z', // Newer
          endDate: '2024-01-03T10:00:00Z',
          sourceName: 'Apple Watch',
        },
      ];

      mockHealthKit.getRespiratoryRateSamples.mockImplementation((options, callback) =>
        callback(null, respRateSamples)
      );
      mockHealthKit.getOxygenSaturationSamples.mockImplementation((options, callback) =>
        callback(null, [])
      );
      mockHealthKit.getVo2MaxSamples.mockImplementation((options, callback) =>
        callback(null, vo2MaxSamples)
      );

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
