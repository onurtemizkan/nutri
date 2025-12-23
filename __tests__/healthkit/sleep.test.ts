/**
 * Sleep Sync Tests
 * Tests for @kingstinct/react-native-healthkit integration
 */

import { mockKingstinctHealthKit, resetMocks, setupDefaultMocks, mockPlatform } from './test-utils';
import { fetchSleepSamples, syncSleepMetrics } from '@/lib/services/healthkit/sleep';

// Sleep analysis value enum (matches HKCategoryValueSleepAnalysis)
const SleepAnalysisValueEnum = {
  INBED: 0,
  ASLEEP: 1, // asleepUnspecified
  AWAKE: 2,
  CORE: 3, // asleepCore (light sleep)
  DEEP: 4, // asleepDeep
  REM: 5, // asleepREM
} as const;

// Helper to create mock sleep samples in @kingstinct/react-native-healthkit format
function createKingstinctSleepSample(
  value: 'DEEP' | 'REM' | 'CORE' | 'AWAKE' | 'INBED' | 'ASLEEP',
  startDate: Date,
  durationHours: number
) {
  const endDate = new Date(startDate.getTime() + durationHours * 60 * 60 * 1000);
  return {
    value: SleepAnalysisValueEnum[value],
    startDate,
    endDate,
    uuid: `uuid_${Date.now()}_${Math.random()}`,
    device: { name: 'Apple Watch' },
    sourceRevision: { source: { name: 'Apple Watch', bundleIdentifier: 'com.apple.health' } },
  };
}

describe('Sleep Sync', () => {
  beforeEach(() => {
    resetMocks();
    setupDefaultMocks();
    mockPlatform.OS = 'ios';
  });

  afterEach(() => {
    mockPlatform.OS = 'ios';
  });

  describe('fetchSleepSamples', () => {
    it('should return empty array on non-iOS platforms', async () => {
      mockPlatform.OS = 'android';

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-02'),
      };

      const result = await fetchSleepSamples(options);
      expect(result).toEqual([]);
    });

    it('should map HealthKit sleep values correctly', async () => {
      const mockSamples = [
        createKingstinctSleepSample('DEEP', new Date('2024-01-01T01:00:00Z'), 1),
        createKingstinctSleepSample('REM', new Date('2024-01-01T02:00:00Z'), 1),
      ];

      mockKingstinctHealthKit.queryCategorySamples.mockResolvedValue(mockSamples);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-02'),
      };

      const result = await fetchSleepSamples(options);

      expect(result).toHaveLength(2);
      expect(result[0].value).toBe('ASLEEPDEEP');
      expect(result[1].value).toBe('ASLEEPREM');
    });
  });

  describe('syncSleepMetrics', () => {
    it('should return empty array when no sleep data', async () => {
      mockKingstinctHealthKit.queryCategorySamples.mockResolvedValue([]);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-02'),
      };

      const result = await syncSleepMetrics(options);
      expect(result).toEqual([]);
    });

    it('should calculate sleep duration from samples', async () => {
      // Simulate a night of sleep: 11PM - 7AM (8 hours)
      const startTime = new Date('2024-01-01T23:00:00Z');
      const mockSamples = [
        // 2 hours deep sleep
        createKingstinctSleepSample('DEEP', startTime, 2),
        // 2 hours REM
        createKingstinctSleepSample(
          'REM',
          new Date(startTime.getTime() + 2 * 60 * 60 * 1000),
          2
        ),
        // 4 hours core/light sleep
        createKingstinctSleepSample(
          'CORE',
          new Date(startTime.getTime() + 4 * 60 * 60 * 1000),
          4
        ),
      ];

      mockKingstinctHealthKit.queryCategorySamples.mockResolvedValue(mockSamples);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-02'),
      };

      const result = await syncSleepMetrics(options);

      // Should have metrics for SLEEP_DURATION, DEEP_SLEEP_DURATION, REM_SLEEP_DURATION
      const metricTypes = result.map((m) => m.metricType);
      expect(metricTypes).toContain('SLEEP_DURATION');
      expect(metricTypes).toContain('DEEP_SLEEP_DURATION');
      expect(metricTypes).toContain('REM_SLEEP_DURATION');

      // Check sleep duration (8 hours total)
      const sleepDuration = result.find((m) => m.metricType === 'SLEEP_DURATION');
      expect(sleepDuration?.value).toBe(8);

      // Check deep sleep (2 hours)
      const deepSleep = result.find((m) => m.metricType === 'DEEP_SLEEP_DURATION');
      expect(deepSleep?.value).toBe(2);

      // Check REM (2 hours)
      const remSleep = result.find((m) => m.metricType === 'REM_SLEEP_DURATION');
      expect(remSleep?.value).toBe(2);
    });

    it('should calculate sleep efficiency correctly', async () => {
      // 8 hours in bed, 6 hours asleep = 75% efficiency
      const startTime = new Date('2024-01-01T23:00:00Z');
      const mockSamples = [
        // 1 hour in bed (not asleep)
        createKingstinctSleepSample('INBED', startTime, 1),
        // 6 hours asleep
        createKingstinctSleepSample(
          'ASLEEP',
          new Date(startTime.getTime() + 1 * 60 * 60 * 1000),
          6
        ),
        // 1 hour awake
        createKingstinctSleepSample(
          'AWAKE',
          new Date(startTime.getTime() + 7 * 60 * 60 * 1000),
          1
        ),
      ];

      mockKingstinctHealthKit.queryCategorySamples.mockResolvedValue(mockSamples);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-02'),
      };

      const result = await syncSleepMetrics(options);

      // Check sleep efficiency (6/8 = 75%)
      const efficiency = result.find((m) => m.metricType === 'SLEEP_EFFICIENCY');
      expect(efficiency?.value).toBe(75);
    });

    it('should group sleep sessions correctly when there are gaps', async () => {
      // Two separate sleep sessions (nap and night sleep)
      const napTime = new Date('2024-01-01T14:00:00Z'); // 2PM nap
      const nightTime = new Date('2024-01-01T23:00:00Z'); // 11PM night sleep

      const mockSamples = [
        // 1 hour nap
        createKingstinctSleepSample('ASLEEP', napTime, 1),
        // 7 hours night sleep (9 hours later)
        createKingstinctSleepSample('ASLEEP', nightTime, 7),
      ];

      mockKingstinctHealthKit.queryCategorySamples.mockResolvedValue(mockSamples);

      const options = {
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-02'),
      };

      const result = await syncSleepMetrics(options);

      // Should have 2 SLEEP_DURATION entries (one for nap, one for night)
      const sleepDurations = result.filter((m) => m.metricType === 'SLEEP_DURATION');
      expect(sleepDurations).toHaveLength(2);
      expect(sleepDurations[0].value).toBe(7); // Night sleep (sorted newest first)
      expect(sleepDurations[1].value).toBe(1); // Nap
    });
  });
});
