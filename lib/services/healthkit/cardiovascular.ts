/**
 * Cardiovascular Metrics Sync
 * Syncs heart rate, resting heart rate, and HRV data from HealthKit
 */

import { Platform } from 'react-native';
import {
  HealthKitSample,
  ProcessedHealthMetric,
  HealthKitSyncOptions,
  METRIC_UNITS,
} from '@/lib/types/healthkit';
import { getHealthKit } from './permissions';

/**
 * Query options for react-native-health
 */
interface HealthKitQueryOptions {
  startDate: string;
  endDate: string;
  ascending?: boolean;
  limit?: number;
}

/**
 * Fetch resting heart rate samples from HealthKit
 */
export async function fetchRestingHeartRate(
  options: HealthKitSyncOptions
): Promise<ProcessedHealthMetric[]> {
  if (Platform.OS !== 'ios') {
    return [];
  }

  const healthKit = await getHealthKit();
  if (!healthKit) {
    return [];
  }

  const queryOptions: HealthKitQueryOptions = {
    startDate: options.startDate.toISOString(),
    endDate: options.endDate.toISOString(),
    ascending: false,
  };

  return new Promise((resolve) => {
    healthKit.getRestingHeartRateSamples(queryOptions, (error, results) => {
      if (error) {
        console.warn('Error fetching resting heart rate:', error);
        resolve([]);
        return;
      }

      const processed = (results || []).map((sample: HealthKitSample) =>
        transformToHealthMetric(sample, 'RESTING_HEART_RATE')
      );
      resolve(processed);
    });
  });
}

/**
 * Fetch heart rate variability (HRV SDNN) samples from HealthKit
 */
export async function fetchHeartRateVariability(
  options: HealthKitSyncOptions
): Promise<ProcessedHealthMetric[]> {
  if (Platform.OS !== 'ios') {
    return [];
  }

  const healthKit = await getHealthKit();
  if (!healthKit) {
    return [];
  }

  const queryOptions: HealthKitQueryOptions = {
    startDate: options.startDate.toISOString(),
    endDate: options.endDate.toISOString(),
    ascending: false,
  };

  return new Promise((resolve) => {
    healthKit.getHeartRateVariabilitySamples(queryOptions, (error, results) => {
      if (error) {
        console.warn('Error fetching HRV:', error);
        resolve([]);
        return;
      }

      const processed = (results || []).map((sample: HealthKitSample) =>
        transformToHealthMetric(sample, 'HEART_RATE_VARIABILITY_SDNN')
      );
      resolve(processed);
    });
  });
}

/**
 * Fetch instantaneous heart rate samples from HealthKit
 * Note: This returns many samples, we may want to aggregate or limit
 */
export async function fetchHeartRateSamples(
  options: HealthKitSyncOptions
): Promise<HealthKitSample[]> {
  if (Platform.OS !== 'ios') {
    return [];
  }

  const healthKit = await getHealthKit();
  if (!healthKit) {
    return [];
  }

  const queryOptions: HealthKitQueryOptions = {
    startDate: options.startDate.toISOString(),
    endDate: options.endDate.toISOString(),
    ascending: false,
    limit: 1000, // Limit to avoid memory issues
  };

  return new Promise((resolve) => {
    healthKit.getHeartRateSamples(queryOptions, (error, results) => {
      if (error) {
        console.warn('Error fetching heart rate samples:', error);
        resolve([]);
        return;
      }
      resolve(results || []);
    });
  });
}

/**
 * Transform a HealthKit sample to our ProcessedHealthMetric format
 */
function transformToHealthMetric(
  sample: HealthKitSample,
  metricType: 'RESTING_HEART_RATE' | 'HEART_RATE_VARIABILITY_SDNN'
): ProcessedHealthMetric {
  return {
    metricType,
    value: sample.value,
    unit: METRIC_UNITS[metricType],
    recordedAt: sample.startDate,
    source: 'apple_health',
    sourceId: sample.id,
    metadata: {
      sourceName: sample.sourceName,
      device: sample.sourceName || 'Apple Watch',
      quality: 'high',
      endDate: sample.endDate,
    },
  };
}

/**
 * Sync all cardiovascular metrics
 */
export async function syncCardiovascularMetrics(
  options: HealthKitSyncOptions
): Promise<ProcessedHealthMetric[]> {
  const [restingHeartRate, hrvSamples] = await Promise.all([
    fetchRestingHeartRate(options),
    fetchHeartRateVariability(options),
  ]);

  // Combine and sort by recordedAt (newest first)
  const allMetrics = [...restingHeartRate, ...hrvSamples].sort(
    (a, b) => new Date(b.recordedAt).getTime() - new Date(a.recordedAt).getTime()
  );

  return allMetrics;
}

/**
 * Get daily average heart rate from samples
 * Useful for aggregating instantaneous HR readings
 */
export function calculateDailyAverageHeartRate(
  samples: HealthKitSample[]
): Map<string, number> {
  const dailyTotals = new Map<string, { sum: number; count: number }>();

  for (const sample of samples) {
    const date = new Date(sample.startDate).toISOString().split('T')[0];
    const existing = dailyTotals.get(date) || { sum: 0, count: 0 };
    existing.sum += sample.value;
    existing.count += 1;
    dailyTotals.set(date, existing);
  }

  const dailyAverages = new Map<string, number>();
  for (const [date, { sum, count }] of dailyTotals) {
    dailyAverages.set(date, Math.round(sum / count));
  }

  return dailyAverages;
}
