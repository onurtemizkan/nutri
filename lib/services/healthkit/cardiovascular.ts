/**
 * Cardiovascular Metrics Sync
 * Syncs heart rate, resting heart rate, and HRV data from HealthKit
 * Using @kingstinct/react-native-healthkit
 */

import { Platform } from 'react-native';
import { queryQuantitySamples } from '@kingstinct/react-native-healthkit';
import {
  ProcessedHealthMetric,
  HealthKitSyncOptions,
  METRIC_UNITS,
} from '@/lib/types/healthkit';

// Type for QuantitySample from the library
interface QuantitySample {
  readonly quantity: number;
  readonly startDate: Date;
  readonly endDate: Date;
  readonly uuid?: string;
  readonly device?: {
    name?: string;
    manufacturer?: string;
    model?: string;
  };
  readonly sourceRevision?: {
    source?: {
      name: string;
      bundleIdentifier: string;
    };
  };
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

  try {
    const samples = await queryQuantitySamples('HKQuantityTypeIdentifierRestingHeartRate', {
      limit: -1,
      filter: {
        date: {
          startDate: options.startDate,
          endDate: options.endDate,
        },
      },
    });

    // Handle undefined or non-array response
    if (!samples || !Array.isArray(samples)) {
      console.warn('queryQuantitySamples returned invalid response for RestingHeartRate');
      return [];
    }

    return samples.map((sample) => transformToHealthMetric(sample, 'RESTING_HEART_RATE'));
  } catch (error) {
    console.warn('Error fetching resting heart rate:', error);
    return [];
  }
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

  try {
    const samples = await queryQuantitySamples('HKQuantityTypeIdentifierHeartRateVariabilitySDNN', {
      limit: -1,
      filter: {
        date: {
          startDate: options.startDate,
          endDate: options.endDate,
        },
      },
    });

    // Handle undefined or non-array response
    if (!samples || !Array.isArray(samples)) {
      console.warn('queryQuantitySamples returned invalid response for HRV');
      return [];
    }

    // HRV SDNN is returned in seconds, convert to milliseconds
    return samples.map((sample) => {
      const valueInMs = sample.quantity < 1 ? sample.quantity * 1000 : sample.quantity;
      return transformToHealthMetric({ ...sample, quantity: valueInMs }, 'HEART_RATE_VARIABILITY_SDNN');
    });
  } catch (error) {
    console.warn('Error fetching HRV:', error);
    return [];
  }
}

/**
 * Fetch instantaneous heart rate samples from HealthKit
 * Note: This returns many samples, we may want to aggregate or limit
 */
export async function fetchHeartRateSamples(
  options: HealthKitSyncOptions
): Promise<QuantitySample[]> {
  if (Platform.OS !== 'ios') {
    return [];
  }

  try {
    const samples = await queryQuantitySamples('HKQuantityTypeIdentifierHeartRate', {
      limit: 1000, // Limit to avoid memory issues
      filter: {
        date: {
          startDate: options.startDate,
          endDate: options.endDate,
        },
      },
    });

    // Handle undefined or non-array response
    if (!samples || !Array.isArray(samples)) {
      console.warn('queryQuantitySamples returned invalid response for HeartRate');
      return [];
    }

    return [...samples];
  } catch (error) {
    console.warn('Error fetching heart rate samples:', error);
    return [];
  }
}

/**
 * Transform a HealthKit sample to our ProcessedHealthMetric format
 */
function transformToHealthMetric(
  sample: QuantitySample,
  metricType: 'RESTING_HEART_RATE' | 'HEART_RATE_VARIABILITY_SDNN'
): ProcessedHealthMetric {
  const sourceName = sample.sourceRevision?.source?.name || sample.device?.name;

  return {
    metricType,
    value: sample.quantity,
    unit: METRIC_UNITS[metricType],
    recordedAt: sample.startDate.toISOString(),
    source: 'apple_health',
    sourceId: sample.uuid,
    metadata: {
      sourceName,
      device: sourceName || 'Apple Watch',
      quality: 'high',
      endDate: sample.endDate.toISOString(),
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
  samples: QuantitySample[]
): Map<string, number> {
  const dailyTotals = new Map<string, { sum: number; count: number }>();

  for (const sample of samples) {
    const date = sample.startDate.toISOString().split('T')[0];
    const existing = dailyTotals.get(date) || { sum: 0, count: 0 };
    existing.sum += sample.quantity;
    existing.count += 1;
    dailyTotals.set(date, existing);
  }

  const dailyAverages = new Map<string, number>();
  for (const [date, { sum, count }] of dailyTotals) {
    dailyAverages.set(date, Math.round(sum / count));
  }

  return dailyAverages;
}
