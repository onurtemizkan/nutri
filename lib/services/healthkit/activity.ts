/**
 * Activity Metrics Sync
 * Syncs steps, active calories, and other activity data from HealthKit
 * Using @kingstinct/react-native-healthkit
 */

import { Platform } from 'react-native';
import {
  HealthKitSample,
  ProcessedHealthMetric,
  HealthKitSyncOptions,
  METRIC_UNITS,
} from '@/lib/types/healthkit';

/**
 * Fetch step count samples from HealthKit
 */
export async function fetchStepCount(
  options: HealthKitSyncOptions
): Promise<ProcessedHealthMetric[]> {
  if (Platform.OS !== 'ios') {
    return [];
  }

  try {
    const { queryQuantitySamples } = await import('@kingstinct/react-native-healthkit');

    const samples = await queryQuantitySamples('HKQuantityTypeIdentifierStepCount', {
      limit: -1,
      filter: {
        date: {
          startDate: options.startDate,
          endDate: options.endDate,
        },
      },
    });

    // Aggregate steps by day
    const dailySteps = aggregateByDay([...samples]);
    return dailySteps.map((daily) => transformStepsToHealthMetric(daily));
  } catch (error) {
    console.warn('Error fetching step count:', error);
    return [];
  }
}

/**
 * Fetch total step count for a date range (single value)
 */
export async function fetchTotalStepCount(
  options: HealthKitSyncOptions
): Promise<number> {
  if (Platform.OS !== 'ios') {
    return 0;
  }

  try {
    const { queryQuantitySamples } = await import('@kingstinct/react-native-healthkit');

    const samples = await queryQuantitySamples('HKQuantityTypeIdentifierStepCount', {
      limit: -1,
      filter: {
        date: {
          startDate: options.startDate,
          endDate: options.endDate,
        },
      },
    });

    return samples.reduce((total, sample) => total + sample.quantity, 0);
  } catch (error) {
    console.warn('Error fetching total step count:', error);
    return 0;
  }
}

/**
 * Fetch active energy burned (active calories) from HealthKit
 */
export async function fetchActiveCalories(
  options: HealthKitSyncOptions
): Promise<ProcessedHealthMetric[]> {
  if (Platform.OS !== 'ios') {
    return [];
  }

  try {
    const { queryQuantitySamples } = await import('@kingstinct/react-native-healthkit');

    const samples = await queryQuantitySamples('HKQuantityTypeIdentifierActiveEnergyBurned', {
      limit: -1,
      filter: {
        date: {
          startDate: options.startDate,
          endDate: options.endDate,
        },
      },
    });

    // Aggregate by day
    const dailyCalories = aggregateByDay([...samples]);
    return dailyCalories.map((daily) => transformCaloriesToHealthMetric(daily));
  } catch (error) {
    console.warn('Error fetching active calories:', error);
    return [];
  }
}

/**
 * Fetch basal (resting) energy burned from HealthKit
 */
export async function fetchBasalCalories(
  options: HealthKitSyncOptions
): Promise<HealthKitSample[]> {
  if (Platform.OS !== 'ios') {
    return [];
  }

  try {
    const { queryQuantitySamples } = await import('@kingstinct/react-native-healthkit');

    const samples = await queryQuantitySamples('HKQuantityTypeIdentifierBasalEnergyBurned', {
      limit: -1,
      filter: {
        date: {
          startDate: options.startDate,
          endDate: options.endDate,
        },
      },
    });

    return [...samples];
  } catch (error) {
    console.warn('Error fetching basal calories:', error);
    return [];
  }
}

/**
 * Fetch distance walked/run from HealthKit
 */
export async function fetchDistanceWalkingRunning(
  options: HealthKitSyncOptions
): Promise<HealthKitSample[]> {
  if (Platform.OS !== 'ios') {
    return [];
  }

  try {
    const { queryQuantitySamples } = await import('@kingstinct/react-native-healthkit');

    const samples = await queryQuantitySamples('HKQuantityTypeIdentifierDistanceWalkingRunning', {
      limit: -1,
      filter: {
        date: {
          startDate: options.startDate,
          endDate: options.endDate,
        },
      },
    });

    return [...samples];
  } catch (error) {
    console.warn('Error fetching distance:', error);
    return [];
  }
}

/**
 * Aggregate samples by day
 */
function aggregateByDay(samples: HealthKitSample[]): {
  date: string;
  value: number;
  sourceName?: string;
}[] {
  const dailyTotals = new Map<string, { value: number; sourceName?: string }>();

  for (const sample of samples) {
    const date = sample.startDate.toISOString().split('T')[0];
    const sourceName = sample.sourceRevision?.source?.name || sample.device?.name;
    const existing = dailyTotals.get(date) || { value: 0, sourceName };
    existing.value += sample.quantity;
    dailyTotals.set(date, existing);
  }

  return Array.from(dailyTotals.entries()).map(([date, data]) => ({
    date,
    value: Math.round(data.value),
    sourceName: data.sourceName,
  }));
}

/**
 * Transform step count to ProcessedHealthMetric
 */
function transformStepsToHealthMetric(daily: {
  date: string;
  value: number;
  sourceName?: string;
}): ProcessedHealthMetric {
  // Create a date at end of day
  const recordedAt = new Date(daily.date);
  recordedAt.setHours(23, 59, 59, 999);

  return {
    metricType: 'STEPS',
    value: Math.round(daily.value),
    unit: METRIC_UNITS.STEPS,
    recordedAt: recordedAt.toISOString(),
    source: 'apple_health',
    metadata: {
      sourceName: daily.sourceName,
      device: daily.sourceName || 'iPhone/Apple Watch',
      quality: 'high',
    },
  };
}

/**
 * Transform calories to ProcessedHealthMetric
 */
function transformCaloriesToHealthMetric(daily: {
  date: string;
  value: number;
  sourceName?: string;
}): ProcessedHealthMetric {
  // Create a date at end of day
  const recordedAt = new Date(daily.date);
  recordedAt.setHours(23, 59, 59, 999);

  return {
    metricType: 'ACTIVE_CALORIES',
    value: Math.round(daily.value),
    unit: METRIC_UNITS.ACTIVE_CALORIES,
    recordedAt: recordedAt.toISOString(),
    source: 'apple_health',
    metadata: {
      sourceName: daily.sourceName,
      device: daily.sourceName || 'Apple Watch',
      quality: 'high',
    },
  };
}

/**
 * Sync all activity metrics
 */
export async function syncActivityMetrics(
  options: HealthKitSyncOptions
): Promise<ProcessedHealthMetric[]> {
  const [steps, activeCalories] = await Promise.all([
    fetchStepCount(options),
    fetchActiveCalories(options),
  ]);

  // Combine and sort by recordedAt (newest first)
  const allMetrics = [...steps, ...activeCalories].sort(
    (a, b) => new Date(b.recordedAt).getTime() - new Date(a.recordedAt).getTime()
  );

  return allMetrics;
}

/**
 * Get today's activity summary
 */
export async function getTodayActivitySummary(): Promise<{
  steps: number;
  activeCalories: number;
}> {
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const now = new Date();

  const [steps, activeCaloriesMetrics] = await Promise.all([
    fetchTotalStepCount({ startDate: today, endDate: now }),
    fetchActiveCalories({ startDate: today, endDate: now }),
  ]);

  const activeCalories = activeCaloriesMetrics.reduce(
    (sum, metric) => sum + metric.value,
    0
  );

  return {
    steps,
    activeCalories: Math.round(activeCalories),
  };
}
