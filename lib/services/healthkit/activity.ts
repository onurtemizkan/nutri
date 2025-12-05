/**
 * Activity Metrics Sync
 * Syncs steps, active calories, and other activity data from HealthKit
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
  period?: number;
  includeManuallyAdded?: boolean;
}

/**
 * Daily step count result
 */
interface DailyStepCount {
  value: number;
  startDate: string;
  endDate: string;
}

/**
 * Fetch daily step counts from HealthKit
 */
export async function fetchStepCount(
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
    period: 1440, // Get daily totals (1440 minutes = 1 day)
    includeManuallyAdded: false,
  };

  return new Promise((resolve) => {
    healthKit.getDailyStepCountSamples(queryOptions, (error, results) => {
      if (error) {
        console.warn('Error fetching step count:', error);
        resolve([]);
        return;
      }

      const processed = (results || []).map((sample: DailyStepCount) =>
        transformStepsToHealthMetric(sample)
      );
      resolve(processed);
    });
  });
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

  const healthKit = await getHealthKit();
  if (!healthKit) {
    return 0;
  }

  const queryOptions: HealthKitQueryOptions = {
    startDate: options.startDate.toISOString(),
    endDate: options.endDate.toISOString(),
    includeManuallyAdded: false,
  };

  return new Promise((resolve) => {
    healthKit.getStepCount(queryOptions, (error, results) => {
      if (error) {
        console.warn('Error fetching total step count:', error);
        resolve(0);
        return;
      }
      resolve(results?.value || 0);
    });
  });
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
    healthKit.getActiveEnergyBurned(queryOptions, (error, results) => {
      if (error) {
        console.warn('Error fetching active calories:', error);
        resolve([]);
        return;
      }

      // Aggregate by day
      const dailyCalories = aggregateByDay(results || []);
      const processed = dailyCalories.map((daily) =>
        transformCaloriesToHealthMetric(daily)
      );
      resolve(processed);
    });
  });
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
    healthKit.getBasalEnergyBurned(queryOptions, (error, results) => {
      if (error) {
        console.warn('Error fetching basal calories:', error);
        resolve([]);
        return;
      }
      resolve(results || []);
    });
  });
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
    healthKit.getDailyDistanceWalkingRunningSamples(queryOptions, (error, results) => {
      if (error) {
        console.warn('Error fetching distance:', error);
        resolve([]);
        return;
      }
      resolve(results || []);
    });
  });
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
    const date = new Date(sample.startDate).toISOString().split('T')[0];
    const existing = dailyTotals.get(date) || { value: 0, sourceName: sample.sourceName };
    existing.value += sample.value;
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
function transformStepsToHealthMetric(sample: DailyStepCount): ProcessedHealthMetric {
  return {
    metricType: 'STEPS',
    value: Math.round(sample.value),
    unit: METRIC_UNITS.STEPS,
    recordedAt: sample.startDate,
    source: 'apple_health',
    metadata: {
      device: 'iPhone/Apple Watch',
      quality: 'high',
      endDate: sample.endDate,
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
  // Create a date at noon for the day
  const recordedAt = new Date(daily.date);
  recordedAt.setHours(23, 59, 59, 999);

  return {
    metricType: 'ACTIVE_CALORIES',
    value: daily.value,
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
