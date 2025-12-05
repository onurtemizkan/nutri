/**
 * Respiratory Metrics Sync
 * Syncs respiratory rate, oxygen saturation, and VO2Max from HealthKit
 */

import { Platform } from 'react-native';
import {
  HealthKitSample,
  ProcessedHealthMetric,
  HealthKitSyncOptions,
  METRIC_UNITS,
  HealthMetricType,
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
 * Fetch respiratory rate samples from HealthKit
 */
export async function fetchRespiratoryRate(
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
    healthKit.getRespiratoryRateSamples(queryOptions, (error, results) => {
      if (error) {
        console.warn('Error fetching respiratory rate:', error);
        resolve([]);
        return;
      }

      const processed = (results || []).map((sample: HealthKitSample) =>
        transformToHealthMetric(sample, 'RESPIRATORY_RATE')
      );
      resolve(processed);
    });
  });
}

/**
 * Fetch oxygen saturation (SpO2) samples from HealthKit
 */
export async function fetchOxygenSaturation(
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
    healthKit.getOxygenSaturationSamples(queryOptions, (error, results) => {
      if (error) {
        console.warn('Error fetching oxygen saturation:', error);
        resolve([]);
        return;
      }

      // SpO2 is returned as a decimal (0.98 for 98%), we convert to percentage
      const processed = (results || []).map((sample: HealthKitSample) => {
        const value = sample.value > 1 ? sample.value : sample.value * 100;
        return transformToHealthMetric({ ...sample, value }, 'OXYGEN_SATURATION');
      });
      resolve(processed);
    });
  });
}

/**
 * Fetch VO2Max samples from HealthKit
 * Note: VO2Max is calculated less frequently (typically after outdoor workouts)
 */
export async function fetchVo2Max(
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
    limit: 100, // VO2Max is updated infrequently
  };

  return new Promise((resolve) => {
    healthKit.getVo2MaxSamples(queryOptions, (error, results) => {
      if (error) {
        console.warn('Error fetching VO2Max:', error);
        resolve([]);
        return;
      }

      const processed = (results || []).map((sample: HealthKitSample) =>
        transformToHealthMetric(sample, 'VO2_MAX')
      );
      resolve(processed);
    });
  });
}

/**
 * Transform a HealthKit sample to our ProcessedHealthMetric format
 */
function transformToHealthMetric(
  sample: HealthKitSample,
  metricType: 'RESPIRATORY_RATE' | 'OXYGEN_SATURATION' | 'VO2_MAX'
): ProcessedHealthMetric {
  return {
    metricType,
    value: sample.value,
    unit: METRIC_UNITS[metricType as HealthMetricType],
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
 * Sync all respiratory metrics
 */
export async function syncRespiratoryMetrics(
  options: HealthKitSyncOptions
): Promise<ProcessedHealthMetric[]> {
  const [respiratoryRate, oxygenSaturation, vo2Max] = await Promise.all([
    fetchRespiratoryRate(options),
    fetchOxygenSaturation(options),
    fetchVo2Max(options),
  ]);

  // Combine and sort by recordedAt (newest first)
  const allMetrics = [...respiratoryRate, ...oxygenSaturation, ...vo2Max].sort(
    (a, b) => new Date(b.recordedAt).getTime() - new Date(a.recordedAt).getTime()
  );

  return allMetrics;
}
