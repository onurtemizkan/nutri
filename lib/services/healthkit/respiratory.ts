/**
 * Respiratory Metrics Sync
 * Syncs respiratory rate, oxygen saturation, and VO2Max from HealthKit
 * Using @kingstinct/react-native-healthkit
 */

import { Platform } from 'react-native';
import { queryQuantitySamples } from '@kingstinct/react-native-healthkit';
import {
  HealthKitSample,
  ProcessedHealthMetric,
  HealthKitSyncOptions,
  METRIC_UNITS,
  HealthMetricType,
} from '@/lib/types/healthkit';

/**
 * Fetch respiratory rate samples from HealthKit
 */
export async function fetchRespiratoryRate(
  options: HealthKitSyncOptions
): Promise<ProcessedHealthMetric[]> {
  if (Platform.OS !== 'ios') {
    return [];
  }

  try {
    const samples = await queryQuantitySamples('HKQuantityTypeIdentifierRespiratoryRate', {
      limit: -1,
      filter: {
        date: {
          startDate: options.startDate,
          endDate: options.endDate,
        },
      },
    });

    if (!samples || !Array.isArray(samples)) {
      console.warn('queryQuantitySamples returned invalid response for RespiratoryRate');
      return [];
    }

    return samples.map((sample) => transformToHealthMetric(sample, 'RESPIRATORY_RATE'));
  } catch (error) {
    console.warn('Error fetching respiratory rate:', error);
    return [];
  }
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

  try {
    const samples = await queryQuantitySamples('HKQuantityTypeIdentifierOxygenSaturation', {
      limit: -1,
      filter: {
        date: {
          startDate: options.startDate,
          endDate: options.endDate,
        },
      },
    });

    if (!samples || !Array.isArray(samples)) {
      console.warn('queryQuantitySamples returned invalid response for OxygenSaturation');
      return [];
    }

    // SpO2 is returned as a decimal (0.98 for 98%), we convert to percentage
    return samples.map((sample) => {
      const value = sample.quantity > 1 ? sample.quantity : sample.quantity * 100;
      return transformToHealthMetric({ ...sample, quantity: value }, 'OXYGEN_SATURATION');
    });
  } catch (error) {
    console.warn('Error fetching oxygen saturation:', error);
    return [];
  }
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

  try {
    const samples = await queryQuantitySamples('HKQuantityTypeIdentifierVO2Max', {
      limit: 100, // VO2Max is updated infrequently
      filter: {
        date: {
          startDate: options.startDate,
          endDate: options.endDate,
        },
      },
    });

    if (!samples || !Array.isArray(samples)) {
      console.warn('queryQuantitySamples returned invalid response for VO2Max');
      return [];
    }

    return samples.map((sample) => transformToHealthMetric(sample, 'VO2_MAX'));
  } catch (error) {
    console.warn('Error fetching VO2Max:', error);
    return [];
  }
}

/**
 * Transform a HealthKit sample to our ProcessedHealthMetric format
 */
function transformToHealthMetric(
  sample: HealthKitSample,
  metricType: 'RESPIRATORY_RATE' | 'OXYGEN_SATURATION' | 'VO2_MAX'
): ProcessedHealthMetric {
  const sourceName = sample.sourceRevision?.source?.name || sample.device?.name;

  return {
    metricType,
    value: sample.quantity,
    unit: METRIC_UNITS[metricType as HealthMetricType],
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
