/**
 * HealthKit Sync Orchestration
 * Coordinates syncing all health data from HealthKit to the backend
 */

import * as SecureStore from 'expo-secure-store';
import {
  ProcessedHealthMetric,
  HealthKitSyncOptions,
  SyncResult,
  SyncStatus,
  SYNC_TIMESTAMP_KEYS,
} from '@/lib/types/healthkit';
import { uploadHealthMetricsInBatches } from '@/lib/api/health-metrics';
import { syncCardiovascularMetrics } from './cardiovascular';
import { syncRespiratoryMetrics } from './respiratory';
import { syncSleepMetrics } from './sleep';
import { syncActivityMetrics } from './activity';
import {
  isHealthKitAvailable,
  isHealthKitInitialized,
  requestHealthKitPermissions,
} from './permissions';

/**
 * Default sync period (30 days)
 */
const DEFAULT_SYNC_DAYS = 30;

/**
 * Batch size for API uploads
 */
const BATCH_SIZE = 50;

/**
 * Current sync state
 */
let syncInProgress = false;

/**
 * Get the last sync timestamp for a category
 */
async function getLastSyncTimestamp(key: string): Promise<Date | null> {
  try {
    const timestamp = await SecureStore.getItemAsync(key);
    if (timestamp) {
      return new Date(timestamp);
    }
  } catch {
    // Ignore errors
  }
  return null;
}

/**
 * Store the last sync timestamp for a category
 */
async function setLastSyncTimestamp(key: string, date: Date): Promise<void> {
  try {
    await SecureStore.setItemAsync(key, date.toISOString());
  } catch (error) {
    console.warn('Failed to store sync timestamp:', error);
  }
}

/**
 * Get sync date range
 * Uses last sync timestamp for incremental sync, or default period for initial sync
 */
async function getSyncDateRange(
  timestampKey: string,
  defaultDays: number = DEFAULT_SYNC_DAYS
): Promise<HealthKitSyncOptions> {
  const lastSync = await getLastSyncTimestamp(timestampKey);

  const endDate = new Date();
  let startDate: Date;

  if (lastSync) {
    // Incremental sync: start from last sync minus 1 day (overlap for safety)
    startDate = new Date(lastSync);
    startDate.setDate(startDate.getDate() - 1);
  } else {
    // Initial sync: go back defaultDays
    startDate = new Date();
    startDate.setDate(startDate.getDate() - defaultDays);
  }

  return { startDate, endDate };
}

/**
 * Sync cardiovascular data (RHR, HRV)
 */
export async function syncCardiovascular(
  onProgress?: (message: string) => void
): Promise<SyncResult> {
  const errors: string[] = [];

  try {
    onProgress?.('Fetching cardiovascular data from HealthKit...');
    const options = await getSyncDateRange(SYNC_TIMESTAMP_KEYS.CARDIOVASCULAR);
    const metrics = await syncCardiovascularMetrics(options);

    if (metrics.length === 0) {
      return {
        success: true,
        metricsCount: 0,
        errors: [],
        lastSyncedDate: new Date(),
      };
    }

    onProgress?.(`Uploading ${metrics.length} cardiovascular metrics...`);
    const result = await uploadHealthMetricsInBatches(metrics, BATCH_SIZE);

    if (result.errors.length > 0) {
      errors.push(...result.errors.map((e) => e.error));
    }

    await setLastSyncTimestamp(SYNC_TIMESTAMP_KEYS.CARDIOVASCULAR, new Date());

    return {
      success: result.totalErrors === 0,
      metricsCount: result.totalCreated + result.totalUpdated,
      errors,
      lastSyncedDate: new Date(),
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    errors.push(errorMessage);
    return {
      success: false,
      metricsCount: 0,
      errors,
    };
  }
}

/**
 * Sync respiratory data (respiratory rate, SpO2, VO2Max)
 */
export async function syncRespiratory(
  onProgress?: (message: string) => void
): Promise<SyncResult> {
  const errors: string[] = [];

  try {
    onProgress?.('Fetching respiratory data from HealthKit...');
    const options = await getSyncDateRange(SYNC_TIMESTAMP_KEYS.RESPIRATORY);
    const metrics = await syncRespiratoryMetrics(options);

    if (metrics.length === 0) {
      return {
        success: true,
        metricsCount: 0,
        errors: [],
        lastSyncedDate: new Date(),
      };
    }

    onProgress?.(`Uploading ${metrics.length} respiratory metrics...`);
    const result = await uploadHealthMetricsInBatches(metrics, BATCH_SIZE);

    if (result.errors.length > 0) {
      errors.push(...result.errors.map((e) => e.error));
    }

    await setLastSyncTimestamp(SYNC_TIMESTAMP_KEYS.RESPIRATORY, new Date());

    return {
      success: result.totalErrors === 0,
      metricsCount: result.totalCreated + result.totalUpdated,
      errors,
      lastSyncedDate: new Date(),
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    errors.push(errorMessage);
    return {
      success: false,
      metricsCount: 0,
      errors,
    };
  }
}

/**
 * Sync sleep data
 */
export async function syncSleep(
  onProgress?: (message: string) => void
): Promise<SyncResult> {
  const errors: string[] = [];

  try {
    onProgress?.('Fetching sleep data from HealthKit...');
    const options = await getSyncDateRange(SYNC_TIMESTAMP_KEYS.SLEEP);
    const metrics = await syncSleepMetrics(options);

    if (metrics.length === 0) {
      return {
        success: true,
        metricsCount: 0,
        errors: [],
        lastSyncedDate: new Date(),
      };
    }

    onProgress?.(`Uploading ${metrics.length} sleep metrics...`);
    const result = await uploadHealthMetricsInBatches(metrics, BATCH_SIZE);

    if (result.errors.length > 0) {
      errors.push(...result.errors.map((e) => e.error));
    }

    await setLastSyncTimestamp(SYNC_TIMESTAMP_KEYS.SLEEP, new Date());

    return {
      success: result.totalErrors === 0,
      metricsCount: result.totalCreated + result.totalUpdated,
      errors,
      lastSyncedDate: new Date(),
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    errors.push(errorMessage);
    return {
      success: false,
      metricsCount: 0,
      errors,
    };
  }
}

/**
 * Sync activity data (steps, calories)
 */
export async function syncActivity(
  onProgress?: (message: string) => void
): Promise<SyncResult> {
  const errors: string[] = [];

  try {
    onProgress?.('Fetching activity data from HealthKit...');
    const options = await getSyncDateRange(SYNC_TIMESTAMP_KEYS.ACTIVITY);
    const metrics = await syncActivityMetrics(options);

    if (metrics.length === 0) {
      return {
        success: true,
        metricsCount: 0,
        errors: [],
        lastSyncedDate: new Date(),
      };
    }

    onProgress?.(`Uploading ${metrics.length} activity metrics...`);
    const result = await uploadHealthMetricsInBatches(metrics, BATCH_SIZE);

    if (result.errors.length > 0) {
      errors.push(...result.errors.map((e) => e.error));
    }

    await setLastSyncTimestamp(SYNC_TIMESTAMP_KEYS.ACTIVITY, new Date());

    return {
      success: result.totalErrors === 0,
      metricsCount: result.totalCreated + result.totalUpdated,
      errors,
      lastSyncedDate: new Date(),
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    errors.push(errorMessage);
    return {
      success: false,
      metricsCount: 0,
      errors,
    };
  }
}

/**
 * Sync all health data
 */
export async function syncAllHealthData(
  onProgress?: (message: string, progress: number) => void
): Promise<{
  success: boolean;
  results: {
    cardiovascular: SyncResult;
    respiratory: SyncResult;
    sleep: SyncResult;
    activity: SyncResult;
  };
  totalMetrics: number;
  errors: string[];
}> {
  if (syncInProgress) {
    return {
      success: false,
      results: {
        cardiovascular: { success: false, metricsCount: 0, errors: ['Sync already in progress'] },
        respiratory: { success: false, metricsCount: 0, errors: ['Sync already in progress'] },
        sleep: { success: false, metricsCount: 0, errors: ['Sync already in progress'] },
        activity: { success: false, metricsCount: 0, errors: ['Sync already in progress'] },
      },
      totalMetrics: 0,
      errors: ['Sync already in progress'],
    };
  }

  syncInProgress = true;

  try {
    // Check if HealthKit is available and initialized
    const isAvailable = await isHealthKitAvailable();
    if (!isAvailable) {
      throw new Error('HealthKit is not available on this device');
    }

    const isInitialized = await isHealthKitInitialized();
    if (!isInitialized) {
      // Try to initialize
      const permGranted = await requestHealthKitPermissions();
      if (!permGranted) {
        throw new Error('HealthKit permissions not granted');
      }
    }

    // Sync each category
    onProgress?.('Syncing cardiovascular data...', 0);
    const cardiovascular = await syncCardiovascular((msg) =>
      onProgress?.(msg, 10)
    );

    onProgress?.('Syncing respiratory data...', 25);
    const respiratory = await syncRespiratory((msg) =>
      onProgress?.(msg, 35)
    );

    onProgress?.('Syncing sleep data...', 50);
    const sleep = await syncSleep((msg) =>
      onProgress?.(msg, 60)
    );

    onProgress?.('Syncing activity data...', 75);
    const activity = await syncActivity((msg) =>
      onProgress?.(msg, 85)
    );

    onProgress?.('Sync complete!', 100);

    const totalMetrics =
      cardiovascular.metricsCount +
      respiratory.metricsCount +
      sleep.metricsCount +
      activity.metricsCount;

    const allErrors = [
      ...cardiovascular.errors,
      ...respiratory.errors,
      ...sleep.errors,
      ...activity.errors,
    ];

    const success =
      cardiovascular.success &&
      respiratory.success &&
      sleep.success &&
      activity.success;

    return {
      success,
      results: {
        cardiovascular,
        respiratory,
        sleep,
        activity,
      },
      totalMetrics,
      errors: allErrors,
    };
  } finally {
    syncInProgress = false;
  }
}

/**
 * Get the current sync status
 */
export async function getSyncStatus(): Promise<SyncStatus> {
  const isAvailable = await isHealthKitAvailable();
  const isAuthorized = await isHealthKitInitialized();

  const [cardiovascular, respiratory, sleep, activity] = await Promise.all([
    getLastSyncTimestamp(SYNC_TIMESTAMP_KEYS.CARDIOVASCULAR),
    getLastSyncTimestamp(SYNC_TIMESTAMP_KEYS.RESPIRATORY),
    getLastSyncTimestamp(SYNC_TIMESTAMP_KEYS.SLEEP),
    getLastSyncTimestamp(SYNC_TIMESTAMP_KEYS.ACTIVITY),
  ]);

  return {
    isAvailable,
    isAuthorized,
    lastSync: {
      cardiovascular: cardiovascular || undefined,
      respiratory: respiratory || undefined,
      sleep: sleep || undefined,
      activity: activity || undefined,
    },
    syncInProgress,
  };
}

/**
 * Check if a sync is currently in progress
 */
export function isSyncInProgress(): boolean {
  return syncInProgress;
}

/**
 * Force a full sync (ignore last sync timestamps)
 */
export async function forceFullSync(
  days: number = DEFAULT_SYNC_DAYS,
  onProgress?: (message: string, progress: number) => void
): Promise<ReturnType<typeof syncAllHealthData>> {
  // Clear sync timestamps to force full sync
  await Promise.all([
    SecureStore.deleteItemAsync(SYNC_TIMESTAMP_KEYS.CARDIOVASCULAR),
    SecureStore.deleteItemAsync(SYNC_TIMESTAMP_KEYS.RESPIRATORY),
    SecureStore.deleteItemAsync(SYNC_TIMESTAMP_KEYS.SLEEP),
    SecureStore.deleteItemAsync(SYNC_TIMESTAMP_KEYS.ACTIVITY),
  ]);

  return syncAllHealthData(onProgress);
}
