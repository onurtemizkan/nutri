/**
 * HealthKit Service
 * Main entry point for HealthKit integration
 *
 * Usage:
 * ```typescript
 * import { healthKitService } from '@/lib/services/healthkit';
 *
 * // Check availability
 * const available = await healthKitService.isAvailable();
 *
 * // Request permissions
 * const result = await healthKitService.requestPermissions();
 *
 * // Sync all data
 * const syncResult = await healthKitService.syncAll((message, progress) => {
 *   console.log(`${message} - ${progress}%`);
 * });
 * ```
 */

// Re-export types
// Import for the convenience service object
import {
  isHealthKitAvailable,
  isHealthKitInitialized,
  requestHealthKitPermissions,
  disconnectHealthKit,
  getHealthKitStatus,
} from './permissions';

import {
  syncAllHealthData,
  syncCardiovascular,
  syncRespiratory,
  syncSleep,
  syncActivity,
  getSyncStatus,
  isSyncInProgress,
  forceFullSync,
} from './sync';

import { getTodayActivitySummary } from './activity';
import { getSleepSummaryForDate } from './sleep';

export * from '@/lib/types/healthkit';

// Re-export permission functions
export {
  isHealthKitAvailable,
  isHealthKitInitialized,
  requestHealthKitPermissions,
  getStoredPermissionStatus,
  disconnectHealthKit,
  getHealthKitStatus,
  ensureHealthKitInitialized,
} from './permissions';

// Re-export sync functions
export {
  syncAllHealthData,
  syncCardiovascular,
  syncRespiratory,
  syncSleep,
  syncActivity,
  getSyncStatus,
  isSyncInProgress,
  forceFullSync,
} from './sync';

// Re-export individual metric functions for granular control
export { syncCardiovascularMetrics, fetchRestingHeartRate, fetchHeartRateVariability } from './cardiovascular';
export { syncRespiratoryMetrics, fetchRespiratoryRate, fetchOxygenSaturation, fetchVo2Max } from './respiratory';
export { syncSleepMetrics, fetchSleepSamples, getSleepSummaryForDate } from './sleep';
export { syncActivityMetrics, fetchStepCount, fetchActiveCalories, getTodayActivitySummary } from './activity';

/**
 * HealthKit Service object
 * Provides a convenient interface for all HealthKit operations
 */
export const healthKitService = {
  // Availability & Permissions
  isAvailable: isHealthKitAvailable,
  isInitialized: isHealthKitInitialized,
  requestPermissions: requestHealthKitPermissions,
  disconnect: disconnectHealthKit,
  getStatus: getHealthKitStatus,

  // Sync Operations
  syncAll: syncAllHealthData,
  syncCardiovascular,
  syncRespiratory,
  syncSleep,
  syncActivity,
  getSyncStatus,
  isSyncInProgress,
  forceFullSync,

  // Quick Access
  getTodayActivity: getTodayActivitySummary,
  getLastNightSleep: async () => {
    const today = new Date();
    return getSleepSummaryForDate(today);
  },

  // Convenience method to connect and sync
  connectAndSync: async (
    onProgress?: (message: string, progress: number) => void
  ) => {
    // Check availability
    const available = await isHealthKitAvailable();
    if (!available) {
      throw new Error('HealthKit is not available on this device');
    }

    // Request permissions
    onProgress?.('Requesting permissions...', 5);
    const permResult = await requestHealthKitPermissions();
    if (!permResult.success) {
      throw new Error(permResult.error || 'Failed to get HealthKit permissions');
    }

    // Sync all data
    return syncAllHealthData(onProgress);
  },
};

export default healthKitService;
