/**
 * HealthKit Permission Management
 * Handles HealthKit availability checks and permission requests
 * Using @kingstinct/react-native-healthkit
 */

import { Platform } from 'react-native';
import * as SecureStore from 'expo-secure-store';
import {
  HEALTHKIT_READ_PERMISSIONS,
  PermissionRequestResult,
  SYNC_TIMESTAMP_KEYS,
} from '@/lib/types/healthkit';

/**
 * Storage key for permission status
 */
const PERMISSION_STATUS_KEY = 'healthkit_permission_status';

/**
 * Check if HealthKit is available on this device
 */
export async function isHealthKitAvailable(): Promise<boolean> {
  if (Platform.OS !== 'ios') {
    return false;
  }

  try {
    const { isHealthDataAvailable } = await import('@kingstinct/react-native-healthkit');
    return isHealthDataAvailable();
  } catch (error) {
    console.warn('HealthKit availability check error:', error);
    return false;
  }
}

/**
 * Get the current permission status from storage
 */
export async function getStoredPermissionStatus(): Promise<string | null> {
  if (Platform.OS !== 'ios') {
    return 'not_available';
  }

  try {
    return await SecureStore.getItemAsync(PERMISSION_STATUS_KEY);
  } catch {
    return null;
  }
}

/**
 * Store permission status
 */
async function setStoredPermissionStatus(status: string): Promise<void> {
  try {
    await SecureStore.setItemAsync(PERMISSION_STATUS_KEY, status);
  } catch (error) {
    console.warn('Failed to store permission status:', error);
  }
}

/**
 * Request HealthKit permissions for reading health data
 */
export async function requestHealthKitPermissions(): Promise<PermissionRequestResult> {
  if (Platform.OS !== 'ios') {
    return {
      success: false,
      granted: {},
      denied: [],
      error: 'HealthKit is only available on iOS',
    };
  }

  try {
    const { isHealthDataAvailable, requestAuthorization } = await import(
      '@kingstinct/react-native-healthkit'
    );

    const isAvailable = isHealthDataAvailable();
    if (!isAvailable) {
      return {
        success: false,
        granted: {},
        denied: [],
        error: 'HealthKit is not available on this device',
      };
    }

    // Request authorization for all read permissions
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const authorized = await requestAuthorization({
      toRead: HEALTHKIT_READ_PERMISSIONS as any,
    });

    if (authorized) {
      await setStoredPermissionStatus('granted');
      return {
        success: true,
        granted: {
          heartRate: true,
          restingHeartRate: true,
          heartRateVariability: true,
          respiratoryRate: true,
          oxygenSaturation: true,
          vo2Max: true,
          sleepAnalysis: true,
          stepCount: true,
          activeEnergy: true,
        },
        denied: [],
      };
    } else {
      await setStoredPermissionStatus('denied');
      return {
        success: false,
        granted: {},
        denied: HEALTHKIT_READ_PERMISSIONS,
        error: 'User denied HealthKit permissions',
      };
    }
  } catch (error) {
    console.warn('HealthKit permission request error:', error);
    await setStoredPermissionStatus('error');
    return {
      success: false,
      granted: {},
      denied: HEALTHKIT_READ_PERMISSIONS,
      error: error instanceof Error ? error.message : 'Failed to request HealthKit permissions',
    };
  }
}

/**
 * Check if HealthKit is initialized and ready to use
 */
export async function isHealthKitInitialized(): Promise<boolean> {
  const isAvailable = await isHealthKitAvailable();
  if (!isAvailable) {
    return false;
  }

  const status = await getStoredPermissionStatus();
  return status === 'granted';
}

/**
 * Initialize HealthKit if not already initialized
 */
export async function ensureHealthKitInitialized(): Promise<boolean> {
  const isInitialized = await isHealthKitInitialized();
  if (isInitialized) {
    // Re-request to ensure connection is active
    const result = await requestHealthKitPermissions();
    return result.success;
  }
  return false;
}

/**
 * Disconnect from HealthKit (clear stored permission status)
 */
export async function disconnectHealthKit(): Promise<void> {
  try {
    await SecureStore.deleteItemAsync(PERMISSION_STATUS_KEY);
  } catch {
    // Ignore errors
  }

  // Clear sync timestamps
  for (const key of Object.values(SYNC_TIMESTAMP_KEYS)) {
    try {
      await SecureStore.deleteItemAsync(key);
    } catch {
      // Ignore errors when clearing timestamps
    }
  }
}

/**
 * Get HealthKit authorization status summary
 */
export async function getHealthKitStatus(): Promise<{
  available: boolean;
  authorized: boolean;
  platform: string;
}> {
  const available = await isHealthKitAvailable();
  const status = await getStoredPermissionStatus();

  return {
    available,
    authorized: status === 'granted',
    platform: Platform.OS,
  };
}
