/**
 * HealthKit Permission Management
 * Handles HealthKit availability checks and permission requests
 */

import { Platform } from 'react-native';
import * as SecureStore from 'expo-secure-store';
import {
  HEALTHKIT_READ_PERMISSIONS,
  HEALTHKIT_WRITE_PERMISSIONS,
  PermissionRequestResult,
  SYNC_TIMESTAMP_KEYS,
} from '@/lib/types/healthkit';

// Type for react-native-health module
type HealthKitModule = typeof import('react-native-health').default;

// Cached HealthKit instance
let AppleHealthKit: HealthKitModule | null = null;
let loadAttempted = false;

/**
 * Get the HealthKit module
 * Uses require() for better Jest compatibility
 */
function loadHealthKit(): HealthKitModule | null {
  if (Platform.OS !== 'ios') {
    return null;
  }

  if (!loadAttempted) {
    loadAttempted = true;
    try {
      // Use require for Jest compatibility
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const module = require('react-native-health');
      AppleHealthKit = module.default || module;

      // Verify the native module is actually available
      if (!AppleHealthKit || typeof AppleHealthKit.isAvailable !== 'function') {
        console.warn(
          'HealthKit native module not available. Are you running in Expo Go? HealthKit requires a development build.'
        );
        AppleHealthKit = null;
      }
    } catch (error) {
      console.warn('Failed to load react-native-health:', error);
      AppleHealthKit = null;
    }
  }

  return AppleHealthKit;
}

/**
 * Get the HealthKit instance (for internal use)
 */
export function getHealthKit(): HealthKitModule | null {
  return loadHealthKit();
}

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

  const healthKit = getHealthKit();
  if (!healthKit) {
    return false;
  }

  return new Promise((resolve) => {
    healthKit.isAvailable((error, available) => {
      if (error) {
        console.warn('HealthKit availability check error:', error);
        resolve(false);
        return;
      }
      resolve(available);
    });
  });
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

  const healthKit = getHealthKit();
  if (!healthKit) {
    return {
      success: false,
      granted: {},
      denied: [],
      error: 'HealthKit native module not available. Please use a development build instead of Expo Go.',
    };
  }

  const isAvailable = await isHealthKitAvailable();
  if (!isAvailable) {
    return {
      success: false,
      granted: {},
      denied: [],
      error: 'HealthKit is not available on this device',
    };
  }

  // Build permissions object for react-native-health
  // Cast is needed because our string array doesn't match library's HealthPermission enum type
  const permissions = {
    permissions: {
      read: HEALTHKIT_READ_PERMISSIONS as unknown as string[],
      write: HEALTHKIT_WRITE_PERMISSIONS,
    },
  } as Parameters<typeof healthKit.initHealthKit>[0];

  return new Promise((resolve) => {
    healthKit.initHealthKit(permissions, (error: string | { message?: string } | null) => {
      if (error) {
        console.warn('HealthKit initialization error:', error);
        setStoredPermissionStatus('denied');
        const errorMessage = typeof error === 'string' ? error : error.message || 'Failed to initialize HealthKit';
        resolve({
          success: false,
          granted: {},
          denied: HEALTHKIT_READ_PERMISSIONS as unknown as string[],
          error: errorMessage,
        });
        return;
      }

      setStoredPermissionStatus('granted');
      resolve({
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
      });
    });
  });
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
    // Re-initialize to ensure connection is active
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
