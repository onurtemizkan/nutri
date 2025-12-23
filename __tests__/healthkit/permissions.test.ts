/**
 * HealthKit Permissions Tests
 */

import {
  mockHealthKit,
  mockKingstinctHealthKit,
  mockSecureStore,
  resetMocks,
  setupDefaultMocks,
  mockPlatform,
} from './test-utils';
import {
  isHealthKitAvailable,
  requestHealthKitPermissions,
  getStoredPermissionStatus,
  disconnectHealthKit,
} from '@/lib/services/healthkit/permissions';

describe('HealthKit Permissions', () => {
  beforeEach(() => {
    resetMocks();
    setupDefaultMocks();
    mockPlatform.OS = 'ios';
  });

  afterEach(() => {
    mockPlatform.OS = 'ios';
  });

  describe('isHealthKitAvailable', () => {
    it('should return false on non-iOS platforms', async () => {
      mockPlatform.OS = 'android';

      const result = await isHealthKitAvailable();
      expect(result).toBe(false);
    });

    it('should return true when HealthKit is available on iOS', async () => {
      // Uses the @kingstinct/react-native-healthkit API
      mockKingstinctHealthKit.isHealthDataAvailable.mockReturnValue(true);

      const result = await isHealthKitAvailable();
      expect(result).toBe(true);
    });

    it('should return false when HealthKit is not available', async () => {
      mockKingstinctHealthKit.isHealthDataAvailable.mockReturnValue(false);

      const result = await isHealthKitAvailable();
      expect(result).toBe(false);
    });

    it('should return false on HealthKit error', async () => {
      mockKingstinctHealthKit.isHealthDataAvailable.mockImplementation(() => {
        throw new Error('HealthKit not available');
      });

      const result = await isHealthKitAvailable();
      expect(result).toBe(false);
    });
  });

  describe('requestHealthKitPermissions', () => {
    it('should return error result on non-iOS platforms', async () => {
      mockPlatform.OS = 'android';

      const result = await requestHealthKitPermissions();
      expect(result.success).toBe(false);
      expect(result.error).toBe('HealthKit is only available on iOS');
    });

    it('should request permissions and save status on success', async () => {
      // Uses the @kingstinct/react-native-healthkit API
      mockKingstinctHealthKit.isHealthDataAvailable.mockReturnValue(true);
      mockKingstinctHealthKit.requestAuthorization.mockResolvedValue(true);

      const result = await requestHealthKitPermissions();

      expect(result.success).toBe(true);
      expect(result.granted).toHaveProperty('heartRate', true);
      expect(result.granted).toHaveProperty('restingHeartRate', true);
      expect(mockKingstinctHealthKit.requestAuthorization).toHaveBeenCalled();
      expect(mockSecureStore.setItemAsync).toHaveBeenCalledWith(
        'healthkit_permission_status',
        'granted'
      );
    });

    it('should return error result when HealthKit is not available', async () => {
      mockKingstinctHealthKit.isHealthDataAvailable.mockReturnValue(false);

      const result = await requestHealthKitPermissions();

      expect(result.success).toBe(false);
      expect(result.error).toBe('HealthKit is not available on this device');
    });

    it('should return error result on permission denial', async () => {
      mockKingstinctHealthKit.isHealthDataAvailable.mockReturnValue(true);
      mockKingstinctHealthKit.requestAuthorization.mockResolvedValue(false);

      const result = await requestHealthKitPermissions();

      expect(result.success).toBe(false);
      expect(result.error).toBe('User denied HealthKit permissions');
    });

    it('should call requestAuthorization with read permissions', async () => {
      mockKingstinctHealthKit.isHealthDataAvailable.mockReturnValue(true);
      mockKingstinctHealthKit.requestAuthorization.mockResolvedValue(true);

      await requestHealthKitPermissions();

      // Verify requestAuthorization was called
      expect(mockKingstinctHealthKit.requestAuthorization).toHaveBeenCalled();
    });
  });

  describe('getStoredPermissionStatus', () => {
    it('should return stored permission status', async () => {
      mockSecureStore.getItemAsync.mockResolvedValue('granted');

      const status = await getStoredPermissionStatus();
      expect(status).toBe('granted');
    });

    it('should return null when no status is stored', async () => {
      mockSecureStore.getItemAsync.mockResolvedValue(null);

      const status = await getStoredPermissionStatus();
      expect(status).toBeNull();
    });

    it('should return not_available on non-iOS', async () => {
      mockPlatform.OS = 'android';

      const status = await getStoredPermissionStatus();
      expect(status).toBe('not_available');
    });
  });

  describe('disconnectHealthKit', () => {
    it('should clear all HealthKit related data', async () => {
      await disconnectHealthKit();

      // Should clear permission status
      expect(mockSecureStore.deleteItemAsync).toHaveBeenCalledWith('healthkit_permission_status');

      // Should clear all sync timestamps
      expect(mockSecureStore.deleteItemAsync).toHaveBeenCalledWith(
        'healthkit_last_sync_cardiovascular'
      );
      expect(mockSecureStore.deleteItemAsync).toHaveBeenCalledWith(
        'healthkit_last_sync_respiratory'
      );
      expect(mockSecureStore.deleteItemAsync).toHaveBeenCalledWith('healthkit_last_sync_sleep');
      expect(mockSecureStore.deleteItemAsync).toHaveBeenCalledWith('healthkit_last_sync_activity');
    });
  });
});
