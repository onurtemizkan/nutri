/**
 * Sync Orchestration Tests
 */

import { mockSecureStore, resetMocks, mockPlatform } from './test-utils';

// Mock the permissions module
const mockIsHealthKitAvailable = jest.fn().mockResolvedValue(true);
const mockIsHealthKitInitialized = jest.fn().mockResolvedValue(true);
const mockRequestHealthKitPermissions = jest.fn().mockResolvedValue(true);

jest.mock('@/lib/services/healthkit/permissions', () => ({
  isHealthKitAvailable: () => mockIsHealthKitAvailable(),
  isHealthKitInitialized: () => mockIsHealthKitInitialized(),
  requestHealthKitPermissions: () => mockRequestHealthKitPermissions(),
}));

// Mock the category sync functions
const mockSyncCardiovascularMetrics = jest.fn().mockResolvedValue([]);
const mockSyncRespiratoryMetrics = jest.fn().mockResolvedValue([]);
const mockSyncSleepMetrics = jest.fn().mockResolvedValue([]);
const mockSyncActivityMetrics = jest.fn().mockResolvedValue([]);

jest.mock('@/lib/services/healthkit/cardiovascular', () => ({
  syncCardiovascularMetrics: () => mockSyncCardiovascularMetrics(),
}));

jest.mock('@/lib/services/healthkit/respiratory', () => ({
  syncRespiratoryMetrics: () => mockSyncRespiratoryMetrics(),
}));

jest.mock('@/lib/services/healthkit/sleep', () => ({
  syncSleepMetrics: () => mockSyncSleepMetrics(),
}));

jest.mock('@/lib/services/healthkit/activity', () => ({
  syncActivityMetrics: () => mockSyncActivityMetrics(),
}));

// Mock the health-metrics API
const mockUploadHealthMetricsInBatches = jest.fn().mockResolvedValue({
  totalCreated: 0,
  totalUpdated: 0,
  totalErrors: 0,
  errors: [],
  batches: [],
});

jest.mock('@/lib/api/health-metrics', () => ({
  uploadHealthMetricsInBatches: (...args: unknown[]) => mockUploadHealthMetricsInBatches(...args),
}));

// Import after mocks are set up
import {
  syncAllHealthData,
  syncCardiovascular,
  syncRespiratory,
  syncSleep,
  syncActivity,
  forceFullSync,
  getSyncStatus,
} from '@/lib/services/healthkit/sync';

describe('Sync Orchestration', () => {
  beforeEach(() => {
    resetMocks();
    mockPlatform.OS = 'ios';
    jest.clearAllMocks();

    // Reset permission mocks
    mockIsHealthKitAvailable.mockResolvedValue(true);
    mockIsHealthKitInitialized.mockResolvedValue(true);
    mockRequestHealthKitPermissions.mockResolvedValue(true);

    // Reset category sync mocks
    mockSyncCardiovascularMetrics.mockResolvedValue([]);
    mockSyncRespiratoryMetrics.mockResolvedValue([]);
    mockSyncSleepMetrics.mockResolvedValue([]);
    mockSyncActivityMetrics.mockResolvedValue([]);

    // Reset upload mock
    mockUploadHealthMetricsInBatches.mockResolvedValue({
      totalCreated: 0,
      totalUpdated: 0,
      totalErrors: 0,
      errors: [],
      batches: [],
    });

    // Reset SecureStore mocks
    mockSecureStore.getItemAsync.mockResolvedValue(null);
    mockSecureStore.setItemAsync.mockResolvedValue(undefined);
    mockSecureStore.deleteItemAsync.mockResolvedValue(undefined);
  });

  afterEach(() => {
    mockPlatform.OS = 'ios';
  });

  describe('syncCardiovascular', () => {
    it('should sync cardiovascular metrics successfully', async () => {
      const mockMetrics = [
        {
          metricType: 'RESTING_HEART_RATE',
          value: 58,
          unit: 'bpm',
          source: 'apple_health',
          recordedAt: '2024-01-01T08:00:00Z',
        },
      ];

      mockSyncCardiovascularMetrics.mockResolvedValue(mockMetrics);
      mockUploadHealthMetricsInBatches.mockResolvedValue({
        totalCreated: 1,
        totalUpdated: 0,
        totalErrors: 0,
        errors: [],
        batches: [{ created: 1, updated: 0, errors: [] }],
      });

      const result = await syncCardiovascular();

      expect(result.success).toBe(true);
      expect(result.metricsCount).toBe(1);
      expect(result.errors).toEqual([]);
      expect(mockUploadHealthMetricsInBatches).toHaveBeenCalledWith(mockMetrics, 50);
    });

    it('should save sync timestamp on success', async () => {
      mockSyncCardiovascularMetrics.mockResolvedValue([
        {
          metricType: 'RESTING_HEART_RATE',
          value: 58,
          unit: 'bpm',
          source: 'apple_health',
          recordedAt: '2024-01-01T08:00:00Z',
        },
      ]);

      await syncCardiovascular();

      expect(mockSecureStore.setItemAsync).toHaveBeenCalledWith(
        'healthkit_last_sync_cardiovascular',
        expect.any(String)
      );
    });

    it('should use last sync timestamp for incremental sync', async () => {
      const lastSync = new Date('2024-01-01T00:00:00Z').toISOString();
      mockSecureStore.getItemAsync.mockResolvedValue(lastSync);

      await syncCardiovascular();

      expect(mockSecureStore.getItemAsync).toHaveBeenCalledWith(
        'healthkit_last_sync_cardiovascular'
      );
    });

    it('should return empty metrics when no data', async () => {
      mockSyncCardiovascularMetrics.mockResolvedValue([]);

      const result = await syncCardiovascular();

      expect(result.success).toBe(true);
      expect(result.metricsCount).toBe(0);
      expect(mockUploadHealthMetricsInBatches).not.toHaveBeenCalled();
    });

    it('should handle upload errors', async () => {
      mockSyncCardiovascularMetrics.mockResolvedValue([
        {
          metricType: 'RESTING_HEART_RATE',
          value: 58,
          unit: 'bpm',
          source: 'apple_health',
          recordedAt: '2024-01-01T08:00:00Z',
        },
      ]);

      mockUploadHealthMetricsInBatches.mockResolvedValue({
        totalCreated: 0,
        totalUpdated: 0,
        totalErrors: 1,
        errors: [{ index: 0, error: 'Failed to upload' }],
        batches: [],
      });

      const result = await syncCardiovascular();

      expect(result.success).toBe(false);
      expect(result.errors).toContain('Failed to upload');
    });
  });

  describe('syncRespiratory', () => {
    it('should sync respiratory metrics successfully', async () => {
      mockSyncRespiratoryMetrics.mockResolvedValue([
        {
          metricType: 'RESPIRATORY_RATE',
          value: 14,
          unit: 'breaths/min',
          source: 'apple_health',
          recordedAt: '2024-01-01T08:00:00Z',
        },
      ]);

      mockUploadHealthMetricsInBatches.mockResolvedValue({
        totalCreated: 1,
        totalUpdated: 0,
        totalErrors: 0,
        errors: [],
        batches: [],
      });

      const result = await syncRespiratory();

      expect(result.success).toBe(true);
      expect(result.metricsCount).toBe(1);
    });
  });

  describe('syncSleep', () => {
    it('should sync sleep metrics successfully', async () => {
      mockSyncSleepMetrics.mockResolvedValue([
        {
          metricType: 'SLEEP_DURATION',
          value: 8,
          unit: 'hours',
          source: 'apple_health',
          recordedAt: '2024-01-01T08:00:00Z',
        },
      ]);

      mockUploadHealthMetricsInBatches.mockResolvedValue({
        totalCreated: 1,
        totalUpdated: 0,
        totalErrors: 0,
        errors: [],
        batches: [],
      });

      const result = await syncSleep();

      expect(result.success).toBe(true);
    });
  });

  describe('syncActivity', () => {
    it('should sync activity metrics successfully', async () => {
      mockSyncActivityMetrics.mockResolvedValue([
        {
          metricType: 'STEPS',
          value: 8500,
          unit: 'steps',
          source: 'apple_health',
          recordedAt: '2024-01-01T08:00:00Z',
        },
      ]);

      mockUploadHealthMetricsInBatches.mockResolvedValue({
        totalCreated: 1,
        totalUpdated: 0,
        totalErrors: 0,
        errors: [],
        batches: [],
      });

      const result = await syncActivity();

      expect(result.success).toBe(true);
      expect(result.metricsCount).toBe(1);
    });
  });

  describe('syncAllHealthData', () => {
    it('should sync all metric categories', async () => {
      mockSyncCardiovascularMetrics.mockResolvedValue([
        {
          metricType: 'RESTING_HEART_RATE',
          value: 58,
          unit: 'bpm',
          source: 'apple_health',
          recordedAt: '2024-01-01T08:00:00Z',
        },
      ]);

      mockUploadHealthMetricsInBatches.mockResolvedValue({
        totalCreated: 1,
        totalUpdated: 0,
        totalErrors: 0,
        errors: [],
        batches: [],
      });

      const result = await syncAllHealthData();

      expect(result.success).toBe(true);
      expect(result.results).toBeDefined();
      expect(result.results.cardiovascular).toBeDefined();
      expect(result.results.respiratory).toBeDefined();
      expect(result.results.sleep).toBeDefined();
      expect(result.results.activity).toBeDefined();
    });

    it('should call progress callback for each category', async () => {
      const progressCallback = jest.fn();
      await syncAllHealthData(progressCallback);

      // Should be called multiple times for progress updates
      expect(progressCallback).toHaveBeenCalled();
      expect(progressCallback.mock.calls.length).toBeGreaterThan(0);
    });

    it('should handle errors gracefully', async () => {
      // Make one category fail
      mockSyncCardiovascularMetrics.mockRejectedValue(new Error('HealthKit error'));

      const result = await syncAllHealthData();

      // Should still complete (partial success)
      expect(result).toBeDefined();
      expect(result.results).toBeDefined();
      expect(result.results.cardiovascular.success).toBe(false);
      expect(result.results.cardiovascular.errors).toContain('HealthKit error');
    });

    it('should fail if HealthKit is not available', async () => {
      mockIsHealthKitAvailable.mockResolvedValue(false);

      await expect(syncAllHealthData()).rejects.toThrow('HealthKit is not available');
    });

    it('should request permissions if not initialized', async () => {
      mockIsHealthKitInitialized.mockResolvedValue(false);
      mockRequestHealthKitPermissions.mockResolvedValue(true);

      const result = await syncAllHealthData();

      expect(mockRequestHealthKitPermissions).toHaveBeenCalled();
      expect(result).toBeDefined();
    });

    it('should fail if permissions are not granted', async () => {
      mockIsHealthKitInitialized.mockResolvedValue(false);
      mockRequestHealthKitPermissions.mockResolvedValue(false);

      await expect(syncAllHealthData()).rejects.toThrow('HealthKit permissions not granted');
    });

    it('should prevent concurrent syncs', async () => {
      // Start first sync
      const firstSync = syncAllHealthData();

      // Try to start second sync immediately
      const secondSync = syncAllHealthData();

      const results = await Promise.all([firstSync, secondSync]);

      // Second sync should fail with "already in progress"
      expect(results[1].errors).toContain('Sync already in progress');
    });
  });

  describe('forceFullSync', () => {
    it('should clear all sync timestamps and perform full sync', async () => {
      await forceFullSync();

      // Should delete all sync timestamps
      expect(mockSecureStore.deleteItemAsync).toHaveBeenCalledWith(
        'healthkit_last_sync_cardiovascular'
      );
      expect(mockSecureStore.deleteItemAsync).toHaveBeenCalledWith(
        'healthkit_last_sync_respiratory'
      );
      expect(mockSecureStore.deleteItemAsync).toHaveBeenCalledWith('healthkit_last_sync_sleep');
      expect(mockSecureStore.deleteItemAsync).toHaveBeenCalledWith('healthkit_last_sync_activity');
    });

    it('should accept custom days parameter', async () => {
      await forceFullSync(60);

      // Should still work with custom days
      expect(mockSecureStore.deleteItemAsync).toHaveBeenCalledTimes(4);
    });
  });

  describe('getSyncStatus', () => {
    it('should return sync status for all categories', async () => {
      const now = new Date().toISOString();
      mockSecureStore.getItemAsync
        .mockResolvedValueOnce(now) // cardiovascular
        .mockResolvedValueOnce(null) // respiratory (never synced)
        .mockResolvedValueOnce(now) // sleep
        .mockResolvedValueOnce(now); // activity

      const status = await getSyncStatus();

      expect(status.isAvailable).toBe(true);
      expect(status.isAuthorized).toBe(true);
      expect(status.lastSync).toBeDefined();
    });

    it('should reflect HealthKit availability', async () => {
      mockIsHealthKitAvailable.mockResolvedValue(false);
      mockIsHealthKitInitialized.mockResolvedValue(false);

      const status = await getSyncStatus();

      expect(status.isAvailable).toBe(false);
      expect(status.isAuthorized).toBe(false);
    });

    it('should track sync in progress state', async () => {
      const status = await getSyncStatus();
      expect(status.syncInProgress).toBe(false);
    });
  });
});
