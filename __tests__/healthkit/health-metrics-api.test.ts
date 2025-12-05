/**
 * Health Metrics API Tests
 */

import { chunkArray, uploadHealthMetricsInBatches } from '@/lib/api/health-metrics';
import { ProcessedHealthMetric } from '@/lib/types/healthkit';
import api from '@/lib/api/client';

// Mock axios
jest.mock('@/lib/api/client', () => ({
  __esModule: true,
  default: {
    get: jest.fn(),
    post: jest.fn(),
    delete: jest.fn(),
  },
}));

const mockApi = api as jest.Mocked<typeof api>;

describe('Health Metrics API', () => {
  describe('chunkArray', () => {
    it('should split array into chunks of specified size', () => {
      const array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const chunks = chunkArray(array, 3);

      expect(chunks).toHaveLength(4);
      expect(chunks[0]).toEqual([1, 2, 3]);
      expect(chunks[1]).toEqual([4, 5, 6]);
      expect(chunks[2]).toEqual([7, 8, 9]);
      expect(chunks[3]).toEqual([10]);
    });

    it('should return single chunk for small arrays', () => {
      const array = [1, 2, 3];
      const chunks = chunkArray(array, 10);

      expect(chunks).toHaveLength(1);
      expect(chunks[0]).toEqual([1, 2, 3]);
    });

    it('should return empty array for empty input', () => {
      const chunks = chunkArray([], 10);
      expect(chunks).toEqual([]);
    });

    it('should handle exact multiple sizes', () => {
      const array = [1, 2, 3, 4, 5, 6];
      const chunks = chunkArray(array, 3);

      expect(chunks).toHaveLength(2);
      expect(chunks[0]).toEqual([1, 2, 3]);
      expect(chunks[1]).toEqual([4, 5, 6]);
    });
  });

  describe('uploadHealthMetricsInBatches', () => {
    beforeEach(() => {
      jest.clearAllMocks();
    });

    it('should upload metrics in batches', async () => {
      const metrics: ProcessedHealthMetric[] = [
        {
          metricType: 'RESTING_HEART_RATE',
          value: 60,
          unit: 'bpm',
          recordedAt: '2024-01-01T00:00:00Z',
          source: 'apple_health',
        },
        {
          metricType: 'STEPS',
          value: 8000,
          unit: 'steps',
          recordedAt: '2024-01-01T00:00:00Z',
          source: 'apple_health',
        },
      ];

      mockApi.post.mockResolvedValue({
        data: { created: 2, updated: 0, errors: [] },
      });

      const result = await uploadHealthMetricsInBatches(metrics, 50);

      expect(mockApi.post).toHaveBeenCalledTimes(1);
      expect(mockApi.post).toHaveBeenCalledWith('/health-metrics/bulk', {
        metrics: metrics,
      });
      expect(result.totalCreated).toBe(2);
      expect(result.totalUpdated).toBe(0);
      expect(result.totalErrors).toBe(0);
    });

    it('should split large arrays into multiple batches', async () => {
      // Create 75 metrics to require 2 batches of 50
      const metrics: ProcessedHealthMetric[] = Array.from({ length: 75 }, (_, i) => ({
        metricType: 'STEPS' as const,
        value: 1000 + i,
        unit: 'steps',
        recordedAt: `2024-01-${String((i % 30) + 1).padStart(2, '0')}T00:00:00Z`,
        source: 'apple_health' as const,
      }));

      mockApi.post.mockResolvedValue({
        data: { created: 50, updated: 0, errors: [] },
      });

      const progressCallback = jest.fn();
      await uploadHealthMetricsInBatches(metrics, 50, progressCallback);

      expect(mockApi.post).toHaveBeenCalledTimes(2);
      expect(progressCallback).toHaveBeenCalledTimes(2);
      expect(progressCallback).toHaveBeenNthCalledWith(1, 1, 2);
      expect(progressCallback).toHaveBeenNthCalledWith(2, 2, 2);
    });

    it('should handle API errors gracefully', async () => {
      const metrics: ProcessedHealthMetric[] = [
        {
          metricType: 'RESTING_HEART_RATE',
          value: 60,
          unit: 'bpm',
          recordedAt: '2024-01-01T00:00:00Z',
          source: 'apple_health',
        },
      ];

      mockApi.post.mockRejectedValue(new Error('Network error'));

      const result = await uploadHealthMetricsInBatches(metrics, 50);

      expect(result.totalErrors).toBe(1);
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0]).toMatchObject({
        batch: 0,
        error: 'Network error',
      });
    });

    it('should aggregate results across batches', async () => {
      const metrics: ProcessedHealthMetric[] = Array.from({ length: 100 }, (_, i) => ({
        metricType: 'STEPS' as const,
        value: 1000 + i,
        unit: 'steps',
        recordedAt: `2024-01-${String((i % 30) + 1).padStart(2, '0')}T00:00:00Z`,
        source: 'apple_health' as const,
      }));

      mockApi.post
        .mockResolvedValueOnce({ data: { created: 40, updated: 10, errors: [] } })
        .mockResolvedValueOnce({ data: { created: 30, updated: 20, errors: [] } });

      const result = await uploadHealthMetricsInBatches(metrics, 50);

      expect(result.totalCreated).toBe(70); // 40 + 30
      expect(result.totalUpdated).toBe(30); // 10 + 20
      expect(result.totalErrors).toBe(0);
    });

    it('should track partial failures', async () => {
      const metrics: ProcessedHealthMetric[] = Array.from({ length: 100 }, (_, i) => ({
        metricType: 'STEPS' as const,
        value: 1000 + i,
        unit: 'steps',
        recordedAt: `2024-01-${String((i % 30) + 1).padStart(2, '0')}T00:00:00Z`,
        source: 'apple_health' as const,
      }));

      mockApi.post
        .mockResolvedValueOnce({
          data: {
            created: 45,
            updated: 0,
            errors: [
              { index: 3, error: 'Invalid value' },
              { index: 7, error: 'Duplicate entry' },
            ],
          },
        })
        .mockResolvedValueOnce({ data: { created: 50, updated: 0, errors: [] } });

      const result = await uploadHealthMetricsInBatches(metrics, 50);

      expect(result.totalCreated).toBe(95);
      expect(result.totalErrors).toBe(2);
      expect(result.errors).toHaveLength(1); // One batch had errors
    });
  });
});
