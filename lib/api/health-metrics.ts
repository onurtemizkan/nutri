/**
 * Health Metrics API Client
 * Interfaces with the backend health metrics endpoints
 */

import api from './client';
import {
  ProcessedHealthMetric,
  BulkHealthMetricsRequest,
  BulkHealthMetricsResponse,
  HealthMetricType,
} from '@/lib/types/healthkit';

/**
 * Health metric from the API
 */
export interface HealthMetric {
  id: string;
  userId: string;
  metricType: HealthMetricType;
  value: number;
  unit: string;
  recordedAt: string;
  source: string;
  sourceId?: string;
  metadata?: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

/**
 * Query parameters for fetching health metrics
 */
export interface GetHealthMetricsParams {
  metricType?: HealthMetricType;
  startDate?: string;
  endDate?: string;
  source?: string;
  limit?: number;
  offset?: number;
}

/**
 * Metric statistics response
 */
export interface MetricStats {
  count: number;
  min: number;
  max: number;
  avg: number;
  latest: number;
  latestDate: string;
}

/**
 * Time series data point
 */
export interface TimeSeriesPoint {
  recordedAt: string;
  value: number;
}

/**
 * Health Metrics API
 */
export const healthMetricsApi = {
  /**
   * Get all health metrics for the user with optional filters
   */
  getAll: async (params?: GetHealthMetricsParams): Promise<HealthMetric[]> => {
    const response = await api.get('/health-metrics', { params });
    return response.data.healthMetrics || response.data;
  },

  /**
   * Get a single health metric by ID
   */
  getById: async (id: string): Promise<HealthMetric> => {
    const response = await api.get(`/health-metrics/${id}`);
    return response.data;
  },

  /**
   * Create a single health metric
   */
  create: async (data: ProcessedHealthMetric): Promise<HealthMetric> => {
    const response = await api.post('/health-metrics', data);
    return response.data;
  },

  /**
   * Create multiple health metrics in bulk
   * The server handles deduplication via upsert
   */
  createBulk: async (
    metrics: ProcessedHealthMetric[]
  ): Promise<BulkHealthMetricsResponse> => {
    const request: BulkHealthMetricsRequest = { metrics };
    const response = await api.post('/health-metrics/bulk', request);
    return response.data;
  },

  /**
   * Get the latest metric of a specific type
   */
  getLatest: async (metricType: HealthMetricType): Promise<HealthMetric | null> => {
    try {
      const response = await api.get(`/health-metrics/latest/${metricType}`);
      return response.data;
    } catch (error) {
      // Return null if no metric found
      return null;
    }
  },

  /**
   * Get daily average for a metric type
   */
  getDailyAverage: async (
    metricType: HealthMetricType,
    date?: string
  ): Promise<{ average: number; count: number }> => {
    const params = date ? { date } : undefined;
    const response = await api.get(`/health-metrics/average/daily/${metricType}`, {
      params,
    });
    return response.data;
  },

  /**
   * Get weekly average for a metric type
   */
  getWeeklyAverage: async (
    metricType: HealthMetricType
  ): Promise<{ average: number; count: number }> => {
    const response = await api.get(`/health-metrics/average/weekly/${metricType}`);
    return response.data;
  },

  /**
   * Get time series data for a metric
   */
  getTimeSeries: async (
    metricType: HealthMetricType,
    params?: { startDate?: string; endDate?: string; limit?: number }
  ): Promise<TimeSeriesPoint[]> => {
    const response = await api.get(`/health-metrics/timeseries/${metricType}`, {
      params,
    });
    return response.data.data || response.data;
  },

  /**
   * Get statistics for a metric type
   */
  getStats: async (metricType: HealthMetricType): Promise<MetricStats> => {
    const response = await api.get(`/health-metrics/stats/${metricType}`);
    return response.data;
  },

  /**
   * Delete a health metric
   */
  delete: async (id: string): Promise<void> => {
    await api.delete(`/health-metrics/${id}`);
  },
};

/**
 * Chunk array into smaller arrays
 */
export function chunkArray<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
}

/**
 * Upload health metrics in batches
 * Splits large arrays into chunks of 50 for API limits
 */
export async function uploadHealthMetricsInBatches(
  metrics: ProcessedHealthMetric[],
  batchSize: number = 50,
  onProgress?: (completed: number, total: number) => void
): Promise<{
  totalCreated: number;
  totalUpdated: number;
  totalErrors: number;
  errors: { batch: number; error: string }[];
}> {
  const chunks = chunkArray(metrics, batchSize);
  let totalCreated = 0;
  let totalUpdated = 0;
  let totalErrors = 0;
  const errors: { batch: number; error: string }[] = [];

  for (let i = 0; i < chunks.length; i++) {
    try {
      const result = await healthMetricsApi.createBulk(chunks[i]);
      totalCreated += result.created;
      totalUpdated += result.updated;
      totalErrors += result.errors.length;

      if (result.errors.length > 0) {
        errors.push({
          batch: i,
          error: `${result.errors.length} items failed`,
        });
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      errors.push({ batch: i, error: errorMessage });
      totalErrors += chunks[i].length;
    }

    // Report progress
    if (onProgress) {
      onProgress(i + 1, chunks.length);
    }
  }

  return {
    totalCreated,
    totalUpdated,
    totalErrors,
    errors,
  };
}
