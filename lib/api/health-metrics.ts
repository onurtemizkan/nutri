/**
 * Health Metrics API Client
 * Unified API functions for health metrics including HealthKit sync and UI operations
 */

import api from './client';
import {
  HealthMetric,
  CreateHealthMetricInput,
  UpdateHealthMetricInput,
  HealthMetricStats,
  TimeSeriesDataPoint,
  HealthMetricType,
  HealthMetricSource,
  AverageResponse,
} from '../types/health-metrics';

// Re-export for HealthKit integration compatibility
import {
  ProcessedHealthMetric,
  BulkHealthMetricsRequest,
  BulkHealthMetricsResponse,
} from '@/lib/types/healthkit';

export type { HealthMetric, CreateHealthMetricInput, UpdateHealthMetricInput, HealthMetricStats, TimeSeriesDataPoint };

/**
 * Query parameters for fetching health metrics
 */
export interface GetHealthMetricsParams {
  metricType?: HealthMetricType;
  startDate?: string;
  endDate?: string;
  source?: HealthMetricSource | string;
  limit?: number;
  offset?: number;
}

/**
 * Health Metrics API client
 */
export const healthMetricsApi = {
  /**
   * Create a new health metric
   * POST /health-metrics
   */
  async create(data: CreateHealthMetricInput | ProcessedHealthMetric): Promise<HealthMetric> {
    const response = await api.post<HealthMetric>('/health-metrics', data);
    return response.data;
  },

  /**
   * Create multiple health metrics in bulk
   * The server handles deduplication via upsert
   * POST /health-metrics/bulk
   */
  async createBulk(
    metrics: (CreateHealthMetricInput | ProcessedHealthMetric)[]
  ): Promise<BulkHealthMetricsResponse> {
    const request: BulkHealthMetricsRequest = { metrics: metrics as ProcessedHealthMetric[] };
    const response = await api.post('/health-metrics/bulk', request);
    return response.data;
  },

  /**
   * Get all health metrics with optional filters
   * GET /health-metrics
   */
  async getAll(params?: GetHealthMetricsParams): Promise<HealthMetric[]> {
    const response = await api.get('/health-metrics', { params });
    return response.data.healthMetrics || response.data;
  },

  /**
   * Get a health metric by ID
   * GET /health-metrics/:id
   */
  async getById(id: string): Promise<HealthMetric> {
    const response = await api.get<HealthMetric>(`/health-metrics/${id}`);
    return response.data;
  },

  /**
   * Get the latest value for a specific metric type
   * GET /health-metrics/latest/:metricType
   */
  async getLatest(metricType: HealthMetricType): Promise<HealthMetric | null> {
    try {
      const response = await api.get<HealthMetric>(`/health-metrics/latest/${metricType}`);
      return response.data;
    } catch (error) {
      // Return null if no metrics found (404)
      if (isNotFoundError(error)) {
        return null;
      }
      throw error;
    }
  },

  /**
   * Get time series data for charts
   * GET /health-metrics/timeseries/:metricType
   */
  async getTimeSeries(
    metricType: HealthMetricType,
    startDate?: string,
    endDate?: string,
    limit?: number
  ): Promise<TimeSeriesDataPoint[]> {
    const params: Record<string, string | number> = {};
    if (startDate) params.startDate = startDate;
    if (endDate) params.endDate = endDate;
    if (limit) params.limit = limit;

    const response = await api.get(`/health-metrics/timeseries/${metricType}`, { params });
    return response.data.data || response.data;
  },

  /**
   * Get statistics (avg, min, max, trend) for a metric type
   * GET /health-metrics/stats/:metricType
   */
  async getStats(metricType: HealthMetricType, days?: number): Promise<HealthMetricStats | null> {
    try {
      const params: Record<string, number> = {};
      if (days) params.days = days;

      const response = await api.get<HealthMetricStats>(
        `/health-metrics/stats/${metricType}`,
        { params }
      );
      return response.data;
    } catch (error) {
      // Return null if no metrics found (404)
      if (isNotFoundError(error)) {
        return null;
      }
      throw error;
    }
  },

  /**
   * Get daily average for a metric type
   * GET /health-metrics/average/daily/:metricType
   */
  async getDailyAverage(metricType: HealthMetricType, date?: string): Promise<AverageResponse | null> {
    try {
      const params: Record<string, string> = {};
      if (date) params.date = date;

      const response = await api.get<AverageResponse>(
        `/health-metrics/average/daily/${metricType}`,
        { params }
      );
      return response.data;
    } catch (error) {
      // Return null if no metrics found (404)
      if (isNotFoundError(error)) {
        return null;
      }
      throw error;
    }
  },

  /**
   * Get weekly average for a metric type
   * GET /health-metrics/average/weekly/:metricType
   */
  async getWeeklyAverage(metricType: HealthMetricType): Promise<AverageResponse | null> {
    try {
      const response = await api.get<AverageResponse>(
        `/health-metrics/average/weekly/${metricType}`
      );
      return response.data;
    } catch (error) {
      // Return null if no metrics found (404)
      if (isNotFoundError(error)) {
        return null;
      }
      throw error;
    }
  },

  /**
   * Update a health metric
   * PUT /health-metrics/:id
   */
  async update(id: string, data: UpdateHealthMetricInput): Promise<HealthMetric> {
    const response = await api.put<HealthMetric>(`/health-metrics/${id}`, data);
    return response.data;
  },

  /**
   * Delete a health metric
   * DELETE /health-metrics/:id
   */
  async delete(id: string): Promise<void> {
    await api.delete(`/health-metrics/${id}`);
  },

  /**
   * Get recent metrics by type (for displaying individual entries)
   * GET /health-metrics?metricType=X&limit=Y
   */
  async getRecentByType(metricType: HealthMetricType, limit: number = 20): Promise<HealthMetric[]> {
    const response = await api.get('/health-metrics', {
      params: {
        metricType,
        limit,
      },
    });
    return response.data.healthMetrics || response.data;
  },

  /**
   * Get dashboard data for key metrics (RHR, HRV, Sleep, Recovery)
   * Fetches latest values and stats for dashboard display
   * @param metricTypes Array of metric types to fetch
   * @param days Number of days for stats calculation (default: 30)
   */
  async getDashboardData(metricTypes: HealthMetricType[], days: number = 30): Promise<
    Record<HealthMetricType, { latest: HealthMetric | null; stats: HealthMetricStats | null }>
  > {
    const results: Record<HealthMetricType, { latest: HealthMetric | null; stats: HealthMetricStats | null }> = {} as Record<HealthMetricType, { latest: HealthMetric | null; stats: HealthMetricStats | null }>;

    // Fetch all metrics in parallel
    await Promise.all(
      metricTypes.map(async (metricType) => {
        const [latest, stats] = await Promise.all([
          this.getLatest(metricType),
          this.getStats(metricType, days),
        ]);
        results[metricType] = { latest, stats };
      })
    );

    return results;
  },
};

/**
 * Helper to check if an error is a 404 Not Found
 */
function isNotFoundError(error: unknown): boolean {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { status?: number } }).response;
    return response?.status === 404;
  }
  return false;
}

/**
 * Chunk array into smaller arrays
 * Used for batch uploads with API limits
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
 * Used primarily by HealthKit sync
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

export default healthMetricsApi;
