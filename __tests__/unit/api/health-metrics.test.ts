/**
 * Unit tests for Health Metrics API Client
 */

import api from '@/lib/api/client';
import { healthMetricsApi } from '@/lib/api/health-metrics';
import {
  HealthMetric,
  HealthMetricStats,
  TimeSeriesDataPoint,
  CreateHealthMetricInput,
} from '@/lib/types/health-metrics';

// Mock the API client
jest.mock('@/lib/api/client');

const mockedApi = api as jest.Mocked<typeof api>;

// Test fixtures
const mockHealthMetric: HealthMetric = {
  id: 'metric-1',
  userId: 'user-1',
  metricType: 'RESTING_HEART_RATE',
  value: 62,
  unit: 'bpm',
  recordedAt: '2024-01-15T08:00:00Z',
  source: 'manual',
  createdAt: '2024-01-15T08:00:00Z',
  updatedAt: '2024-01-15T08:00:00Z',
};

const mockHealthMetric2: HealthMetric = {
  id: 'metric-2',
  userId: 'user-1',
  metricType: 'HEART_RATE_VARIABILITY_SDNN',
  value: 45,
  unit: 'ms',
  recordedAt: '2024-01-15T09:00:00Z',
  source: 'apple_health',
  createdAt: '2024-01-15T09:00:00Z',
  updatedAt: '2024-01-15T09:00:00Z',
};

const mockStats: HealthMetricStats = {
  average: 65,
  min: 58,
  max: 72,
  count: 30,
  trend: 'down',
  percentChange: -3.5,
};

const mockTimeSeries: TimeSeriesDataPoint[] = [
  { date: '2024-01-10', value: 64 },
  { date: '2024-01-11', value: 62 },
  { date: '2024-01-12', value: 65 },
  { date: '2024-01-13', value: 61 },
  { date: '2024-01-14', value: 63 },
  { date: '2024-01-15', value: 62 },
];

describe('healthMetricsApi', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('create()', () => {
    const createInput: CreateHealthMetricInput = {
      metricType: 'RESTING_HEART_RATE',
      value: 62,
      unit: 'bpm',
      recordedAt: '2024-01-15T08:00:00Z',
      source: 'manual',
    };

    it('should successfully create a health metric', async () => {
      // Arrange
      mockedApi.post.mockResolvedValueOnce({ data: mockHealthMetric });

      // Act
      const result = await healthMetricsApi.create(createInput);

      // Assert
      expect(result).toEqual(mockHealthMetric);
      expect(mockedApi.post).toHaveBeenCalledWith('/health-metrics', createInput);
      expect(mockedApi.post).toHaveBeenCalledTimes(1);
    });

    it('should send correct payload format to POST /health-metrics', async () => {
      // Arrange
      mockedApi.post.mockResolvedValueOnce({ data: mockHealthMetric });

      // Act
      await healthMetricsApi.create(createInput);

      // Assert
      const [[endpoint, payload]] = mockedApi.post.mock.calls;
      expect(endpoint).toBe('/health-metrics');
      expect(payload).toMatchObject({
        metricType: 'RESTING_HEART_RATE',
        value: 62,
        unit: 'bpm',
        source: 'manual',
      });
    });

    it('should handle validation errors (400)', async () => {
      // Arrange
      const error = {
        response: {
          status: 400,
          data: { error: 'Validation failed' },
        },
      };
      mockedApi.post.mockRejectedValueOnce(error);

      // Act & Assert
      await expect(healthMetricsApi.create(createInput)).rejects.toEqual(error);
    });

    it('should handle auth errors (401)', async () => {
      // Arrange
      const error = {
        response: {
          status: 401,
          data: { error: 'Unauthorized' },
        },
      };
      mockedApi.post.mockRejectedValueOnce(error);

      // Act & Assert
      await expect(healthMetricsApi.create(createInput)).rejects.toEqual(error);
    });
  });

  describe('createBulk()', () => {
    it('should create multiple health metrics', async () => {
      // Arrange
      const metrics: CreateHealthMetricInput[] = [
        { metricType: 'RESTING_HEART_RATE', value: 62, unit: 'bpm', recordedAt: '2024-01-15T08:00:00Z', source: 'manual' },
        { metricType: 'HEART_RATE_VARIABILITY_SDNN', value: 45, unit: 'ms', recordedAt: '2024-01-15T09:00:00Z', source: 'manual' },
      ];
      const response = { created: 2, metrics: [mockHealthMetric, mockHealthMetric2] };
      mockedApi.post.mockResolvedValueOnce({ data: response });

      // Act
      const result = await healthMetricsApi.createBulk(metrics);

      // Assert
      expect(result).toEqual(response);
      expect(mockedApi.post).toHaveBeenCalledWith('/health-metrics/bulk', { metrics });
    });
  });

  describe('getAll()', () => {
    it('should return array of metrics', async () => {
      // Arrange
      const metrics = [mockHealthMetric, mockHealthMetric2];
      mockedApi.get.mockResolvedValueOnce({ data: metrics });

      // Act
      const result = await healthMetricsApi.getAll();

      // Assert
      expect(result).toEqual(metrics);
      expect(mockedApi.get).toHaveBeenCalledWith('/health-metrics', { params: undefined });
    });

    it('should pass query params (metricType, startDate, endDate, source, limit)', async () => {
      // Arrange
      const params = {
        metricType: 'RESTING_HEART_RATE' as const,
        startDate: '2024-01-01',
        endDate: '2024-01-31',
        source: 'manual' as const,
        limit: 10,
      };
      mockedApi.get.mockResolvedValueOnce({ data: [mockHealthMetric] });

      // Act
      await healthMetricsApi.getAll(params);

      // Assert
      expect(mockedApi.get).toHaveBeenCalledWith('/health-metrics', { params });
    });

    it('should handle empty results', async () => {
      // Arrange
      mockedApi.get.mockResolvedValueOnce({ data: [] });

      // Act
      const result = await healthMetricsApi.getAll();

      // Assert
      expect(result).toEqual([]);
    });

    it('should handle network errors', async () => {
      // Arrange
      const error = new Error('Network error');
      mockedApi.get.mockRejectedValueOnce(error);

      // Act & Assert
      await expect(healthMetricsApi.getAll()).rejects.toThrow('Network error');
    });
  });

  describe('getById()', () => {
    it('should return single metric by ID', async () => {
      // Arrange
      mockedApi.get.mockResolvedValueOnce({ data: mockHealthMetric });

      // Act
      const result = await healthMetricsApi.getById('metric-1');

      // Assert
      expect(result).toEqual(mockHealthMetric);
      expect(mockedApi.get).toHaveBeenCalledWith('/health-metrics/metric-1');
    });

    it('should handle 404 not found', async () => {
      // Arrange
      const error = {
        response: {
          status: 404,
          data: { error: 'Not found' },
        },
      };
      mockedApi.get.mockRejectedValueOnce(error);

      // Act & Assert
      await expect(healthMetricsApi.getById('nonexistent')).rejects.toEqual(error);
    });
  });

  describe('getLatest()', () => {
    it('should return latest metric for type', async () => {
      // Arrange
      mockedApi.get.mockResolvedValueOnce({ data: mockHealthMetric });

      // Act
      const result = await healthMetricsApi.getLatest('RESTING_HEART_RATE');

      // Assert
      expect(result).toEqual(mockHealthMetric);
      expect(mockedApi.get).toHaveBeenCalledWith('/health-metrics/latest/RESTING_HEART_RATE');
    });

    it('should return null when no metrics exist (404)', async () => {
      // Arrange
      const error = {
        response: {
          status: 404,
          data: { error: 'No metrics found' },
        },
      };
      mockedApi.get.mockRejectedValueOnce(error);

      // Act
      const result = await healthMetricsApi.getLatest('RESTING_HEART_RATE');

      // Assert
      expect(result).toBeNull();
    });

    it('should throw on non-404 errors', async () => {
      // Arrange
      const error = {
        response: {
          status: 500,
          data: { error: 'Server error' },
        },
      };
      mockedApi.get.mockRejectedValueOnce(error);

      // Act & Assert
      await expect(healthMetricsApi.getLatest('RESTING_HEART_RATE')).rejects.toEqual(error);
    });
  });

  describe('getTimeSeries()', () => {
    it('should return array of data points for chart', async () => {
      // Arrange
      mockedApi.get.mockResolvedValueOnce({ data: mockTimeSeries });

      // Act
      const result = await healthMetricsApi.getTimeSeries('RESTING_HEART_RATE');

      // Assert
      expect(result).toEqual(mockTimeSeries);
      expect(mockedApi.get).toHaveBeenCalledWith(
        '/health-metrics/timeseries/RESTING_HEART_RATE',
        { params: {} }
      );
    });

    it('should pass date range params', async () => {
      // Arrange
      mockedApi.get.mockResolvedValueOnce({ data: mockTimeSeries });
      const startDate = '2024-01-10';
      const endDate = '2024-01-15';

      // Act
      await healthMetricsApi.getTimeSeries('RESTING_HEART_RATE', startDate, endDate);

      // Assert
      expect(mockedApi.get).toHaveBeenCalledWith(
        '/health-metrics/timeseries/RESTING_HEART_RATE',
        { params: { startDate, endDate } }
      );
    });

    it('should handle empty data gracefully', async () => {
      // Arrange
      mockedApi.get.mockResolvedValueOnce({ data: [] });

      // Act
      const result = await healthMetricsApi.getTimeSeries('RESTING_HEART_RATE');

      // Assert
      expect(result).toEqual([]);
    });
  });

  describe('getStats()', () => {
    it('should return stats with average, min, max, trend', async () => {
      // Arrange
      mockedApi.get.mockResolvedValueOnce({ data: mockStats });

      // Act
      const result = await healthMetricsApi.getStats('RESTING_HEART_RATE');

      // Assert
      expect(result).toEqual(mockStats);
      expect(result?.average).toBe(65);
      expect(result?.min).toBe(58);
      expect(result?.max).toBe(72);
      expect(result?.trend).toBe('down');
      expect(result?.percentChange).toBe(-3.5);
    });

    it('should pass days param', async () => {
      // Arrange
      mockedApi.get.mockResolvedValueOnce({ data: mockStats });

      // Act
      await healthMetricsApi.getStats('RESTING_HEART_RATE', 7);

      // Assert
      expect(mockedApi.get).toHaveBeenCalledWith(
        '/health-metrics/stats/RESTING_HEART_RATE',
        { params: { days: 7 } }
      );
    });

    it('should return null when no data (404)', async () => {
      // Arrange
      const error = {
        response: {
          status: 404,
        },
      };
      mockedApi.get.mockRejectedValueOnce(error);

      // Act
      const result = await healthMetricsApi.getStats('RESTING_HEART_RATE');

      // Assert
      expect(result).toBeNull();
    });
  });

  describe('getDailyAverage()', () => {
    it('should return average and count', async () => {
      // Arrange
      const averageResponse = { average: 63, count: 5 };
      mockedApi.get.mockResolvedValueOnce({ data: averageResponse });

      // Act
      const result = await healthMetricsApi.getDailyAverage('RESTING_HEART_RATE');

      // Assert
      expect(result).toEqual(averageResponse);
      expect(mockedApi.get).toHaveBeenCalledWith(
        '/health-metrics/average/daily/RESTING_HEART_RATE',
        { params: {} }
      );
    });

    it('should handle date param for daily', async () => {
      // Arrange
      const averageResponse = { average: 63, count: 5 };
      mockedApi.get.mockResolvedValueOnce({ data: averageResponse });
      const date = '2024-01-15';

      // Act
      await healthMetricsApi.getDailyAverage('RESTING_HEART_RATE', date);

      // Assert
      expect(mockedApi.get).toHaveBeenCalledWith(
        '/health-metrics/average/daily/RESTING_HEART_RATE',
        { params: { date } }
      );
    });

    it('should return null when no data (404)', async () => {
      // Arrange
      const error = { response: { status: 404 } };
      mockedApi.get.mockRejectedValueOnce(error);

      // Act
      const result = await healthMetricsApi.getDailyAverage('RESTING_HEART_RATE');

      // Assert
      expect(result).toBeNull();
    });
  });

  describe('getWeeklyAverage()', () => {
    it('should return average and count', async () => {
      // Arrange
      const averageResponse = { average: 64, count: 21 };
      mockedApi.get.mockResolvedValueOnce({ data: averageResponse });

      // Act
      const result = await healthMetricsApi.getWeeklyAverage('RESTING_HEART_RATE');

      // Assert
      expect(result).toEqual(averageResponse);
      expect(mockedApi.get).toHaveBeenCalledWith(
        '/health-metrics/average/weekly/RESTING_HEART_RATE'
      );
    });

    it('should return null when no data (404)', async () => {
      // Arrange
      const error = { response: { status: 404 } };
      mockedApi.get.mockRejectedValueOnce(error);

      // Act
      const result = await healthMetricsApi.getWeeklyAverage('RESTING_HEART_RATE');

      // Assert
      expect(result).toBeNull();
    });
  });

  describe('delete()', () => {
    it('should successfully delete metric', async () => {
      // Arrange
      mockedApi.delete.mockResolvedValueOnce({ data: { message: 'Deleted' } });

      // Act
      await healthMetricsApi.delete('metric-1');

      // Assert
      expect(mockedApi.delete).toHaveBeenCalledWith('/health-metrics/metric-1');
    });

    it('should handle 404 errors', async () => {
      // Arrange
      const error = {
        response: {
          status: 404,
          data: { error: 'Not found' },
        },
      };
      mockedApi.delete.mockRejectedValueOnce(error);

      // Act & Assert
      await expect(healthMetricsApi.delete('nonexistent')).rejects.toEqual(error);
    });
  });

  describe('getDashboardData()', () => {
    it('should fetch data for all specified metric types', async () => {
      // Arrange
      const metricTypes = ['RESTING_HEART_RATE', 'HEART_RATE_VARIABILITY_SDNN'] as const;

      // Mock getLatest and getStats calls
      mockedApi.get
        .mockResolvedValueOnce({ data: mockHealthMetric }) // getLatest RHR
        .mockResolvedValueOnce({ data: mockStats }) // getStats RHR
        .mockResolvedValueOnce({ data: mockHealthMetric2 }) // getLatest HRV
        .mockResolvedValueOnce({ data: mockStats }); // getStats HRV

      // Act
      const result = await healthMetricsApi.getDashboardData([...metricTypes]);

      // Assert
      expect(result.RESTING_HEART_RATE).toBeDefined();
      expect(result.RESTING_HEART_RATE.latest).toEqual(mockHealthMetric);
      expect(result.RESTING_HEART_RATE.stats).toEqual(mockStats);
      expect(result.HEART_RATE_VARIABILITY_SDNN).toBeDefined();
    });

    it('should handle missing data gracefully', async () => {
      // Arrange
      const metricTypes = ['RESTING_HEART_RATE'] as const;
      const notFoundError = { response: { status: 404 } };

      mockedApi.get
        .mockRejectedValueOnce(notFoundError) // getLatest returns 404
        .mockRejectedValueOnce(notFoundError); // getStats returns 404

      // Act
      const result = await healthMetricsApi.getDashboardData([...metricTypes]);

      // Assert
      expect(result.RESTING_HEART_RATE.latest).toBeNull();
      expect(result.RESTING_HEART_RATE.stats).toBeNull();
    });
  });
});
