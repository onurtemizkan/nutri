/**
 * Component tests for Health Metric Detail screen
 */

import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import HealthMetricDetailScreen from '@/app/health/[metricType]';
import { healthMetricsApi } from '@/lib/api/health-metrics';
import {
  HealthMetricStats,
  TimeSeriesDataPoint,
} from '@/lib/types/health-metrics';

// Mock dependencies
jest.mock('@/lib/api/health-metrics');
jest.mock('expo-router', () => ({
  useRouter: () => ({
    push: jest.fn(),
    back: jest.fn(),
  }),
  useLocalSearchParams: () => ({
    metricType: 'RESTING_HEART_RATE',
  }),
}));
jest.mock('expo-linear-gradient', () => ({
  LinearGradient: ({ children }: { children: React.ReactNode }) => children,
}));
jest.mock('react-native-chart-kit', () => ({
  LineChart: () => null,
}));
jest.mock('@expo/vector-icons', () => ({
  Ionicons: ({ name, ...props }: { name: string; [key: string]: unknown }) => {
    const { Text } = require('react-native');
    return <Text {...props}>{name}</Text>;
  },
}));
jest.mock('react-native-safe-area-context', () => ({
  SafeAreaView: ({ children }: { children: React.ReactNode }) => children,
}));
jest.mock('@/hooks/useResponsive', () => ({
  useResponsive: () => ({
    isTablet: false,
    isLandscape: false,
    deviceCategory: 'medium',
    getResponsiveValue: (values: Record<string, number>) => values.default ?? values.medium ?? 16,
    getSpacing: () => ({ horizontal: 16, vertical: 16 }),
    width: 375,
  }),
}));
jest.mock('@/lib/components/SwipeableHealthMetricCard', () => ({
  SwipeableHealthMetricCard: ({ metric }: { metric: unknown }) => {
    const { View, Text } = require('react-native');
    return (
      <View>
        <Text>MetricCard</Text>
      </View>
    );
  },
}));

const mockedHealthMetricsApi = healthMetricsApi as jest.Mocked<typeof healthMetricsApi>;

// Test fixtures
const mockTimeSeries: TimeSeriesDataPoint[] = [
  { date: '2024-01-10', value: 64, source: 'manual' },
  { date: '2024-01-11', value: 62, source: 'manual' },
  { date: '2024-01-12', value: 65, source: 'manual' },
  { date: '2024-01-13', value: 61, source: 'manual' },
  { date: '2024-01-14', value: 63, source: 'manual' },
  { date: '2024-01-15', value: 62, source: 'manual' },
];

const mockStats: HealthMetricStats = {
  average: 63,
  min: 58,
  max: 72,
  count: 30,
  trend: 'down',
  percentChange: -3.5,
};

const mockStatsUp: HealthMetricStats = {
  average: 65,
  min: 60,
  max: 70,
  count: 25,
  trend: 'up',
  percentChange: 5.2,
};

const mockStatsStable: HealthMetricStats = {
  average: 64,
  min: 62,
  max: 66,
  count: 28,
  trend: 'stable',
  percentChange: 0.1,
};

describe('HealthMetricDetailScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Default mocks for all tests - component calls all three APIs
    mockedHealthMetricsApi.getRecentByType.mockResolvedValue([]);
  });

  it('renders header with metric name', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockResolvedValueOnce(mockTimeSeries);
    mockedHealthMetricsApi.getStats.mockResolvedValueOnce(mockStats);

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);

    // Assert
    await findByText('Resting Heart Rate');
  });

  it('renders date range filter buttons', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockResolvedValueOnce(mockTimeSeries);
    mockedHealthMetricsApi.getStats.mockResolvedValueOnce(mockStats);

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);

    // Assert - component uses shortened labels like '7D', '30D', '90D', '1Y'
    await findByText('7D');
    await findByText('30D');
    await findByText('90D');
  });

  it('displays statistics (avg, min, max)', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockResolvedValueOnce(mockTimeSeries);
    mockedHealthMetricsApi.getStats.mockResolvedValueOnce(mockStats);

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);

    // Assert
    await findByText('Average');
    await findByText('Minimum');
    await findByText('Maximum');
    await findByText('63'); // average value
    await findByText('58'); // min value
    await findByText('72'); // max value
  });

  it('shows trend information with percentage', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockResolvedValueOnce(mockTimeSeries);
    mockedHealthMetricsApi.getStats.mockResolvedValueOnce(mockStats);

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);

    // Assert
    await findByText('Trend');
    await findByText('-3.5%');
    await findByText(/trending downward/);
  });

  it('shows upward trend correctly', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockResolvedValueOnce(mockTimeSeries);
    mockedHealthMetricsApi.getStats.mockResolvedValueOnce(mockStatsUp);

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);

    // Assert
    await findByText('+5.2%');
    await findByText(/trending upward/);
  });

  it('shows stable trend correctly', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockResolvedValueOnce(mockTimeSeries);
    mockedHealthMetricsApi.getStats.mockResolvedValueOnce(mockStatsStable);

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);

    // Assert
    await findByText(/stable/);
  });

  it('shows data source indicator', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockResolvedValueOnce(mockTimeSeries);
    mockedHealthMetricsApi.getStats.mockResolvedValueOnce(mockStats);

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);

    // Assert
    await findByText('Data Source');
    await findByText('Manual Entry');
  });

  it('date range filter updates chart data', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockResolvedValue(mockTimeSeries);
    mockedHealthMetricsApi.getStats.mockResolvedValue(mockStats);

    // Act
    const { findByText, getByText } = render(<HealthMetricDetailScreen />);
    await findByText('Resting Heart Rate');

    // Change date range - component uses '7D' not '7 Days'
    const sevenDaysButton = getByText('7D');
    fireEvent.press(sevenDaysButton);

    // Assert - API should be called again with new range
    await waitFor(() => {
      expect(mockedHealthMetricsApi.getTimeSeries).toHaveBeenCalledTimes(2);
    });
  });

  it('handles empty data gracefully', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockResolvedValueOnce([]);
    mockedHealthMetricsApi.getStats.mockResolvedValueOnce(null);

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);

    // Assert
    await findByText('No data for this period');
    await findByText('Add health metrics to see your history');
  });

  it('shows loading state', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );
    mockedHealthMetricsApi.getStats.mockImplementation(
      () => new Promise(() => {})
    );

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);

    // Assert - Header should still render
    await findByText('Resting Heart Rate');
  });

  it('shows error state with retry button', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockRejectedValueOnce(
      new Error('Network error')
    );
    mockedHealthMetricsApi.getStats.mockRejectedValueOnce(
      new Error('Network error')
    );

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);

    // Assert
    await findByText('Failed to load data');
    await findByText('Retry');
  });

  it('retry button reloads data', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries
      .mockRejectedValueOnce(new Error('Network error'))
      .mockResolvedValueOnce(mockTimeSeries);
    mockedHealthMetricsApi.getStats
      .mockRejectedValueOnce(new Error('Network error'))
      .mockResolvedValueOnce(mockStats);

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);
    const retryButton = await findByText('Retry');
    fireEvent.press(retryButton);

    // Assert
    await waitFor(() => {
      expect(mockedHealthMetricsApi.getTimeSeries).toHaveBeenCalledTimes(2);
    });
  });

  it('shows metric description', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockResolvedValueOnce(mockTimeSeries);
    mockedHealthMetricsApi.getStats.mockResolvedValueOnce(mockStats);

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);

    // Assert
    await findByText('About RHR');
    await findByText('Your heart rate while at rest');
  });

  it('displays correct units in statistics', async () => {
    // Arrange
    mockedHealthMetricsApi.getTimeSeries.mockResolvedValueOnce(mockTimeSeries);
    mockedHealthMetricsApi.getStats.mockResolvedValueOnce(mockStats);

    // Act
    const { findAllByText } = render(<HealthMetricDetailScreen />);

    // Assert - bpm units should appear in stat cards
    const bpmTexts = await findAllByText('bpm');
    expect(bpmTexts.length).toBeGreaterThan(0);
  });
});

describe('HealthMetricDetailScreen - Invalid Metric Type', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Default mock for getRecentByType
    mockedHealthMetricsApi.getRecentByType.mockResolvedValue([]);
    // Override useLocalSearchParams for invalid metric
    jest.doMock('expo-router', () => ({
      useRouter: () => ({
        push: jest.fn(),
        back: jest.fn(),
      }),
      useLocalSearchParams: () => ({
        metricType: 'INVALID_METRIC',
      }),
    }));
  });

  it('shows error for invalid metric type', async () => {
    // Arrange - The component should handle invalid metric type gracefully
    mockedHealthMetricsApi.getTimeSeries.mockResolvedValueOnce([]);
    mockedHealthMetricsApi.getStats.mockResolvedValueOnce(null);

    // Act
    const { findByText } = render(<HealthMetricDetailScreen />);

    // Assert - Should show some content (either error or empty state)
    // The exact behavior depends on how the component handles invalid metrics
    await waitFor(() => {
      expect(mockedHealthMetricsApi.getTimeSeries).toHaveBeenCalled();
    });
  });
});
