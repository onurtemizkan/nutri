import { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { LineChart } from 'react-native-chart-kit';
import { healthMetricsApi } from '@/lib/api/health-metrics';
import {
  HealthMetric,
  HealthMetricStats,
  TimeSeriesDataPoint,
  HealthMetricType,
  HealthMetricSource,
  METRIC_CONFIG,
  SOURCE_CONFIG,
} from '@/lib/types/health-metrics';
import { SwipeableHealthMetricCard } from '@/lib/components/SwipeableHealthMetricCard';
import { showAlert } from '@/lib/utils/alert';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';

type DateRange = '7d' | '30d' | '90d' | '1y';

export default function HealthMetricDetailScreen() {
  const { metricType } = useLocalSearchParams<{ metricType: string }>();
  const router = useRouter();
  const { isTablet, getSpacing, width: screenWidth } = useResponsive();
  const responsiveSpacing = getSpacing();

  // Calculate chart width based on responsive layout
  const effectiveContentWidth = isTablet
    ? Math.min(screenWidth, FORM_MAX_WIDTH)
    : screenWidth;
  const chartWidth = effectiveContentWidth - responsiveSpacing.horizontal * 2 - spacing.md * 2;

  const [timeSeries, setTimeSeries] = useState<TimeSeriesDataPoint[]>([]);
  const [stats, setStats] = useState<HealthMetricStats | null>(null);
  const [recentMetrics, setRecentMetrics] = useState<HealthMetric[]>([]);
  const [dateRange, setDateRange] = useState<DateRange>('30d');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Validate metricType
  const validMetricType = metricType as HealthMetricType;
  const config = METRIC_CONFIG[validMetricType];

  const getDaysFromRange = (range: DateRange): number => {
    switch (range) {
      case '7d':
        return 7;
      case '30d':
        return 30;
      case '90d':
        return 90;
      case '1y':
        return 365;
    }
  };

  const getStartDate = (range: DateRange): string => {
    const days = getDaysFromRange(range);
    const date = new Date();
    date.setDate(date.getDate() - days);
    return date.toISOString();
  };

  const loadData = useCallback(async () => {
    if (!validMetricType || !config) {
      setError('Invalid metric type');
      setIsLoading(false);
      return;
    }

    try {
      setError(null);
      const startDate = getStartDate(dateRange);
      const endDate = new Date().toISOString();

      const [timeSeriesData, statsData, recentData] = await Promise.all([
        healthMetricsApi.getTimeSeries(validMetricType, startDate, endDate),
        healthMetricsApi.getStats(validMetricType, getDaysFromRange(dateRange)),
        healthMetricsApi.getRecentByType(validMetricType, 20),
      ]);

      setTimeSeries(timeSeriesData);
      setStats(statsData);
      setRecentMetrics(recentData);
    } catch (err) {
      console.error('Failed to load metric data:', err);
      setError('Failed to load data');
    } finally {
      setIsLoading(false);
    }
  }, [validMetricType, dateRange, config]);

  useEffect(() => {
    setIsLoading(true);
    loadData();
  }, [loadData]);

  const dateRanges: DateRange[] = ['7d', '30d', '90d', '1y'];

  const formatValue = (value: number): string => {
    if (!config) return value.toString();

    // Format based on metric type
    if (
      validMetricType === 'SLEEP_DURATION' ||
      validMetricType === 'DEEP_SLEEP_DURATION' ||
      validMetricType === 'REM_SLEEP_DURATION'
    ) {
      const hours = Math.floor(value);
      const minutes = Math.round((value - hours) * 60);
      return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
    }

    // For percentages and scores, show as integers
    if (config.unit === '%' || config.unit === 'pts') {
      return Math.round(value).toString();
    }

    // For heart rate and similar, show as integers
    if (config.unit === 'bpm' || config.unit === 'ms') {
      return Math.round(value).toString();
    }

    return value.toFixed(1);
  };

  const getTrendIcon = (trend: 'up' | 'down' | 'stable' | undefined) => {
    switch (trend) {
      case 'up':
        return 'trending-up';
      case 'down':
        return 'trending-down';
      default:
        return 'remove';
    }
  };

  const getTrendColor = (trend: 'up' | 'down' | 'stable' | undefined) => {
    if (!validMetricType) return colors.text.tertiary;

    const isLowerBetter =
      validMetricType === 'RESTING_HEART_RATE' || validMetricType === 'STRESS_LEVEL';

    switch (trend) {
      case 'up':
        return isLowerBetter ? colors.status.error : colors.status.success;
      case 'down':
        return isLowerBetter ? colors.status.success : colors.status.error;
      default:
        return colors.text.tertiary;
    }
  };

  const handleEditMetric = (metric: HealthMetric) => {
    router.push(`/edit-health-metric/${metric.id}`);
  };

  const handleDeleteMetric = (metric: HealthMetric) => {
    showAlert(
      'Delete Entry',
      `Are you sure you want to delete this ${config?.shortName || 'metric'} entry?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await healthMetricsApi.delete(metric.id);
              // Refresh data after deletion
              loadData();
            } catch (err) {
              showAlert('Error', getErrorMessage(err, 'Failed to delete entry'));
            }
          },
        },
      ]
    );
  };

  // Prepare chart data
  const chartData = {
    labels:
      timeSeries.length > 0
        ? timeSeries.slice(-7).map((d) => {
            const date = new Date(d.date);
            return `${date.getMonth() + 1}/${date.getDate()}`;
          })
        : ['--'],
    datasets: [
      {
        data:
          timeSeries.length > 0
            ? timeSeries.slice(-7).map((d) => d.value)
            : [0],
        color: (opacity = 1) => `rgba(139, 92, 246, ${opacity})`,
        strokeWidth: 2,
      },
    ],
  };

  const chartConfig = {
    backgroundColor: colors.background.tertiary,
    backgroundGradientFrom: colors.background.tertiary,
    backgroundGradientTo: colors.background.tertiary,
    decimalPlaces: config?.unit === 'bpm' || config?.unit === 'ms' ? 0 : 1,
    color: (opacity = 1) => `rgba(139, 92, 246, ${opacity})`,
    labelColor: (opacity = 1) => `rgba(156, 163, 175, ${opacity})`,
    style: {
      borderRadius: borderRadius.lg,
    },
    propsForDots: {
      r: '4',
      strokeWidth: '2',
      stroke: colors.primary.main,
    },
  };

  // Get primary source from time series
  const primarySource: HealthMetricSource | undefined =
    timeSeries.length > 0 ? timeSeries[0].source : undefined;
  const sourceConfig = primarySource ? SOURCE_CONFIG[primarySource] : null;

  if (!config) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle-outline" size={64} color={colors.status.error} />
          <Text style={styles.errorTitle}>Invalid Metric Type</Text>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Text style={styles.backButtonText}>Go Back</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} testID="health-metric-detail-screen">
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.headerBackButton} testID="health-metric-detail-back-button">
          <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle} numberOfLines={1}>
          {config.displayName}
        </Text>
        <View style={styles.headerSpacer} />
      </View>

      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={[
          styles.scrollContent,
          { paddingHorizontal: responsiveSpacing.horizontal },
          isTablet && styles.scrollContentTablet
        ]}
      >
        <View style={styles.content}>
          {/* Date Range Selector */}
          <View style={styles.dateRangeContainer}>
            {dateRanges.map((range) => (
              <TouchableOpacity
                key={range}
                style={styles.dateRangeButton}
                onPress={() => setDateRange(range)}
                activeOpacity={0.8}
              >
                {dateRange === range ? (
                  <LinearGradient
                    colors={gradients.primary}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={styles.dateRangeButtonActive}
                  >
                    <Text style={styles.dateRangeTextActive}>
                      {range === '7d' ? '7D' : range === '30d' ? '30D' : range === '90d' ? '90D' : '1Y'}
                    </Text>
                  </LinearGradient>
                ) : (
                  <View style={styles.dateRangeButtonInactive}>
                    <Text style={styles.dateRangeText}>
                      {range === '7d' ? '7D' : range === '30d' ? '30D' : range === '90d' ? '90D' : '1Y'}
                    </Text>
                  </View>
                )}
              </TouchableOpacity>
            ))}
          </View>

          {/* Loading State */}
          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={colors.primary.main} />
            </View>
          )}

          {/* Error State */}
          {error && !isLoading && (
            <View style={styles.errorCard}>
              <Ionicons name="alert-circle-outline" size={24} color={colors.status.error} />
              <Text style={styles.errorText}>{error}</Text>
              <TouchableOpacity onPress={loadData} style={styles.retryButton}>
                <Text style={styles.retryText}>Retry</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* No Data State */}
          {!isLoading && !error && timeSeries.length === 0 && (
            <View style={styles.emptyCard}>
              <Ionicons name="analytics-outline" size={48} color={colors.text.disabled} />
              <Text style={styles.emptyTitle}>No data for this period</Text>
              <Text style={styles.emptySubtitle}>
                Add health metrics to see your history
              </Text>
            </View>
          )}

          {/* Chart */}
          {!isLoading && !error && timeSeries.length > 0 && (
            <View style={styles.chartCard}>
              <LineChart
                data={chartData}
                width={chartWidth}
                height={200}
                chartConfig={chartConfig}
                bezier
                style={styles.chart}
                withInnerLines={false}
                withOuterLines={true}
                withVerticalLines={false}
                withHorizontalLines={true}
                withDots={true}
                withShadow={false}
              />
            </View>
          )}

          {/* Statistics */}
          {!isLoading && !error && stats && (
            <View style={styles.statsContainer}>
              <View style={styles.statCard}>
                <Text style={styles.statLabel}>Average</Text>
                <Text style={styles.statValue}>{formatValue(stats.average)}</Text>
                <Text style={styles.statUnit}>{config.unit}</Text>
              </View>

              <View style={styles.statCard}>
                <Text style={styles.statLabel}>Minimum</Text>
                <Text style={styles.statValue}>{formatValue(stats.min)}</Text>
                <Text style={styles.statUnit}>{config.unit}</Text>
              </View>

              <View style={styles.statCard}>
                <Text style={styles.statLabel}>Maximum</Text>
                <Text style={styles.statValue}>{formatValue(stats.max)}</Text>
                <Text style={styles.statUnit}>{config.unit}</Text>
              </View>
            </View>
          )}

          {/* Trend */}
          {!isLoading && !error && stats && (
            <View style={styles.trendCard}>
              <View style={styles.trendHeader}>
                <Text style={styles.trendTitle}>Trend</Text>
                <View style={styles.trendBadge}>
                  <Ionicons
                    name={getTrendIcon(stats.trend)}
                    size={20}
                    color={getTrendColor(stats.trend)}
                  />
                  {stats.percentChange !== undefined && (
                    <Text style={[styles.trendPercentage, { color: getTrendColor(stats.trend) }]}>
                      {stats.percentChange >= 0 ? '+' : ''}
                      {stats.percentChange.toFixed(1)}%
                    </Text>
                  )}
                </View>
              </View>
              <Text style={styles.trendDescription}>
                {stats.trend === 'up'
                  ? 'Your values are trending upward'
                  : stats.trend === 'down'
                  ? 'Your values are trending downward'
                  : 'Your values are stable'}
                {' compared to the previous period.'}
              </Text>
            </View>
          )}

          {/* Data Source */}
          {!isLoading && !error && sourceConfig && (
            <View style={styles.sourceCard}>
              <Text style={styles.sourceTitle}>Data Source</Text>
              <View style={styles.sourceContent}>
                <Ionicons
                  name={sourceConfig.icon as keyof typeof Ionicons.glyphMap}
                  size={20}
                  color={colors.primary.main}
                />
                <Text style={styles.sourceName}>{sourceConfig.displayName}</Text>
              </View>
            </View>
          )}

          {/* Description */}
          {config.description && (
            <View style={styles.descriptionCard}>
              <Text style={styles.descriptionTitle}>About {config.shortName}</Text>
              <Text style={styles.descriptionText}>{config.description}</Text>
            </View>
          )}

          {/* Recent Entries */}
          {!isLoading && !error && recentMetrics.length > 0 && (
            <View style={styles.recentEntriesSection}>
              <Text style={styles.recentEntriesTitle}>Recent Entries</Text>
              <Text style={styles.recentEntriesHint}>
                Swipe left or long-press to edit or delete
              </Text>
              {recentMetrics.map((metric) => (
                <SwipeableHealthMetricCard
                  key={metric.id}
                  metric={metric}
                  onEdit={handleEditMetric}
                  onDelete={handleDeleteMetric}
                />
              ))}
            </View>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
    backgroundColor: colors.background.secondary,
  },
  headerBackButton: {
    padding: spacing.xs,
    marginRight: spacing.sm,
  },
  headerTitle: {
    flex: 1,
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    textAlign: 'center',
  },
  headerSpacer: {
    width: 32,
  },

  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: spacing.lg,
  },
  scrollContentTablet: {
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
  },
  content: {
    paddingVertical: spacing.lg,
    paddingBottom: spacing['3xl'],
  },

  // Date Range Selector
  dateRangeContainer: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginBottom: spacing.xl,
  },
  dateRangeButton: {
    flex: 1,
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  dateRangeButtonActive: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
    alignItems: 'center',
  },
  dateRangeButtonInactive: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
    backgroundColor: colors.background.tertiary,
    alignItems: 'center',
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  dateRangeText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
  },
  dateRangeTextActive: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Loading
  loadingContainer: {
    padding: spacing['3xl'],
    alignItems: 'center',
  },

  // Error
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
  },
  errorTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginTop: spacing.lg,
    marginBottom: spacing.lg,
  },
  backButton: {
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.md,
  },
  backButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  errorCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.status.error,
    marginBottom: spacing.lg,
  },
  errorText: {
    fontSize: typography.fontSize.md,
    color: colors.status.error,
    marginTop: spacing.sm,
    marginBottom: spacing.md,
  },
  retryButton: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.special.highlight,
    borderRadius: borderRadius.md,
  },
  retryText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },

  // Empty
  emptyCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  emptyTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  emptySubtitle: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    textAlign: 'center',
  },

  // Chart
  chartCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    marginBottom: spacing.lg,
    ...shadows.sm,
  },
  chart: {
    borderRadius: borderRadius.lg,
  },

  // Statistics
  statsContainer: {
    flexDirection: 'row',
    gap: spacing.md,
    marginBottom: spacing.lg,
  },
  statCard: {
    flex: 1,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  statLabel: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: spacing.xs,
  },
  statValue: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  statUnit: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    marginTop: spacing.xs,
  },

  // Trend
  trendCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    marginBottom: spacing.lg,
  },
  trendHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  trendTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  trendBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    backgroundColor: colors.special.highlight,
    borderRadius: borderRadius.sm,
  },
  trendPercentage: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.bold,
  },
  trendDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    lineHeight: 20,
  },

  // Source
  sourceCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    marginBottom: spacing.lg,
  },
  sourceTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.sm,
  },
  sourceContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  sourceName: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },

  // Description
  descriptionCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  descriptionTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.sm,
  },
  descriptionText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    lineHeight: 20,
  },

  // Recent Entries
  recentEntriesSection: {
    marginTop: spacing.lg,
  },
  recentEntriesTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  recentEntriesHint: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginBottom: spacing.md,
  },
});
