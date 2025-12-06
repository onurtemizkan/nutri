import { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { healthMetricsApi } from '@/lib/api/health-metrics';
import {
  HealthMetric,
  HealthMetricStats,
  HealthMetricType,
  METRIC_CONFIG,
  DASHBOARD_METRICS,
} from '@/lib/types/health-metrics';
import { useAuth } from '@/lib/context/AuthContext';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';

type TimeRange = 'today' | 'week' | 'month';

interface MetricData {
  latest: HealthMetric | null;
  stats: HealthMetricStats | null;
}

export default function HealthScreen() {
  const [metrics, setMetrics] = useState<Record<HealthMetricType, MetricData>>({} as Record<HealthMetricType, MetricData>);
  const [timeRange, setTimeRange] = useState<TimeRange>('today');
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { user } = useAuth();
  const router = useRouter();

  const loadHealthData = useCallback(async () => {
    if (!user) {
      setIsLoading(false);
      return;
    }

    try {
      setError(null);
      const data = await healthMetricsApi.getDashboardData(DASHBOARD_METRICS);
      setMetrics(data);
    } catch (err) {
      console.error('Failed to load health data:', err);
      setError('Failed to load health data');
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadHealthData();
    setRefreshing(false);
  }, [loadHealthData]);

  useEffect(() => {
    loadHealthData();
  }, [loadHealthData]);

  const timeRanges: TimeRange[] = ['today', 'week', 'month'];

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

  const getTrendColor = (trend: 'up' | 'down' | 'stable' | undefined, metricType: HealthMetricType) => {
    // For most metrics, up is good. But for RHR, down is better
    const isLowerBetter = metricType === 'RESTING_HEART_RATE' || metricType === 'STRESS_LEVEL';

    switch (trend) {
      case 'up':
        return isLowerBetter ? colors.status.error : colors.status.success;
      case 'down':
        return isLowerBetter ? colors.status.success : colors.status.error;
      default:
        return colors.text.tertiary;
    }
  };

  const formatValue = (value: number | undefined, metricType: HealthMetricType): string => {
    if (value === undefined) return '--';

    const config = METRIC_CONFIG[metricType];

    // Format based on metric type
    if (metricType === 'SLEEP_DURATION' || metricType === 'DEEP_SLEEP_DURATION' || metricType === 'REM_SLEEP_DURATION') {
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

  const hasAnyData = metrics && Object.values(metrics).some(m => m?.latest !== null);

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={colors.primary.main}
            colors={[colors.primary.main]}
          />
        }
      >
        <View style={styles.content}>
          {/* Header */}
          <View style={styles.header}>
            <View>
              <Text style={styles.title}>Health</Text>
              <Text style={styles.subtitle}>
                {new Date().toLocaleDateString('en-US', {
                  weekday: 'long',
                  month: 'long',
                  day: 'numeric',
                })}
              </Text>
            </View>
          </View>

          {/* Time Range Selector */}
          <View style={styles.timeRangeContainer}>
            {timeRanges.map((range) => (
              <TouchableOpacity
                key={range}
                style={styles.timeRangeButton}
                onPress={() => setTimeRange(range)}
                activeOpacity={0.8}
              >
                {timeRange === range ? (
                  <LinearGradient
                    colors={gradients.primary}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={styles.timeRangeButtonActive}
                  >
                    <Text style={styles.timeRangeTextActive}>
                      {range.charAt(0).toUpperCase() + range.slice(1)}
                    </Text>
                  </LinearGradient>
                ) : (
                  <View style={styles.timeRangeButtonInactive}>
                    <Text style={styles.timeRangeText}>
                      {range.charAt(0).toUpperCase() + range.slice(1)}
                    </Text>
                  </View>
                )}
              </TouchableOpacity>
            ))}
          </View>

          {/* Error State */}
          {error && (
            <View style={styles.errorContainer}>
              <Ionicons name="alert-circle-outline" size={24} color={colors.status.error} />
              <Text style={styles.errorText}>{error}</Text>
              <TouchableOpacity onPress={loadHealthData} style={styles.retryButton}>
                <Text style={styles.retryText}>Retry</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Empty State */}
          {!error && !hasAnyData && (
            <View style={styles.emptyContainer}>
              <Ionicons name="heart-outline" size={64} color={colors.text.disabled} />
              <Text style={styles.emptyTitle}>No health data yet</Text>
              <Text style={styles.emptySubtitle}>
                Add your first health metric to start tracking
              </Text>
              <TouchableOpacity
                style={styles.emptyButton}
                onPress={() => router.push('/health/add' as `/health/${string}`)}
                activeOpacity={0.8}
              >
                <LinearGradient
                  colors={gradients.primary}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.emptyButtonGradient}
                >
                  <Text style={styles.emptyButtonText}>Add Health Metric</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          )}

          {/* Metric Cards Grid */}
          {!error && hasAnyData && (
            <View style={styles.metricsGrid}>
              {DASHBOARD_METRICS.map((metricType) => {
                const config = METRIC_CONFIG[metricType];
                const data = metrics[metricType];
                const value = data?.latest?.value;
                const trend = data?.stats?.trend;
                const percentChange = data?.stats?.percentChange;

                return (
                  <TouchableOpacity
                    key={metricType}
                    style={styles.metricCard}
                    onPress={() => router.push(`/health/${metricType}` as `/health/${string}`)}
                    activeOpacity={0.7}
                  >
                    <View style={styles.metricHeader}>
                      <View style={styles.metricIconContainer}>
                        <Ionicons
                          name={config.icon as keyof typeof Ionicons.glyphMap}
                          size={20}
                          color={colors.primary.main}
                        />
                      </View>
                      {trend && (
                        <View style={styles.trendContainer}>
                          <Ionicons
                            name={getTrendIcon(trend)}
                            size={16}
                            color={getTrendColor(trend, metricType)}
                          />
                          {percentChange !== undefined && (
                            <Text
                              style={[
                                styles.trendText,
                                { color: getTrendColor(trend, metricType) },
                              ]}
                            >
                              {Math.abs(percentChange).toFixed(1)}%
                            </Text>
                          )}
                        </View>
                      )}
                    </View>

                    <Text style={styles.metricLabel}>{config.shortName}</Text>

                    <View style={styles.metricValueContainer}>
                      <Text style={styles.metricValue}>
                        {formatValue(value, metricType)}
                      </Text>
                      {value !== undefined && (
                        <Text style={styles.metricUnit}>{config.unit}</Text>
                      )}
                    </View>

                    {data?.stats && (
                      <Text style={styles.metricAverage}>
                        Avg: {formatValue(data.stats.average, metricType)} {config.unit}
                      </Text>
                    )}
                  </TouchableOpacity>
                );
              })}
            </View>
          )}

          {/* All Metrics Section */}
          {!error && hasAnyData && (
            <View style={styles.allMetricsSection}>
              <TouchableOpacity
                style={styles.allMetricsButton}
                onPress={() => router.push('/health/add' as `/health/${string}`)}
                activeOpacity={0.8}
              >
                <Ionicons name="add-circle-outline" size={24} color={colors.primary.main} />
                <Text style={styles.allMetricsText}>Add Health Metric</Text>
                <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
              </TouchableOpacity>
            </View>
          )}
        </View>
      </ScrollView>

      {/* Floating Add Button */}
      {hasAnyData && (
        <TouchableOpacity
          style={styles.fab}
          onPress={() => router.push('/health/add' as `/health/${string}`)}
          activeOpacity={0.8}
        >
          <LinearGradient
            colors={gradients.primary}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.fabGradient}
          >
            <Ionicons name="add" size={28} color={colors.text.primary} />
          </LinearGradient>
        </TouchableOpacity>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: 100,
  },

  // Header
  header: {
    marginBottom: spacing.xl,
  },
  title: {
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    fontWeight: typography.fontWeight.medium,
  },

  // Time Range Selector
  timeRangeContainer: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginBottom: spacing.xl,
  },
  timeRangeButton: {
    flex: 1,
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  timeRangeButtonActive: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
    alignItems: 'center',
  },
  timeRangeButtonInactive: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
    backgroundColor: colors.background.tertiary,
    alignItems: 'center',
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  timeRangeText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
  },
  timeRangeTextActive: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Error State
  errorContainer: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.status.error,
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

  // Empty State
  emptyContainer: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  emptyTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  emptySubtitle: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    textAlign: 'center',
    marginBottom: spacing.lg,
  },
  emptyButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  emptyButtonGradient: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
  },
  emptyButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Metrics Grid
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.md,
    marginBottom: spacing.xl,
  },
  metricCard: {
    width: '47%',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  metricHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  metricIconContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
  },
  trendContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  trendText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
  },
  metricLabel: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: spacing.xs,
  },
  metricValueContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: spacing.xs,
  },
  metricValue: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  metricUnit: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginLeft: spacing.xs,
  },
  metricAverage: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
  },

  // All Metrics Section
  allMetricsSection: {
    marginTop: spacing.md,
  },
  allMetricsButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  allMetricsText: {
    flex: 1,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginLeft: spacing.md,
  },

  // FAB
  fab: {
    position: 'absolute',
    bottom: spacing.xl,
    right: spacing.xl,
    width: 56,
    height: 56,
    borderRadius: 28,
    overflow: 'hidden',
    ...shadows.xl,
  },
  fabGradient: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
});
