import { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { LineChart } from 'react-native-chart-kit';
import { weightApi } from '@/lib/api/weight';
import {
  WeightTrendsResult,
  WeightProgressResult,
  WeightRecord,
  formatWeight,
  formatWeightChange,
  getBmiColor,
  WeightUnit,
} from '@/lib/types/weight';
import { useAuth } from '@/lib/context/AuthContext';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { SwipeableWeightCard } from '@/lib/components/SwipeableWeightCard';
import { showAlert } from '@/lib/utils/alert';

type TimeRange = '7d' | '30d' | '90d';

const TIME_RANGE_DAYS: Record<TimeRange, number> = {
  '7d': 7,
  '30d': 30,
  '90d': 90,
};

export default function WeightScreen() {
  const [trends, setTrends] = useState<WeightTrendsResult | null>(null);
  const [progress, setProgress] = useState<WeightProgressResult | null>(null);
  const [recentRecords, setRecentRecords] = useState<WeightRecord[]>([]);
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [unit] = useState<WeightUnit>('kg'); // TODO: Get from user preferences
  const { user } = useAuth();
  const router = useRouter();
  const { isTablet, getResponsiveValue, width } = useResponsive();

  // Responsive values
  const contentPadding = getResponsiveValue({
    small: spacing.md,
    medium: spacing.lg,
    large: spacing.lg,
    tablet: spacing.xl,
    default: spacing.lg,
  });
  const fabSize = getResponsiveValue({
    small: 52,
    medium: 56,
    large: 60,
    tablet: 64,
    default: 56,
  });

  const loadWeightData = useCallback(async () => {
    if (!user) {
      setIsLoading(false);
      return;
    }

    try {
      setError(null);
      const days = TIME_RANGE_DAYS[timeRange];

      // Fetch trends, progress, and recent records in parallel
      const [trendsData, progressData, recordsData] = await Promise.all([
        weightApi.getTrends({ days }),
        weightApi.getProgress(),
        weightApi.getAll({ limit: 20 }),
      ]);

      setTrends(trendsData);
      setProgress(progressData);
      setRecentRecords(recordsData);
    } catch (err) {
      console.error('Failed to load weight data:', err);
      setError('Failed to load weight data');
    } finally {
      setIsLoading(false);
    }
  }, [user, timeRange]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadWeightData();
    setRefreshing(false);
  }, [loadWeightData]);

  // Reload data when screen comes into focus
  useFocusEffect(
    useCallback(() => {
      loadWeightData();
    }, [loadWeightData])
  );

  const handleEdit = (record: WeightRecord) => {
    router.push(`/weight/edit/${record.id}`);
  };

  const handleDelete = async (record: WeightRecord) => {
    showAlert(
      'Delete Weight Record',
      `Are you sure you want to delete this weight record (${formatWeight(record.weight, unit)})?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await weightApi.delete(record.id);
              await loadWeightData();
            } catch (err) {
              console.error('Failed to delete weight record:', err);
              showAlert('Error', 'Failed to delete weight record');
            }
          },
        },
      ]
    );
  };

  const timeRanges: TimeRange[] = ['7d', '30d', '90d'];
  const timeRangeLabels: Record<TimeRange, string> = {
    '7d': '7 Days',
    '30d': '30 Days',
    '90d': '90 Days',
  };

  // Prepare chart data
  const getChartData = () => {
    if (!trends || trends.movingAverage7Day.length === 0) {
      return null;
    }

    const dataPoints = trends.movingAverage7Day.slice(-10); // Last 10 points for readability
    const labels = dataPoints.map((p) => {
      const date = new Date(p.date);
      return `${date.getMonth() + 1}/${date.getDate()}`;
    });
    const values = dataPoints.map((p) => p.value);

    return {
      labels,
      datasets: [{ data: values }],
    };
  };

  const chartData = getChartData();
  const chartWidth = width - contentPadding * 2;
  const hasAnyData = recentRecords.length > 0;

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container} testID="weight-screen">
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} testID="weight-screen">
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
        <View style={[styles.content, { padding: contentPadding }]}>
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.headerLeft}>
              <TouchableOpacity
                onPress={() => router.back()}
                style={styles.backButton}
                accessibilityLabel="Go back"
              >
                <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
              </TouchableOpacity>
              <View>
                <Text style={styles.title}>Weight</Text>
                <Text style={styles.subtitle}>Track your progress</Text>
              </View>
            </View>
            {hasAnyData && (
              <TouchableOpacity
                onPress={() => router.push('/weight/goal')}
                style={styles.goalButton}
                accessibilityLabel="Set weight goal"
              >
                <Ionicons name="flag-outline" size={20} color={colors.primary.main} />
              </TouchableOpacity>
            )}
          </View>

          {/* Error State */}
          {error && (
            <View style={styles.errorContainer}>
              <Ionicons name="alert-circle-outline" size={24} color={colors.status.error} />
              <Text style={styles.errorText}>{error}</Text>
              <TouchableOpacity onPress={loadWeightData} style={styles.retryButton}>
                <Text style={styles.retryText}>Retry</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Empty State */}
          {!error && !hasAnyData && (
            <View style={styles.emptyContainer}>
              <Ionicons name="scale-outline" size={64} color={colors.text.disabled} />
              <Text style={styles.emptyTitle}>No weight records yet</Text>
              <Text style={styles.emptySubtitle}>
                Start tracking your weight to see trends and progress
              </Text>
              <TouchableOpacity
                style={styles.emptyButton}
                onPress={() => router.push('/weight/add')}
                activeOpacity={0.8}
              >
                <LinearGradient
                  colors={gradients.primary}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.emptyButtonGradient}
                >
                  <Text style={styles.emptyButtonText}>Add Weight</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          )}

          {/* Progress Card */}
          {!error && hasAnyData && progress && (
            <View style={styles.progressCard}>
              <View style={styles.progressHeader}>
                <Text style={styles.progressTitle}>Current Progress</Text>
                {progress.isOnTrack !== null && (
                  <View
                    style={[
                      styles.statusBadge,
                      {
                        backgroundColor: progress.isOnTrack
                          ? colors.status.success + '20'
                          : colors.status.warning + '20',
                      },
                    ]}
                  >
                    <Text
                      style={[
                        styles.statusText,
                        {
                          color: progress.isOnTrack ? colors.status.success : colors.status.warning,
                        },
                      ]}
                    >
                      {progress.isOnTrack ? 'On Track' : 'Off Track'}
                    </Text>
                  </View>
                )}
              </View>

              <View style={styles.progressRow}>
                <View style={styles.progressItem}>
                  <Text style={styles.progressLabel}>Current</Text>
                  <Text style={styles.progressValue}>
                    {progress.currentWeight !== null
                      ? formatWeight(progress.currentWeight, unit, 1)
                      : '--'}
                  </Text>
                </View>
                <View style={styles.progressItem}>
                  <Text style={styles.progressLabel}>Goal</Text>
                  <Text style={styles.progressValue}>
                    {progress.goalWeight !== null
                      ? formatWeight(progress.goalWeight, unit, 1)
                      : '--'}
                  </Text>
                </View>
                <View style={styles.progressItem}>
                  <Text style={styles.progressLabel}>Remaining</Text>
                  <Text
                    style={[
                      styles.progressValue,
                      {
                        color:
                          progress.remainingWeight !== null && progress.remainingWeight <= 0
                            ? colors.status.success
                            : colors.text.primary,
                      },
                    ]}
                  >
                    {progress.remainingWeight !== null
                      ? formatWeightChange(progress.remainingWeight, unit)
                      : '--'}
                  </Text>
                </View>
              </View>

              {/* Progress Bar */}
              {progress.progressPercentage !== null && (
                <View style={styles.progressBarContainer}>
                  <View style={styles.progressBarBackground}>
                    <View
                      style={[
                        styles.progressBarFill,
                        {
                          width: `${Math.min(100, Math.max(0, progress.progressPercentage))}%`,
                        },
                      ]}
                    />
                  </View>
                  <Text style={styles.progressPercentage}>
                    {Math.round(progress.progressPercentage)}%
                  </Text>
                </View>
              )}

              {/* BMI */}
              {progress.bmi !== null && (
                <View style={styles.bmiContainer}>
                  <Text style={styles.bmiLabel}>BMI</Text>
                  <View style={styles.bmiValueContainer}>
                    <Text style={styles.bmiValue}>{progress.bmi}</Text>
                    <View
                      style={[
                        styles.bmiCategoryBadge,
                        { backgroundColor: getBmiColor(progress.bmiCategory) + '20' },
                      ]}
                    >
                      <Text
                        style={[
                          styles.bmiCategoryText,
                          { color: getBmiColor(progress.bmiCategory) },
                        ]}
                      >
                        {progress.bmiCategory}
                      </Text>
                    </View>
                  </View>
                </View>
              )}
            </View>
          )}

          {/* Time Range Selector */}
          {!error && hasAnyData && (
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
                      <Text style={styles.timeRangeTextActive}>{timeRangeLabels[range]}</Text>
                    </LinearGradient>
                  ) : (
                    <View style={styles.timeRangeButtonInactive}>
                      <Text style={styles.timeRangeText}>{timeRangeLabels[range]}</Text>
                    </View>
                  )}
                </TouchableOpacity>
              ))}
            </View>
          )}

          {/* Trend Chart */}
          {!error && hasAnyData && chartData && (
            <View style={styles.chartCard}>
              <Text style={styles.chartTitle}>Weight Trend</Text>
              <LineChart
                data={chartData}
                width={chartWidth - spacing.lg * 2}
                height={200}
                chartConfig={{
                  backgroundColor: colors.background.tertiary,
                  backgroundGradientFrom: colors.background.tertiary,
                  backgroundGradientTo: colors.background.tertiary,
                  decimalPlaces: 1,
                  color: (opacity = 1) => `rgba(99, 102, 241, ${opacity})`,
                  labelColor: () => colors.text.tertiary,
                  style: { borderRadius: 16 },
                  propsForDots: {
                    r: '4',
                    strokeWidth: '2',
                    stroke: colors.primary.main,
                  },
                  propsForBackgroundLines: {
                    stroke: colors.border.secondary,
                    strokeDasharray: '0',
                  },
                }}
                bezier
                style={styles.chart}
                withInnerLines={true}
                withOuterLines={false}
                withVerticalLabels={true}
                withHorizontalLabels={true}
              />

              {/* Statistics */}
              {trends && (
                <View style={styles.statsContainer}>
                  <View style={styles.statItem}>
                    <Text style={styles.statLabel}>Min</Text>
                    <Text style={styles.statValue}>
                      {trends.minWeight !== null ? formatWeight(trends.minWeight, unit, 1) : '--'}
                    </Text>
                  </View>
                  <View style={styles.statItem}>
                    <Text style={styles.statLabel}>Max</Text>
                    <Text style={styles.statValue}>
                      {trends.maxWeight !== null ? formatWeight(trends.maxWeight, unit, 1) : '--'}
                    </Text>
                  </View>
                  <View style={styles.statItem}>
                    <Text style={styles.statLabel}>Avg</Text>
                    <Text style={styles.statValue}>
                      {trends.averageWeight !== null
                        ? formatWeight(trends.averageWeight, unit, 1)
                        : '--'}
                    </Text>
                  </View>
                  <View style={styles.statItem}>
                    <Text style={styles.statLabel}>Change</Text>
                    <Text
                      style={[
                        styles.statValue,
                        {
                          color:
                            trends.totalChange !== null
                              ? trends.totalChange < 0
                                ? colors.status.success
                                : trends.totalChange > 0
                                  ? colors.status.error
                                  : colors.text.primary
                              : colors.text.primary,
                        },
                      ]}
                    >
                      {trends.totalChange !== null
                        ? formatWeightChange(trends.totalChange, unit)
                        : '--'}
                    </Text>
                  </View>
                </View>
              )}
            </View>
          )}

          {/* Recent Records */}
          {!error && hasAnyData && (
            <View style={styles.recordsSection}>
              <Text style={styles.sectionTitle}>Recent Records</Text>
              {recentRecords.map((record) => (
                <SwipeableWeightCard
                  key={record.id}
                  record={record}
                  unit={unit}
                  onEdit={handleEdit}
                  onDelete={handleDelete}
                />
              ))}
            </View>
          )}
        </View>
      </ScrollView>

      {/* Floating Add Button */}
      <TouchableOpacity
        style={[styles.fab, { width: fabSize, height: fabSize, borderRadius: fabSize / 2 }]}
        onPress={() => router.push('/weight/add')}
        activeOpacity={0.8}
      >
        <LinearGradient
          colors={gradients.primary}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.fabGradient}
        >
          <Ionicons name="add" size={fabSize * 0.5} color={colors.text.primary} />
        </LinearGradient>
      </TouchableOpacity>
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
    paddingBottom: 100,
  },

  // Header
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  backButton: {
    padding: spacing.xs,
  },
  title: {
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    fontWeight: typography.fontWeight.medium,
  },
  goalButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
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

  // Progress Card
  progressCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  progressTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  statusBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
  },
  statusText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
  },
  progressRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  progressItem: {
    alignItems: 'center',
    flex: 1,
  },
  progressLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  progressValue: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  progressBarContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  progressBarBackground: {
    flex: 1,
    height: 8,
    backgroundColor: colors.border.secondary,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: colors.primary.main,
    borderRadius: 4,
  },
  progressPercentage: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
    minWidth: 40,
    textAlign: 'right',
  },
  bmiContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  bmiLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  bmiValueContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  bmiValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  bmiCategoryBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
  },
  bmiCategoryText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
  },

  // Time Range Selector
  timeRangeContainer: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginBottom: spacing.lg,
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

  // Chart Card
  chartCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  chartTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  chart: {
    marginVertical: spacing.sm,
    borderRadius: 16,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  statItem: {
    alignItems: 'center',
  },
  statLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  statValue: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },

  // Records Section
  recordsSection: {
    marginTop: spacing.md,
  },
  sectionTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.md,
  },

  // FAB
  fab: {
    position: 'absolute',
    bottom: spacing.xl,
    right: spacing.xl,
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
