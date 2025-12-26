import React, { useState, useCallback, useMemo } from 'react';
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
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { GoalCard, StreakCard, WeeklyTrendCard } from '@/lib/components/GoalCard';
import { CircularProgress } from '@/lib/components/CircularProgress';
import {
  goalsApi,
  HistoricalProgress,
  DailyGoalProgress,
  WeeklyGoalSummary,
} from '@/lib/api/goals';
import { getErrorMessage } from '@/lib/utils/errorHandling';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

type TimeRange = '7d' | '30d' | '90d';

export default function GoalsScreen() {
  const [data, setData] = useState<HistoricalProgress | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedRange, setSelectedRange] = useState<TimeRange>('7d');
  const router = useRouter();

  const getDaysForRange = (range: TimeRange): number => {
    switch (range) {
      case '7d':
        return 7;
      case '30d':
        return 30;
      case '90d':
        return 90;
    }
  };

  const loadData = useCallback(async () => {
    try {
      setError(null);
      const days = getDaysForRange(selectedRange);
      const historicalData = await goalsApi.getHistoricalProgress(days);
      setData(historicalData);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to load goal progress'));
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  }, [selectedRange]);

  useFocusEffect(
    useCallback(() => {
      setIsLoading(true);
      loadData();
    }, [loadData])
  );

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadData();
  }, [loadData]);

  const handleRangeChange = useCallback((range: TimeRange) => {
    setSelectedRange(range);
    setIsLoading(true);
  }, []);

  const handleBackPress = useCallback(() => {
    router.back();
  }, [router]);

  // Calculate statistics
  const statistics = useMemo(() => {
    if (!data?.daily.length) return null;

    const trackedDays = data.daily.filter((d) => d.calories.consumed > 0 || d.protein.consumed > 0);

    const daysGoalsMet = trackedDays.filter((d) => d.allGoalsMet).length;
    const successRate = trackedDays.length > 0 ? (daysGoalsMet / trackedDays.length) * 100 : 0;

    const avgCalories =
      trackedDays.length > 0
        ? trackedDays.reduce((sum, d) => sum + d.calories.percentage, 0) / trackedDays.length
        : 0;
    const avgProtein =
      trackedDays.length > 0
        ? trackedDays.reduce((sum, d) => sum + d.protein.percentage, 0) / trackedDays.length
        : 0;
    const avgCarbs =
      trackedDays.length > 0
        ? trackedDays.reduce((sum, d) => sum + d.carbs.percentage, 0) / trackedDays.length
        : 0;
    const avgFat =
      trackedDays.length > 0
        ? trackedDays.reduce((sum, d) => sum + d.fat.percentage, 0) / trackedDays.length
        : 0;

    return {
      trackedDays: trackedDays.length,
      daysGoalsMet,
      successRate,
      avgCalories,
      avgProtein,
      avgCarbs,
      avgFat,
    };
  }, [data?.daily]);

  // Today's progress (first item in daily array, most recent)
  const todayProgress = useMemo(() => {
    if (!data?.daily.length) return null;
    return data.daily[data.daily.length - 1]; // Last item is most recent
  }, [data?.daily]);

  if (isLoading && !data) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Loading goal progress...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error && !data) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={handleBackPress} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Goals</Text>
          <View style={styles.headerRight} />
        </View>
        <View style={styles.errorContainer}>
          <Ionicons name="warning" size={48} color={colors.semantic.warning} />
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity onPress={loadData} style={styles.retryButton}>
            <Text style={styles.retryText}>Retry</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={handleBackPress} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Goals</Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={colors.primary.main}
          />
        }
      >
        <View style={styles.content}>
          {/* Time Range Selector */}
          <View style={styles.rangeSelector}>
            {(['7d', '30d', '90d'] as TimeRange[]).map((range) => (
              <TouchableOpacity
                key={range}
                style={[styles.rangeButton, selectedRange === range && styles.rangeButtonActive]}
                onPress={() => handleRangeChange(range)}
              >
                <Text
                  style={[
                    styles.rangeButtonText,
                    selectedRange === range && styles.rangeButtonTextActive,
                  ]}
                >
                  {range === '7d' ? '7 Days' : range === '30d' ? '30 Days' : '90 Days'}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          {/* Streak Card */}
          {data?.streak && (
            <StreakCard
              currentStreak={data.streak.currentStreak}
              longestStreak={data.streak.longestStreak}
            />
          )}

          {/* Overview Stats */}
          {statistics && (
            <View style={styles.statsCard}>
              <Text style={styles.statsTitle}>Overview</Text>
              <View style={styles.statsGrid}>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{statistics.trackedDays}</Text>
                  <Text style={styles.statLabel}>Days Tracked</Text>
                </View>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{statistics.daysGoalsMet}</Text>
                  <Text style={styles.statLabel}>Goals Met</Text>
                </View>
                <View style={styles.statItem}>
                  <Text style={[styles.statValue, { color: colors.semantic.success }]}>
                    {Math.round(statistics.successRate)}%
                  </Text>
                  <Text style={styles.statLabel}>Success Rate</Text>
                </View>
              </View>
            </View>
          )}

          {/* Today's Progress */}
          {todayProgress && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Today's Progress</Text>
              <View style={styles.goalsGrid}>
                <View style={styles.goalRow}>
                  <View style={styles.goalCardWrapper}>
                    <GoalCard type="calories" progress={todayProgress.calories} />
                  </View>
                  <View style={styles.goalCardWrapper}>
                    <GoalCard type="protein" progress={todayProgress.protein} />
                  </View>
                </View>
                <View style={styles.goalRow}>
                  <View style={styles.goalCardWrapper}>
                    <GoalCard type="carbs" progress={todayProgress.carbs} />
                  </View>
                  <View style={styles.goalCardWrapper}>
                    <GoalCard type="fat" progress={todayProgress.fat} />
                  </View>
                </View>
              </View>
            </View>
          )}

          {/* Average Progress */}
          {statistics && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Average Progress</Text>
              <View style={styles.averageCard}>
                <View style={styles.averageRow}>
                  <View style={styles.averageItem}>
                    <CircularProgress
                      percentage={statistics.avgCalories}
                      size={60}
                      strokeWidth={6}
                      color={colors.primary.main}
                      gradientEndColor={colors.primary.light}
                      gradientId="avgCalories"
                    >
                      <Text style={styles.avgPercentage}>
                        {Math.round(statistics.avgCalories)}%
                      </Text>
                    </CircularProgress>
                    <Text style={styles.avgLabel}>Calories</Text>
                  </View>
                  <View style={styles.averageItem}>
                    <CircularProgress
                      percentage={statistics.avgProtein}
                      size={60}
                      strokeWidth={6}
                      color={colors.semantic.success}
                      gradientEndColor="#059669"
                      gradientId="avgProtein"
                    >
                      <Text style={styles.avgPercentage}>{Math.round(statistics.avgProtein)}%</Text>
                    </CircularProgress>
                    <Text style={styles.avgLabel}>Protein</Text>
                  </View>
                  <View style={styles.averageItem}>
                    <CircularProgress
                      percentage={statistics.avgCarbs}
                      size={60}
                      strokeWidth={6}
                      color="#EC4899"
                      gradientEndColor="#F59E0B"
                      gradientId="avgCarbs"
                    >
                      <Text style={styles.avgPercentage}>{Math.round(statistics.avgCarbs)}%</Text>
                    </CircularProgress>
                    <Text style={styles.avgLabel}>Carbs</Text>
                  </View>
                  <View style={styles.averageItem}>
                    <CircularProgress
                      percentage={statistics.avgFat}
                      size={60}
                      strokeWidth={6}
                      color={colors.secondary.main}
                      gradientEndColor={colors.secondary.cyan}
                      gradientId="avgFat"
                    >
                      <Text style={styles.avgPercentage}>{Math.round(statistics.avgFat)}%</Text>
                    </CircularProgress>
                    <Text style={styles.avgLabel}>Fat</Text>
                  </View>
                </View>
              </View>
            </View>
          )}

          {/* Weekly Summaries */}
          {data?.weeklySummaries && data.weeklySummaries.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Weekly History</Text>
              {data.weeklySummaries.map((week, index) => (
                <WeeklySummaryCard key={week.weekStart} summary={week} isFirst={index === 0} />
              ))}
            </View>
          )}

          {/* Monthly Summaries */}
          {data?.monthlySummaries && data.monthlySummaries.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Monthly History</Text>
              {data.monthlySummaries.map((month) => (
                <MonthlySummaryCard key={month.month} summary={month} />
              ))}
            </View>
          )}

          {/* Daily History Chart */}
          {data?.daily && data.daily.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Daily History</Text>
              <DailyHistoryChart data={data.daily} />
            </View>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

// ============================================================================
// SUB-COMPONENTS
// ============================================================================

interface WeeklySummaryCardProps {
  summary: WeeklyGoalSummary;
  isFirst: boolean;
}

const WeeklySummaryCard = React.memo(function WeeklySummaryCard({
  summary,
  isFirst,
}: WeeklySummaryCardProps) {
  const weekLabel = isFirst
    ? 'This Week'
    : `Week of ${new Date(summary.weekStart).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;

  return (
    <View style={styles.summaryCard}>
      <View style={styles.summaryHeader}>
        <Text style={styles.summaryLabel}>{weekLabel}</Text>
        <View
          style={[
            styles.trendBadge,
            {
              backgroundColor:
                summary.trend === 'improving'
                  ? colors.special.successLight
                  : summary.trend === 'declining'
                    ? colors.special.warningLight
                    : colors.background.elevated,
            },
          ]}
        >
          <Ionicons
            name={
              summary.trend === 'improving'
                ? 'trending-up'
                : summary.trend === 'declining'
                  ? 'trending-down'
                  : 'remove'
            }
            size={14}
            color={
              summary.trend === 'improving'
                ? colors.semantic.success
                : summary.trend === 'declining'
                  ? colors.semantic.warning
                  : colors.text.tertiary
            }
          />
        </View>
      </View>

      <View style={styles.summaryStats}>
        <View style={styles.summaryStatItem}>
          <Text style={styles.summaryStatValue}>{summary.daysTracked}</Text>
          <Text style={styles.summaryStatLabel}>Tracked</Text>
        </View>
        <View style={styles.summaryStatItem}>
          <Text style={styles.summaryStatValue}>{summary.daysGoalsMet}</Text>
          <Text style={styles.summaryStatLabel}>Goals Met</Text>
        </View>
        <View style={styles.summaryStatItem}>
          <Text style={styles.summaryStatValue}>
            {Math.round(summary.averageProgress.calories)}%
          </Text>
          <Text style={styles.summaryStatLabel}>Avg Cal</Text>
        </View>
        <View style={styles.summaryStatItem}>
          <Text style={styles.summaryStatValue}>
            {Math.round(summary.averageProgress.protein)}%
          </Text>
          <Text style={styles.summaryStatLabel}>Avg Pro</Text>
        </View>
      </View>
    </View>
  );
});

interface MonthlySummaryCardProps {
  summary: {
    month: string;
    daysTracked: number;
    daysGoalsMet: number;
    successRate: number;
    averageProgress: {
      calories: number;
      protein: number;
      carbs: number;
      fat: number;
    };
  };
}

const MonthlySummaryCard = React.memo(function MonthlySummaryCard({
  summary,
}: MonthlySummaryCardProps) {
  const [year, month] = summary.month.split('-');
  const monthName = new Date(parseInt(year), parseInt(month) - 1).toLocaleDateString('en-US', {
    month: 'long',
    year: 'numeric',
  });

  return (
    <View style={styles.summaryCard}>
      <View style={styles.summaryHeader}>
        <Text style={styles.summaryLabel}>{monthName}</Text>
        <Text style={[styles.successRateBadge, { color: colors.semantic.success }]}>
          {Math.round(summary.successRate)}% success
        </Text>
      </View>

      <View style={styles.summaryStats}>
        <View style={styles.summaryStatItem}>
          <Text style={styles.summaryStatValue}>{summary.daysTracked}</Text>
          <Text style={styles.summaryStatLabel}>Days</Text>
        </View>
        <View style={styles.summaryStatItem}>
          <Text style={styles.summaryStatValue}>{summary.daysGoalsMet}</Text>
          <Text style={styles.summaryStatLabel}>Met</Text>
        </View>
        <View style={styles.summaryStatItem}>
          <Text style={styles.summaryStatValue}>
            {Math.round(summary.averageProgress.calories)}%
          </Text>
          <Text style={styles.summaryStatLabel}>Cal</Text>
        </View>
        <View style={styles.summaryStatItem}>
          <Text style={styles.summaryStatValue}>
            {Math.round(summary.averageProgress.protein)}%
          </Text>
          <Text style={styles.summaryStatLabel}>Pro</Text>
        </View>
      </View>
    </View>
  );
});

interface DailyHistoryChartProps {
  data: DailyGoalProgress[];
}

const DailyHistoryChart = React.memo(function DailyHistoryChart({ data }: DailyHistoryChartProps) {
  // Get the last 7 days for display
  const recentData = data.slice(-7);
  const maxValue = 100;

  return (
    <View style={styles.chartCard}>
      <View style={styles.chartContainer}>
        {recentData.map((day, index) => {
          const date = new Date(day.date);
          const dayLabel = date.toLocaleDateString('en-US', { weekday: 'short' });
          const avgProgress =
            (day.calories.percentage +
              day.protein.percentage +
              day.carbs.percentage +
              day.fat.percentage) /
            4;
          const barHeight = Math.min((avgProgress / maxValue) * 100, 100);

          return (
            <View key={day.date} style={styles.chartBarContainer}>
              <View style={styles.chartBar}>
                <LinearGradient
                  colors={day.allGoalsMet ? gradients.success : gradients.primary}
                  start={{ x: 0, y: 1 }}
                  end={{ x: 0, y: 0 }}
                  style={[styles.chartBarFill, { height: `${barHeight}%` }]}
                />
              </View>
              <Text style={styles.chartLabel}>{dayLabel}</Text>
              {day.allGoalsMet && (
                <Ionicons
                  name="checkmark-circle"
                  size={14}
                  color={colors.semantic.success}
                  style={styles.chartCheck}
                />
              )}
            </View>
          );
        })}
      </View>
      <View style={styles.chartLegend}>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: colors.primary.main }]} />
          <Text style={styles.legendText}>In Progress</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: colors.semantic.success }]} />
          <Text style={styles.legendText}>Goals Met</Text>
        </View>
      </View>
    </View>
  );
});

// ============================================================================
// STYLES
// ============================================================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.md,
  },
  loadingText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
    gap: spacing.md,
  },
  errorText: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
    textAlign: 'center',
  },
  retryButton: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.md,
  },
  retryText: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    fontWeight: typography.fontWeight.semibold,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  backButton: {
    padding: spacing.sm,
  },
  headerTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  headerRight: {
    width: 40,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: spacing.md,
    paddingBottom: spacing.xl * 2,
  },
  rangeSelector: {
    flexDirection: 'row',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.xs,
    marginBottom: spacing.md,
  },
  rangeButton: {
    flex: 1,
    paddingVertical: spacing.sm,
    alignItems: 'center',
    borderRadius: borderRadius.sm,
  },
  rangeButtonActive: {
    backgroundColor: colors.primary.main,
  },
  rangeButtonText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.tertiary,
  },
  rangeButtonTextActive: {
    color: colors.text.primary,
  },
  section: {
    marginTop: spacing.lg,
  },
  sectionTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  statsCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginTop: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  statsTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.md,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  statLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  goalsGrid: {
    gap: spacing.sm,
  },
  goalRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  goalCardWrapper: {
    flex: 1,
  },
  averageCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  averageRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  averageItem: {
    alignItems: 'center',
    gap: spacing.xs,
  },
  avgPercentage: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  avgLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  summaryCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  summaryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  summaryLabel: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  trendBadge: {
    padding: spacing.xs,
    borderRadius: borderRadius.sm,
  },
  successRateBadge: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
  },
  summaryStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  summaryStatItem: {
    alignItems: 'center',
  },
  summaryStatValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  summaryStatLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  chartCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  chartContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'flex-end',
    height: 120,
    marginBottom: spacing.sm,
  },
  chartBarContainer: {
    alignItems: 'center',
    flex: 1,
  },
  chartBar: {
    width: 24,
    height: 100,
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.xs,
    overflow: 'hidden',
    justifyContent: 'flex-end',
  },
  chartBarFill: {
    width: '100%',
    borderRadius: borderRadius.xs,
  },
  chartLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  chartCheck: {
    position: 'absolute',
    top: -8,
  },
  chartLegend: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.lg,
    marginTop: spacing.sm,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  legendText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
});
