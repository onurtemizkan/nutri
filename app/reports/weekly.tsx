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
  Share,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { LineChart, BarChart } from 'react-native-chart-kit';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { reportsApi } from '@/lib/api/reports';
import {
  WeeklyReport,
  DailyBreakdown,
  TrendComparison,
  MetricTrend,
  TopFood,
  ReportInsight,
  ReportAchievement,
} from '@/lib/types/reports';
import { getErrorMessage } from '@/lib/utils/errorHandling';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CHART_WIDTH = SCREEN_WIDTH - spacing.md * 2 - spacing.md * 2;

export default function WeeklyReportScreen() {
  const [report, setReport] = useState<WeeklyReport | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentDate, setCurrentDate] = useState<string>(() => {
    const today = new Date();
    return today.toISOString().split('T')[0];
  });
  const router = useRouter();

  const loadReport = useCallback(async () => {
    try {
      setError(null);
      const data = await reportsApi.getWeeklyReport(currentDate);
      setReport(data);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to load weekly report'));
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  }, [currentDate]);

  useFocusEffect(
    useCallback(() => {
      setIsLoading(true);
      loadReport();
    }, [loadReport])
  );

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadReport();
  }, [loadReport]);

  const handleBackPress = useCallback(() => {
    router.back();
  }, [router]);

  const handlePreviousWeek = useCallback(() => {
    const prevDate = reportsApi.getPreviousWeekDate(currentDate);
    setCurrentDate(prevDate);
    setIsLoading(true);
  }, [currentDate]);

  const handleNextWeek = useCallback(() => {
    const nextDate = reportsApi.getNextWeekDate(currentDate);
    setCurrentDate(nextDate);
    setIsLoading(true);
  }, [currentDate]);

  const handleShare = useCallback(async () => {
    if (!report) return;

    try {
      const dateRange = reportsApi.formatDateRange(report.periodStart, report.periodEnd);
      const message =
        `Weekly Nutrition Report (${dateRange})\n\n` +
        `Calories: ${Math.round(report.averages.calories)} avg/day\n` +
        `Protein: ${Math.round(report.averages.protein)}g avg/day\n` +
        `Carbs: ${Math.round(report.averages.carbs)}g avg/day\n` +
        `Fat: ${Math.round(report.averages.fat)}g avg/day\n\n` +
        `Goals Met: ${report.daysGoalsMet}/${report.dailyBreakdowns.length} days (${Math.round(report.goalCompletionRate)}%)\n` +
        `Current Streak: ${report.streak.current} days`;

      await Share.share({ message });
    } catch (err) {
      console.error('Failed to share report:', err);
    }
  }, [report]);

  const handleExport = useCallback(() => {
    router.push(`/reports/export?type=weekly&date=${currentDate}`);
  }, [router, currentDate]);

  const isCurrentWeek = useMemo(() => {
    return reportsApi.isCurrentWeek(currentDate);
  }, [currentDate]);

  const weekRange = useMemo(() => {
    if (!report) return '';
    return reportsApi.formatDateRange(report.periodStart, report.periodEnd);
  }, [report]);

  // Chart data for daily calories
  const caloriesChartData = useMemo(() => {
    if (!report?.dailyBreakdowns) return null;

    const labels = report.dailyBreakdowns.map((d) => {
      const date = new Date(d.date);
      return date.toLocaleDateString('en-US', { weekday: 'short' });
    });

    const data = report.dailyBreakdowns.map((d) => d.calories);

    return {
      labels,
      datasets: [{ data, color: () => colors.primary.main, strokeWidth: 2 }],
    };
  }, [report?.dailyBreakdowns]);

  // Chart data for macros
  const macrosChartData = useMemo(() => {
    if (!report?.dailyBreakdowns) return null;

    const labels = report.dailyBreakdowns.map((d) => {
      const date = new Date(d.date);
      return date.toLocaleDateString('en-US', { weekday: 'short' });
    });

    return {
      labels,
      datasets: [
        {
          data: report.dailyBreakdowns.map((d) => d.protein),
          color: () => colors.semantic.success,
          strokeWidth: 2,
        },
        {
          data: report.dailyBreakdowns.map((d) => d.carbs),
          color: () => '#EC4899',
          strokeWidth: 2,
        },
        {
          data: report.dailyBreakdowns.map((d) => d.fat),
          color: () => colors.secondary.main,
          strokeWidth: 2,
        },
      ],
    };
  }, [report?.dailyBreakdowns]);

  if (isLoading && !report) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Loading weekly report...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error && !report) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={handleBackPress} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Weekly Report</Text>
          <View style={styles.headerRight} />
        </View>
        <View style={styles.errorContainer}>
          <Ionicons name="warning" size={48} color={colors.semantic.warning} />
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity onPress={loadReport} style={styles.retryButton}>
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
        <Text style={styles.headerTitle}>Weekly Report</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity onPress={handleShare} style={styles.headerButton}>
            <Ionicons name="share-outline" size={22} color={colors.text.primary} />
          </TouchableOpacity>
          <TouchableOpacity onPress={handleExport} style={styles.headerButton}>
            <Ionicons name="download-outline" size={22} color={colors.text.primary} />
          </TouchableOpacity>
        </View>
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
          {/* Week Selector */}
          <View style={styles.weekSelector}>
            <TouchableOpacity onPress={handlePreviousWeek} style={styles.weekNavButton}>
              <LinearGradient
                colors={[colors.background.tertiary, colors.background.elevated]}
                style={styles.weekNavGradient}
              >
                <Ionicons name="chevron-back" size={20} color={colors.text.primary} />
              </LinearGradient>
            </TouchableOpacity>

            <View style={styles.weekInfo}>
              <Text style={styles.weekLabel}>{weekRange}</Text>
              {isCurrentWeek && <Text style={styles.currentWeekBadge}>Current Week</Text>}
            </View>

            <TouchableOpacity
              onPress={handleNextWeek}
              style={[styles.weekNavButton, isCurrentWeek && styles.weekNavButtonDisabled]}
              disabled={isCurrentWeek}
            >
              <LinearGradient
                colors={
                  isCurrentWeek
                    ? [colors.background.secondary, colors.background.secondary]
                    : [colors.background.tertiary, colors.background.elevated]
                }
                style={styles.weekNavGradient}
              >
                <Ionicons
                  name="chevron-forward"
                  size={20}
                  color={isCurrentWeek ? colors.text.disabled : colors.text.primary}
                />
              </LinearGradient>
            </TouchableOpacity>
          </View>

          {report && (
            <>
              {/* Summary Stats */}
              <View style={styles.statsContainer}>
                <View style={styles.statCard}>
                  <Text style={styles.statValue}>{Math.round(report.averages.calories)}</Text>
                  <Text style={styles.statLabel}>Avg Cal/Day</Text>
                </View>
                <View style={styles.statCard}>
                  <Text style={styles.statValue}>
                    {report.daysGoalsMet}/{report.dailyBreakdowns.length}
                  </Text>
                  <Text style={styles.statLabel}>Goals Met</Text>
                </View>
                <View style={styles.statCard}>
                  <Text style={[styles.statValue, { color: colors.semantic.success }]}>
                    {Math.round(report.goalCompletionRate)}%
                  </Text>
                  <Text style={styles.statLabel}>Success Rate</Text>
                </View>
              </View>

              {/* Streak Section */}
              <View style={styles.streakCard}>
                <View style={styles.streakContent}>
                  <Ionicons name="flame" size={32} color={colors.semantic.warning} />
                  <View style={styles.streakInfo}>
                    <Text style={styles.streakValue}>{report.streak.current}</Text>
                    <Text style={styles.streakLabel}>Day Streak</Text>
                  </View>
                </View>
                <View style={styles.streakBest}>
                  <Text style={styles.streakBestLabel}>Best</Text>
                  <Text style={styles.streakBestValue}>{report.streak.longest}</Text>
                </View>
              </View>

              {/* Trends Section */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Trends vs Last Week</Text>
                <View style={styles.trendsGrid}>
                  <TrendBadge label="Calories" trend={report.trends.calories} />
                  <TrendBadge label="Protein" trend={report.trends.protein} />
                  <TrendBadge label="Carbs" trend={report.trends.carbs} />
                  <TrendBadge label="Fat" trend={report.trends.fat} />
                </View>
              </View>

              {/* Calories Chart */}
              {caloriesChartData && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Daily Calories</Text>
                  <View style={styles.chartCard}>
                    <LineChart
                      data={caloriesChartData}
                      width={CHART_WIDTH}
                      height={180}
                      chartConfig={{
                        backgroundColor: colors.background.tertiary,
                        backgroundGradientFrom: colors.background.tertiary,
                        backgroundGradientTo: colors.background.tertiary,
                        decimalPlaces: 0,
                        color: (opacity = 1) => `rgba(139, 92, 246, ${opacity})`,
                        labelColor: () => colors.text.tertiary,
                        style: { borderRadius: borderRadius.md },
                        propsForDots: {
                          r: '5',
                          strokeWidth: '2',
                          stroke: colors.primary.main,
                        },
                      }}
                      bezier
                      style={styles.chart}
                    />
                  </View>
                </View>
              )}

              {/* Macros Chart */}
              {macrosChartData && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Daily Macros</Text>
                  <View style={styles.chartCard}>
                    <LineChart
                      data={macrosChartData}
                      width={CHART_WIDTH}
                      height={180}
                      chartConfig={{
                        backgroundColor: colors.background.tertiary,
                        backgroundGradientFrom: colors.background.tertiary,
                        backgroundGradientTo: colors.background.tertiary,
                        decimalPlaces: 0,
                        color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
                        labelColor: () => colors.text.tertiary,
                        style: { borderRadius: borderRadius.md },
                        propsForDots: { r: '4', strokeWidth: '1' },
                      }}
                      bezier
                      style={styles.chart}
                    />
                    <View style={styles.chartLegend}>
                      <View style={styles.legendItem}>
                        <View
                          style={[styles.legendDot, { backgroundColor: colors.semantic.success }]}
                        />
                        <Text style={styles.legendText}>Protein</Text>
                      </View>
                      <View style={styles.legendItem}>
                        <View style={[styles.legendDot, { backgroundColor: '#EC4899' }]} />
                        <Text style={styles.legendText}>Carbs</Text>
                      </View>
                      <View style={styles.legendItem}>
                        <View
                          style={[styles.legendDot, { backgroundColor: colors.secondary.main }]}
                        />
                        <Text style={styles.legendText}>Fat</Text>
                      </View>
                    </View>
                  </View>
                </View>
              )}

              {/* Macro Distribution */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Macro Distribution</Text>
                <View style={styles.macroDistCard}>
                  <View style={styles.macroDistBar}>
                    <View
                      style={[
                        styles.macroDistSegment,
                        {
                          width: `${report.macroDistribution.proteinPercent}%`,
                          backgroundColor: colors.semantic.success,
                          borderTopLeftRadius: borderRadius.sm,
                          borderBottomLeftRadius: borderRadius.sm,
                        },
                      ]}
                    />
                    <View
                      style={[
                        styles.macroDistSegment,
                        {
                          width: `${report.macroDistribution.carbsPercent}%`,
                          backgroundColor: '#EC4899',
                        },
                      ]}
                    />
                    <View
                      style={[
                        styles.macroDistSegment,
                        {
                          width: `${report.macroDistribution.fatPercent}%`,
                          backgroundColor: colors.secondary.main,
                          borderTopRightRadius: borderRadius.sm,
                          borderBottomRightRadius: borderRadius.sm,
                        },
                      ]}
                    />
                  </View>
                  <View style={styles.macroDistLabels}>
                    <View style={styles.macroDistItem}>
                      <View
                        style={[styles.legendDot, { backgroundColor: colors.semantic.success }]}
                      />
                      <Text style={styles.macroDistLabel}>
                        Protein {Math.round(report.macroDistribution.proteinPercent)}%
                      </Text>
                    </View>
                    <View style={styles.macroDistItem}>
                      <View style={[styles.legendDot, { backgroundColor: '#EC4899' }]} />
                      <Text style={styles.macroDistLabel}>
                        Carbs {Math.round(report.macroDistribution.carbsPercent)}%
                      </Text>
                    </View>
                    <View style={styles.macroDistItem}>
                      <View
                        style={[styles.legendDot, { backgroundColor: colors.secondary.main }]}
                      />
                      <Text style={styles.macroDistLabel}>
                        Fat {Math.round(report.macroDistribution.fatPercent)}%
                      </Text>
                    </View>
                  </View>
                </View>
              </View>

              {/* Daily Breakdown */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Daily Breakdown</Text>
                {report.dailyBreakdowns.map((day) => (
                  <DayBreakdownCard key={day.date} day={day} />
                ))}
              </View>

              {/* Top Foods */}
              {report.topFoods.length > 0 && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Top Foods</Text>
                  {report.topFoods.slice(0, 5).map((food, index) => (
                    <TopFoodCard key={food.name} food={food} rank={index + 1} />
                  ))}
                </View>
              )}

              {/* AI Insights */}
              {report.insights.length > 0 && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Insights</Text>
                  {report.insights.map((insight) => (
                    <InsightCard key={insight.id} insight={insight} />
                  ))}
                </View>
              )}

              {/* Achievements */}
              {report.achievements.length > 0 && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Achievements</Text>
                  <View style={styles.achievementsGrid}>
                    {report.achievements.map((achievement) => (
                      <AchievementCard key={achievement.id} achievement={achievement} />
                    ))}
                  </View>
                </View>
              )}
            </>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

// ============================================================================
// SUB-COMPONENTS
// ============================================================================

interface TrendBadgeProps {
  label: string;
  trend: MetricTrend;
}

const TrendBadge = React.memo(function TrendBadge({ label, trend }: TrendBadgeProps) {
  const isPositive = trend.trend === 'up';
  const isNegative = trend.trend === 'down';
  const trendColor = isPositive
    ? colors.semantic.success
    : isNegative
      ? colors.semantic.warning
      : colors.text.tertiary;
  const bgColor = isPositive
    ? colors.special.successLight
    : isNegative
      ? colors.special.warningLight
      : colors.background.elevated;

  return (
    <View style={[styles.trendBadge, { backgroundColor: bgColor }]}>
      <Text style={styles.trendLabel}>{label}</Text>
      <View style={styles.trendValueRow}>
        <Ionicons
          name={isPositive ? 'trending-up' : isNegative ? 'trending-down' : 'remove'}
          size={16}
          color={trendColor}
        />
        <Text style={[styles.trendPercent, { color: trendColor }]}>
          {trend.percentChange > 0 ? '+' : ''}
          {Math.round(trend.percentChange)}%
        </Text>
      </View>
    </View>
  );
});

interface DayBreakdownCardProps {
  day: DailyBreakdown;
}

const DayBreakdownCard = React.memo(function DayBreakdownCard({ day }: DayBreakdownCardProps) {
  const date = new Date(day.date);
  const dayName = date.toLocaleDateString('en-US', { weekday: 'short' });
  const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

  return (
    <View style={styles.dayCard}>
      <View style={styles.dayInfo}>
        <Text style={styles.dayName}>{dayName}</Text>
        <Text style={styles.dayDate}>{dateStr}</Text>
      </View>
      <View style={styles.dayStats}>
        <View style={styles.dayStat}>
          <Text style={styles.dayStatValue}>{Math.round(day.calories)}</Text>
          <Text style={styles.dayStatLabel}>cal</Text>
        </View>
        <View style={styles.dayStat}>
          <Text style={styles.dayStatValue}>{Math.round(day.protein)}g</Text>
          <Text style={styles.dayStatLabel}>pro</Text>
        </View>
        <View style={styles.dayStat}>
          <Text style={styles.dayStatValue}>{Math.round(day.carbs)}g</Text>
          <Text style={styles.dayStatLabel}>carb</Text>
        </View>
        <View style={styles.dayStat}>
          <Text style={styles.dayStatValue}>{Math.round(day.fat)}g</Text>
          <Text style={styles.dayStatLabel}>fat</Text>
        </View>
      </View>
      <View style={styles.dayCompletion}>
        <Text
          style={[
            styles.dayCompletionText,
            { color: day.goalCompletion >= 100 ? colors.semantic.success : colors.text.tertiary },
          ]}
        >
          {Math.round(day.goalCompletion)}%
        </Text>
        {day.goalCompletion >= 100 && (
          <Ionicons name="checkmark-circle" size={16} color={colors.semantic.success} />
        )}
      </View>
    </View>
  );
});

interface TopFoodCardProps {
  food: TopFood;
  rank: number;
}

const TopFoodCard = React.memo(function TopFoodCard({ food, rank }: TopFoodCardProps) {
  return (
    <View style={styles.topFoodCard}>
      <View style={styles.topFoodRank}>
        <Text style={styles.topFoodRankText}>{rank}</Text>
      </View>
      <View style={styles.topFoodInfo}>
        <Text style={styles.topFoodName} numberOfLines={1}>
          {food.name}
        </Text>
        <Text style={styles.topFoodCount}>{food.count}x this week</Text>
      </View>
      <View style={styles.topFoodCalories}>
        <Text style={styles.topFoodCaloriesValue}>{Math.round(food.avgCaloriesPerServing)}</Text>
        <Text style={styles.topFoodCaloriesLabel}>cal avg</Text>
      </View>
    </View>
  );
});

interface InsightCardProps {
  insight: ReportInsight;
}

const InsightCard = React.memo(function InsightCard({ insight }: InsightCardProps) {
  const iconName = getCategoryIcon(insight.category);
  const priorityColor =
    insight.priority === 'high'
      ? colors.semantic.error
      : insight.priority === 'medium'
        ? colors.semantic.warning
        : colors.text.tertiary;

  return (
    <View style={styles.insightCard}>
      <View style={[styles.insightIcon, { backgroundColor: colors.special.highlight }]}>
        <Ionicons name={iconName} size={20} color={colors.primary.main} />
      </View>
      <View style={styles.insightContent}>
        <Text style={styles.insightTitle}>{insight.title}</Text>
        <Text style={styles.insightDescription}>{insight.description}</Text>
      </View>
      <View style={[styles.priorityDot, { backgroundColor: priorityColor }]} />
    </View>
  );
});

interface AchievementCardProps {
  achievement: ReportAchievement;
}

const AchievementCard = React.memo(function AchievementCard({ achievement }: AchievementCardProps) {
  const iconName = getAchievementIcon(achievement.type);

  return (
    <View style={styles.achievementCard}>
      <LinearGradient colors={gradients.primary} style={styles.achievementIconContainer}>
        <Ionicons name={iconName} size={24} color={colors.text.primary} />
      </LinearGradient>
      <Text style={styles.achievementTitle} numberOfLines={2}>
        {achievement.title}
      </Text>
    </View>
  );
});

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function getCategoryIcon(category: ReportInsight['category']): keyof typeof Ionicons.glyphMap {
  switch (category) {
    case 'nutrition':
      return 'nutrition-outline';
    case 'health':
      return 'heart-outline';
    case 'activity':
      return 'fitness-outline';
    case 'achievement':
      return 'trophy-outline';
    case 'recommendation':
      return 'bulb-outline';
    default:
      return 'information-circle-outline';
  }
}

function getAchievementIcon(type: ReportAchievement['type']): keyof typeof Ionicons.glyphMap {
  switch (type) {
    case 'streak':
      return 'flame';
    case 'milestone':
      return 'flag';
    case 'goal':
      return 'checkmark-circle';
    case 'improvement':
      return 'trending-up';
    default:
      return 'star';
  }
}

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
  headerActions: {
    flexDirection: 'row',
    gap: spacing.xs,
  },
  headerButton: {
    padding: spacing.sm,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: spacing.md,
    paddingBottom: spacing.xl * 2,
  },
  weekSelector: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.lg,
  },
  weekNavButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  weekNavButtonDisabled: {
    opacity: 0.5,
  },
  weekNavGradient: {
    padding: spacing.sm,
    borderRadius: borderRadius.md,
  },
  weekInfo: {
    alignItems: 'center',
  },
  weekLabel: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  currentWeekBadge: {
    fontSize: typography.fontSize.xs,
    color: colors.primary.main,
    marginTop: spacing.xs,
  },
  statsContainer: {
    flexDirection: 'row',
    gap: spacing.sm,
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
  streakCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  streakContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  streakInfo: {
    alignItems: 'flex-start',
  },
  streakValue: {
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  streakLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  streakBest: {
    alignItems: 'center',
  },
  streakBestLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  streakBestValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.secondary,
  },
  section: {
    marginBottom: spacing.lg,
  },
  sectionTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  trendsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  trendBadge: {
    flex: 1,
    minWidth: '45%',
    padding: spacing.md,
    borderRadius: borderRadius.md,
    alignItems: 'center',
  },
  trendLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  trendValueRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  trendPercent: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
  },
  chartCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  chart: {
    borderRadius: borderRadius.md,
    marginLeft: -spacing.md,
  },
  chartLegend: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.lg,
    marginTop: spacing.md,
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
  macroDistCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  macroDistBar: {
    flexDirection: 'row',
    height: 24,
    borderRadius: borderRadius.sm,
    overflow: 'hidden',
    marginBottom: spacing.md,
  },
  macroDistSegment: {
    height: '100%',
  },
  macroDistLabels: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  macroDistItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  macroDistLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.secondary,
  },
  dayCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  dayInfo: {
    width: 60,
  },
  dayName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  dayDate: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  dayStats: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  dayStat: {
    alignItems: 'center',
  },
  dayStatValue: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  dayStatLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  dayCompletion: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    width: 60,
    justifyContent: 'flex-end',
  },
  dayCompletionText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
  },
  topFoodCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  topFoodRank: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.background.elevated,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  topFoodRankText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.secondary,
  },
  topFoodInfo: {
    flex: 1,
  },
  topFoodName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  topFoodCount: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  topFoodCalories: {
    alignItems: 'flex-end',
  },
  topFoodCaloriesValue: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.primary.main,
  },
  topFoodCaloriesLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  insightCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  insightIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  insightContent: {
    flex: 1,
  },
  insightTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  insightDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    lineHeight: 20,
  },
  priorityDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginLeft: spacing.sm,
    marginTop: 6,
  },
  achievementsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  achievementCard: {
    width: '48%',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  achievementIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.sm,
  },
  achievementTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    textAlign: 'center',
  },
});
