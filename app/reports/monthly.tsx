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
  MonthlyReport,
  WeeklyBreakdown,
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

export default function MonthlyReportScreen() {
  const [report, setReport] = useState<MonthlyReport | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentMonth, setCurrentMonth] = useState<string>(() => {
    const today = new Date();
    return `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}`;
  });
  const router = useRouter();

  const loadReport = useCallback(async () => {
    try {
      setError(null);
      const data = await reportsApi.getMonthlyReport(currentMonth);
      setReport(data);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to load monthly report'));
    } finally {
      setIsLoading(false);
      setRefreshing(false);
    }
  }, [currentMonth]);

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

  const handlePreviousMonth = useCallback(() => {
    const prevMonth = reportsApi.getPreviousMonth(currentMonth);
    setCurrentMonth(prevMonth);
    setIsLoading(true);
  }, [currentMonth]);

  const handleNextMonth = useCallback(() => {
    const nextMonth = reportsApi.getNextMonth(currentMonth);
    setCurrentMonth(nextMonth);
    setIsLoading(true);
  }, [currentMonth]);

  const handleShare = useCallback(async () => {
    if (!report) return;

    try {
      const monthLabel = reportsApi.formatMonth(report.month);
      const message =
        `Monthly Nutrition Report (${monthLabel})\n\n` +
        `Calories: ${Math.round(report.averages.calories)} avg/day\n` +
        `Protein: ${Math.round(report.averages.protein)}g avg/day\n` +
        `Carbs: ${Math.round(report.averages.carbs)}g avg/day\n` +
        `Fat: ${Math.round(report.averages.fat)}g avg/day\n\n` +
        `Days Tracked: ${report.totalDaysTracked}\n` +
        `Goals Met: ${report.daysGoalsMet} days (${Math.round(report.goalCompletionRate)}%)\n` +
        `Longest Streak: ${report.streak.longest} days`;

      await Share.share({ message });
    } catch (err) {
      console.error('Failed to share report:', err);
    }
  }, [report]);

  const handleExport = useCallback(() => {
    router.push(`/reports/export?type=monthly&month=${currentMonth}`);
  }, [router, currentMonth]);

  const isCurrentMonth = useMemo(() => {
    return reportsApi.isCurrentMonth(currentMonth);
  }, [currentMonth]);

  const monthLabel = useMemo(() => {
    return reportsApi.formatMonth(currentMonth);
  }, [currentMonth]);

  // Weekly comparison chart data
  const weeklyChartData = useMemo(() => {
    if (!report?.weeklyBreakdowns) return null;

    const labels = report.weeklyBreakdowns.map((w) => `W${w.weekNumber}`);
    const data = report.weeklyBreakdowns.map((w) => w.averages.calories);

    return {
      labels,
      datasets: [
        { data: data.length > 0 ? data : [0], color: () => colors.primary.main, strokeWidth: 2 },
      ],
    };
  }, [report?.weeklyBreakdowns]);

  // Goal completion bar chart data
  const goalChartData = useMemo(() => {
    if (!report?.weeklyBreakdowns) return null;

    const labels = report.weeklyBreakdowns.map((w) => `W${w.weekNumber}`);
    const data = report.weeklyBreakdowns.map((w) => w.goalCompletionRate);

    return {
      labels,
      datasets: [{ data: data.length > 0 ? data : [0] }],
    };
  }, [report?.weeklyBreakdowns]);

  // Daily calories trend
  const dailyCaloriesData = useMemo(() => {
    if (!report?.dailyBreakdowns || report.dailyBreakdowns.length === 0) return null;

    // Sample every other day for monthly view to keep chart readable
    const sampledData = report.dailyBreakdowns.filter(
      (_, i) => i % 2 === 0 || i === report.dailyBreakdowns.length - 1
    );

    const labels = sampledData.map((d) => {
      const date = new Date(d.date);
      return date.getDate().toString();
    });

    return {
      labels,
      datasets: [
        {
          data: sampledData.map((d) => d.calories),
          color: () => colors.primary.main,
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
          <Text style={styles.loadingText}>Loading monthly report...</Text>
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
          <Text style={styles.headerTitle}>Monthly Report</Text>
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
        <Text style={styles.headerTitle}>Monthly Report</Text>
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
          {/* Month Selector */}
          <View style={styles.monthSelector}>
            <TouchableOpacity onPress={handlePreviousMonth} style={styles.monthNavButton}>
              <LinearGradient
                colors={[colors.background.tertiary, colors.background.elevated]}
                style={styles.monthNavGradient}
              >
                <Ionicons name="chevron-back" size={20} color={colors.text.primary} />
              </LinearGradient>
            </TouchableOpacity>

            <View style={styles.monthInfo}>
              <Text style={styles.monthLabel}>{monthLabel}</Text>
              {isCurrentMonth && <Text style={styles.currentMonthBadge}>Current Month</Text>}
            </View>

            <TouchableOpacity
              onPress={handleNextMonth}
              style={[styles.monthNavButton, isCurrentMonth && styles.monthNavButtonDisabled]}
              disabled={isCurrentMonth}
            >
              <LinearGradient
                colors={
                  isCurrentMonth
                    ? [colors.background.secondary, colors.background.secondary]
                    : [colors.background.tertiary, colors.background.elevated]
                }
                style={styles.monthNavGradient}
              >
                <Ionicons
                  name="chevron-forward"
                  size={20}
                  color={isCurrentMonth ? colors.text.disabled : colors.text.primary}
                />
              </LinearGradient>
            </TouchableOpacity>
          </View>

          {report && (
            <>
              {/* Summary Stats */}
              <View style={styles.statsContainer}>
                <View style={styles.statCard}>
                  <Text style={styles.statValue}>{report.totalDaysTracked}</Text>
                  <Text style={styles.statLabel}>Days Tracked</Text>
                </View>
                <View style={styles.statCard}>
                  <Text style={styles.statValue}>{report.daysGoalsMet}</Text>
                  <Text style={styles.statLabel}>Goals Met</Text>
                </View>
                <View style={styles.statCard}>
                  <Text style={[styles.statValue, { color: colors.semantic.success }]}>
                    {Math.round(report.goalCompletionRate)}%
                  </Text>
                  <Text style={styles.statLabel}>Success Rate</Text>
                </View>
              </View>

              {/* Averages Card */}
              <View style={styles.averagesCard}>
                <Text style={styles.cardTitle}>Daily Averages</Text>
                <View style={styles.averagesGrid}>
                  <View style={styles.averageItem}>
                    <Text style={styles.averageValue}>{Math.round(report.averages.calories)}</Text>
                    <Text style={styles.averageLabel}>Calories</Text>
                  </View>
                  <View style={styles.averageItem}>
                    <Text style={[styles.averageValue, { color: colors.semantic.success }]}>
                      {Math.round(report.averages.protein)}g
                    </Text>
                    <Text style={styles.averageLabel}>Protein</Text>
                  </View>
                  <View style={styles.averageItem}>
                    <Text style={[styles.averageValue, { color: '#EC4899' }]}>
                      {Math.round(report.averages.carbs)}g
                    </Text>
                    <Text style={styles.averageLabel}>Carbs</Text>
                  </View>
                  <View style={styles.averageItem}>
                    <Text style={[styles.averageValue, { color: colors.secondary.main }]}>
                      {Math.round(report.averages.fat)}g
                    </Text>
                    <Text style={styles.averageLabel}>Fat</Text>
                  </View>
                </View>
              </View>

              {/* Streak Section */}
              <View style={styles.streakCard}>
                <View style={styles.streakContent}>
                  <Ionicons name="flame" size={32} color={colors.semantic.warning} />
                  <View style={styles.streakInfo}>
                    <Text style={styles.streakValue}>{report.streak.current}</Text>
                    <Text style={styles.streakLabel}>Current Streak</Text>
                  </View>
                </View>
                <View style={styles.streakStats}>
                  <View style={styles.streakStatItem}>
                    <Text style={styles.streakStatValue}>{report.streak.longest}</Text>
                    <Text style={styles.streakStatLabel}>Longest</Text>
                  </View>
                  {report.streak.longestThisMonth !== undefined && (
                    <View style={styles.streakStatItem}>
                      <Text style={styles.streakStatValue}>{report.streak.longestThisMonth}</Text>
                      <Text style={styles.streakStatLabel}>This Month</Text>
                    </View>
                  )}
                </View>
              </View>

              {/* Trends vs Last Month */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Trends vs Last Month</Text>
                <View style={styles.trendsGrid}>
                  <TrendBadge label="Calories" trend={report.trends.calories} />
                  <TrendBadge label="Protein" trend={report.trends.protein} />
                  <TrendBadge label="Carbs" trend={report.trends.carbs} />
                  <TrendBadge label="Fat" trend={report.trends.fat} />
                </View>
              </View>

              {/* Weekly Comparison Chart */}
              {weeklyChartData && report.weeklyBreakdowns.length > 0 && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Weekly Calories</Text>
                  <View style={styles.chartCard}>
                    <BarChart
                      data={weeklyChartData}
                      width={CHART_WIDTH}
                      height={180}
                      chartConfig={{
                        backgroundColor: colors.background.tertiary,
                        backgroundGradientFrom: colors.background.tertiary,
                        backgroundGradientTo: colors.background.tertiary,
                        decimalPlaces: 0,
                        color: (opacity = 1) => `rgba(139, 92, 246, ${opacity})`,
                        labelColor: () => colors.text.tertiary,
                        barPercentage: 0.6,
                        style: { borderRadius: borderRadius.md },
                      }}
                      style={styles.chart}
                      yAxisLabel=""
                      yAxisSuffix=""
                    />
                  </View>
                </View>
              )}

              {/* Daily Calories Trend */}
              {dailyCaloriesData && report.dailyBreakdowns.length > 0 && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Daily Calorie Trend</Text>
                  <View style={styles.chartCard}>
                    <LineChart
                      data={dailyCaloriesData}
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
                          r: '3',
                          strokeWidth: '1',
                          stroke: colors.primary.main,
                        },
                      }}
                      bezier
                      style={styles.chart}
                    />
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

              {/* Weekly Breakdowns */}
              {report.weeklyBreakdowns.length > 0 && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Weekly Summary</Text>
                  {report.weeklyBreakdowns.map((week) => (
                    <WeeklyBreakdownCard key={week.weekNumber} week={week} />
                  ))}
                </View>
              )}

              {/* Best Days */}
              {report.bestDays.length > 0 && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Best Days</Text>
                  {report.bestDays.slice(0, 3).map((day) => (
                    <DayHighlightCard key={day.date} day={day} type="best" />
                  ))}
                </View>
              )}

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

              {/* Year over Year (if available) */}
              {report.yearOverYear.available && report.yearOverYear.previousYearAvg && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Year over Year</Text>
                  <View style={styles.yoyCard}>
                    <View style={styles.yoyItem}>
                      <Text style={styles.yoyLabel}>Last Year Avg Calories</Text>
                      <Text style={styles.yoyValue}>
                        {Math.round(report.yearOverYear.previousYearAvg.calories)}
                      </Text>
                    </View>
                    {report.yearOverYear.percentChange !== null && (
                      <View style={styles.yoyChange}>
                        <Ionicons
                          name={
                            report.yearOverYear.percentChange >= 0 ? 'trending-up' : 'trending-down'
                          }
                          size={20}
                          color={
                            report.yearOverYear.percentChange >= 0
                              ? colors.semantic.success
                              : colors.semantic.warning
                          }
                        />
                        <Text
                          style={[
                            styles.yoyChangeText,
                            {
                              color:
                                report.yearOverYear.percentChange >= 0
                                  ? colors.semantic.success
                                  : colors.semantic.warning,
                            },
                          ]}
                        >
                          {report.yearOverYear.percentChange >= 0 ? '+' : ''}
                          {Math.round(report.yearOverYear.percentChange)}%
                        </Text>
                      </View>
                    )}
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

interface WeeklyBreakdownCardProps {
  week: WeeklyBreakdown;
}

const WeeklyBreakdownCard = React.memo(function WeeklyBreakdownCard({
  week,
}: WeeklyBreakdownCardProps) {
  const startDate = new Date(week.weekStart);
  const endDate = new Date(week.weekEnd);
  const dateRange = `${startDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${endDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`;

  return (
    <View style={styles.weekCard}>
      <View style={styles.weekHeader}>
        <View>
          <Text style={styles.weekTitle}>Week {week.weekNumber}</Text>
          <Text style={styles.weekDates}>{dateRange}</Text>
        </View>
        <View style={styles.weekCompletion}>
          <Text
            style={[
              styles.weekCompletionText,
              {
                color:
                  week.goalCompletionRate >= 70 ? colors.semantic.success : colors.text.tertiary,
              },
            ]}
          >
            {Math.round(week.goalCompletionRate)}%
          </Text>
          {week.goalCompletionRate >= 70 && (
            <Ionicons name="checkmark-circle" size={16} color={colors.semantic.success} />
          )}
        </View>
      </View>
      <View style={styles.weekStats}>
        <View style={styles.weekStat}>
          <Text style={styles.weekStatValue}>{Math.round(week.averages.calories)}</Text>
          <Text style={styles.weekStatLabel}>cal/day</Text>
        </View>
        <View style={styles.weekStat}>
          <Text style={styles.weekStatValue}>{Math.round(week.averages.protein)}g</Text>
          <Text style={styles.weekStatLabel}>protein</Text>
        </View>
        <View style={styles.weekStat}>
          <Text style={styles.weekStatValue}>{week.daysTracked}</Text>
          <Text style={styles.weekStatLabel}>days</Text>
        </View>
        <View style={styles.weekStat}>
          <Text style={styles.weekStatValue}>{week.daysGoalsMet}</Text>
          <Text style={styles.weekStatLabel}>goals met</Text>
        </View>
      </View>
    </View>
  );
});

interface DayHighlightCardProps {
  day: DailyBreakdown;
  type: 'best' | 'worst';
}

const DayHighlightCard = React.memo(function DayHighlightCard({
  day,
  type,
}: DayHighlightCardProps) {
  const date = new Date(day.date);
  const dateStr = date.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  });

  return (
    <View style={styles.dayHighlightCard}>
      <View style={styles.dayHighlightHeader}>
        <Ionicons
          name={type === 'best' ? 'star' : 'alert-circle'}
          size={20}
          color={type === 'best' ? colors.semantic.success : colors.semantic.warning}
        />
        <Text style={styles.dayHighlightDate}>{dateStr}</Text>
      </View>
      <View style={styles.dayHighlightStats}>
        <Text style={styles.dayHighlightCalories}>{Math.round(day.calories)} cal</Text>
        <Text style={styles.dayHighlightMacros}>
          {Math.round(day.protein)}g P | {Math.round(day.carbs)}g C | {Math.round(day.fat)}g F
        </Text>
      </View>
      <View style={styles.dayHighlightCompletion}>
        <Text style={[styles.dayHighlightCompletionText, { color: colors.semantic.success }]}>
          {Math.round(day.goalCompletion)}% goal
        </Text>
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
        <Text style={styles.topFoodCount}>{food.count}x this month</Text>
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
  monthSelector: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.lg,
  },
  monthNavButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  monthNavButtonDisabled: {
    opacity: 0.5,
  },
  monthNavGradient: {
    padding: spacing.sm,
    borderRadius: borderRadius.md,
  },
  monthInfo: {
    alignItems: 'center',
  },
  monthLabel: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  currentMonthBadge: {
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
  averagesCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  cardTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.md,
  },
  averagesGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  averageItem: {
    alignItems: 'center',
  },
  averageValue: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  averageLabel: {
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
  streakStats: {
    flexDirection: 'row',
    gap: spacing.lg,
  },
  streakStatItem: {
    alignItems: 'center',
  },
  streakStatValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.secondary,
  },
  streakStatLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
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
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  weekCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  weekHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: spacing.sm,
  },
  weekTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  weekDates: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  weekCompletion: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  weekCompletionText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
  },
  weekStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  weekStat: {
    alignItems: 'center',
  },
  weekStatValue: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  weekStatLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  dayHighlightCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    flexDirection: 'row',
    alignItems: 'center',
  },
  dayHighlightHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    width: 120,
  },
  dayHighlightDate: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
  },
  dayHighlightStats: {
    flex: 1,
  },
  dayHighlightCalories: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  dayHighlightMacros: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  dayHighlightCompletion: {
    alignItems: 'flex-end',
  },
  dayHighlightCompletionText: {
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
  yoyCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  yoyItem: {},
  yoyLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  yoyValue: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginTop: spacing.xs,
  },
  yoyChange: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  yoyChangeText: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
  },
});
