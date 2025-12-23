import { useState, useCallback, useRef } from 'react';
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
import { useFocusEffect } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { mealsApi } from '@/lib/api/meals';
import { supplementsApi } from '@/lib/api/supplements';
import { healthMetricsApi } from '@/lib/api/health-metrics';
import { DailySummary, Meal, TodaySupplementStatus } from '@/lib/types';
import {
  HealthMetricType,
  HealthMetric,
  HealthMetricStats,
  DASHBOARD_METRICS,
  METRIC_CONFIG,
} from '@/lib/types/health-metrics';
import { useAuth } from '@/lib/context/AuthContext';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { SwipeableMealCard } from '@/lib/components/SwipeableMealCard';
import { SupplementTracker } from '@/lib/components/SupplementTracker';
import { DailyMicronutrientSummary } from '@/lib/components/DailyMicronutrientSummary';
import { showAlert } from '@/lib/utils/alert';
import { getErrorMessage } from '@/lib/utils/errorHandling';

type HealthTrendsData = Record<
  HealthMetricType,
  { latest: HealthMetric | null; stats: HealthMetricStats | null }
>;

export default function HomeScreen() {
  const [summary, setSummary] = useState<DailySummary | null>(null);
  const [supplementStatus, setSupplementStatus] = useState<TodaySupplementStatus | null>(null);
  const [healthTrends, setHealthTrends] = useState<HealthTrendsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [supplementKey, setSupplementKey] = useState(0);
  const { user } = useAuth();
  const router = useRouter();
  const { isTablet, deviceCategory, getResponsiveValue, scale } = useResponsive();

  // Responsive values
  const contentPadding = getResponsiveValue({
    small: spacing.md,
    medium: spacing.lg,
    large: spacing.lg,
    tablet: spacing.xl,
    default: spacing.lg,
  });
  const calorieRingSize = getResponsiveValue({
    small: 120,
    medium: 140,
    large: 160,
    tablet: 180,
    default: 140,
  });
  const fabSize = getResponsiveValue({
    small: 52,
    medium: 56,
    large: 60,
    tablet: 64,
    default: 56,
  });

  const loadSummary = useCallback(async () => {
    // Only load if user is authenticated
    if (!user) {
      setIsLoading(false);
      return;
    }

    try {
      // Load meals, supplements, and health trends in parallel
      const [mealsData, supplementsData, trendsData] = await Promise.all([
        mealsApi.getDailySummary(),
        supplementsApi.getTodayStatus().catch(() => null), // Don't fail if supplements fail
        healthMetricsApi.getDashboardData(DASHBOARD_METRICS, 7).catch(() => null), // 7-day trends
      ]);
      setSummary(mealsData);
      setSupplementStatus(supplementsData);
      setHealthTrends(trendsData);
    } catch (error) {
      console.error('Failed to load summary:', error);
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadSummary();
    // Refresh supplement tracker
    setSupplementKey(k => k + 1);
    setRefreshing(false);
  }, [loadSummary]);

  // Use useFocusEffect to reload data when screen comes into focus
  // This ensures data is refreshed after adding/editing a meal or supplement
  useFocusEffect(
    useCallback(() => {
      loadSummary();
      // Refresh supplement tracker by incrementing key
      setSupplementKey(k => k + 1);
    }, [loadSummary])
  );

  const calorieProgress =
    summary && summary.goals
      ? (summary.totalCalories / summary.goals.goalCalories) * 100
      : 0;

  const getMealsByType = (type: string) => {
    return summary?.meals.filter((meal) => meal.mealType === type) || [];
  };

  const handleEditMeal = (meal: Meal) => {
    router.push(`/edit-meal/${meal.id}`);
  };

  const handleDeleteMeal = (meal: Meal) => {
    showAlert(
      'Delete Meal',
      `Are you sure you want to delete "${meal.name}"?`,
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await mealsApi.deleteMeal(meal.id);
              // Refresh the summary to reflect changes
              await loadSummary();
            } catch (error) {
              showAlert('Error', getErrorMessage(error, 'Failed to delete meal'));
            }
          },
        },
      ]
    );
  };

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
    <SafeAreaView style={styles.container} testID="home-screen">
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
            <View>
              <Text style={styles.greeting}>Hello, {user?.name}!</Text>
              <Text style={styles.date}>{new Date().toLocaleDateString('en-US', {
                weekday: 'long',
                month: 'long',
                day: 'numeric'
              })}</Text>
            </View>
          </View>

          {/* Calorie Summary Card */}
          <View style={[styles.summaryCard, isTablet && styles.summaryCardTablet]} testID="home-calorie-summary">
            <LinearGradient
              colors={gradients.primary}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={[
                styles.calorieRing,
                { width: calorieRingSize, height: calorieRingSize, borderRadius: calorieRingSize / 2 }
              ]}
            >
              <View style={[
                styles.calorieContent,
                { width: calorieRingSize - 24, height: calorieRingSize - 24, borderRadius: (calorieRingSize - 24) / 2 }
              ]}>
                <Text style={styles.calorieValue}>{Math.round(summary?.totalCalories || 0)}</Text>
                <Text style={styles.calorieLabel}>/ {summary?.goals?.goalCalories || 2000}</Text>
              </View>
            </LinearGradient>
            <Text style={styles.calorieSubtext}>Calories</Text>
            <View style={styles.progressBar}>
              <LinearGradient
                colors={gradients.primary}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={[
                  styles.progressFill,
                  { width: `${Math.min(calorieProgress, 100)}%` }
                ]}
              />
            </View>
          </View>

          {/* Macros */}
          <View style={[styles.macrosContainer, isTablet && styles.macrosContainerTablet]} testID="home-macros-container">
            <View style={styles.macroCard}>
              <Text style={styles.macroValue}>{Math.round(summary?.totalProtein || 0)}g</Text>
              <Text style={styles.macroLabel}>Protein</Text>
              <View style={styles.macroProgress}>
                <LinearGradient
                  colors={gradients.success}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={[
                    styles.macroProgressFill,
                    {
                      width: `${Math.min(((summary?.totalProtein || 0) / (summary?.goals?.goalProtein || 1)) * 100, 100)}%`,
                    }
                  ]}
                />
              </View>
            </View>

            <View style={styles.macroCard}>
              <Text style={styles.macroValue}>{Math.round(summary?.totalCarbs || 0)}g</Text>
              <Text style={styles.macroLabel}>Carbs</Text>
              <View style={styles.macroProgress}>
                <LinearGradient
                  colors={gradients.accent}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={[
                    styles.macroProgressFill,
                    {
                      width: `${Math.min(((summary?.totalCarbs || 0) / (summary?.goals?.goalCarbs || 1)) * 100, 100)}%`,
                    }
                  ]}
                />
              </View>
            </View>

            <View style={styles.macroCard}>
              <Text style={styles.macroValue}>{Math.round(summary?.totalFat || 0)}g</Text>
              <Text style={styles.macroLabel}>Fat</Text>
              <View style={styles.macroProgress}>
                <LinearGradient
                  colors={gradients.secondary}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={[
                    styles.macroProgressFill,
                    {
                      width: `${Math.min(((summary?.totalFat || 0) / (summary?.goals?.goalFat || 1)) * 100, 100)}%`,
                    }
                  ]}
                />
              </View>
            </View>
          </View>

          {/* 7-Day Health Trends */}
          {healthTrends && Object.keys(healthTrends).length > 0 && (
            <View style={styles.trendsSection} testID="home-health-trends">
              <View style={styles.trendsSectionHeader}>
                <Text style={styles.sectionTitle}>7-Day Trends</Text>
                <TouchableOpacity
                  onPress={() => router.push('/(tabs)/health')}
                  style={styles.viewAllButton}
                >
                  <Text style={styles.viewAllText}>View All</Text>
                  <Ionicons name="chevron-forward" size={16} color={colors.primary.main} />
                </TouchableOpacity>
              </View>
              <View style={[styles.trendsContainer, isTablet && styles.trendsContainerTablet]}>
                {DASHBOARD_METRICS.map((metricType) => {
                  const data = healthTrends[metricType];
                  if (!data?.latest) return null;

                  const config = METRIC_CONFIG[metricType];
                  const trendIcon = data.stats?.trend === 'up'
                    ? 'trending-up'
                    : data.stats?.trend === 'down'
                      ? 'trending-down'
                      : 'remove';
                  const trendColor = data.stats?.trend === 'up'
                    ? (metricType === 'RESTING_HEART_RATE' ? colors.status.warning : colors.status.success)
                    : data.stats?.trend === 'down'
                      ? (metricType === 'RESTING_HEART_RATE' ? colors.status.success : colors.status.warning)
                      : colors.text.tertiary;

                  return (
                    <TouchableOpacity
                      key={metricType}
                      style={styles.trendCard}
                      onPress={() => router.push(`/health/${metricType}`)}
                      activeOpacity={0.7}
                    >
                      <View style={styles.trendCardHeader}>
                        <Ionicons
                          name={config.icon as keyof typeof Ionicons.glyphMap}
                          size={18}
                          color={colors.primary.main}
                        />
                        <Text style={styles.trendCardLabel}>{config.shortName}</Text>
                      </View>
                      <View style={styles.trendCardValue}>
                        <Text style={styles.trendValueText}>
                          {metricType === 'SLEEP_DURATION'
                            ? `${data.latest.value.toFixed(1)}h`
                            : Math.round(data.latest.value)}
                        </Text>
                        <Text style={styles.trendUnitText}>{config.unit}</Text>
                      </View>
                      {data.stats && (
                        <View style={styles.trendIndicator}>
                          <Ionicons name={trendIcon} size={14} color={trendColor} />
                          <Text style={[styles.trendChangeText, { color: trendColor }]}>
                            {Math.abs(data.stats.percentChange).toFixed(0)}%
                          </Text>
                        </View>
                      )}
                    </TouchableOpacity>
                  );
                })}
              </View>
            </View>
          )}

          {/* Daily Micronutrient Summary (Food + Supplements) */}
          <DailyMicronutrientSummary
            meals={summary?.meals || []}
            supplements={supplementStatus?.supplements}
          />

          {/* Today's Meals */}
          <View style={styles.mealsSection} testID="home-meals-section">
            <Text style={styles.sectionTitle}>Today's Meals</Text>

            {['breakfast', 'lunch', 'dinner', 'snack'].map((mealType) => {
              const meals = getMealsByType(mealType);
              return (
                <View key={mealType} style={styles.mealTypeSection}>
                  <Text style={styles.mealTypeTitle}>
                    {mealType.charAt(0).toUpperCase() + mealType.slice(1)}
                  </Text>
                  {meals.length > 0 ? (
                    meals.map((meal: Meal) => (
                      <SwipeableMealCard
                        key={meal.id}
                        meal={meal}
                        onEdit={handleEditMeal}
                        onDelete={handleDeleteMeal}
                      />
                    ))
                  ) : (
                    <Text style={styles.noMeals}>No meals yet</Text>
                  )}
                </View>
              );
            })}
          </View>

          {/* Today's Supplements */}
          <SupplementTracker
            key={supplementKey}
            onRefreshNeeded={() => setSupplementKey(k => k + 1)}
          />
        </View>
      </ScrollView>

      {/* Floating Add Button */}
      <TouchableOpacity
        style={[
          styles.fab,
          { width: fabSize, height: fabSize, borderRadius: fabSize / 2 }
        ]}
        onPress={() => router.push('/add-meal')}
        activeOpacity={0.8}
        testID="home-add-meal-fab"
      >
        <LinearGradient
          colors={gradients.primary}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.fabGradient}
        >
          <Text style={[styles.fabText, { fontSize: fabSize * 0.5 }]}>+</Text>
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
    marginBottom: spacing.xl,
  },
  greeting: {
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
    letterSpacing: -0.5,
  },
  date: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    fontWeight: typography.fontWeight.medium,
  },

  // Summary Card
  summaryCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    alignItems: 'center',
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.md,
  },
  summaryCardTablet: {
    maxWidth: 500,
    alignSelf: 'center',
    width: '100%',
  },
  calorieRing: {
    width: 140,
    height: 140,
    borderRadius: 70,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
    ...shadows.glow,
  },
  calorieContent: {
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    width: 116,
    height: 116,
    borderRadius: 58,
    justifyContent: 'center',
  },
  calorieValue: {
    fontSize: typography.fontSize['4xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  calorieLabel: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
  calorieSubtext: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
    marginBottom: spacing.md,
    fontWeight: typography.fontWeight.medium,
  },
  progressBar: {
    width: '100%',
    height: 6,
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.xs,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: borderRadius.xs,
  },

  // Macros
  macrosContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.xl,
    gap: spacing.md,
  },
  macrosContainerTablet: {
    maxWidth: 600,
    alignSelf: 'center',
    width: '100%',
  },
  macroCard: {
    flex: 1,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  macroValue: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  macroLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginBottom: spacing.sm,
    fontWeight: typography.fontWeight.semibold,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  macroProgress: {
    width: '100%',
    height: 4,
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.xs,
    overflow: 'hidden',
  },
  macroProgressFill: {
    height: '100%',
    borderRadius: borderRadius.xs,
  },

  // Health Trends
  trendsSection: {
    marginBottom: spacing.xl,
  },
  trendsSectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  viewAllButton: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  viewAllText: {
    fontSize: typography.fontSize.sm,
    color: colors.primary.main,
    fontWeight: typography.fontWeight.medium,
  },
  trendsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  trendsContainerTablet: {
    maxWidth: 600,
    alignSelf: 'center',
    width: '100%',
  },
  trendCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    width: '48%',
    minWidth: 140,
    ...shadows.sm,
  },
  trendCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginBottom: spacing.sm,
  },
  trendCardLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    fontWeight: typography.fontWeight.semibold,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  trendCardValue: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: spacing.xs,
  },
  trendValueText: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  trendUnitText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginLeft: spacing.xs,
  },
  trendIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  trendChangeText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
  },

  // Meals Section
  mealsSection: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.md,
    letterSpacing: -0.5,
  },
  mealTypeSection: {
    marginBottom: spacing.lg,
  },
  mealTypeTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
    textTransform: 'capitalize',
  },
  mealCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.sm,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  mealInfo: {
    flex: 1,
  },
  mealName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  mealMacros: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  mealCalories: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.primary.main,
  },
  noMeals: {
    fontSize: typography.fontSize.sm,
    color: colors.text.disabled,
    fontStyle: 'italic',
    paddingVertical: spacing.sm,
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
  fabText: {
    color: colors.text.primary,
    fontWeight: typography.fontWeight.regular,
  },
});
