import { useState, useCallback, useMemo, memo } from 'react';
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
import Svg, { Circle, Defs, LinearGradient as SvgLinearGradient, Stop } from 'react-native-svg';
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

/**
 * Valid meal types for the application
 * Using a type and constant array ensures compile-time type safety
 */
type MealType = 'breakfast' | 'lunch' | 'dinner' | 'snack';
const MEAL_TYPES: readonly MealType[] = ['breakfast', 'lunch', 'dinner', 'snack'] as const;

/**
 * Memoized meal type section component
 * Prevents re-renders when other parts of the screen update
 */
interface MealTypeSectionProps {
  mealType: MealType;
  meals: Meal[];
  onEdit: (meal: Meal) => void;
  onDelete: (meal: Meal) => void;
}

const MealTypeSection = memo(function MealTypeSection({
  mealType,
  meals,
  onEdit,
  onDelete,
}: MealTypeSectionProps) {
  const capitalizedType = mealType.charAt(0).toUpperCase() + mealType.slice(1);

  return (
    <View style={styles.mealTypeSection}>
      <Text style={styles.mealTypeTitle}>{capitalizedType}</Text>
      {meals.length > 0 ? (
        meals.map((meal) => (
          <SwipeableMealCard
            key={meal.id}
            meal={meal}
            onEdit={onEdit}
            onDelete={onDelete}
          />
        ))
      ) : (
        <Text style={styles.noMeals}>No meals yet</Text>
      )}
    </View>
  );
});

/**
 * Memoized trend card component
 * Prevents re-renders when other parts of the screen update
 */
interface TrendCardProps {
  metricType: HealthMetricType;
  data: { latest: HealthMetric | null; stats: HealthMetricStats | null };
  onPress: () => void;
}

const TrendCard = memo(function TrendCard({
  metricType,
  data,
  onPress,
}: TrendCardProps) {
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
      style={styles.trendCard}
      onPress={onPress}
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
});

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

  // Memoize expensive calculations to prevent recalculation on every render
  const calorieProgress = useMemo(() => {
    return summary && summary.goals
      ? (summary.totalCalories / summary.goals.goalCalories) * 100
      : 0;
  }, [summary?.totalCalories, summary?.goals?.goalCalories]);

  // Memoize meals grouped by type to prevent re-filtering on every render
  const mealsByType = useMemo((): Record<MealType, Meal[]> => {
    if (!summary?.meals) return { breakfast: [], lunch: [], dinner: [], snack: [] };
    return {
      breakfast: summary.meals.filter((meal) => meal.mealType === 'breakfast'),
      lunch: summary.meals.filter((meal) => meal.mealType === 'lunch'),
      dinner: summary.meals.filter((meal) => meal.mealType === 'dinner'),
      snack: summary.meals.filter((meal) => meal.mealType === 'snack'),
    };
  }, [summary?.meals]);

  // Memoize date string to prevent reformatting on every render
  const formattedDate = useMemo(() => {
    return new Date().toLocaleDateString('en-US', {
      weekday: 'long',
      month: 'long',
      day: 'numeric',
    });
  }, []);

  // Memoize handlers to prevent child re-renders
  const handleEditMeal = useCallback((meal: Meal) => {
    router.push(`/edit-meal/${meal.id}`);
  }, [router]);

  // Create stable references for trend card navigation
  const handleTrendPress = useCallback((metricType: HealthMetricType) => {
    router.push(`/health/${metricType}`);
  }, [router]);

  const handleViewAllTrends = useCallback(() => {
    router.push('/(tabs)/health');
  }, [router]);

  const handleAddMeal = useCallback(() => {
    router.push('/add-meal');
  }, [router]);

  const handleDeleteMeal = useCallback((meal: Meal) => {
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
  }, [loadSummary]);

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
              <Text style={styles.date}>{formattedDate}</Text>
            </View>
          </View>

          {/* Nutrition Summary - Macros + Calories */}
          <View style={[styles.nutritionCard, isTablet && styles.nutritionCardTablet]} testID="home-calorie-summary">
            {/* Macros - Left Side */}
            <View style={styles.macrosColumn} testID="home-macros-container">
              <View style={styles.macroRow}>
                <View style={styles.macroInfo}>
                  <Text style={styles.macroLabel}>Protein</Text>
                  <Text style={styles.macroValue}>{Math.round(summary?.totalProtein || 0)}g</Text>
                </View>
                <View style={styles.macroProgressContainer}>
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
              </View>

              <View style={styles.macroRow}>
                <View style={styles.macroInfo}>
                  <Text style={styles.macroLabel}>Carbs</Text>
                  <Text style={styles.macroValue}>{Math.round(summary?.totalCarbs || 0)}g</Text>
                </View>
                <View style={styles.macroProgressContainer}>
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
              </View>

              <View style={styles.macroRow}>
                <View style={styles.macroInfo}>
                  <Text style={styles.macroLabel}>Fat</Text>
                  <Text style={styles.macroValue}>{Math.round(summary?.totalFat || 0)}g</Text>
                </View>
                <View style={styles.macroProgressContainer}>
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
            </View>

            {/* Calories - Right Side */}
            <View style={styles.calorieSection}>
              <View style={styles.calorieRingContainer}>
                <Svg width={88} height={88} style={styles.calorieSvg}>
                  <Defs>
                    <SvgLinearGradient id="calorieGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <Stop offset="0%" stopColor={colors.primary.main} />
                      <Stop offset="100%" stopColor={colors.primary.light} />
                    </SvgLinearGradient>
                  </Defs>
                  {/* Background circle */}
                  <Circle
                    cx={44}
                    cy={44}
                    r={38}
                    stroke={colors.background.elevated}
                    strokeWidth={8}
                    fill="transparent"
                  />
                  {/* Progress circle */}
                  <Circle
                    cx={44}
                    cy={44}
                    r={38}
                    stroke="url(#calorieGradient)"
                    strokeWidth={8}
                    fill="transparent"
                    strokeLinecap="round"
                    strokeDasharray={2 * Math.PI * 38}
                    strokeDashoffset={2 * Math.PI * 38 * (1 - Math.min(calorieProgress / 100, 1))}
                    rotation={-90}
                    origin="44, 44"
                  />
                </Svg>
                <View style={styles.calorieContent}>
                  <Text style={styles.calorieValue}>{Math.round(summary?.totalCalories || 0)}</Text>
                  <Text style={styles.calorieGoal}>/ {summary?.goals?.goalCalories || 2000}</Text>
                </View>
              </View>
              <Text style={styles.calorieLabel}>Calories</Text>
            </View>
          </View>

          {/* Daily Micronutrient Summary (Food + Supplements) */}
          <DailyMicronutrientSummary
            meals={summary?.meals || []}
            supplements={supplementStatus?.supplements}
          />

          {/* 7-Day Health Trends */}
          {healthTrends && Object.keys(healthTrends).length > 0 && (
            <View style={styles.trendsSection} testID="home-health-trends">
              <View style={styles.trendsSectionHeader}>
                <Text style={styles.sectionTitle}>7-Day Trends</Text>
                <TouchableOpacity
                  onPress={handleViewAllTrends}
                  style={styles.viewAllButton}
                >
                  <Text style={styles.viewAllText}>View All</Text>
                  <Ionicons name="chevron-forward" size={16} color={colors.primary.main} />
                </TouchableOpacity>
              </View>
              <View style={[styles.trendsContainer, isTablet && styles.trendsContainerTablet]}>
                {DASHBOARD_METRICS.map((metricType) => (
                  <TrendCard
                    key={metricType}
                    metricType={metricType}
                    data={healthTrends[metricType]}
                    onPress={() => handleTrendPress(metricType)}
                  />
                ))}
              </View>
            </View>
          )}

          {/* Today's Meals */}
          <View style={styles.mealsSection} testID="home-meals-section">
            <Text style={styles.sectionTitle}>Today's Meals</Text>

            {MEAL_TYPES.map((mealType) => {
              const meals = mealsByType[mealType];
              return (
                <MealTypeSection
                  key={mealType}
                  mealType={mealType}
                  meals={meals}
                  onEdit={handleEditMeal}
                  onDelete={handleDeleteMeal}
                />
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
        onPress={handleAddMeal}
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

  // Nutrition Card (Combined Macros + Calories)
  nutritionCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    flexDirection: 'row',
    alignItems: 'center',
    ...shadows.md,
  },
  nutritionCardTablet: {
    maxWidth: 600,
    alignSelf: 'center',
    width: '100%',
  },

  // Macros - Left Side
  macrosColumn: {
    flex: 1,
    gap: spacing.sm,
  },
  macroRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  macroInfo: {
    width: 70,
  },
  macroLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    fontWeight: typography.fontWeight.semibold,
    textTransform: 'uppercase',
    letterSpacing: 0.3,
  },
  macroValue: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  macroProgressContainer: {
    flex: 1,
  },
  macroProgress: {
    height: 6,
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.xs,
    overflow: 'hidden',
  },
  macroProgressFill: {
    height: '100%',
    borderRadius: borderRadius.xs,
  },

  // Calories - Right Side
  calorieSection: {
    alignItems: 'center',
    marginLeft: spacing.md,
  },
  calorieRingContainer: {
    width: 88,
    height: 88,
    justifyContent: 'center',
    alignItems: 'center',
  },
  calorieSvg: {
    position: 'absolute',
  },
  calorieContent: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  calorieValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  calorieGoal: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  calorieLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.secondary,
    marginTop: spacing.xs,
    fontWeight: typography.fontWeight.medium,
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
