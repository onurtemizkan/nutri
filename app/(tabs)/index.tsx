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
import { mealsApi } from '@/lib/api/meals';
import { DailySummary, Meal } from '@/lib/types';
import { useAuth } from '@/lib/context/AuthContext';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';

export default function HomeScreen() {
  const [summary, setSummary] = useState<DailySummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { user } = useAuth();
  const router = useRouter();

  const loadSummary = useCallback(async () => {
    // Only load if user is authenticated
    if (!user) {
      setIsLoading(false);
      return;
    }

    try {
      const data = await mealsApi.getDailySummary();
      setSummary(data);
    } catch (error) {
      console.error('Failed to load summary:', error);
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadSummary();
    setRefreshing(false);
  }, [loadSummary]);

  useEffect(() => {
    loadSummary();
  }, [loadSummary]);

  const calorieProgress =
    summary && summary.goals
      ? (summary.totalCalories / summary.goals.goalCalories) * 100
      : 0;

  const getMealsByType = (type: string) => {
    return summary?.meals.filter((meal) => meal.mealType === type) || [];
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
              <Text style={styles.greeting}>Hello, {user?.name}!</Text>
              <Text style={styles.date}>{new Date().toLocaleDateString('en-US', {
                weekday: 'long',
                month: 'long',
                day: 'numeric'
              })}</Text>
            </View>
          </View>

          {/* Calorie Summary Card */}
          <View style={styles.summaryCard}>
            <LinearGradient
              colors={gradients.primary}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={styles.calorieRing}
            >
              <View style={styles.calorieContent}>
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
          <View style={styles.macrosContainer}>
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

          {/* Today's Meals */}
          <View style={styles.mealsSection}>
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
                      <View key={meal.id} style={styles.mealCard}>
                        <View style={styles.mealInfo}>
                          <Text style={styles.mealName}>{meal.name}</Text>
                          <Text style={styles.mealMacros}>
                            P: {Math.round(meal.protein)}g • C: {Math.round(meal.carbs)}g • F: {Math.round(meal.fat)}g
                          </Text>
                        </View>
                        <Text style={styles.mealCalories}>{Math.round(meal.calories)} cal</Text>
                      </View>
                    ))
                  ) : (
                    <Text style={styles.noMeals}>No meals yet</Text>
                  )}
                </View>
              );
            })}
          </View>
        </View>
      </ScrollView>

      {/* Floating Add Button */}
      <TouchableOpacity
        style={styles.fab}
        onPress={() => router.push('/add-meal')}
        activeOpacity={0.8}
      >
        <LinearGradient
          colors={gradients.primary}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.fabGradient}
        >
          <Text style={styles.fabText}>+</Text>
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
    padding: spacing.lg,
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
  fabText: {
    fontSize: 32,
    color: colors.text.primary,
    fontWeight: typography.fontWeight.regular,
  },
});
