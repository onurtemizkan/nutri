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
import { mealsApi } from '@/lib/api/meals';
import { DailySummary, Meal } from '@/lib/types';
import { useAuth } from '@/lib/context/AuthContext';

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
          <ActivityIndicator size="large" color="#3b5998" />
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
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
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
            <View style={styles.calorieRing}>
              <View style={styles.calorieContent}>
                <Text style={styles.calorieValue}>{Math.round(summary?.totalCalories || 0)}</Text>
                <Text style={styles.calorieLabel}>/ {summary?.goals?.goalCalories || 2000}</Text>
              </View>
            </View>
            <Text style={styles.calorieSubtext}>Calories</Text>
            <View style={styles.progressBar}>
              <View
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
                <View
                  style={[
                    styles.macroProgressFill,
                    {
                      width: `${Math.min(((summary?.totalProtein || 0) / (summary?.goals?.goalProtein || 1)) * 100, 100)}%`,
                      backgroundColor: '#4CAF50'
                    }
                  ]}
                />
              </View>
            </View>

            <View style={styles.macroCard}>
              <Text style={styles.macroValue}>{Math.round(summary?.totalCarbs || 0)}g</Text>
              <Text style={styles.macroLabel}>Carbs</Text>
              <View style={styles.macroProgress}>
                <View
                  style={[
                    styles.macroProgressFill,
                    {
                      width: `${Math.min(((summary?.totalCarbs || 0) / (summary?.goals?.goalCarbs || 1)) * 100, 100)}%`,
                      backgroundColor: '#FF9800'
                    }
                  ]}
                />
              </View>
            </View>

            <View style={styles.macroCard}>
              <Text style={styles.macroValue}>{Math.round(summary?.totalFat || 0)}g</Text>
              <Text style={styles.macroLabel}>Fat</Text>
              <View style={styles.macroProgress}>
                <View
                  style={[
                    styles.macroProgressFill,
                    {
                      width: `${Math.min(((summary?.totalFat || 0) / (summary?.goals?.goalFat || 1)) * 100, 100)}%`,
                      backgroundColor: '#F44336'
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
      >
        <Text style={styles.fabText}>+</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
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
    padding: 20,
    paddingBottom: 100,
  },
  header: {
    marginBottom: 24,
  },
  greeting: {
    fontSize: 28,
    fontWeight: '700',
    color: '#000',
    marginBottom: 4,
  },
  date: {
    fontSize: 14,
    color: '#666',
  },
  summaryCard: {
    backgroundColor: '#fff',
    borderRadius: 20,
    padding: 24,
    alignItems: 'center',
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 3,
  },
  calorieRing: {
    width: 140,
    height: 140,
    borderRadius: 70,
    borderWidth: 12,
    borderColor: '#3b5998',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
  },
  calorieContent: {
    alignItems: 'center',
  },
  calorieValue: {
    fontSize: 36,
    fontWeight: '700',
    color: '#000',
  },
  calorieLabel: {
    fontSize: 16,
    color: '#666',
  },
  calorieSubtext: {
    fontSize: 16,
    color: '#666',
    marginBottom: 12,
  },
  progressBar: {
    width: '100%',
    height: 6,
    backgroundColor: '#e0e0e0',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#3b5998',
    borderRadius: 3,
  },
  macrosContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
    gap: 12,
  },
  macroCard: {
    flex: 1,
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  macroValue: {
    fontSize: 20,
    fontWeight: '700',
    color: '#000',
    marginBottom: 4,
  },
  macroLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 8,
  },
  macroProgress: {
    width: '100%',
    height: 4,
    backgroundColor: '#e0e0e0',
    borderRadius: 2,
    overflow: 'hidden',
  },
  macroProgressFill: {
    height: '100%',
    borderRadius: 2,
  },
  mealsSection: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#000',
    marginBottom: 16,
  },
  mealTypeSection: {
    marginBottom: 20,
  },
  mealTypeTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#666',
    marginBottom: 12,
  },
  mealCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  mealInfo: {
    flex: 1,
  },
  mealName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
    marginBottom: 4,
  },
  mealMacros: {
    fontSize: 12,
    color: '#666',
  },
  mealCalories: {
    fontSize: 16,
    fontWeight: '700',
    color: '#3b5998',
  },
  noMeals: {
    fontSize: 14,
    color: '#999',
    fontStyle: 'italic',
  },
  fab: {
    position: 'absolute',
    bottom: 24,
    right: 24,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#3b5998',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  fabText: {
    fontSize: 32,
    color: '#fff',
    fontWeight: '300',
  },
});
