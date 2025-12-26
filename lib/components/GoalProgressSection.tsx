import React, { memo, useState, useCallback } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useRouter } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography, shadows, gradients } from '@/lib/theme/colors';
import { GoalCard, StreakCard, WeeklyTrendCard } from './GoalCard';
import { goalsApi, GoalProgressDashboard } from '@/lib/api/goals';
import { getErrorMessage } from '@/lib/utils/errorHandling';

interface GoalProgressSectionProps {
  /** Callback when data changes */
  onDataLoaded?: (data: GoalProgressDashboard) => void;
}

/**
 * Goal Progress Section component for the dashboard home screen
 * Shows today's progress, streak, and weekly trends
 */
export const GoalProgressSection = memo(function GoalProgressSection({
  onDataLoaded,
}: GoalProgressSectionProps) {
  const [data, setData] = useState<GoalProgressDashboard | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const loadData = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const dashboardData = await goalsApi.getDashboardProgress();
      setData(dashboardData);
      onDataLoaded?.(dashboardData);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to load goal progress'));
    } finally {
      setIsLoading(false);
    }
  }, [onDataLoaded]);

  // Reload data when screen comes into focus
  useFocusEffect(
    useCallback(() => {
      loadData();
    }, [loadData])
  );

  const handleViewAllPress = useCallback(() => {
    router.push('/goals');
  }, [router]);

  if (isLoading && !data) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="small" color={colors.primary.main} />
      </View>
    );
  }

  if (error && !data) {
    return (
      <View style={styles.errorContainer}>
        <Ionicons name="warning" size={24} color={colors.semantic.warning} />
        <Text style={styles.errorText}>{error}</Text>
        <TouchableOpacity onPress={loadData} style={styles.retryButton}>
          <Text style={styles.retryText}>Retry</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!data) {
    return null;
  }

  const { today, streak, weeklyAverage, weeklyTrend } = data;

  return (
    <View style={styles.container}>
      {/* Section Header */}
      <View style={styles.header}>
        <Text style={styles.sectionTitle}>Goal Progress</Text>
        <TouchableOpacity onPress={handleViewAllPress} style={styles.viewAllButton}>
          <Text style={styles.viewAllText}>View All</Text>
          <Ionicons name="chevron-forward" size={16} color={colors.primary.main} />
        </TouchableOpacity>
      </View>

      {/* Streak Card */}
      <StreakCard
        currentStreak={streak.currentStreak}
        longestStreak={streak.longestStreak}
        onPress={handleViewAllPress}
      />

      {/* Today's Goals Grid */}
      <View style={styles.goalsGrid}>
        <View style={styles.goalRow}>
          <View style={styles.goalCardWrapper}>
            <GoalCard
              type="calories"
              progress={today.calories}
              compact
              onPress={handleViewAllPress}
            />
          </View>
          <View style={styles.goalCardWrapper}>
            <GoalCard
              type="protein"
              progress={today.protein}
              compact
              onPress={handleViewAllPress}
            />
          </View>
        </View>
        <View style={styles.goalRow}>
          <View style={styles.goalCardWrapper}>
            <GoalCard type="carbs" progress={today.carbs} compact onPress={handleViewAllPress} />
          </View>
          <View style={styles.goalCardWrapper}>
            <GoalCard type="fat" progress={today.fat} compact onPress={handleViewAllPress} />
          </View>
        </View>
      </View>

      {/* Weekly Trend */}
      <WeeklyTrendCard
        trend={weeklyTrend}
        averageProgress={weeklyAverage}
        onPress={handleViewAllPress}
      />

      {/* Motivational Message */}
      {today.allGoalsMet && (
        <View style={styles.celebrationBanner}>
          <Ionicons name="trophy" size={20} color={colors.semantic.success} />
          <Text style={styles.celebrationText}>All goals met today! Keep it up!</Text>
        </View>
      )}
    </View>
  );
});

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.xl,
  },
  loadingContainer: {
    padding: spacing.xl,
    alignItems: 'center',
  },
  errorContainer: {
    backgroundColor: colors.special.warningLight,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    alignItems: 'center',
    gap: spacing.sm,
  },
  errorText: {
    fontSize: typography.fontSize.sm,
    color: colors.semantic.warning,
    textAlign: 'center',
  },
  retryButton: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
  },
  retryText: {
    fontSize: typography.fontSize.sm,
    color: colors.primary.main,
    fontWeight: typography.fontWeight.semibold,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  sectionTitle: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    letterSpacing: -0.5,
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
  goalsGrid: {
    marginTop: spacing.md,
    marginBottom: spacing.md,
    gap: spacing.sm,
  },
  goalRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  goalCardWrapper: {
    flex: 1,
  },
  celebrationBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    backgroundColor: colors.special.successLight,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginTop: spacing.md,
    borderWidth: 1,
    borderColor: colors.semantic.success + '30',
  },
  celebrationText: {
    fontSize: typography.fontSize.sm,
    color: colors.semantic.success,
    fontWeight: typography.fontWeight.semibold,
  },
});

export default GoalProgressSection;
