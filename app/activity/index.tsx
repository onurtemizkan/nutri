/**
 * Activity List Screen
 *
 * Displays activity history with weekly summary, category filter,
 * and pull-to-refresh functionality.
 */

import { useState, useCallback } from 'react';
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
import { activitiesApi } from '@/lib/api/activities';
import {
  Activity,
  ActivityType,
  ActivityCategory,
  WeeklySummary,
  ACTIVITY_TYPE_CONFIG,
  INTENSITY_CONFIG,
  CATEGORY_DISPLAY_NAMES,
  getActivitiesByCategory,
  formatDuration,
  formatDistance,
} from '@/lib/types/activities';
import { useAuth } from '@/lib/context/AuthContext';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { EmptyState } from '@/lib/components/ui/EmptyState';

type FilterCategory = ActivityCategory | 'all';

export default function ActivityListScreen() {
  const [activities, setActivities] = useState<Activity[]>([]);
  const [weeklySummary, setWeeklySummary] = useState<WeeklySummary | null>(null);
  const [filter, setFilter] = useState<FilterCategory>('all');
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { user } = useAuth();
  const router = useRouter();
  const { isTablet, isLandscape, getResponsiveValue, width } = useResponsive();

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

  const loadData = useCallback(async () => {
    if (!user) {
      setIsLoading(false);
      return;
    }

    try {
      setError(null);
      const [activitiesData, summaryData] = await Promise.all([
        activitiesApi.getAll({ limit: 50 }),
        activitiesApi.getWeeklySummary().catch(() => null),
      ]);
      setActivities(activitiesData);
      setWeeklySummary(summaryData);
    } catch (err) {
      console.error('Failed to load activities:', err);
      setError('Failed to load activities');
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  }, [loadData]);

  useFocusEffect(
    useCallback(() => {
      loadData();
    }, [loadData])
  );

  // Filter activities by category
  const filteredActivities = activities.filter((activity) => {
    if (filter === 'all') return true;
    const config = ACTIVITY_TYPE_CONFIG[activity.activityType];
    return config.category === filter;
  });

  // Group activities by date
  const groupedActivities = filteredActivities.reduce(
    (groups, activity) => {
      const date = new Date(activity.startedAt).toLocaleDateString('en-US', {
        weekday: 'long',
        month: 'short',
        day: 'numeric',
      });
      if (!groups[date]) {
        groups[date] = [];
      }
      groups[date].push(activity);
      return groups;
    },
    {} as Record<string, Activity[]>
  );

  const renderFilterButton = (category: FilterCategory, label: string) => {
    const isActive = filter === category;
    return (
      <TouchableOpacity
        key={category}
        onPress={() => setFilter(category)}
        style={[styles.filterButton, isActive && styles.filterButtonActive]}
        accessibilityLabel={`Filter by ${label}`}
        accessibilityState={{ selected: isActive }}
        testID={`activity-filter-${category}`}
      >
        {isActive ? (
          <LinearGradient
            colors={gradients.primary}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.filterButtonGradient}
          >
            <Text style={styles.filterButtonTextActive}>{label}</Text>
          </LinearGradient>
        ) : (
          <Text style={styles.filterButtonText}>{label}</Text>
        )}
      </TouchableOpacity>
    );
  };

  const renderActivityItem = (activity: Activity) => {
    const typeConfig = ACTIVITY_TYPE_CONFIG[activity.activityType];
    const intensityConfig = INTENSITY_CONFIG[activity.intensity];
    const startTime = new Date(activity.startedAt).toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
    });

    return (
      <TouchableOpacity
        key={activity.id}
        style={styles.activityCard}
        onPress={() => router.push(`/activity/${activity.id}`)}
        accessibilityLabel={`${typeConfig.displayName}, ${formatDuration(activity.duration)}, ${activity.caloriesBurned || 0} calories`}
        testID={`activity-item-${activity.id}`}
      >
        <View style={[styles.activityIconContainer, { backgroundColor: typeConfig.color + '20' }]}>
          <Ionicons
            name={typeConfig.icon as keyof typeof Ionicons.glyphMap}
            size={24}
            color={typeConfig.color}
          />
        </View>
        <View style={styles.activityInfo}>
          <Text style={styles.activityName}>{typeConfig.displayName}</Text>
          <View style={styles.activityMeta}>
            <Text style={styles.activityTime}>{startTime}</Text>
            <View
              style={[styles.intensityBadge, { backgroundColor: intensityConfig.color + '20' }]}
            >
              <Text style={[styles.intensityText, { color: intensityConfig.color }]}>
                {intensityConfig.shortName}
              </Text>
            </View>
          </View>
        </View>
        <View style={styles.activityStats}>
          <Text style={styles.activityDuration}>{formatDuration(activity.duration)}</Text>
          {activity.caloriesBurned && (
            <Text style={styles.activityCalories}>{activity.caloriesBurned} cal</Text>
          )}
          {activity.distance && (
            <Text style={styles.activityDistance}>{formatDistance(activity.distance)}</Text>
          )}
        </View>
        <Ionicons name="chevron-forward" size={20} color={colors.text.disabled} />
      </TouchableOpacity>
    );
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Loading activities...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']} testID="activity-screen">
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[styles.scrollContent, { paddingHorizontal: contentPadding }]}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={colors.primary.main}
          />
        }
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Activities</Text>
          <Text style={styles.headerSubtitle}>Track your workouts</Text>
        </View>

        {/* Weekly Summary Card */}
        {weeklySummary && (
          <View style={styles.summaryCard} testID="activity-weekly-summary">
            <LinearGradient
              colors={gradients.primary}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={styles.summaryGradient}
            >
              <Text style={styles.summaryTitle}>This Week</Text>
              <View style={styles.summaryStats}>
                <View style={styles.summaryStat}>
                  <Text style={styles.summaryValue}>{weeklySummary.workoutCount}</Text>
                  <Text style={styles.summaryLabel}>Workouts</Text>
                </View>
                <View style={styles.summaryDivider} />
                <View style={styles.summaryStat}>
                  <Text style={styles.summaryValue}>
                    {formatDuration(weeklySummary.totalMinutes)}
                  </Text>
                  <Text style={styles.summaryLabel}>Duration</Text>
                </View>
                <View style={styles.summaryDivider} />
                <View style={styles.summaryStat}>
                  <Text style={styles.summaryValue}>
                    {weeklySummary.totalCalories.toLocaleString()}
                  </Text>
                  <Text style={styles.summaryLabel}>Calories</Text>
                </View>
              </View>
            </LinearGradient>
          </View>
        )}

        {/* Category Filter */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.filterContainer}
          contentContainerStyle={styles.filterContent}
        >
          {renderFilterButton('all', 'All')}
          {renderFilterButton('cardio', 'Cardio')}
          {renderFilterButton('strength', 'Strength')}
          {renderFilterButton('sports', 'Sports')}
          {renderFilterButton('other', 'Other')}
        </ScrollView>

        {/* Error State */}
        {error && (
          <View style={styles.errorContainer}>
            <Ionicons name="alert-circle-outline" size={48} color={colors.status.error} />
            <Text style={styles.errorText}>{error}</Text>
            <TouchableOpacity style={styles.retryButton} onPress={loadData}>
              <Text style={styles.retryButtonText}>Retry</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Empty State */}
        {!error && filteredActivities.length === 0 && (
          <EmptyState
            icon="fitness-outline"
            title="No activities logged yet"
            description="Track your first workout to start monitoring your exercise."
            actionLabel="Log an activity"
            onAction={() => router.push('/activity/add')}
            testID="activity-empty-state"
          />
        )}

        {/* Activities List by Date */}
        {Object.entries(groupedActivities).map(([date, dayActivities]) => (
          <View key={date} style={styles.dateGroup}>
            <Text style={styles.dateHeader}>{date}</Text>
            {dayActivities.map(renderActivityItem)}
          </View>
        ))}

        {/* Bottom padding for FAB */}
        <View style={{ height: 80 }} />
      </ScrollView>

      {/* Floating Action Button */}
      <TouchableOpacity
        style={[styles.fab, { width: fabSize, height: fabSize, borderRadius: fabSize / 2 }]}
        onPress={() => router.push('/activity/add')}
        accessibilityLabel="Add new activity"
        testID="activity-add-button"
      >
        <LinearGradient
          colors={gradients.primary}
          style={[styles.fabGradient, { borderRadius: fabSize / 2 }]}
        >
          <Ionicons name="add" size={28} color={colors.text.primary} />
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
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: spacing.xl,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
  },
  header: {
    marginTop: spacing.lg,
    marginBottom: spacing.lg,
  },
  headerTitle: {
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.bold as '700',
    color: colors.text.primary,
  },
  headerSubtitle: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
    marginTop: spacing.xs,
  },
  summaryCard: {
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
    marginBottom: spacing.lg,
    ...shadows.md,
  },
  summaryGradient: {
    padding: spacing.lg,
  },
  summaryTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  summaryStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  summaryStat: {
    alignItems: 'center',
  },
  summaryValue: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold as '700',
    color: colors.text.primary,
  },
  summaryLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.primary,
    opacity: 0.8,
    marginTop: spacing.xs,
  },
  summaryDivider: {
    width: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
  },
  filterContainer: {
    marginBottom: spacing.lg,
  },
  filterContent: {
    paddingVertical: spacing.xs,
    gap: spacing.sm,
  },
  filterButton: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.full,
    backgroundColor: colors.background.tertiary,
    marginRight: spacing.sm,
  },
  filterButtonActive: {
    backgroundColor: 'transparent',
  },
  filterButtonGradient: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.full,
  },
  filterButtonText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium as '500',
    color: colors.text.secondary,
  },
  filterButtonTextActive: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: spacing['2xl'],
  },
  errorText: {
    fontSize: typography.fontSize.md,
    color: colors.status.error,
    marginTop: spacing.md,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.md,
  },
  retryButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
  },
  dateGroup: {
    marginBottom: spacing.lg,
  },
  dateHeader: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.tertiary,
    marginBottom: spacing.sm,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  activityCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  activityIconContainer: {
    width: 48,
    height: 48,
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  activityInfo: {
    flex: 1,
  },
  activityName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
  },
  activityMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.xs,
    gap: spacing.sm,
  },
  activityTime: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  intensityBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
  },
  intensityText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold as '600',
  },
  activityStats: {
    alignItems: 'flex-end',
    marginRight: spacing.sm,
  },
  activityDuration: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
  },
  activityCalories: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  activityDistance: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  fab: {
    position: 'absolute',
    right: spacing.lg,
    bottom: spacing.lg,
    ...shadows.xl,
  },
  fabGradient: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
});
