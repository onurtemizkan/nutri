import { useState, useEffect, useCallback } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useRouter } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { weightApi } from '@/lib/api/weight';
import {
  WeightSummary,
  formatWeight,
  formatWeightChange,
  getBmiColor,
  WeightUnit,
} from '@/lib/types/weight';
import { colors, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';

interface WeightWidgetProps {
  unit?: WeightUnit;
}

export function WeightWidget({ unit = 'kg' }: WeightWidgetProps) {
  const router = useRouter();
  const [summary, setSummary] = useState<WeightSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(false);

  const loadData = useCallback(async () => {
    try {
      setError(false);
      const data = await weightApi.getSummary();
      setSummary(data);
    } catch (err) {
      console.warn('Failed to load weight summary:', err);
      setError(true);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Reload on focus
  useFocusEffect(
    useCallback(() => {
      loadData();
    }, [loadData])
  );

  const handlePress = () => {
    router.push('/weight');
  };

  const handleAddPress = () => {
    router.push('/weight/add');
  };

  // Loading state
  if (isLoading) {
    return (
      <TouchableOpacity style={styles.container} onPress={handlePress} activeOpacity={0.7}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="small" color={colors.primary.main} />
        </View>
      </TouchableOpacity>
    );
  }

  // Empty state - no weight records yet
  if (!summary || summary.currentWeight === null) {
    return (
      <TouchableOpacity style={styles.container} onPress={handleAddPress} activeOpacity={0.7}>
        <View style={styles.emptyContainer}>
          <View style={styles.emptyIconContainer}>
            <Ionicons name="scale-outline" size={24} color={colors.text.disabled} />
          </View>
          <View style={styles.emptyInfo}>
            <Text style={styles.emptyTitle}>Track Your Weight</Text>
            <Text style={styles.emptySubtitle}>Tap to add your first weight</Text>
          </View>
          <Ionicons name="add-circle" size={24} color={colors.primary.main} />
        </View>
      </TouchableOpacity>
    );
  }

  // Has data
  return (
    <TouchableOpacity style={styles.container} onPress={handlePress} activeOpacity={0.7}>
      <View style={styles.header}>
        <View style={styles.iconContainer}>
          <Ionicons name="scale-outline" size={20} color={colors.primary.main} />
        </View>
        <Text style={styles.title}>Weight</Text>
        <TouchableOpacity onPress={handleAddPress} style={styles.addButton}>
          <Ionicons name="add" size={20} color={colors.primary.main} />
        </TouchableOpacity>
      </View>

      <View style={styles.content}>
        {/* Current Weight */}
        <View style={styles.mainValue}>
          <Text style={styles.weightValue}>{formatWeight(summary.currentWeight, unit, 1)}</Text>
          {summary.weeklyChange !== null && (
            <View
              style={[
                styles.changeBadge,
                {
                  backgroundColor:
                    summary.weeklyChange < 0
                      ? colors.status.success + '20'
                      : summary.weeklyChange > 0
                        ? colors.status.warning + '20'
                        : colors.background.tertiary,
                },
              ]}
            >
              <Ionicons
                name={
                  summary.weeklyChange < 0
                    ? 'trending-down'
                    : summary.weeklyChange > 0
                      ? 'trending-up'
                      : 'remove'
                }
                size={14}
                color={
                  summary.weeklyChange < 0
                    ? colors.status.success
                    : summary.weeklyChange > 0
                      ? colors.status.warning
                      : colors.text.tertiary
                }
              />
              <Text
                style={[
                  styles.changeText,
                  {
                    color:
                      summary.weeklyChange < 0
                        ? colors.status.success
                        : summary.weeklyChange > 0
                          ? colors.status.warning
                          : colors.text.tertiary,
                  },
                ]}
              >
                {formatWeightChange(summary.weeklyChange, unit)}
              </Text>
            </View>
          )}
        </View>

        {/* Progress and BMI */}
        <View style={styles.statsRow}>
          {/* Goal Progress */}
          {summary.goalWeight !== null && summary.progressPercentage !== null && (
            <View style={styles.statItem}>
              <View style={styles.progressBarBackground}>
                <View
                  style={[
                    styles.progressBarFill,
                    {
                      width: `${Math.min(100, Math.max(0, summary.progressPercentage))}%`,
                    },
                  ]}
                />
              </View>
              <Text style={styles.statLabel}>
                {Math.round(summary.progressPercentage)}% to goal
              </Text>
            </View>
          )}

          {/* BMI */}
          {summary.bmi !== null && summary.bmiCategory && (
            <View style={styles.statItem}>
              <View style={styles.bmiContainer}>
                <Text style={styles.bmiValue}>{summary.bmi}</Text>
                <View
                  style={[
                    styles.bmiCategoryBadge,
                    { backgroundColor: getBmiColor(summary.bmiCategory) + '20' },
                  ]}
                >
                  <Text
                    style={[styles.bmiCategoryText, { color: getBmiColor(summary.bmiCategory) }]}
                  >
                    {summary.bmiCategory}
                  </Text>
                </View>
              </View>
              <Text style={styles.statLabel}>BMI</Text>
            </View>
          )}
        </View>

        {/* Last recorded */}
        {summary.lastRecordDate && (
          <Text style={styles.lastRecorded}>
            Last recorded {formatLastRecorded(summary.lastRecordDate)}
          </Text>
        )}
      </View>

      <View style={styles.chevronContainer}>
        <Ionicons name="chevron-forward" size={16} color={colors.text.disabled} />
      </View>
    </TouchableOpacity>
  );
}

function formatLastRecorded(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return 'today';
  if (diffDays === 1) return 'yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  loadingContainer: {
    height: 100,
    justifyContent: 'center',
    alignItems: 'center',
  },

  // Empty State
  emptyContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  emptyIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.background.secondary,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  emptyInfo: {
    flex: 1,
  },
  emptyTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  emptySubtitle: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  iconContainer: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.sm,
  },
  title: {
    flex: 1,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  addButton: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
  },

  // Content
  content: {
    flex: 1,
  },
  mainValue: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  weightValue: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginRight: spacing.sm,
  },
  changeBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
    gap: 4,
  },
  changeText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
  },

  // Stats Row
  statsRow: {
    flexDirection: 'row',
    gap: spacing.lg,
    marginBottom: spacing.sm,
  },
  statItem: {
    flex: 1,
  },
  progressBarBackground: {
    height: 6,
    backgroundColor: colors.border.secondary,
    borderRadius: 3,
    overflow: 'hidden',
    marginBottom: spacing.xs,
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: colors.primary.main,
    borderRadius: 3,
  },
  statLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  bmiContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginBottom: spacing.xs,
  },
  bmiValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  bmiCategoryBadge: {
    paddingHorizontal: spacing.xs,
    paddingVertical: 2,
    borderRadius: borderRadius.xs,
  },
  bmiCategoryText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
  },

  // Last Recorded
  lastRecorded: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    marginTop: spacing.xs,
  },

  // Chevron
  chevronContainer: {
    position: 'absolute',
    right: spacing.md,
    top: '50%',
    marginTop: -8,
  },
});
