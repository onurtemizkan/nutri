import React, { memo } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, gradients, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';
import { CircularProgress } from './CircularProgress';
import type { MacroProgress } from '@/lib/api/goals';

// ============================================================================
// TYPES
// ============================================================================

export type GoalType = 'calories' | 'protein' | 'carbs' | 'fat';

interface GoalCardProps {
  /** Type of goal (calories, protein, carbs, fat) */
  type: GoalType;
  /** Progress data for the goal */
  progress: MacroProgress;
  /** Whether to show detailed information */
  showDetails?: boolean;
  /** Callback when card is pressed */
  onPress?: () => void;
  /** Custom size for compact display */
  compact?: boolean;
}

// ============================================================================
// GOAL CONFIG
// ============================================================================

interface GoalConfig {
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
  unit: string;
  color: string;
  gradientColors: readonly [string, string];
  gradientId: string;
}

const GOAL_CONFIG: Record<GoalType, GoalConfig> = {
  calories: {
    label: 'Calories',
    icon: 'flame',
    unit: 'kcal',
    color: colors.primary.main,
    gradientColors: gradients.primary as [string, string],
    gradientId: 'caloriesGradient',
  },
  protein: {
    label: 'Protein',
    icon: 'fish',
    unit: 'g',
    color: colors.semantic.success,
    gradientColors: gradients.success as [string, string],
    gradientId: 'proteinGradient',
  },
  carbs: {
    label: 'Carbs',
    icon: 'pizza',
    unit: 'g',
    color: '#EC4899', // Pink (from accent gradient)
    gradientColors: gradients.accent as [string, string],
    gradientId: 'carbsGradient',
  },
  fat: {
    label: 'Fat',
    icon: 'water',
    unit: 'g',
    color: colors.secondary.main,
    gradientColors: gradients.secondary as [string, string],
    gradientId: 'fatGradient',
  },
};

// ============================================================================
// COMPONENT
// ============================================================================

/**
 * Reusable card component for displaying goal progress
 * Shows circular progress with consumed/goal values
 */
export const GoalCard = memo(function GoalCard({
  type,
  progress,
  showDetails = true,
  onPress,
  compact = false,
}: GoalCardProps) {
  const config = GOAL_CONFIG[type];

  // Determine status
  const getStatusInfo = () => {
    if (progress.isMet) {
      return {
        icon: 'checkmark-circle' as const,
        color: colors.semantic.success,
        text: 'Goal met!',
      };
    }
    if (progress.isOnTrack) {
      return { icon: 'time' as const, color: colors.semantic.warning, text: 'On track' };
    }
    if (progress.percentage >= 50) {
      return { icon: 'trending-up' as const, color: colors.text.tertiary, text: 'Halfway there' };
    }
    return { icon: 'arrow-forward' as const, color: colors.text.tertiary, text: 'Keep going' };
  };

  const status = getStatusInfo();

  const CardContent = (
    <View style={[styles.card, compact && styles.cardCompact]}>
      <View style={styles.header}>
        <View style={styles.labelContainer}>
          <Ionicons name={config.icon} size={compact ? 14 : 18} color={config.color} />
          <Text style={[styles.label, compact && styles.labelCompact]}>{config.label}</Text>
        </View>
        {progress.isMet && (
          <Ionicons
            name="checkmark-circle"
            size={compact ? 14 : 16}
            color={colors.semantic.success}
          />
        )}
      </View>

      <View style={styles.progressContainer}>
        <CircularProgress
          percentage={progress.percentage}
          size={compact ? 60 : 80}
          strokeWidth={compact ? 6 : 8}
          color={config.gradientColors[0]}
          gradientEndColor={config.gradientColors[1]}
          gradientId={config.gradientId}
        >
          <Text style={[styles.percentageText, compact && styles.percentageTextCompact]}>
            {Math.round(progress.percentage)}%
          </Text>
        </CircularProgress>
      </View>

      {showDetails && (
        <View style={styles.details}>
          <View style={styles.valuesRow}>
            <Text style={[styles.consumed, compact && styles.consumedCompact]}>
              {Math.round(progress.consumed)}
            </Text>
            <Text style={[styles.separator, compact && styles.separatorCompact]}>/</Text>
            <Text style={[styles.goal, compact && styles.goalCompact]}>
              {progress.goal}
              {config.unit}
            </Text>
          </View>

          {!compact && (
            <View style={styles.statusRow}>
              <Ionicons name={status.icon} size={12} color={status.color} />
              <Text style={[styles.statusText, { color: status.color }]}>{status.text}</Text>
            </View>
          )}
        </View>
      )}
    </View>
  );

  if (onPress) {
    return (
      <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
        {CardContent}
      </TouchableOpacity>
    );
  }

  return CardContent;
});

// ============================================================================
// STREAK CARD
// ============================================================================

interface StreakCardProps {
  currentStreak: number;
  longestStreak: number;
  onPress?: () => void;
}

/**
 * Card component for displaying streak information
 */
export const StreakCard = memo(function StreakCard({
  currentStreak,
  longestStreak,
  onPress,
}: StreakCardProps) {
  const CardContent = (
    <View style={styles.streakCard}>
      <LinearGradient
        colors={gradients.primary}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.streakGradient}
      >
        <View style={styles.streakContent}>
          <Ionicons name="flame" size={32} color={colors.text.primary} />
          <View style={styles.streakInfo}>
            <Text style={styles.streakValue}>{currentStreak}</Text>
            <Text style={styles.streakLabel}>day streak</Text>
          </View>
        </View>

        <View style={styles.streakDivider} />

        <View style={styles.longestStreak}>
          <Ionicons name="trophy" size={16} color="rgba(255,255,255,0.8)" />
          <Text style={styles.longestStreakText}>Best: {longestStreak} days</Text>
        </View>
      </LinearGradient>
    </View>
  );

  if (onPress) {
    return (
      <TouchableOpacity onPress={onPress} activeOpacity={0.8}>
        {CardContent}
      </TouchableOpacity>
    );
  }

  return CardContent;
});

// ============================================================================
// WEEKLY TREND CARD
// ============================================================================

interface WeeklyTrendCardProps {
  trend: 'improving' | 'declining' | 'stable';
  averageProgress: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
  onPress?: () => void;
}

/**
 * Card component for displaying weekly trend summary
 */
export const WeeklyTrendCard = memo(function WeeklyTrendCard({
  trend,
  averageProgress,
  onPress,
}: WeeklyTrendCardProps) {
  const getTrendInfo = () => {
    switch (trend) {
      case 'improving':
        return {
          icon: 'trending-up' as const,
          color: colors.semantic.success,
          text: 'Improving',
          description: 'Great progress this week!',
        };
      case 'declining':
        return {
          icon: 'trending-down' as const,
          color: colors.semantic.warning,
          text: 'Declining',
          description: 'Focus on your goals',
        };
      default:
        return {
          icon: 'remove' as const,
          color: colors.text.tertiary,
          text: 'Stable',
          description: 'Consistent tracking',
        };
    }
  };

  const trendInfo = getTrendInfo();
  const overallAverage =
    (averageProgress.calories +
      averageProgress.protein +
      averageProgress.carbs +
      averageProgress.fat) /
    4;

  const CardContent = (
    <View style={styles.trendCard}>
      <View style={styles.trendHeader}>
        <View style={styles.trendTitleRow}>
          <Text style={styles.trendTitle}>Weekly Progress</Text>
          <View style={[styles.trendBadge, { backgroundColor: trendInfo.color + '20' }]}>
            <Ionicons name={trendInfo.icon} size={14} color={trendInfo.color} />
            <Text style={[styles.trendBadgeText, { color: trendInfo.color }]}>
              {trendInfo.text}
            </Text>
          </View>
        </View>
        <Text style={styles.trendDescription}>{trendInfo.description}</Text>
      </View>

      <View style={styles.trendProgress}>
        <View style={styles.trendBar}>
          <LinearGradient
            colors={gradients.primary}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={[styles.trendBarFill, { width: `${Math.min(overallAverage, 100)}%` }]}
          />
        </View>
        <Text style={styles.trendPercentage}>{Math.round(overallAverage)}% avg</Text>
      </View>

      <View style={styles.macroBreakdown}>
        <MacroMini label="Cal" value={averageProgress.calories} color={colors.primary.main} />
        <MacroMini label="Pro" value={averageProgress.protein} color={colors.semantic.success} />
        <MacroMini label="Carbs" value={averageProgress.carbs} color="#EC4899" />
        <MacroMini label="Fat" value={averageProgress.fat} color={colors.secondary.main} />
      </View>
    </View>
  );

  if (onPress) {
    return (
      <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
        {CardContent}
      </TouchableOpacity>
    );
  }

  return CardContent;
});

// Mini component for macro breakdown
const MacroMini = memo(function MacroMini({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <View style={styles.macroMini}>
      <View style={[styles.macroMiniDot, { backgroundColor: color }]} />
      <Text style={styles.macroMiniLabel}>{label}</Text>
      <Text style={styles.macroMiniValue}>{Math.round(value)}%</Text>
    </View>
  );
});

// ============================================================================
// STYLES
// ============================================================================

const styles = StyleSheet.create({
  // Goal Card
  card: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  cardCompact: {
    padding: spacing.sm,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  labelContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  label: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  labelCompact: {
    fontSize: typography.fontSize.xs,
  },
  progressContainer: {
    alignItems: 'center',
    marginVertical: spacing.sm,
  },
  percentageText: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  percentageTextCompact: {
    fontSize: typography.fontSize.md,
  },
  details: {
    alignItems: 'center',
    marginTop: spacing.xs,
  },
  valuesRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
  },
  consumed: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  consumedCompact: {
    fontSize: typography.fontSize.md,
  },
  separator: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
    marginHorizontal: spacing.xs,
  },
  separatorCompact: {
    fontSize: typography.fontSize.sm,
  },
  goal: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  goalCompact: {
    fontSize: typography.fontSize.xs,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginTop: spacing.xs,
  },
  statusText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
  },

  // Streak Card
  streakCard: {
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
    ...shadows.md,
  },
  streakGradient: {
    padding: spacing.md,
  },
  streakContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  streakInfo: {
    flex: 1,
  },
  streakValue: {
    fontSize: typography.fontSize['4xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    lineHeight: typography.fontSize['4xl'] * 1.1,
  },
  streakLabel: {
    fontSize: typography.fontSize.sm,
    color: 'rgba(255,255,255,0.8)',
    fontWeight: typography.fontWeight.medium,
  },
  streakDivider: {
    height: 1,
    backgroundColor: 'rgba(255,255,255,0.2)',
    marginVertical: spacing.md,
  },
  longestStreak: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  longestStreakText: {
    fontSize: typography.fontSize.sm,
    color: 'rgba(255,255,255,0.8)',
    fontWeight: typography.fontWeight.medium,
  },

  // Weekly Trend Card
  trendCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  trendHeader: {
    marginBottom: spacing.md,
  },
  trendTitleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  trendTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  trendBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
  },
  trendBadgeText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
  },
  trendDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  trendProgress: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  trendBar: {
    flex: 1,
    height: 8,
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.xs,
    overflow: 'hidden',
  },
  trendBarFill: {
    height: '100%',
    borderRadius: borderRadius.xs,
  },
  trendPercentage: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    minWidth: 60,
    textAlign: 'right',
  },
  macroBreakdown: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  macroMini: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  macroMiniDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  macroMiniLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  macroMiniValue: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
  },
});

export default GoalCard;
