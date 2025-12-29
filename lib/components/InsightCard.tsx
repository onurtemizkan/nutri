/**
 * InsightCard Component
 * Displays ML-generated insights with priority indication, swipe actions, and feedback
 */

import React, { memo, useCallback, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  PanResponder,
  Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, spacing, borderRadius, typography, shadows, gradients } from '@/lib/theme/colors';
import {
  MLInsight,
  InsightPriority,
  getPriorityColor,
  getInsightTypeIcon,
  getInsightTypeLabel,
  formatConfidence,
  formatCorrelation,
  getCorrelationStrength,
} from '@/lib/types/insights';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const SWIPE_THRESHOLD = SCREEN_WIDTH * 0.25;

// ============================================================================
// TYPES
// ============================================================================

interface InsightCardProps {
  insight: MLInsight;
  onPress?: () => void;
  onDismiss?: () => void;
  onFeedback?: (helpful: boolean) => void;
  onMarkViewed?: () => void;
  compact?: boolean;
}

// ============================================================================
// PRIORITY CONFIG
// ============================================================================

interface PriorityConfig {
  backgroundColor: string;
  borderColor: string;
  gradientColors: readonly [string, string];
}

const PRIORITY_CONFIG: Record<InsightPriority, PriorityConfig> = {
  CRITICAL: {
    backgroundColor: 'rgba(220, 38, 38, 0.1)',
    borderColor: 'rgba(220, 38, 38, 0.3)',
    gradientColors: ['#DC2626', '#EF4444'] as const,
  },
  HIGH: {
    backgroundColor: 'rgba(245, 158, 11, 0.1)',
    borderColor: 'rgba(245, 158, 11, 0.3)',
    gradientColors: ['#F59E0B', '#FBBF24'] as const,
  },
  MEDIUM: {
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    borderColor: 'rgba(59, 130, 246, 0.3)',
    gradientColors: ['#3B82F6', '#60A5FA'] as const,
  },
  LOW: {
    backgroundColor: colors.background.tertiary,
    borderColor: colors.border.secondary,
    gradientColors: ['#6B7280', '#9CA3AF'] as const,
  },
};

// ============================================================================
// COMPONENT
// ============================================================================

export const InsightCard = memo(function InsightCard({
  insight,
  onPress,
  onDismiss,
  onFeedback,
  onMarkViewed,
  compact = false,
}: InsightCardProps) {
  const [showFeedback, setShowFeedback] = useState(false);
  const translateX = React.useRef(new Animated.Value(0)).current;
  const priorityConfig = PRIORITY_CONFIG[insight.priority];
  const priorityColor = getPriorityColor(insight.priority);
  const typeIcon = getInsightTypeIcon(insight.insightType) as keyof typeof Ionicons.glyphMap;

  // Handle swipe to dismiss
  const panResponder = React.useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => false,
      onMoveShouldSetPanResponder: (_, gestureState) => {
        return Math.abs(gestureState.dx) > 10 && Math.abs(gestureState.dy) < 20;
      },
      onPanResponderMove: (_, gestureState) => {
        if (gestureState.dx < 0) {
          translateX.setValue(gestureState.dx);
        }
      },
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dx < -SWIPE_THRESHOLD && onDismiss) {
          Animated.timing(translateX, {
            toValue: -SCREEN_WIDTH,
            duration: 200,
            useNativeDriver: true,
          }).start(() => onDismiss());
        } else {
          Animated.spring(translateX, {
            toValue: 0,
            useNativeDriver: true,
            friction: 8,
          }).start();
        }
      },
    })
  ).current;

  const handlePress = useCallback(() => {
    if (!insight.viewed && onMarkViewed) {
      onMarkViewed();
    }
    onPress?.();
  }, [insight.viewed, onMarkViewed, onPress]);

  const handleFeedbackPress = useCallback(
    (helpful: boolean) => {
      onFeedback?.(helpful);
      setShowFeedback(false);
    },
    [onFeedback]
  );

  // Compact card for lists
  if (compact) {
    return (
      <TouchableOpacity
        onPress={handlePress}
        activeOpacity={0.7}
        style={[styles.compactCard, !insight.viewed && styles.unviewedCard]}
      >
        <View style={styles.compactContent}>
          <View style={[styles.priorityIndicator, { backgroundColor: priorityColor }]} />
          <View style={styles.compactIconContainer}>
            <Ionicons name={typeIcon} size={16} color={priorityColor} />
          </View>
          <View style={styles.compactTextContainer}>
            <Text style={styles.compactTitle} numberOfLines={1}>
              {insight.title}
            </Text>
            <Text style={styles.compactMeta}>
              {getInsightTypeLabel(insight.insightType)} • {formatConfidence(insight.confidence)}{' '}
              confidence
            </Text>
          </View>
          {!insight.viewed && <View style={styles.unreadDot} />}
          <Ionicons name="chevron-forward" size={16} color={colors.text.tertiary} />
        </View>
      </TouchableOpacity>
    );
  }

  // Full card
  return (
    <View style={styles.swipeContainer}>
      {/* Dismiss action background */}
      <View style={styles.dismissBackground}>
        <Ionicons name="close-circle" size={24} color={colors.text.primary} />
        <Text style={styles.dismissText}>Dismiss</Text>
      </View>

      <Animated.View
        style={[styles.cardContainer, { transform: [{ translateX }] }]}
        {...panResponder.panHandlers}
      >
        <TouchableOpacity
          onPress={handlePress}
          activeOpacity={0.9}
          style={[
            styles.card,
            {
              backgroundColor: priorityConfig.backgroundColor,
              borderColor: priorityConfig.borderColor,
            },
            !insight.viewed && styles.unviewedCard,
          ]}
        >
          {/* Header */}
          <View style={styles.header}>
            <View style={styles.typeContainer}>
              <LinearGradient
                colors={priorityConfig.gradientColors}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.iconGradient}
              >
                <Ionicons name={typeIcon} size={18} color={colors.text.primary} />
              </LinearGradient>
              <View>
                <Text style={styles.typeLabel}>{getInsightTypeLabel(insight.insightType)}</Text>
                <Text style={[styles.priorityBadge, { color: priorityColor }]}>
                  {insight.priority} Priority
                </Text>
              </View>
            </View>
            {!insight.viewed && <View style={styles.unreadDotLarge} />}
          </View>

          {/* Title */}
          <Text style={styles.title}>{insight.title}</Text>

          {/* Description */}
          <Text style={styles.description} numberOfLines={3}>
            {insight.description}
          </Text>

          {/* Correlation info if available */}
          {insight.correlation !== null && (
            <View style={styles.correlationContainer}>
              <View style={styles.correlationItem}>
                <Ionicons name="analytics" size={14} color={colors.text.tertiary} />
                <Text style={styles.correlationLabel}>Correlation:</Text>
                <Text style={[styles.correlationValue, { color: priorityColor }]}>
                  {formatCorrelation(insight.correlation)} (
                  {getCorrelationStrength(insight.correlation)})
                </Text>
              </View>
            </View>
          )}

          {/* Recommendation */}
          <View style={styles.recommendationContainer}>
            <Ionicons name="bulb-outline" size={16} color={colors.semantic.success} />
            <Text style={styles.recommendation} numberOfLines={2}>
              {insight.recommendation}
            </Text>
          </View>

          {/* Footer */}
          <View style={styles.footer}>
            <View style={styles.metaContainer}>
              <Text style={styles.confidence}>
                {formatConfidence(insight.confidence)} confidence
              </Text>
              <Text style={styles.separator}>•</Text>
              <Text style={styles.dataPoints}>{insight.dataPoints} data points</Text>
            </View>

            {/* Feedback buttons */}
            {insight.helpful === null && !showFeedback && (
              <TouchableOpacity style={styles.feedbackButton} onPress={() => setShowFeedback(true)}>
                <Ionicons name="thumbs-up-outline" size={14} color={colors.text.tertiary} />
                <Text style={styles.feedbackButtonText}>Was this helpful?</Text>
              </TouchableOpacity>
            )}

            {showFeedback && (
              <View style={styles.feedbackActions}>
                <TouchableOpacity
                  style={[styles.feedbackActionButton, styles.helpfulButton]}
                  onPress={() => handleFeedbackPress(true)}
                >
                  <Ionicons name="thumbs-up" size={14} color={colors.semantic.success} />
                  <Text style={[styles.feedbackActionText, { color: colors.semantic.success }]}>
                    Yes
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.feedbackActionButton, styles.notHelpfulButton]}
                  onPress={() => handleFeedbackPress(false)}
                >
                  <Ionicons name="thumbs-down" size={14} color={colors.semantic.error} />
                  <Text style={[styles.feedbackActionText, { color: colors.semantic.error }]}>
                    No
                  </Text>
                </TouchableOpacity>
              </View>
            )}

            {insight.helpful !== null && (
              <View style={styles.feedbackGiven}>
                <Ionicons
                  name={insight.helpful ? 'checkmark-circle' : 'close-circle'}
                  size={14}
                  color={insight.helpful ? colors.semantic.success : colors.semantic.error}
                />
                <Text
                  style={[
                    styles.feedbackGivenText,
                    {
                      color: insight.helpful ? colors.semantic.success : colors.semantic.error,
                    },
                  ]}
                >
                  {insight.helpful ? 'Marked helpful' : 'Not helpful'}
                </Text>
              </View>
            )}
          </View>
        </TouchableOpacity>
      </Animated.View>
    </View>
  );
});

// ============================================================================
// INSIGHT SUMMARY CARD
// ============================================================================

interface InsightSummaryCardProps {
  totalInsights: number;
  unviewedCount: number;
  highPriorityCount: number;
  onPress?: () => void;
}

export const InsightSummaryCard = memo(function InsightSummaryCard({
  totalInsights,
  unviewedCount,
  highPriorityCount,
  onPress,
}: InsightSummaryCardProps) {
  return (
    <TouchableOpacity onPress={onPress} activeOpacity={0.8} style={styles.summaryCard}>
      <LinearGradient
        colors={gradients.primary}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.summaryGradient}
      >
        <View style={styles.summaryContent}>
          <View style={styles.summaryIconContainer}>
            <Ionicons name="bulb" size={28} color={colors.text.primary} />
          </View>
          <View style={styles.summaryTextContainer}>
            <Text style={styles.summaryTitle}>
              {unviewedCount > 0 ? `${unviewedCount} New Insights` : `${totalInsights} Insights`}
            </Text>
            <Text style={styles.summarySubtitle}>
              {highPriorityCount > 0
                ? `${highPriorityCount} high priority`
                : 'Personalized recommendations'}
            </Text>
          </View>
          <Ionicons name="chevron-forward" size={20} color={colors.text.primary} />
        </View>
      </LinearGradient>
    </TouchableOpacity>
  );
});

// ============================================================================
// STYLES
// ============================================================================

const styles = StyleSheet.create({
  // Swipe container
  swipeContainer: {
    position: 'relative',
    marginBottom: spacing.md,
  },
  dismissBackground: {
    position: 'absolute',
    right: 0,
    top: 0,
    bottom: 0,
    width: '100%',
    backgroundColor: colors.semantic.error,
    borderRadius: borderRadius.lg,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    paddingRight: spacing.lg,
    gap: spacing.sm,
  },
  dismissText: {
    color: colors.text.primary,
    fontWeight: typography.fontWeight.semibold,
    fontSize: typography.fontSize.sm,
  },
  cardContainer: {
    backgroundColor: colors.background.primary,
    borderRadius: borderRadius.lg,
  },

  // Card
  card: {
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    ...shadows.sm,
  },
  unviewedCard: {
    borderLeftWidth: 3,
    borderLeftColor: colors.primary.main,
  },

  // Header
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: spacing.sm,
  },
  typeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  iconGradient: {
    width: 36,
    height: 36,
    borderRadius: borderRadius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  typeLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  priorityBadge: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    textTransform: 'uppercase',
  },
  unreadDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.primary.main,
  },
  unreadDotLarge: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: colors.primary.main,
  },

  // Content
  title: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  description: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    lineHeight: typography.fontSize.sm * 1.5,
    marginBottom: spacing.md,
  },

  // Correlation
  correlationContainer: {
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.sm,
    padding: spacing.sm,
    marginBottom: spacing.md,
  },
  correlationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  correlationLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  correlationValue: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
  },

  // Recommendation
  recommendationContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderRadius: borderRadius.sm,
    padding: spacing.sm,
    marginBottom: spacing.md,
  },
  recommendation: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.semantic.success,
    lineHeight: typography.fontSize.sm * 1.4,
  },

  // Footer
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
    paddingTop: spacing.sm,
  },
  metaContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  confidence: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  separator: {
    color: colors.text.disabled,
  },
  dataPoints: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },

  // Feedback
  feedbackButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
  },
  feedbackButtonText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  feedbackActions: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  feedbackActionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
    borderRadius: borderRadius.sm,
  },
  helpfulButton: {
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
  },
  notHelpfulButton: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
  },
  feedbackActionText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
  },
  feedbackGiven: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  feedbackGivenText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
  },

  // Compact card
  compactCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.sm,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  compactContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  priorityIndicator: {
    width: 3,
    height: '100%',
    borderRadius: 1.5,
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
  },
  compactIconContainer: {
    width: 28,
    height: 28,
    borderRadius: borderRadius.sm,
    backgroundColor: colors.background.elevated,
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: spacing.xs,
  },
  compactTextContainer: {
    flex: 1,
  },
  compactTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: 2,
  },
  compactMeta: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },

  // Summary card
  summaryCard: {
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
    marginBottom: spacing.md,
    ...shadows.md,
  },
  summaryGradient: {
    padding: spacing.md,
  },
  summaryContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  summaryIconContainer: {
    width: 48,
    height: 48,
    borderRadius: borderRadius.md,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  summaryTextContainer: {
    flex: 1,
  },
  summaryTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  summarySubtitle: {
    fontSize: typography.fontSize.sm,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 2,
  },
});

export default InsightCard;
