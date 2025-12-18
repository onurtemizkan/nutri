import React, { useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
} from 'react-native';
import { Swipeable } from 'react-native-gesture-handler';
import { Ionicons } from '@expo/vector-icons';
import { HealthMetric, METRIC_CONFIG, SOURCE_CONFIG } from '@/lib/types/health-metrics';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';

interface SwipeableHealthMetricCardProps {
  metric: HealthMetric;
  onEdit: (metric: HealthMetric) => void;
  onDelete: (metric: HealthMetric) => void;
}

const ACTION_WIDTH = 72;

export function SwipeableHealthMetricCard({
  metric,
  onEdit,
  onDelete,
}: SwipeableHealthMetricCardProps) {
  const swipeableRef = useRef<Swipeable>(null);
  const config = METRIC_CONFIG[metric.metricType];
  const sourceConfig = SOURCE_CONFIG[metric.source];

  const handleEdit = () => {
    swipeableRef.current?.close();
    onEdit(metric);
  };

  const handleDelete = () => {
    swipeableRef.current?.close();
    onDelete(metric);
  };

  const handleLongPress = () => {
    showAlert(
      config.displayName,
      'What would you like to do?',
      [
        {
          text: 'Edit',
          onPress: () => onEdit(metric),
        },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => onDelete(metric),
        },
        {
          text: 'Cancel',
          style: 'cancel',
        },
      ]
    );
  };

  const formatValue = (value: number): string => {
    // Format based on metric type
    if (
      metric.metricType === 'SLEEP_DURATION' ||
      metric.metricType === 'DEEP_SLEEP_DURATION' ||
      metric.metricType === 'REM_SLEEP_DURATION'
    ) {
      const hours = Math.floor(value);
      const minutes = Math.round((value - hours) * 60);
      return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
    }

    // For percentages and scores, show as integers
    if (config.unit === '%' || config.unit === 'pts') {
      return Math.round(value).toString();
    }

    // For heart rate and similar, show as integers
    if (config.unit === 'bpm' || config.unit === 'ms') {
      return Math.round(value).toString();
    }

    return value.toFixed(1);
  };

  const formatDateTime = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  };

  const renderRightActions = (
    progress: Animated.AnimatedInterpolation<number>,
    _dragX: Animated.AnimatedInterpolation<number>
  ) => {
    const translateX = progress.interpolate({
      inputRange: [0, 1],
      outputRange: [ACTION_WIDTH * 2, 0],
    });

    return (
      <Animated.View
        style={[styles.actionsContainer, { transform: [{ translateX }] }]}
      >
        <TouchableOpacity
          style={[styles.actionButton, styles.editAction]}
          onPress={handleEdit}
          activeOpacity={0.8}
          accessibilityLabel={`Edit ${config.displayName}`}
          accessibilityRole="button"
        >
          <Ionicons name="pencil" size={20} color={colors.text.primary} />
          <Text style={styles.actionText}>Edit</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.deleteAction]}
          onPress={handleDelete}
          activeOpacity={0.8}
          accessibilityLabel={`Delete ${config.displayName}`}
          accessibilityRole="button"
        >
          <Ionicons name="trash" size={20} color={colors.text.primary} />
          <Text style={styles.actionText}>Delete</Text>
        </TouchableOpacity>
      </Animated.View>
    );
  };

  return (
    <Swipeable
      ref={swipeableRef}
      renderRightActions={renderRightActions}
      overshootRight={false}
      friction={2}
      rightThreshold={40}
    >
      <TouchableOpacity
        style={styles.card}
        onLongPress={handleLongPress}
        delayLongPress={500}
        activeOpacity={0.9}
        accessibilityLabel={`${config.displayName}, ${formatValue(metric.value)} ${config.unit} recorded at ${formatDateTime(metric.recordedAt)}. Long press for options.`}
        accessibilityRole="button"
        accessibilityHint="Swipe left or long press to edit or delete"
      >
        <View style={styles.iconContainer}>
          <Ionicons
            name={config.icon as keyof typeof Ionicons.glyphMap}
            size={20}
            color={colors.primary.main}
          />
        </View>
        <View style={styles.content}>
          <Text style={styles.dateTime}>{formatDateTime(metric.recordedAt)}</Text>
        </View>
        <View style={styles.valueContainer}>
          <Text style={styles.value}>{formatValue(metric.value)}</Text>
          <Text style={styles.unit}>{config.unit}</Text>
        </View>
        <View style={styles.sourceIcon}>
          <Ionicons
            name={sourceConfig.icon as keyof typeof Ionicons.glyphMap}
            size={16}
            color={colors.text.disabled}
          />
        </View>
      </TouchableOpacity>
    </Swipeable>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.sm,
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  iconContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
  },
  dateTime: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
  },
  valueContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginLeft: spacing.sm,
  },
  sourceIcon: {
    marginLeft: spacing.sm,
    opacity: 0.5,
  },
  value: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.primary.main,
  },
  unit: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginLeft: spacing.xs,
  },

  // Swipe Actions
  actionsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  actionButton: {
    width: ACTION_WIDTH,
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: spacing.md,
  },
  editAction: {
    backgroundColor: colors.primary.main,
    borderTopLeftRadius: borderRadius.md,
    borderBottomLeftRadius: borderRadius.md,
  },
  deleteAction: {
    backgroundColor: colors.status.error,
    borderTopRightRadius: borderRadius.md,
    borderBottomRightRadius: borderRadius.md,
  },
  actionText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginTop: spacing.xs,
  },
});
