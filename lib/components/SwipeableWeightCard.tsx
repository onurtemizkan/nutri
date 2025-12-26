import React, { useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Animated } from 'react-native';
import { Swipeable } from 'react-native-gesture-handler';
import { Ionicons } from '@expo/vector-icons';
import { WeightRecord, WeightUnit, formatWeight } from '@/lib/types/weight';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';

interface SwipeableWeightCardProps {
  record: WeightRecord;
  unit: WeightUnit;
  onEdit: (record: WeightRecord) => void;
  onDelete: (record: WeightRecord) => void;
}

const ACTION_WIDTH = 72;

/**
 * Format the recorded date/time for display
 */
function formatRecordedAt(dateStr: string): { date: string; time: string } {
  const date = new Date(dateStr);
  const now = new Date();
  const isToday =
    date.getDate() === now.getDate() &&
    date.getMonth() === now.getMonth() &&
    date.getFullYear() === now.getFullYear();

  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  const isYesterday =
    date.getDate() === yesterday.getDate() &&
    date.getMonth() === yesterday.getMonth() &&
    date.getFullYear() === yesterday.getFullYear();

  let dateLabel: string;
  if (isToday) {
    dateLabel = 'Today';
  } else if (isYesterday) {
    dateLabel = 'Yesterday';
  } else {
    dateLabel = date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
  }

  const time = date.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });

  return { date: dateLabel, time };
}

export function SwipeableWeightCard({ record, unit, onEdit, onDelete }: SwipeableWeightCardProps) {
  const swipeableRef = useRef<Swipeable>(null);
  const { date, time } = formatRecordedAt(record.recordedAt);

  const handleEdit = () => {
    swipeableRef.current?.close();
    onEdit(record);
  };

  const handleDelete = () => {
    swipeableRef.current?.close();
    onDelete(record);
  };

  const handleLongPress = () => {
    showAlert(formatWeight(record.weight, unit), 'What would you like to do?', [
      {
        text: 'Edit',
        onPress: () => onEdit(record),
      },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => onDelete(record),
      },
      {
        text: 'Cancel',
        style: 'cancel',
      },
    ]);
  };

  const renderRightActions = (
    progress: Animated.AnimatedInterpolation<number>,
    dragX: Animated.AnimatedInterpolation<number>
  ) => {
    // Animation for the actions sliding in
    const translateX = progress.interpolate({
      inputRange: [0, 1],
      outputRange: [ACTION_WIDTH * 2, 0],
    });

    return (
      <Animated.View style={[styles.actionsContainer, { transform: [{ translateX }] }]}>
        <TouchableOpacity
          style={[styles.actionButton, styles.editAction]}
          onPress={handleEdit}
          activeOpacity={0.8}
          accessibilityLabel={`Edit weight record from ${date}`}
          accessibilityRole="button"
        >
          <Ionicons name="pencil" size={20} color={colors.text.primary} />
          <Text style={styles.actionText}>Edit</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.deleteAction]}
          onPress={handleDelete}
          activeOpacity={0.8}
          accessibilityLabel={`Delete weight record from ${date}`}
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
        accessibilityLabel={`Weight ${formatWeight(record.weight, unit)} recorded ${date} at ${time}. Long press for options.`}
        accessibilityRole="button"
        accessibilityHint="Swipe left or long press to edit or delete"
      >
        <View style={styles.iconContainer}>
          <Ionicons name="scale-outline" size={20} color={colors.primary.main} />
        </View>
        <View style={styles.info}>
          <Text style={styles.weight}>{formatWeight(record.weight, unit, 1)}</Text>
          <Text style={styles.dateTime}>
            {date} at {time}
          </Text>
        </View>
        <Ionicons name="chevron-forward" size={20} color={colors.text.disabled} />
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
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  info: {
    flex: 1,
  },
  weight: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  dateTime: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
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
