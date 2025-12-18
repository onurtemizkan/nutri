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
import { Meal } from '@/lib/types';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { formatMealTime } from '@/lib/utils/formatters';
import { showAlert } from '@/lib/utils/alert';
import { MicronutrientDisplay } from './MicronutrientDisplay';

interface SwipeableMealCardProps {
  meal: Meal;
  onEdit: (meal: Meal) => void;
  onDelete: (meal: Meal) => void;
}

const ACTION_WIDTH = 72;

export function SwipeableMealCard({
  meal,
  onEdit,
  onDelete,
}: SwipeableMealCardProps) {
  const swipeableRef = useRef<Swipeable>(null);

  const handleEdit = () => {
    swipeableRef.current?.close();
    onEdit(meal);
  };

  const handleDelete = () => {
    swipeableRef.current?.close();
    onDelete(meal);
  };

  const handleLongPress = () => {
    showAlert(
      meal.name,
      'What would you like to do?',
      [
        {
          text: 'Edit',
          onPress: () => onEdit(meal),
        },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => onDelete(meal),
        },
        {
          text: 'Cancel',
          style: 'cancel',
        },
      ]
    );
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
      <Animated.View
        style={[styles.actionsContainer, { transform: [{ translateX }] }]}
      >
        <TouchableOpacity
          style={[styles.actionButton, styles.editAction]}
          onPress={handleEdit}
          activeOpacity={0.8}
          accessibilityLabel={`Edit ${meal.name}`}
          accessibilityRole="button"
        >
          <Ionicons name="pencil" size={20} color={colors.text.primary} />
          <Text style={styles.actionText}>Edit</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.actionButton, styles.deleteAction]}
          onPress={handleDelete}
          activeOpacity={0.8}
          accessibilityLabel={`Delete ${meal.name}`}
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
        style={styles.mealCard}
        onLongPress={handleLongPress}
        delayLongPress={500}
        activeOpacity={0.9}
        accessibilityLabel={`${meal.name}, ${Math.round(meal.calories)} calories at ${formatMealTime(meal.consumedAt)}. Long press for options.`}
        accessibilityRole="button"
        accessibilityHint="Swipe left or long press to edit or delete"
      >
        <View style={styles.mealInfo}>
          <View style={styles.mealHeader}>
            <Text style={styles.mealName} numberOfLines={1}>
              {meal.name}
            </Text>
            <Text style={styles.mealTime}>{formatMealTime(meal.consumedAt)}</Text>
          </View>
          <Text style={styles.mealMacros}>
            P: {Math.round(meal.protein)}g • C: {Math.round(meal.carbs)}g • F: {Math.round(meal.fat)}g
          </Text>
          <MicronutrientDisplay meal={meal} compact />
        </View>
        <Text style={styles.mealCalories}>{Math.round(meal.calories)} cal</Text>
      </TouchableOpacity>
    </Swipeable>
  );
}

const styles = StyleSheet.create({
  // Meal Card (matches index.tsx styles)
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
    marginRight: spacing.md,
  },
  mealHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  mealName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    flex: 1,
    marginRight: spacing.sm,
  },
  mealTime: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.tertiary,
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
