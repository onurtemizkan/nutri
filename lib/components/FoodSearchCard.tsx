/**
 * Food Search Card Component
 *
 * Displays a food item from USDA search results with nutrition preview
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography, shadows } from '../theme/colors';
import { USDAFood, USDADataType } from '../types/foods';

interface FoodSearchCardProps {
  food: USDAFood;
  onPress: (food: USDAFood) => void;
  showDataType?: boolean;
}

// Get badge color based on data type
const getDataTypeBadge = (dataType: USDADataType): { label: string; color: string } => {
  switch (dataType) {
    case 'Foundation':
      return { label: 'USDA', color: colors.status.success };
    case 'SR Legacy':
      return { label: 'USDA', color: colors.status.success };
    case 'Branded':
      return { label: 'Brand', color: colors.secondary.main };
    case 'Survey (FNDDS)':
      return { label: 'Recipe', color: colors.status.warning };
    case 'Experimental':
      return { label: 'Exp', color: colors.text.tertiary };
    default:
      return { label: 'Food', color: colors.text.tertiary };
  }
};

// Format number with appropriate precision
const formatNumber = (value: number | undefined, decimals: number = 0): string => {
  if (value === undefined || value === null) return '--';
  return value.toFixed(decimals);
};

// Calculate macro percentages
const getMacroPercentages = (
  protein: number,
  carbs: number,
  fat: number
): { protein: number; carbs: number; fat: number } => {
  const totalCalories = protein * 4 + carbs * 4 + fat * 9;
  if (totalCalories === 0) {
    return { protein: 33, carbs: 33, fat: 34 };
  }
  return {
    protein: Math.round((protein * 4 / totalCalories) * 100),
    carbs: Math.round((carbs * 4 / totalCalories) * 100),
    fat: Math.round((fat * 9 / totalCalories) * 100),
  };
};

export function FoodSearchCard({ food, onPress, showDataType = true }: FoodSearchCardProps) {
  const badge = getDataTypeBadge(food.dataType);
  const macros = getMacroPercentages(food.protein, food.carbs, food.fat);

  return (
    <TouchableOpacity
      style={styles.card}
      onPress={() => onPress(food)}
      activeOpacity={0.7}
      accessibilityLabel={`${food.description}, ${formatNumber(food.calories)} calories`}
      accessibilityRole="button"
    >
      <View style={styles.content}>
        {/* Header with name and badge */}
        <View style={styles.header}>
          <View style={styles.titleContainer}>
            <Text style={styles.name} numberOfLines={2}>
              {food.description}
            </Text>
            {food.brandOwner && (
              <Text style={styles.brand} numberOfLines={1}>
                {food.brandOwner}
              </Text>
            )}
          </View>
          {showDataType && (
            <View style={[styles.badge, { backgroundColor: badge.color + '20' }]}>
              <Text style={[styles.badgeText, { color: badge.color }]}>{badge.label}</Text>
            </View>
          )}
        </View>

        {/* Nutrition preview */}
        <View style={styles.nutritionRow}>
          <View style={styles.calorieContainer}>
            <Text style={styles.calorieValue}>{formatNumber(food.calories)}</Text>
            <Text style={styles.calorieLabel}>kcal</Text>
          </View>

          {/* Macro bar */}
          <View style={styles.macroSection}>
            <View style={styles.macroBar}>
              <View
                style={[
                  styles.macroSegment,
                  { flex: macros.protein, backgroundColor: colors.status.info },
                ]}
              />
              <View
                style={[
                  styles.macroSegment,
                  { flex: macros.carbs, backgroundColor: colors.status.warning },
                ]}
              />
              <View
                style={[
                  styles.macroSegment,
                  { flex: macros.fat, backgroundColor: colors.status.error },
                ]}
              />
            </View>
            <View style={styles.macroLabels}>
              <Text style={styles.macroText}>
                <Text style={{ color: colors.status.info }}>P</Text> {formatNumber(food.protein)}g
              </Text>
              <Text style={styles.macroText}>
                <Text style={{ color: colors.status.warning }}>C</Text> {formatNumber(food.carbs)}g
              </Text>
              <Text style={styles.macroText}>
                <Text style={{ color: colors.status.error }}>F</Text> {formatNumber(food.fat)}g
              </Text>
            </View>
          </View>

          {/* Chevron */}
          <Ionicons
            name="chevron-forward"
            size={20}
            color={colors.text.tertiary}
            style={styles.chevron}
          />
        </View>

        {/* Serving size if available */}
        {food.servingSize && food.servingSizeUnit && (
          <Text style={styles.servingSize}>
            Per {food.servingSize} {food.servingSizeUnit}
          </Text>
        )}
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    marginBottom: spacing.sm,
    ...shadows.sm,
  },
  content: {
    padding: spacing.md,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: spacing.sm,
  },
  titleContainer: {
    flex: 1,
    marginRight: spacing.sm,
  },
  name: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
    lineHeight: typography.fontSize.md * 1.3,
  },
  brand: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  badge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
  },
  badgeText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold as '600',
  },
  nutritionRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  calorieContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginRight: spacing.md,
    minWidth: 70,
  },
  calorieValue: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold as '700',
    color: colors.text.primary,
  },
  calorieLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginLeft: 2,
  },
  macroSection: {
    flex: 1,
  },
  macroBar: {
    flexDirection: 'row',
    height: 6,
    borderRadius: borderRadius.full,
    overflow: 'hidden',
    backgroundColor: colors.background.elevated,
    marginBottom: spacing.xs,
  },
  macroSegment: {
    height: '100%',
  },
  macroLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  macroText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  chevron: {
    marginLeft: spacing.sm,
  },
  servingSize: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    marginTop: spacing.xs,
  },
});

export default FoodSearchCard;
