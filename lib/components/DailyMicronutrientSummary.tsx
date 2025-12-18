/**
 * Daily Micronutrient Summary Component
 *
 * Shows a compact summary of all micronutrients consumed today
 * from both meals and supplements, displayed as small badges
 * with Daily Value percentages.
 */

import React, { useMemo } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';
import type { Meal, SupplementStatus } from '@/lib/types';

interface DailyMicronutrientSummaryProps {
  meals: Meal[];
  supplements?: SupplementStatus[];
  onPress?: () => void;
}

// Daily Values for percentage calculation (FDA reference)
const DAILY_VALUES: Record<string, { value: number; unit: string }> = {
  // Vitamins
  vitaminA: { value: 900, unit: 'mcg' },
  vitaminC: { value: 90, unit: 'mg' },
  vitaminD: { value: 20, unit: 'mcg' },
  vitaminE: { value: 15, unit: 'mg' },
  vitaminK: { value: 120, unit: 'mcg' },
  vitaminB6: { value: 1.7, unit: 'mg' },
  vitaminB12: { value: 2.4, unit: 'mcg' },
  folate: { value: 400, unit: 'mcg' },
  thiamin: { value: 1.2, unit: 'mg' },
  riboflavin: { value: 1.3, unit: 'mg' },
  niacin: { value: 16, unit: 'mg' },
  // Minerals
  calcium: { value: 1300, unit: 'mg' },
  iron: { value: 18, unit: 'mg' },
  magnesium: { value: 420, unit: 'mg' },
  zinc: { value: 11, unit: 'mg' },
  potassium: { value: 4700, unit: 'mg' },
  sodium: { value: 2300, unit: 'mg' },
  phosphorus: { value: 1250, unit: 'mg' },
  // Special
  omega3: { value: 1600, unit: 'mg' },
  fiber: { value: 28, unit: 'g' },
};

// Short labels for compact display
const SHORT_LABELS: Record<string, string> = {
  vitaminA: 'A',
  vitaminC: 'C',
  vitaminD: 'D',
  vitaminE: 'E',
  vitaminK: 'K',
  vitaminB6: 'B6',
  vitaminB12: 'B12',
  folate: 'Fol',
  thiamin: 'B1',
  riboflavin: 'B2',
  niacin: 'B3',
  calcium: 'Ca',
  iron: 'Fe',
  magnesium: 'Mg',
  zinc: 'Zn',
  potassium: 'K+',
  sodium: 'Na',
  phosphorus: 'P',
  omega3: 'O3',
  fiber: 'Fib',
};

// Full labels for accessibility
const FULL_LABELS: Record<string, string> = {
  vitaminA: 'Vitamin A',
  vitaminC: 'Vitamin C',
  vitaminD: 'Vitamin D',
  vitaminE: 'Vitamin E',
  vitaminK: 'Vitamin K',
  vitaminB6: 'Vitamin B6',
  vitaminB12: 'Vitamin B12',
  folate: 'Folate',
  thiamin: 'Thiamin',
  riboflavin: 'Riboflavin',
  niacin: 'Niacin',
  calcium: 'Calcium',
  iron: 'Iron',
  magnesium: 'Magnesium',
  zinc: 'Zinc',
  potassium: 'Potassium',
  sodium: 'Sodium',
  phosphorus: 'Phosphorus',
  omega3: 'Omega-3',
  fiber: 'Fiber',
};

// Nutrient categories for organization
type NutrientCategory = 'vitamins' | 'minerals' | 'other';

const NUTRIENT_CATEGORIES: Record<string, NutrientCategory> = {
  vitaminA: 'vitamins',
  vitaminC: 'vitamins',
  vitaminD: 'vitamins',
  vitaminE: 'vitamins',
  vitaminK: 'vitamins',
  vitaminB6: 'vitamins',
  vitaminB12: 'vitamins',
  folate: 'vitamins',
  thiamin: 'vitamins',
  riboflavin: 'vitamins',
  niacin: 'vitamins',
  calcium: 'minerals',
  iron: 'minerals',
  magnesium: 'minerals',
  zinc: 'minerals',
  potassium: 'minerals',
  sodium: 'minerals',
  phosphorus: 'minerals',
  omega3: 'other',
  fiber: 'other',
};

interface NutrientData {
  key: string;
  label: string;
  fullLabel: string;
  percent: number;
  fromFood: number;
  fromSupplements: number;
  total: number;
  category: NutrientCategory;
}

function getColorForPercent(percent: number): string {
  if (percent >= 100) return colors.status.success;
  if (percent >= 50) return colors.primary.main;
  if (percent >= 25) return colors.status.warning;
  return colors.text.tertiary;
}

function getBgColorForPercent(percent: number): string {
  if (percent >= 100) return `${colors.status.success}20`;
  if (percent >= 50) return `${colors.primary.main}20`;
  if (percent >= 25) return `${colors.status.warning}20`;
  return colors.background.elevated;
}

// Icon for source indicator
function getSourceIcon(fromFood: number, fromSupplements: number): string | null {
  if (fromFood > 0 && fromSupplements > 0) return 'leaf'; // Both sources
  if (fromSupplements > 0) return 'medical'; // Supplements only
  if (fromFood > 0) return 'nutrition'; // Food only
  return null;
}

/**
 * Aggregates micronutrients from all meals
 */
function aggregateMealNutrients(meals: Meal[]): Record<string, number> {
  const totals: Record<string, number> = {};

  const nutrientKeys = Object.keys(DAILY_VALUES);

  meals.forEach(meal => {
    nutrientKeys.forEach(key => {
      const value = (meal as unknown as Record<string, number | undefined>)[key];
      if (typeof value === 'number' && value > 0) {
        totals[key] = (totals[key] || 0) + value;
      }
    });
  });

  return totals;
}

/**
 * Aggregates micronutrients from all supplements
 */
function aggregateSupplementNutrients(supplements: SupplementStatus[]): Record<string, number> {
  const totals: Record<string, number> = {};

  const nutrientKeys = Object.keys(DAILY_VALUES).filter(k => k !== 'fiber'); // Supplements don't have fiber

  supplements.forEach(({ supplement }) => {
    nutrientKeys.forEach(key => {
      const value = (supplement as unknown as Record<string, number | undefined>)[key];
      if (typeof value === 'number' && value > 0) {
        totals[key] = (totals[key] || 0) + value;
      }
    });
  });

  return totals;
}

export function DailyMicronutrientSummary({
  meals,
  supplements = [],
  onPress,
}: DailyMicronutrientSummaryProps) {
  const nutrientData = useMemo(() => {
    const fromFood = aggregateMealNutrients(meals);
    const fromSupplements = aggregateSupplementNutrients(supplements);

    const allNutrients: NutrientData[] = [];

    Object.keys(DAILY_VALUES).forEach(key => {
      const foodValue = fromFood[key] || 0;
      const suppValue = fromSupplements[key] || 0;
      const total = foodValue + suppValue;

      if (total > 0) {
        const dv = DAILY_VALUES[key].value;
        const percent = Math.round((total / dv) * 100);

        allNutrients.push({
          key,
          label: SHORT_LABELS[key],
          fullLabel: FULL_LABELS[key],
          percent,
          fromFood: foodValue,
          fromSupplements: suppValue,
          total,
          category: NUTRIENT_CATEGORIES[key],
        });
      }
    });

    // Sort by percentage descending (most significant first)
    return allNutrients.sort((a, b) => b.percent - a.percent);
  }, [meals, supplements]);

  // No nutrients to show
  if (nutrientData.length === 0) {
    return null;
  }

  // Categorize nutrients
  const vitamins = nutrientData.filter(n => n.category === 'vitamins');
  const minerals = nutrientData.filter(n => n.category === 'minerals');
  const other = nutrientData.filter(n => n.category === 'other');

  // Calculate overall stats
  const nutrientsOver100 = nutrientData.filter(n => n.percent >= 100).length;
  const nutrientsOver50 = nutrientData.filter(n => n.percent >= 50).length;

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={onPress}
      activeOpacity={onPress ? 0.7 : 1}
      accessibilityLabel="Daily micronutrient summary"
      accessibilityHint={onPress ? 'Tap to view detailed breakdown' : undefined}
    >
      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Ionicons name="sparkles" size={16} color={colors.primary.main} />
          <Text style={styles.title}>Today's Micronutrients</Text>
        </View>
        <View style={styles.statsRow}>
          {nutrientsOver100 > 0 && (
            <View style={styles.statBadge}>
              <Ionicons name="checkmark-circle" size={12} color={colors.status.success} />
              <Text style={styles.statText}>{nutrientsOver100} at 100%+</Text>
            </View>
          )}
        </View>
      </View>

      {/* Vitamins Row */}
      {vitamins.length > 0 && (
        <View style={styles.categorySection}>
          <Text style={styles.categoryLabel}>Vitamins</Text>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.nutrientsRow}
          >
            {vitamins.map(nutrient => (
              <NutrientBadge key={nutrient.key} nutrient={nutrient} />
            ))}
          </ScrollView>
        </View>
      )}

      {/* Minerals Row */}
      {minerals.length > 0 && (
        <View style={styles.categorySection}>
          <Text style={styles.categoryLabel}>Minerals</Text>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.nutrientsRow}
          >
            {minerals.map(nutrient => (
              <NutrientBadge key={nutrient.key} nutrient={nutrient} />
            ))}
          </ScrollView>
        </View>
      )}

      {/* Other Row */}
      {other.length > 0 && (
        <View style={styles.categorySection}>
          <Text style={styles.categoryLabel}>Other</Text>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.nutrientsRow}
          >
            {other.map(nutrient => (
              <NutrientBadge key={nutrient.key} nutrient={nutrient} />
            ))}
          </ScrollView>
        </View>
      )}

      {onPress && (
        <View style={styles.footer}>
          <Text style={styles.footerText}>Tap for detailed breakdown</Text>
          <Ionicons name="chevron-forward" size={14} color={colors.text.disabled} />
        </View>
      )}
    </TouchableOpacity>
  );
}

interface NutrientBadgeProps {
  nutrient: NutrientData;
}

function NutrientBadge({ nutrient }: NutrientBadgeProps) {
  const sourceIcon = getSourceIcon(nutrient.fromFood, nutrient.fromSupplements);

  return (
    <View
      style={[
        styles.nutrientBadge,
        { backgroundColor: getBgColorForPercent(nutrient.percent) },
      ]}
      accessibilityLabel={`${nutrient.fullLabel}: ${nutrient.percent}% daily value`}
    >
      <View style={styles.badgeTop}>
        <Text
          style={[
            styles.nutrientLabel,
            { color: getColorForPercent(nutrient.percent) },
          ]}
        >
          {nutrient.label}
        </Text>
        {sourceIcon && (
          <Ionicons
            name={sourceIcon as keyof typeof Ionicons.glyphMap}
            size={8}
            color={getColorForPercent(nutrient.percent)}
            style={styles.sourceIcon}
          />
        )}
      </View>
      <Text
        style={[
          styles.nutrientPercent,
          { color: getColorForPercent(nutrient.percent) },
        ]}
      >
        {nutrient.percent > 999 ? '999+' : nutrient.percent}%
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  title: {
    fontSize: typography.fontSize.sm,
    color: colors.text.primary,
    fontWeight: typography.fontWeight.semibold,
  },
  statsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  statBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: `${colors.status.success}15`,
    paddingHorizontal: spacing.xs,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
    gap: 3,
  },
  statText: {
    fontSize: 10,
    color: colors.status.success,
    fontWeight: typography.fontWeight.medium,
  },
  categorySection: {
    marginBottom: spacing.sm,
  },
  categoryLabel: {
    fontSize: 10,
    color: colors.text.tertiary,
    fontWeight: typography.fontWeight.semibold,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: spacing.xs,
  },
  nutrientsRow: {
    flexDirection: 'row',
    gap: spacing.xs,
    paddingRight: spacing.md,
  },
  nutrientBadge: {
    alignItems: 'center',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
    minWidth: 44,
  },
  badgeTop: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  nutrientLabel: {
    fontSize: 11,
    fontWeight: typography.fontWeight.bold,
  },
  sourceIcon: {
    marginLeft: 2,
  },
  nutrientPercent: {
    fontSize: 10,
    fontWeight: typography.fontWeight.semibold,
    marginTop: 1,
  },
  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: spacing.xs,
    gap: spacing.xs,
  },
  footerText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
  },
});
