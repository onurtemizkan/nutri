/**
 * Supplement Nutrition Summary Component
 *
 * Shows a compact summary of micronutrients from supplements
 * displayed as small badges with Daily Value percentages.
 */

import React, { useMemo } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';
import type { Supplement, SupplementStatus } from '@/lib/types';
import type { SupplementNutrients } from '@/lib/utils/supplementMicronutrients';

interface SupplementNutritionSummaryProps {
  supplements: SupplementStatus[];
  onPress?: () => void;
}

// Daily Values for percentage calculation
const DAILY_VALUES: Record<keyof SupplementNutrients, number> = {
  vitaminA: 900,
  vitaminC: 90,
  vitaminD: 20,
  vitaminE: 15,
  vitaminK: 120,
  vitaminB6: 1.7,
  vitaminB12: 2.4,
  folate: 400,
  thiamin: 1.2,
  riboflavin: 1.3,
  niacin: 16,
  calcium: 1300,
  iron: 18,
  magnesium: 420,
  zinc: 11,
  potassium: 4700,
  sodium: 2300,
  phosphorus: 1250,
  omega3: 1600,
};

// Short labels for compact display
const SHORT_LABELS: Record<keyof SupplementNutrients, string> = {
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
};

// Full labels for tooltip/accessibility
const FULL_LABELS: Record<keyof SupplementNutrients, string> = {
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
};

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

/**
 * Aggregates micronutrients from all supplements
 */
function aggregateNutrients(supplements: SupplementStatus[]): Record<keyof SupplementNutrients, number> {
  const totals: Partial<Record<keyof SupplementNutrients, number>> = {};

  supplements.forEach(({ supplement }) => {
    const nutrientKeys: (keyof SupplementNutrients)[] = [
      'vitaminA', 'vitaminC', 'vitaminD', 'vitaminE', 'vitaminK',
      'vitaminB6', 'vitaminB12', 'folate', 'thiamin', 'riboflavin', 'niacin',
      'calcium', 'iron', 'magnesium', 'zinc', 'potassium', 'sodium', 'phosphorus',
      'omega3',
    ];

    nutrientKeys.forEach(key => {
      const value = (supplement as unknown as Record<string, number | undefined>)[key];
      if (typeof value === 'number' && value > 0) {
        totals[key] = (totals[key] || 0) + value;
      }
    });
  });

  return totals as Record<keyof SupplementNutrients, number>;
}

export function SupplementNutritionSummary({ supplements, onPress }: SupplementNutritionSummaryProps) {
  const aggregated = useMemo(() => aggregateNutrients(supplements), [supplements]);

  // Convert to array with DV percentages, sorted by percentage
  const nutrientData = useMemo(() => {
    return Object.entries(aggregated)
      .filter(([, value]) => value > 0)
      .map(([key, value]) => {
        const dv = DAILY_VALUES[key as keyof SupplementNutrients];
        const percent = Math.round((value / dv) * 100);
        return {
          key: key as keyof SupplementNutrients,
          value,
          percent,
          label: SHORT_LABELS[key as keyof SupplementNutrients],
          fullLabel: FULL_LABELS[key as keyof SupplementNutrients],
        };
      })
      .sort((a, b) => b.percent - a.percent);
  }, [aggregated]);

  // No nutrients to show
  if (nutrientData.length === 0) {
    return null;
  }

  // Show top 6 nutrients (most significant)
  const displayNutrients = nutrientData.slice(0, 6);
  const remainingCount = nutrientData.length - 6;

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={onPress}
      activeOpacity={onPress ? 0.7 : 1}
      accessibilityLabel="Supplement nutrition summary"
      accessibilityHint={onPress ? 'Tap to view details' : undefined}
    >
      <View style={styles.header}>
        <View style={styles.titleRow}>
          <Ionicons name="nutrition-outline" size={12} color={colors.text.tertiary} />
          <Text style={styles.title}>Daily Nutrients from Supplements</Text>
        </View>
        {onPress && (
          <Ionicons name="chevron-forward" size={14} color={colors.text.disabled} />
        )}
      </View>

      <View style={styles.nutrientsRow}>
        {displayNutrients.map(nutrient => (
          <View
            key={nutrient.key}
            style={[
              styles.nutrientBadge,
              { backgroundColor: getBgColorForPercent(nutrient.percent) },
            ]}
            accessibilityLabel={`${nutrient.fullLabel}: ${nutrient.percent}% daily value`}
          >
            <Text
              style={[
                styles.nutrientLabel,
                { color: getColorForPercent(nutrient.percent) },
              ]}
            >
              {nutrient.label}
            </Text>
            <Text
              style={[
                styles.nutrientPercent,
                { color: getColorForPercent(nutrient.percent) },
              ]}
            >
              {nutrient.percent > 999 ? '999+' : nutrient.percent}%
            </Text>
          </View>
        ))}
        {remainingCount > 0 && (
          <View style={styles.moreIndicator}>
            <Text style={styles.moreText}>+{remainingCount}</Text>
          </View>
        )}
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.sm,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  title: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    fontWeight: typography.fontWeight.medium,
  },
  nutrientsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.xs,
  },
  nutrientBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.xs,
    paddingVertical: 3,
    borderRadius: borderRadius.sm,
    gap: 3,
  },
  nutrientLabel: {
    fontSize: 10,
    fontWeight: typography.fontWeight.bold,
  },
  nutrientPercent: {
    fontSize: 10,
    fontWeight: typography.fontWeight.semibold,
  },
  moreIndicator: {
    backgroundColor: colors.background.elevated,
    paddingHorizontal: spacing.xs,
    paddingVertical: 3,
    borderRadius: borderRadius.sm,
  },
  moreText: {
    fontSize: 10,
    color: colors.text.tertiary,
    fontWeight: typography.fontWeight.medium,
  },
});
