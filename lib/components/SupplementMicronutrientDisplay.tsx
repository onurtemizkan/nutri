/**
 * Supplement Micronutrient Display Component
 *
 * Shows the micronutrient content of a supplement with
 * Daily Value percentages and color-coded indicators.
 */

import React, { useMemo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';
import type { Supplement } from '@/lib/types';
import type { SupplementNutrients } from '@/lib/utils/supplementMicronutrients';
import {
  getSupplementNutrientLabel,
  formatSupplementNutrient,
  hasSupplementMicronutrients,
  sanitizeSupplementNutrients,
} from '@/lib/utils/supplementMicronutrients';

interface SupplementMicronutrientDisplayProps {
  supplement: Supplement;
  compact?: boolean;
}

// Daily Values for percentage calculation
const DAILY_VALUES: Partial<Record<keyof SupplementNutrients, number>> = {
  vitaminA: 900, // mcg
  vitaminC: 90, // mg
  vitaminD: 20, // mcg
  vitaminE: 15, // mg
  vitaminK: 120, // mcg
  vitaminB6: 1.7, // mg
  vitaminB12: 2.4, // mcg
  folate: 400, // mcg
  thiamin: 1.2, // mg
  riboflavin: 1.3, // mg
  niacin: 16, // mg
  calcium: 1300, // mg
  iron: 18, // mg
  magnesium: 420, // mg
  zinc: 11, // mg
  potassium: 4700, // mg
  sodium: 2300, // mg
  phosphorus: 1250, // mg
  omega3: 1600, // mg (recommendation varies)
};

function calculateDVPercent(
  value: number | undefined,
  dv: number | undefined
): number | null {
  if (value === undefined || value === null || !dv) return null;
  return Math.round((value / dv) * 100);
}

function getDVColor(percent: number | null): string {
  if (percent === null) return colors.text.tertiary;
  if (percent >= 100) return colors.status.success;
  if (percent >= 50) return colors.primary.main;
  if (percent >= 20) return colors.status.warning;
  return colors.text.tertiary;
}

/**
 * Extracts micronutrient data from supplement
 */
function extractNutrients(supplement: Supplement): SupplementNutrients {
  return {
    vitaminA: supplement.vitaminA,
    vitaminC: supplement.vitaminC,
    vitaminD: supplement.vitaminD,
    vitaminE: supplement.vitaminE,
    vitaminK: supplement.vitaminK,
    vitaminB6: supplement.vitaminB6,
    vitaminB12: supplement.vitaminB12,
    folate: supplement.folate,
    thiamin: supplement.thiamin,
    riboflavin: supplement.riboflavin,
    niacin: supplement.niacin,
    calcium: supplement.calcium,
    iron: supplement.iron,
    magnesium: supplement.magnesium,
    zinc: supplement.zinc,
    potassium: supplement.potassium,
    sodium: supplement.sodium,
    phosphorus: supplement.phosphorus,
    omega3: supplement.omega3,
  };
}

export function SupplementMicronutrientDisplay({
  supplement,
  compact = false,
}: SupplementMicronutrientDisplayProps) {
  const rawNutrients = useMemo(() => extractNutrients(supplement), [supplement]);
  const sanitizedNutrients = useMemo(
    () => sanitizeSupplementNutrients(rawNutrients),
    [rawNutrients]
  );

  if (!hasSupplementMicronutrients(sanitizedNutrients)) {
    return null;
  }

  // Get list of nutrients that have values
  const nutrientEntries = Object.entries(sanitizedNutrients)
    .filter(([, value]) => value !== undefined && value > 0)
    .sort((a, b) => {
      // Sort by Daily Value percentage (highest first)
      const aDV = DAILY_VALUES[a[0] as keyof SupplementNutrients];
      const bDV = DAILY_VALUES[b[0] as keyof SupplementNutrients];
      const aPercent = aDV ? ((a[1] as number) / aDV) * 100 : 0;
      const bPercent = bDV ? ((b[1] as number) / bDV) * 100 : 0;
      return bPercent - aPercent;
    }) as Array<[keyof SupplementNutrients, number]>;

  if (nutrientEntries.length === 0) {
    return null;
  }

  // Compact mode - just show badges for key nutrients
  if (compact) {
    const topNutrients = nutrientEntries.slice(0, 3);

    return (
      <View style={styles.compactContainer}>
        {topNutrients.map(([key, value]) => {
          const dvPercent = calculateDVPercent(
            value,
            DAILY_VALUES[key as keyof SupplementNutrients]
          );
          const shortLabel = getShortLabel(key);

          return (
            <View
              key={key}
              style={[
                styles.compactBadge,
                { borderColor: getDVColor(dvPercent) },
              ]}
            >
              <Text
                style={[styles.compactLabel, { color: getDVColor(dvPercent) }]}
              >
                {shortLabel}
              </Text>
              {dvPercent !== null && (
                <Text
                  style={[styles.compactPercent, { color: getDVColor(dvPercent) }]}
                >
                  {dvPercent}%
                </Text>
              )}
            </View>
          );
        })}
        {nutrientEntries.length > 3 && (
          <Text style={styles.compactMore}>+{nutrientEntries.length - 3}</Text>
        )}
      </View>
    );
  }

  // Full mode - show all nutrients in a card
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Ionicons name="nutrition" size={16} color={colors.primary.main} />
        <Text style={styles.headerTitle}>Micronutrients</Text>
      </View>

      <View style={styles.nutrientList}>
        {nutrientEntries.map(([key, value]) => {
          const dv = DAILY_VALUES[key as keyof SupplementNutrients];
          const dvPercent = calculateDVPercent(value, dv);

          return (
            <View key={key} style={styles.nutrientRow}>
              <Text style={styles.nutrientLabel}>
                {getSupplementNutrientLabel(key)}
              </Text>
              <View style={styles.nutrientValues}>
                <Text style={styles.nutrientValue}>
                  {formatSupplementNutrient(key, value)}
                </Text>
                {dvPercent !== null && (
                  <Text
                    style={[styles.dvPercent, { color: getDVColor(dvPercent) }]}
                  >
                    {dvPercent}% DV
                  </Text>
                )}
              </View>
            </View>
          );
        })}
      </View>

      <View style={styles.disclaimer}>
        <Text style={styles.disclaimerText}>
          * % DV = % Daily Value based on a 2,000 calorie diet
        </Text>
      </View>
    </View>
  );
}

/**
 * Gets short label for compact display
 */
function getShortLabel(nutrient: keyof SupplementNutrients): string {
  const shortLabels: Record<keyof SupplementNutrients, string> = {
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
    omega3: 'ω3',
  };

  return shortLabels[nutrient] || nutrient;
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    padding: spacing.md,
    marginTop: spacing.md,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginBottom: spacing.sm,
  },
  headerTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  nutrientList: {},
  nutrientRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: spacing.xs,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  nutrientLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    flex: 1,
  },
  nutrientValues: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  nutrientValue: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
    minWidth: 60,
    textAlign: 'right',
  },
  dvPercent: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
    minWidth: 50,
    textAlign: 'right',
  },
  disclaimer: {
    marginTop: spacing.sm,
    paddingTop: spacing.xs,
  },
  disclaimerText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    fontStyle: 'italic',
  },

  // Compact styles
  compactContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    flexWrap: 'wrap',
  },
  compactBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    paddingHorizontal: spacing.xs,
    paddingVertical: 2,
    borderRadius: borderRadius.xs,
    borderWidth: 1,
    gap: 2,
  },
  compactLabel: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
  },
  compactPercent: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
  },
  compactMore: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginLeft: spacing.xs,
  },
});
