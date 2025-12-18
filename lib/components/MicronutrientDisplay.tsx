import React, { useState, useMemo } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, LayoutAnimation } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Meal } from '@/lib/types';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';
import {
  sanitizeNutritionData,
  formatNutrientValue,
  getNutrientUnit,
  NutritionData,
  NutrientKey,
} from '@/lib/utils/nutritionSanitizer';

interface MicronutrientDisplayProps {
  meal: Meal;
  compact?: boolean;
}

interface NutrientInfo {
  key: NutrientKey;
  label: string;
  dv?: number; // Daily Value for percentage calculation
}

// Organized by category with Daily Values (where applicable)
const MINERALS: NutrientInfo[] = [
  { key: 'sodium', label: 'Sodium', dv: 2300 },
  { key: 'potassium', label: 'Potassium', dv: 4700 },
  { key: 'calcium', label: 'Calcium', dv: 1300 },
  { key: 'iron', label: 'Iron', dv: 18 },
  { key: 'magnesium', label: 'Magnesium', dv: 420 },
  { key: 'zinc', label: 'Zinc', dv: 11 },
  { key: 'phosphorus', label: 'Phosphorus', dv: 1250 },
];

const VITAMINS: NutrientInfo[] = [
  { key: 'vitaminA', label: 'Vitamin A', dv: 900 },
  { key: 'vitaminC', label: 'Vitamin C', dv: 90 },
  { key: 'vitaminD', label: 'Vitamin D', dv: 20 },
  { key: 'vitaminE', label: 'Vitamin E', dv: 15 },
  { key: 'vitaminK', label: 'Vitamin K', dv: 120 },
  { key: 'vitaminB6', label: 'Vitamin B6', dv: 1.7 },
  { key: 'vitaminB12', label: 'Vitamin B12', dv: 2.4 },
  { key: 'folate', label: 'Folate', dv: 400 },
  { key: 'thiamin', label: 'Thiamin (B1)', dv: 1.2 },
  { key: 'riboflavin', label: 'Riboflavin (B2)', dv: 1.3 },
  { key: 'niacin', label: 'Niacin (B3)', dv: 16 },
];

const FAT_BREAKDOWN: NutrientInfo[] = [
  { key: 'saturatedFat', label: 'Saturated Fat', dv: 20 },
  { key: 'transFat', label: 'Trans Fat' },
  { key: 'cholesterol', label: 'Cholesterol', dv: 300 },
];

function calculateDVPercent(value: number | undefined, dv: number | undefined): number | null {
  if (value === undefined || value === null || !dv) return null;
  return Math.round((value / dv) * 100);
}

function getDVColor(percent: number | null): string {
  if (percent === null) return colors.text.tertiary;
  if (percent >= 20) return colors.status.success;
  if (percent >= 10) return colors.status.warning;
  return colors.text.tertiary;
}

function NutrientRow({ nutrient, sanitizedData }: { nutrient: NutrientInfo; sanitizedData: NutritionData }) {
  const value = sanitizedData[nutrient.key];
  const dvPercent = calculateDVPercent(value, nutrient.dv);

  if (value === undefined || value === null) return null;

  const unit = getNutrientUnit(nutrient.key);
  const formattedValue = formatNutrientValue(nutrient.key, value);

  return (
    <View style={styles.nutrientRow}>
      <Text style={styles.nutrientLabel}>{nutrient.label}</Text>
      <View style={styles.nutrientValues}>
        <Text style={styles.nutrientValue}>{formattedValue}{unit}</Text>
        {dvPercent !== null && (
          <Text style={[styles.dvPercent, { color: getDVColor(dvPercent) }]}>
            {dvPercent}% DV
          </Text>
        )}
      </View>
    </View>
  );
}

function NutrientSection({
  title,
  nutrients,
  sanitizedData,
}: {
  title: string;
  nutrients: NutrientInfo[];
  sanitizedData: NutritionData;
}) {
  const hasAnyValue = nutrients.some(n => {
    const value = sanitizedData[n.key];
    return value !== undefined && value !== null;
  });

  if (!hasAnyValue) return null;

  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {nutrients.map(nutrient => (
        <NutrientRow key={nutrient.key} nutrient={nutrient} sanitizedData={sanitizedData} />
      ))}
    </View>
  );
}

export function MicronutrientDisplay({ meal, compact = false }: MicronutrientDisplayProps) {
  const [expanded, setExpanded] = useState(false);

  // Sanitize meal data to remove outliers - memoized for performance
  const sanitizedData = useMemo(() => {
    const rawData: NutritionData = {
      calories: meal.calories,
      protein: meal.protein,
      carbs: meal.carbs,
      fat: meal.fat,
      fiber: meal.fiber,
      sugar: meal.sugar,
      saturatedFat: meal.saturatedFat,
      transFat: meal.transFat,
      cholesterol: meal.cholesterol,
      sodium: meal.sodium,
      potassium: meal.potassium,
      calcium: meal.calcium,
      iron: meal.iron,
      magnesium: meal.magnesium,
      zinc: meal.zinc,
      phosphorus: meal.phosphorus,
      vitaminA: meal.vitaminA,
      vitaminC: meal.vitaminC,
      vitaminD: meal.vitaminD,
      vitaminE: meal.vitaminE,
      vitaminK: meal.vitaminK,
      vitaminB6: meal.vitaminB6,
      vitaminB12: meal.vitaminB12,
      folate: meal.folate,
      thiamin: meal.thiamin,
      riboflavin: meal.riboflavin,
      niacin: meal.niacin,
    };
    // Sanitize as per-serving data since meal data represents a single serving
    return sanitizeNutritionData(rawData, 'perServing').data;
  }, [meal]);

  // Check if any micronutrients are available after sanitization
  const hasMicronutrients = [...MINERALS, ...VITAMINS, ...FAT_BREAKDOWN].some(n => {
    const value = sanitizedData[n.key];
    return value !== undefined && value !== null;
  });

  if (!hasMicronutrients) {
    return null;
  }

  // For compact mode, show a summary of key nutrients
  if (compact) {
    const keyNutrients = [
      { key: 'vitaminC' as NutrientKey, label: 'C', value: sanitizedData.vitaminC },
      { key: 'vitaminD' as NutrientKey, label: 'D', value: sanitizedData.vitaminD },
      { key: 'iron' as NutrientKey, label: 'Fe', value: sanitizedData.iron },
      { key: 'calcium' as NutrientKey, label: 'Ca', value: sanitizedData.calcium },
    ].filter(n => n.value !== undefined && n.value !== null);

    if (keyNutrients.length === 0) return null;

    return (
      <View style={styles.compactContainer}>
        {keyNutrients.slice(0, 4).map((n) => (
          <View key={n.key} style={styles.compactBadge}>
            <Text style={styles.compactLabel}>{n.label}</Text>
          </View>
        ))}
        {keyNutrients.length > 4 && (
          <Text style={styles.compactMore}>+{keyNutrients.length - 4}</Text>
        )}
      </View>
    );
  }

  const toggleExpanded = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setExpanded(!expanded);
  };

  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.header}
        onPress={toggleExpanded}
        activeOpacity={0.7}
        accessibilityRole="button"
        accessibilityLabel={expanded ? 'Collapse micronutrients' : 'Expand micronutrients'}
      >
        <View style={styles.headerLeft}>
          <Ionicons name="nutrition" size={18} color={colors.primary.main} />
          <Text style={styles.headerTitle}>Micronutrients</Text>
        </View>
        <Ionicons
          name={expanded ? 'chevron-up' : 'chevron-down'}
          size={20}
          color={colors.text.tertiary}
        />
      </TouchableOpacity>

      {expanded && (
        <View style={styles.content}>
          <NutrientSection title="Fat Breakdown" nutrients={FAT_BREAKDOWN} sanitizedData={sanitizedData} />
          <NutrientSection title="Minerals" nutrients={MINERALS} sanitizedData={sanitizedData} />
          <NutrientSection title="Vitamins" nutrients={VITAMINS} sanitizedData={sanitizedData} />

          <View style={styles.disclaimer}>
            <Text style={styles.disclaimerText}>
              * % DV = % Daily Value based on a 2,000 calorie diet
            </Text>
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.md,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  headerTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  content: {
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  section: {
    marginTop: spacing.md,
  },
  sectionTitle: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: spacing.sm,
  },
  nutrientRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: spacing.xs,
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
    marginTop: spacing.md,
    paddingTop: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  disclaimerText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    fontStyle: 'italic',
  },

  // Compact mode styles
  compactContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginTop: spacing.xs,
  },
  compactBadge: {
    backgroundColor: colors.primary.main + '20',
    paddingHorizontal: spacing.xs,
    paddingVertical: 2,
    borderRadius: borderRadius.xs,
  },
  compactLabel: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },
  compactMore: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
});
