/**
 * Classification Result Component
 *
 * Displays food classification results with USDA search matches
 * Includes confirmation UI and "search instead" option
 */

import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { colors, gradients, spacing, borderRadius, typography, shadows } from '../theme/colors';
import { USDAFood, FoodClassification } from '../types/foods';

interface ClassificationResultProps {
  classification: FoodClassification;
  usdaMatches: USDAFood[];
  portionEstimate?: {
    estimated_grams: number;
    quality: 'low' | 'medium' | 'high';
  };
  onConfirm: (food: USDAFood) => void;
  onSearchInstead: (query: string) => void;
  onReportIncorrect: () => void;
}

// Constants
const MAX_CONFIRM_BUTTON_TEXT_LENGTH = 30;

// Format confidence as percentage
const formatConfidence = (confidence: number): string => {
  return `${Math.round(confidence * 100)}%`;
};

// Get confidence color
const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.8) return colors.status.success;
  if (confidence >= 0.5) return colors.status.warning;
  return colors.status.error;
};

// Format food name for display
const formatFoodName = (name: string): string => {
  return name
    .replace(/_/g, ' ')
    .split(' ')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
};

export function ClassificationResult({
  classification,
  usdaMatches,
  portionEstimate,
  onConfirm,
  onSearchInstead,
  onReportIncorrect,
}: ClassificationResultProps) {
  const isLowConfidence = classification.confidence < 0.8;
  const confidenceColor = getConfidenceColor(classification.confidence);

  return (
    <View style={styles.container}>
      {/* Classification Header */}
      <View style={styles.header}>
        <View style={styles.categoryBadge}>
          <Ionicons
            name="restaurant-outline"
            size={16}
            color={colors.primary.main}
          />
          <Text style={styles.categoryText}>
            {formatFoodName(classification.category)}
          </Text>
        </View>

        {/* Confidence Bar */}
        <View style={styles.confidenceSection}>
          <View style={styles.confidenceBarContainer}>
            <View
              style={[
                styles.confidenceBar,
                {
                  width: `${classification.confidence * 100}%`,
                  backgroundColor: confidenceColor,
                },
              ]}
            />
          </View>
          <Text style={[styles.confidenceText, { color: confidenceColor }]}>
            {formatConfidence(classification.confidence)} confident
          </Text>
        </View>
      </View>

      {/* Low Confidence Warning */}
      {isLowConfidence && (
        <View style={styles.warningBanner}>
          <Ionicons
            name="warning-outline"
            size={20}
            color={colors.status.warning}
          />
          <Text style={styles.warningText}>
            Low confidence detection. Please verify or search for the correct food.
          </Text>
        </View>
      )}

      {/* Portion Estimate */}
      {portionEstimate && (
        <View style={styles.portionInfo}>
          <Ionicons name="resize-outline" size={16} color={colors.text.tertiary} />
          <Text style={styles.portionText}>
            Estimated portion: ~{Math.round(portionEstimate.estimated_grams)}g
            <Text style={styles.portionQuality}>
              {' '}({portionEstimate.quality} accuracy)
            </Text>
          </Text>
        </View>
      )}

      {/* Alternative Classifications */}
      {classification.alternatives.length > 0 && (
        <View style={styles.alternativesSection}>
          <Text style={styles.sectionLabel}>Could also be:</Text>
          <View style={styles.alternativeChips}>
            {classification.alternatives.slice(0, 3).map((alt, index) => (
              <TouchableOpacity
                key={index}
                style={styles.alternativeChip}
                onPress={() => onSearchInstead(alt.category)}
              >
                <Text style={styles.alternativeChipText}>
                  {formatFoodName(alt.category)}
                </Text>
                <Text style={styles.alternativeChipConfidence}>
                  {formatConfidence(alt.confidence)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      )}

      {/* USDA Matches */}
      <View style={styles.matchesSection}>
        <Text style={styles.sectionTitle}>Matching Foods</Text>
        <Text style={styles.sectionSubtitle}>
          Select the best match from USDA database
        </Text>

        <ScrollView
          style={styles.matchesList}
          showsVerticalScrollIndicator={false}
          nestedScrollEnabled
        >
          {usdaMatches.slice(0, 5).map((food, index) => (
            <TouchableOpacity
              key={food.fdcId}
              style={[
                styles.matchCard,
                index === 0 && styles.matchCardFirst,
              ]}
              onPress={() => onConfirm(food)}
              activeOpacity={0.7}
            >
              <View style={styles.matchContent}>
                <View style={styles.matchHeader}>
                  <Text style={styles.matchName} numberOfLines={2}>
                    {food.description}
                  </Text>
                  {index === 0 && (
                    <View style={styles.topMatchBadge}>
                      <Ionicons
                        name="star"
                        size={12}
                        color={colors.status.success}
                      />
                      <Text style={styles.topMatchText}>Best Match</Text>
                    </View>
                  )}
                </View>

                {food.brandOwner && (
                  <Text style={styles.matchBrand}>{food.brandOwner}</Text>
                )}

                {/* Nutrition Preview */}
                <View style={styles.nutritionRow}>
                  <View style={styles.nutritionItem}>
                    <Text style={styles.nutritionValue}>{Math.round(food.calories)}</Text>
                    <Text style={styles.nutritionLabel}>cal</Text>
                  </View>
                  <View style={styles.nutritionItem}>
                    <Text style={styles.nutritionValue}>{Math.round(food.protein)}g</Text>
                    <Text style={styles.nutritionLabel}>protein</Text>
                  </View>
                  <View style={styles.nutritionItem}>
                    <Text style={styles.nutritionValue}>{Math.round(food.carbs)}g</Text>
                    <Text style={styles.nutritionLabel}>carbs</Text>
                  </View>
                  <View style={styles.nutritionItem}>
                    <Text style={styles.nutritionValue}>{Math.round(food.fat)}g</Text>
                    <Text style={styles.nutritionLabel}>fat</Text>
                  </View>
                </View>

                {food.servingSize && food.servingSizeUnit && (
                  <Text style={styles.servingSize}>
                    Per {food.servingSize} {food.servingSizeUnit}
                  </Text>
                )}
              </View>

              <Ionicons
                name="chevron-forward"
                size={20}
                color={colors.text.tertiary}
              />
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      {/* Actions */}
      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.searchButton}
          onPress={() => onSearchInstead(classification.category)}
          activeOpacity={0.7}
        >
          <Ionicons name="search-outline" size={20} color={colors.primary.main} />
          <Text style={styles.searchButtonText}>Search for different food</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.reportButton}
          onPress={onReportIncorrect}
          activeOpacity={0.7}
        >
          <Ionicons
            name="flag-outline"
            size={16}
            color={colors.text.tertiary}
          />
          <Text style={styles.reportButtonText}>Report misclassification</Text>
        </TouchableOpacity>
      </View>

      {/* Quick Confirm for Top Match */}
      {usdaMatches.length > 0 && classification.confidence >= 0.8 && (
        <TouchableOpacity
          style={styles.confirmButton}
          onPress={() => onConfirm(usdaMatches[0])}
          activeOpacity={0.8}
        >
          <LinearGradient
            colors={gradients.primary as [string, string]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.confirmButtonGradient}
          >
            <Ionicons name="checkmark-circle" size={20} color={colors.text.primary} />
            <Text style={styles.confirmButtonText}>
              Confirm: {usdaMatches[0].description.slice(0, MAX_CONFIRM_BUTTON_TEXT_LENGTH)}
              {usdaMatches[0].description.length > MAX_CONFIRM_BUTTON_TEXT_LENGTH ? '...' : ''}
            </Text>
          </LinearGradient>
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  header: {
    padding: spacing.md,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    marginBottom: spacing.md,
  },
  categoryBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginBottom: spacing.sm,
  },
  categoryText: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
  },
  confidenceSection: {
    gap: spacing.xs,
  },
  confidenceBarContainer: {
    height: 8,
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.full,
    overflow: 'hidden',
  },
  confidenceBar: {
    height: '100%',
    borderRadius: borderRadius.full,
  },
  confidenceText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium as '500',
  },
  warningBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    padding: spacing.md,
    backgroundColor: colors.status.warning + '15',
    borderRadius: borderRadius.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.status.warning + '40',
  },
  warningText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.status.warning,
  },
  portionInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginBottom: spacing.md,
  },
  portionText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },
  portionQuality: {
    color: colors.text.tertiary,
  },
  alternativesSection: {
    marginBottom: spacing.md,
  },
  sectionLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  alternativeChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.xs,
  },
  alternativeChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.full,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  alternativeChipText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.secondary,
  },
  alternativeChipConfidence: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  matchesSection: {
    flex: 1,
    marginBottom: spacing.md,
  },
  sectionTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  sectionSubtitle: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.md,
  },
  matchesList: {
    maxHeight: 300,
  },
  matchCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.sm,
    ...shadows.sm,
  },
  matchCardFirst: {
    borderWidth: 1,
    borderColor: colors.status.success + '40',
  },
  matchContent: {
    flex: 1,
  },
  matchHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: spacing.xs,
  },
  matchName: {
    flex: 1,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
    marginRight: spacing.sm,
  },
  topMatchBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    backgroundColor: colors.status.success + '20',
    paddingHorizontal: spacing.xs,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
  },
  topMatchText: {
    fontSize: typography.fontSize.xs,
    color: colors.status.success,
    fontWeight: typography.fontWeight.medium as '500',
  },
  matchBrand: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginBottom: spacing.sm,
  },
  nutritionRow: {
    flexDirection: 'row',
    gap: spacing.md,
    marginBottom: spacing.xs,
  },
  nutritionItem: {
    alignItems: 'center',
  },
  nutritionValue: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.bold as '700',
    color: colors.text.primary,
  },
  nutritionLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  servingSize: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
  },
  actions: {
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  searchButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    padding: spacing.md,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.primary.main + '40',
  },
  searchButtonText: {
    fontSize: typography.fontSize.md,
    color: colors.primary.main,
    fontWeight: typography.fontWeight.medium as '500',
  },
  reportButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
    padding: spacing.sm,
  },
  reportButtonText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  confirmButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    ...shadows.md,
  },
  confirmButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
  },
  confirmButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
  },
});

export default ClassificationResult;
