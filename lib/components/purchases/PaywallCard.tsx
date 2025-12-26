/**
 * PaywallCard Component
 *
 * A subscription option card displaying product info, pricing,
 * trial eligibility, and a prominent CTA button.
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

import { colors, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';
import type { SubscriptionProduct } from '@/lib/services/purchases/types';

interface PaywallCardProps {
  product: SubscriptionProduct;
  isSelected: boolean;
  isLoading: boolean;
  isBestValue?: boolean;
  onSelect: () => void;
  onPurchase: () => void;
}

export function PaywallCard({
  product,
  isSelected,
  isLoading,
  isBestValue = false,
  onSelect,
  onPurchase,
}: PaywallCardProps) {
  const hasIntroOffer = product.isEligibleForIntroOffer && product.introOffer;
  const isFreeTrial = hasIntroOffer && product.introOffer?.type === 'free_trial';

  return (
    <TouchableOpacity
      style={[styles.container, isSelected && styles.containerSelected]}
      onPress={onSelect}
      activeOpacity={0.8}
      disabled={isLoading}
      accessibilityRole="button"
      accessibilityLabel={`${product.title}, ${product.displayPrice}${isFreeTrial ? ', with free trial' : ''}`}
      accessibilityState={{ selected: isSelected }}
    >
      {/* Best value badge */}
      {isBestValue && (
        <View style={styles.badge}>
          <LinearGradient
            colors={[colors.primary.main, colors.primary.gradient.end]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.badgeGradient}
          >
            <Text style={styles.badgeText}>Best Value</Text>
          </LinearGradient>
        </View>
      )}

      {/* Selection indicator */}
      <View style={styles.header}>
        <View style={[styles.radioOuter, isSelected && styles.radioOuterSelected]}>
          {isSelected && <View style={styles.radioInner} />}
        </View>

        <View style={styles.headerContent}>
          <Text style={styles.title}>{product.title}</Text>
          <Text style={styles.description}>{product.description}</Text>
        </View>
      </View>

      {/* Pricing */}
      <View style={styles.pricingContainer}>
        <Text style={styles.price}>{product.displayPrice}</Text>

        {/* Show per-month equivalent for yearly */}
        {product.period === 'yearly' && (
          <Text style={styles.monthlyEquivalent}>
            ({formatMonthlyEquivalent(product.priceValue, product.currency)}/mo)
          </Text>
        )}
      </View>

      {/* Intro offer banner */}
      {hasIntroOffer && (
        <View style={styles.introOfferBanner}>
          <Ionicons
            name={isFreeTrial ? 'gift-outline' : 'pricetag-outline'}
            size={16}
            color={colors.semantic.success}
          />
          <Text style={styles.introOfferText}>{formatIntroOffer(product.introOffer!)}</Text>
        </View>
      )}

      {/* Purchase button (only shown when selected) */}
      {isSelected && (
        <TouchableOpacity
          style={styles.purchaseButton}
          onPress={onPurchase}
          disabled={isLoading}
          accessibilityRole="button"
          accessibilityLabel={`Subscribe to ${product.title}`}
        >
          <LinearGradient
            colors={[colors.primary.main, colors.primary.gradient.end]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.purchaseButtonGradient}
          >
            {isLoading ? (
              <ActivityIndicator color={colors.text.primary} size="small" />
            ) : (
              <Text style={styles.purchaseButtonText}>
                {isFreeTrial ? 'Start Free Trial' : 'Subscribe Now'}
              </Text>
            )}
          </LinearGradient>
        </TouchableOpacity>
      )}
    </TouchableOpacity>
  );
}

/**
 * Format intro offer for display
 */
function formatIntroOffer(offer: NonNullable<SubscriptionProduct['introOffer']>): string {
  switch (offer.type) {
    case 'free_trial':
      return `${offer.cycles} ${offer.period} free trial`;
    case 'pay_up_front':
      return `${offer.price} for ${offer.cycles} ${offer.period}`;
    case 'pay_as_you_go':
      return `${offer.price}/${offer.period} for ${offer.cycles} periods`;
    default:
      return 'Special offer available';
  }
}

/**
 * Format monthly equivalent price for yearly subscriptions
 */
function formatMonthlyEquivalent(yearlyPrice: number, currency: string): string {
  const monthly = yearlyPrice / 12;

  // Get currency symbol
  const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });

  return formatter.format(monthly);
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.lg,
    borderWidth: 2,
    borderColor: colors.border.secondary,
    padding: spacing.lg,
    marginBottom: spacing.md,
    ...shadows.md,
  },
  containerSelected: {
    borderColor: colors.primary.main,
    backgroundColor: colors.special.highlight,
  },
  badge: {
    position: 'absolute',
    top: -12,
    right: spacing.md,
  },
  badgeGradient: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
  },
  badgeText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.bold,
    textTransform: 'uppercase',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: spacing.md,
  },
  radioOuter: {
    width: 24,
    height: 24,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: colors.border.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
    marginTop: 2,
  },
  radioOuterSelected: {
    borderColor: colors.primary.main,
  },
  radioInner: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: colors.primary.main,
  },
  headerContent: {
    flex: 1,
  },
  title: {
    ...typography.h3,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  description: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
  },
  pricingContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: spacing.sm,
    paddingLeft: 40, // Align with header content
  },
  price: {
    ...typography.h2,
    color: colors.text.primary,
  },
  monthlyEquivalent: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
    marginLeft: spacing.sm,
  },
  introOfferBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.special.successLight,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
    marginLeft: 40, // Align with header content
    marginBottom: spacing.md,
  },
  introOfferText: {
    ...typography.bodySmall,
    color: colors.semantic.success,
    fontWeight: typography.fontWeight.medium,
    marginLeft: spacing.sm,
  },
  purchaseButton: {
    marginTop: spacing.sm,
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
  },
  purchaseButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  purchaseButtonText: {
    ...typography.button,
    color: colors.text.primary,
  },
});

export default PaywallCard;
