/**
 * PaywallScreen
 *
 * Full-screen paywall UI displaying subscription options,
 * feature comparison, and purchase flow.
 */

import React, { useEffect, useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  SafeAreaView,
  Linking,
  Alert,
  Platform,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

import { colors, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';
import { PaywallCard } from '@/lib/components/purchases/PaywallCard';
import {
  initializePurchaseService,
  fetchSubscriptionProducts,
  purchaseSubscription,
  restorePurchases,
  getPurchaseState,
} from '@/lib/services/purchases';
import { syncEntitlements } from '@/lib/services/purchases/entitlements';
import type { SubscriptionProduct, PurchaseState } from '@/lib/services/purchases/types';
import { FREE_TIER_FEATURES, PRO_TIER_FEATURES } from '@/lib/services/purchases/products';

/** Feature item for comparison table */
interface FeatureItem {
  name: string;
  freeValue: boolean | string;
  proValue: boolean | string;
}

/** Features to compare */
const FEATURE_COMPARISON: FeatureItem[] = [
  { name: 'Basic meal tracking', freeValue: true, proValue: true },
  { name: 'Daily nutrition summary', freeValue: true, proValue: true },
  { name: 'Unlimited meal history', freeValue: false, proValue: true },
  { name: 'ML-powered insights', freeValue: false, proValue: true },
  { name: 'Advanced analytics', freeValue: false, proValue: true },
  { name: 'Custom nutrition goals', freeValue: false, proValue: true },
  { name: 'Export your data', freeValue: false, proValue: true },
  { name: 'Priority support', freeValue: false, proValue: true },
];

export default function PaywallScreen() {
  const router = useRouter();

  // State
  const [products, setProducts] = useState<SubscriptionProduct[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isPurchasing, setIsPurchasing] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize and fetch products
  useEffect(() => {
    async function loadProducts() {
      try {
        setIsLoading(true);
        setError(null);

        // Initialize purchase service
        const initialized = await initializePurchaseService();
        if (!initialized) {
          setError('Unable to connect to the App Store. Please try again.');
          return;
        }

        // Fetch products
        const fetchedProducts = await fetchSubscriptionProducts();
        if (fetchedProducts.length === 0) {
          setError('No subscription options available. Please try again later.');
          return;
        }

        setProducts(fetchedProducts);

        // Auto-select the yearly plan (best value)
        const yearlyProduct = fetchedProducts.find((p) => p.period === 'yearly');
        if (yearlyProduct) {
          setSelectedProductId(yearlyProduct.productId);
        } else if (fetchedProducts.length > 0) {
          setSelectedProductId(fetchedProducts[0].productId);
        }
      } catch (err) {
        console.error('[Paywall] Error loading products:', err);
        setError('Failed to load subscription options. Please try again.');
      } finally {
        setIsLoading(false);
      }
    }

    loadProducts();
  }, []);

  // Handle purchase
  const handlePurchase = useCallback(async () => {
    if (!selectedProductId) return;

    try {
      setIsPurchasing(true);
      setError(null);

      const result = await purchaseSubscription(selectedProductId);

      if (result.success) {
        // Sync entitlements
        await syncEntitlements();

        Alert.alert(
          'Welcome to Pro!',
          'Your subscription is now active. Enjoy all the premium features!',
          [
            {
              text: 'Get Started',
              onPress: () => router.back(),
            },
          ]
        );
      } else if (result.error) {
        if (result.error.code === 'E_USER_CANCELLED') {
          // User cancelled - don't show error
          return;
        }
        if (result.error.code === 'E_DEFERRED') {
          Alert.alert(
            'Purchase Pending',
            "Your purchase requires approval. You'll get access once it's approved.",
            [{ text: 'OK' }]
          );
          return;
        }
        setError(result.error.userMessage);
      }
    } catch (err) {
      console.error('[Paywall] Purchase error:', err);
      setError('An error occurred during purchase. Please try again.');
    } finally {
      setIsPurchasing(false);
    }
  }, [selectedProductId, router]);

  // Handle restore purchases
  const handleRestore = useCallback(async () => {
    try {
      setIsRestoring(true);
      setError(null);

      const result = await restorePurchases();

      if (result.success) {
        await syncEntitlements();
        Alert.alert('Purchases Restored', 'Your subscription has been restored successfully!', [
          {
            text: 'Continue',
            onPress: () => router.back(),
          },
        ]);
      } else if (result.error) {
        if (result.error.code === 'E_PURCHASE_FAILED') {
          Alert.alert('No Purchases Found', "We couldn't find any previous purchases to restore.", [
            { text: 'OK' },
          ]);
        } else {
          setError(result.error.userMessage);
        }
      }
    } catch (err) {
      console.error('[Paywall] Restore error:', err);
      setError('Failed to restore purchases. Please try again.');
    } finally {
      setIsRestoring(false);
    }
  }, [router]);

  // Handle close
  const handleClose = useCallback(() => {
    router.back();
  }, [router]);

  // Render loading state
  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Loading subscription options...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          onPress={handleClose}
          style={styles.closeButton}
          accessibilityRole="button"
          accessibilityLabel="Close"
        >
          <Ionicons name="close" size={28} color={colors.text.primary} />
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Hero section */}
        <View style={styles.heroSection}>
          <LinearGradient
            colors={[colors.primary.main, colors.primary.gradient.end]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.heroIcon}
          >
            <Ionicons name="star" size={32} color={colors.text.primary} />
          </LinearGradient>
          <Text style={styles.heroTitle}>Upgrade to Pro</Text>
          <Text style={styles.heroSubtitle}>
            Unlock the full potential of your nutrition journey
          </Text>
        </View>

        {/* Feature comparison */}
        <View style={styles.featureSection}>
          <Text style={styles.sectionTitle}>Compare Plans</Text>
          <View style={styles.featureTable}>
            {/* Header row */}
            <View style={styles.featureHeaderRow}>
              <Text style={[styles.featureHeaderCell, styles.featureNameCell]}>Feature</Text>
              <Text style={styles.featureHeaderCell}>Free</Text>
              <Text style={[styles.featureHeaderCell, styles.proHeaderCell]}>Pro</Text>
            </View>

            {/* Feature rows */}
            {FEATURE_COMPARISON.map((feature, index) => (
              <View
                key={feature.name}
                style={[
                  styles.featureRow,
                  index === FEATURE_COMPARISON.length - 1 && styles.featureRowLast,
                ]}
              >
                <Text style={[styles.featureCell, styles.featureNameCell]}>{feature.name}</Text>
                <View style={styles.featureCell}>
                  {renderFeatureValue(feature.freeValue, false)}
                </View>
                <View style={[styles.featureCell, styles.proCellBg]}>
                  {renderFeatureValue(feature.proValue, true)}
                </View>
              </View>
            ))}
          </View>
        </View>

        {/* Subscription options */}
        <View style={styles.subscriptionSection}>
          <Text style={styles.sectionTitle}>Choose Your Plan</Text>

          {error && (
            <View style={styles.errorBanner}>
              <Ionicons name="alert-circle" size={20} color={colors.semantic.error} />
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}

          {products.map((product) => (
            <PaywallCard
              key={product.productId}
              product={product}
              isSelected={selectedProductId === product.productId}
              isLoading={isPurchasing && selectedProductId === product.productId}
              isBestValue={product.period === 'yearly'}
              onSelect={() => setSelectedProductId(product.productId)}
              onPurchase={handlePurchase}
            />
          ))}
        </View>

        {/* Restore purchases */}
        <TouchableOpacity
          style={styles.restoreButton}
          onPress={handleRestore}
          disabled={isRestoring}
          accessibilityRole="button"
          accessibilityLabel="Restore Purchases"
        >
          {isRestoring ? (
            <ActivityIndicator size="small" color={colors.text.tertiary} />
          ) : (
            <Text style={styles.restoreButtonText}>Restore Purchases</Text>
          )}
        </TouchableOpacity>

        {/* Legal footer */}
        <View style={styles.legalSection}>
          <Text style={styles.legalText}>
            Payment will be charged to your Apple ID account at confirmation of purchase.
            Subscription automatically renews unless it is canceled at least 24 hours before the end
            of the current period. Your account will be charged for renewal within 24 hours prior to
            the end of the current period. You can manage and cancel your subscriptions by going to
            your account settings in the App Store after purchase.
          </Text>

          <View style={styles.legalLinks}>
            <TouchableOpacity
              onPress={() => router.push('/terms')}
              accessibilityRole="link"
              accessibilityLabel="Terms of Service"
            >
              <Text style={styles.legalLink}>Terms of Service</Text>
            </TouchableOpacity>
            <Text style={styles.legalSeparator}>|</Text>
            <TouchableOpacity
              onPress={() => router.push('/privacy')}
              accessibilityRole="link"
              accessibilityLabel="Privacy Policy"
            >
              <Text style={styles.legalLink}>Privacy Policy</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

/**
 * Render feature value (checkmark, X, or text)
 */
function renderFeatureValue(value: boolean | string, isPro: boolean) {
  if (typeof value === 'string') {
    return <Text style={styles.featureValueText}>{value}</Text>;
  }

  if (value) {
    return (
      <Ionicons
        name="checkmark-circle"
        size={20}
        color={isPro ? colors.semantic.success : colors.text.tertiary}
      />
    );
  }

  return <Ionicons name="close-circle" size={20} color={colors.text.disabled} />;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  loadingText: {
    ...typography.body,
    color: colors.text.tertiary,
    marginTop: spacing.md,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  closeButton: {
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 22,
    backgroundColor: colors.surface.elevated,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing['2xl'],
  },
  heroSection: {
    alignItems: 'center',
    paddingVertical: spacing.xl,
  },
  heroIcon: {
    width: 72,
    height: 72,
    borderRadius: 36,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.lg,
    ...shadows.glow,
  },
  heroTitle: {
    ...typography.h1,
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: spacing.sm,
  },
  heroSubtitle: {
    ...typography.body,
    color: colors.text.tertiary,
    textAlign: 'center',
    maxWidth: 280,
  },
  sectionTitle: {
    ...typography.h3,
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  featureSection: {
    marginBottom: spacing.xl,
  },
  featureTable: {
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
    ...shadows.sm,
  },
  featureHeaderRow: {
    flexDirection: 'row',
    backgroundColor: colors.surface.elevated,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  featureHeaderCell: {
    ...typography.bodySmall,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    flex: 1,
    textAlign: 'center',
  },
  featureNameCell: {
    flex: 2,
    textAlign: 'left',
    paddingLeft: spacing.md,
  },
  proHeaderCell: {
    color: colors.primary.main,
  },
  featureRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  featureRowLast: {
    borderBottomWidth: 0,
  },
  featureCell: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  proCellBg: {
    backgroundColor: colors.special.highlight,
  },
  featureValueText: {
    ...typography.bodySmall,
    color: colors.text.secondary,
  },
  subscriptionSection: {
    marginBottom: spacing.lg,
  },
  errorBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.special.errorLight,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
    marginBottom: spacing.md,
  },
  errorText: {
    ...typography.bodySmall,
    color: colors.semantic.error,
    marginLeft: spacing.sm,
    flex: 1,
  },
  restoreButton: {
    alignItems: 'center',
    paddingVertical: spacing.md,
    marginBottom: spacing.lg,
  },
  restoreButtonText: {
    ...typography.body,
    color: colors.text.tertiary,
    textDecorationLine: 'underline',
  },
  legalSection: {
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  legalText: {
    ...typography.caption,
    color: colors.text.disabled,
    textAlign: 'center',
    lineHeight: 18,
    marginBottom: spacing.md,
  },
  legalLinks: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  legalLink: {
    ...typography.caption,
    color: colors.primary.light,
    textDecorationLine: 'underline',
  },
  legalSeparator: {
    ...typography.caption,
    color: colors.text.disabled,
    marginHorizontal: spacing.sm,
  },
});
