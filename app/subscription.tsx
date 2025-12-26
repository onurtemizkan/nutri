/**
 * ManageSubscriptionScreen
 *
 * Displays current subscription status and provides access
 * to App Store subscription management.
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
  RefreshControl,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

import { colors, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';
import { SubscriptionBadge } from '@/lib/components/purchases/SubscriptionBadge';
import { getSubscriptionInfo, restorePurchases } from '@/lib/services/purchases';
import {
  syncEntitlements,
  getExpirationInfo,
  checkIsInTrialPeriod,
  getTrialDaysRemaining,
} from '@/lib/services/purchases/entitlements';
import {
  APP_STORE_SUBSCRIPTION_URL,
  PRODUCT_DISPLAY_NAMES,
} from '@/lib/services/purchases/products';
import type { LocalSubscriptionInfo } from '@/lib/services/purchases/types';

export default function ManageSubscriptionScreen() {
  const router = useRouter();

  // State
  const [subscriptionInfo, setSubscriptionInfo] = useState<LocalSubscriptionInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);
  const [expirationInfo, setExpirationInfo] = useState<{
    isExpiring: boolean;
    expiresAt: Date | null;
    daysRemaining: number | null;
  } | null>(null);
  const [isInTrial, setIsInTrial] = useState(false);
  const [trialDaysRemaining, setTrialDaysRemaining] = useState<number | null>(null);

  // Load subscription info
  const loadSubscriptionInfo = useCallback(async (showRefreshIndicator = false) => {
    try {
      if (showRefreshIndicator) {
        setIsRefreshing(true);
      } else {
        setIsLoading(true);
      }

      // Sync with backend first
      await syncEntitlements();

      // Get subscription info
      const info = await getSubscriptionInfo();
      setSubscriptionInfo(info);

      // Get expiration info
      const expInfo = await getExpirationInfo();
      setExpirationInfo(expInfo);

      // Check trial status
      const trialStatus = await checkIsInTrialPeriod();
      setIsInTrial(trialStatus);

      if (trialStatus) {
        const remaining = await getTrialDaysRemaining();
        setTrialDaysRemaining(remaining);
      }
    } catch (error) {
      console.error('[Subscription] Error loading info:', error);
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadSubscriptionInfo();
  }, [loadSubscriptionInfo]);

  // Handle refresh
  const handleRefresh = useCallback(() => {
    loadSubscriptionInfo(true);
  }, [loadSubscriptionInfo]);

  // Handle manage subscription
  const handleManageSubscription = useCallback(async () => {
    try {
      const canOpen = await Linking.canOpenURL(APP_STORE_SUBSCRIPTION_URL);
      if (canOpen) {
        await Linking.openURL(APP_STORE_SUBSCRIPTION_URL);
      } else {
        Alert.alert(
          'Cannot Open Settings',
          'Please go to Settings > Apple ID > Subscriptions to manage your subscription.',
          [{ text: 'OK' }]
        );
      }
    } catch (error) {
      console.error('[Subscription] Error opening subscription management:', error);
      Alert.alert('Error', 'Unable to open subscription management.');
    }
  }, []);

  // Handle restore purchases
  const handleRestorePurchases = useCallback(async () => {
    try {
      setIsRestoring(true);

      const result = await restorePurchases();

      if (result.success) {
        await loadSubscriptionInfo();
        Alert.alert('Success', 'Your purchases have been restored.');
      } else if (result.error) {
        if (result.error.code === 'E_PURCHASE_FAILED') {
          Alert.alert('No Purchases Found', "We couldn't find any previous purchases to restore.");
        } else {
          Alert.alert('Error', result.error.userMessage);
        }
      }
    } catch (error) {
      console.error('[Subscription] Restore error:', error);
      Alert.alert('Error', 'Failed to restore purchases. Please try again.');
    } finally {
      setIsRestoring(false);
    }
  }, [loadSubscriptionInfo]);

  // Handle upgrade
  const handleUpgrade = useCallback(() => {
    router.push('/paywall');
  }, [router]);

  // Handle back
  const handleBack = useCallback(() => {
    router.back();
  }, [router]);

  // Get status display
  const getStatusDisplay = () => {
    if (!subscriptionInfo) return { text: 'Unknown', color: colors.text.tertiary };

    switch (subscriptionInfo.status) {
      case 'active':
        return { text: 'Active', color: colors.semantic.success };
      case 'in_grace_period':
        return { text: 'Grace Period', color: colors.semantic.warning };
      case 'in_billing_retry':
        return { text: 'Billing Issue', color: colors.semantic.warning };
      case 'expired':
        return { text: 'Expired', color: colors.semantic.error };
      case 'revoked':
        return { text: 'Revoked', color: colors.semantic.error };
      case 'refunded':
        return { text: 'Refunded', color: colors.semantic.error };
      default:
        return { text: 'Free', color: colors.text.tertiary };
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Loading subscription info...</Text>
        </View>
      </SafeAreaView>
    );
  }

  const isPro = subscriptionInfo?.tier === 'pro' && subscriptionInfo?.status === 'active';
  const statusDisplay = getStatusDisplay();

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          onPress={handleBack}
          style={styles.backButton}
          accessibilityRole="button"
          accessibilityLabel="Go back"
        >
          <Ionicons name="chevron-back" size={28} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Subscription</Text>
        <View style={styles.headerSpacer} />
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={isRefreshing}
            onRefresh={handleRefresh}
            tintColor={colors.primary.main}
          />
        }
      >
        {/* Current plan card */}
        <View style={styles.planCard}>
          <View style={styles.planHeader}>
            <View>
              <Text style={styles.planLabel}>Current Plan</Text>
              <View style={styles.planTitleRow}>
                <Text style={styles.planTitle}>{isPro ? 'Nutri Pro' : 'Free'}</Text>
                {isPro && <SubscriptionBadge size="small" style={styles.badge} />}
              </View>
            </View>
            <View style={[styles.statusBadge, { backgroundColor: `${statusDisplay.color}20` }]}>
              <View style={[styles.statusDot, { backgroundColor: statusDisplay.color }]} />
              <Text style={[styles.statusText, { color: statusDisplay.color }]}>
                {statusDisplay.text}
              </Text>
            </View>
          </View>

          {/* Subscription details */}
          {isPro && subscriptionInfo && (
            <View style={styles.detailsContainer}>
              {/* Product name */}
              {subscriptionInfo.productId && (
                <View style={styles.detailRow}>
                  <Ionicons name="pricetag-outline" size={18} color={colors.text.tertiary} />
                  <Text style={styles.detailLabel}>Plan:</Text>
                  <Text style={styles.detailValue}>
                    {PRODUCT_DISPLAY_NAMES[subscriptionInfo.productId] ||
                      subscriptionInfo.productId}
                  </Text>
                </View>
              )}

              {/* Trial status */}
              {isInTrial && trialDaysRemaining !== null && (
                <View style={styles.detailRow}>
                  <Ionicons name="gift-outline" size={18} color={colors.semantic.success} />
                  <Text style={styles.detailLabel}>Trial:</Text>
                  <Text style={[styles.detailValue, { color: colors.semantic.success }]}>
                    {trialDaysRemaining} days remaining
                  </Text>
                </View>
              )}

              {/* Renewal date */}
              {expirationInfo?.expiresAt && (
                <View style={styles.detailRow}>
                  <Ionicons name="calendar-outline" size={18} color={colors.text.tertiary} />
                  <Text style={styles.detailLabel}>
                    {subscriptionInfo.autoRenewEnabled ? 'Renews:' : 'Expires:'}
                  </Text>
                  <Text style={styles.detailValue}>{formatDate(expirationInfo.expiresAt)}</Text>
                </View>
              )}

              {/* Auto-renew status */}
              <View style={styles.detailRow}>
                <Ionicons
                  name={subscriptionInfo.autoRenewEnabled ? 'refresh' : 'close-circle-outline'}
                  size={18}
                  color={
                    subscriptionInfo.autoRenewEnabled
                      ? colors.semantic.success
                      : colors.semantic.warning
                  }
                />
                <Text style={styles.detailLabel}>Auto-renew:</Text>
                <Text
                  style={[
                    styles.detailValue,
                    {
                      color: subscriptionInfo.autoRenewEnabled
                        ? colors.semantic.success
                        : colors.semantic.warning,
                    },
                  ]}
                >
                  {subscriptionInfo.autoRenewEnabled ? 'On' : 'Off'}
                </Text>
              </View>

              {/* Expiring soon warning */}
              {expirationInfo?.isExpiring && (
                <View style={styles.warningBanner}>
                  <Ionicons name="warning" size={18} color={colors.semantic.warning} />
                  <Text style={styles.warningText}>
                    Your subscription is expiring soon. Enable auto-renew to continue enjoying Pro
                    features.
                  </Text>
                </View>
              )}
            </View>
          )}

          {/* Actions */}
          <View style={styles.actionsContainer}>
            {isPro ? (
              <TouchableOpacity
                style={styles.manageButton}
                onPress={handleManageSubscription}
                accessibilityRole="button"
                accessibilityLabel="Manage Subscription"
              >
                <Ionicons name="settings-outline" size={20} color={colors.primary.main} />
                <Text style={styles.manageButtonText}>Manage Subscription</Text>
                <Ionicons name="open-outline" size={16} color={colors.text.tertiary} />
              </TouchableOpacity>
            ) : (
              <TouchableOpacity
                style={styles.upgradeButton}
                onPress={handleUpgrade}
                accessibilityRole="button"
                accessibilityLabel="Upgrade to Pro"
              >
                <LinearGradient
                  colors={[colors.primary.main, colors.primary.gradient.end]}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.upgradeButtonGradient}
                >
                  <Ionicons name="star" size={20} color={colors.text.primary} />
                  <Text style={styles.upgradeButtonText}>Upgrade to Pro</Text>
                </LinearGradient>
              </TouchableOpacity>
            )}
          </View>
        </View>

        {/* Free tier features (when not Pro) */}
        {!isPro && (
          <View style={styles.featuresCard}>
            <Text style={styles.featuresTitle}>Upgrade to unlock:</Text>
            <View style={styles.featuresList}>
              {PRO_FEATURES.map((feature, index) => (
                <View key={index} style={styles.featureItem}>
                  <Ionicons name="checkmark-circle" size={20} color={colors.primary.main} />
                  <Text style={styles.featureText}>{feature}</Text>
                </View>
              ))}
            </View>
          </View>
        )}

        {/* Restore purchases */}
        <TouchableOpacity
          style={styles.restoreButton}
          onPress={handleRestorePurchases}
          disabled={isRestoring}
          accessibilityRole="button"
          accessibilityLabel="Restore Purchases"
        >
          {isRestoring ? (
            <ActivityIndicator size="small" color={colors.text.tertiary} />
          ) : (
            <>
              <Ionicons name="refresh-outline" size={18} color={colors.text.tertiary} />
              <Text style={styles.restoreButtonText}>Restore Purchases</Text>
            </>
          )}
        </TouchableOpacity>

        {/* Help text */}
        <Text style={styles.helpText}>
          Subscriptions are managed through the App Store. To cancel or modify your subscription,
          tap "Manage Subscription" above or go to Settings {'>'} Apple ID {'>'} Subscriptions.
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

/** Pro features list for upgrade prompt */
const PRO_FEATURES = [
  'Unlimited meal history',
  'ML-powered nutrition insights',
  'Advanced analytics & trends',
  'Custom nutrition goals',
  'Export your data',
  'Priority support',
];

/**
 * Format date for display
 */
function formatDate(date: Date): string {
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
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
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  backButton: {
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerTitle: {
    ...typography.h3,
    color: colors.text.primary,
  },
  headerSpacer: {
    width: 44,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: spacing.lg,
  },
  planCard: {
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    ...shadows.md,
  },
  planHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: spacing.md,
  },
  planLabel: {
    ...typography.caption,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  planTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  planTitle: {
    ...typography.h2,
    color: colors.text.primary,
  },
  badge: {
    marginLeft: spacing.sm,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: spacing.xs,
  },
  statusText: {
    ...typography.caption,
    fontWeight: typography.fontWeight.medium,
  },
  detailsContainer: {
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  detailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  detailLabel: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
    marginLeft: spacing.sm,
    marginRight: spacing.xs,
  },
  detailValue: {
    ...typography.bodySmall,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
  },
  warningBanner: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: colors.special.warningLight,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginTop: spacing.sm,
  },
  warningText: {
    ...typography.bodySmall,
    color: colors.semantic.warning,
    marginLeft: spacing.sm,
    flex: 1,
  },
  actionsContainer: {
    marginTop: spacing.lg,
  },
  manageButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.special.highlight,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.primary.main,
  },
  manageButtonText: {
    ...typography.button,
    color: colors.primary.main,
    marginHorizontal: spacing.sm,
  },
  upgradeButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  upgradeButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
  },
  upgradeButtonText: {
    ...typography.button,
    color: colors.text.primary,
    marginLeft: spacing.sm,
  },
  featuresCard: {
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    ...shadows.sm,
  },
  featuresTitle: {
    ...typography.h3,
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  featuresList: {
    gap: spacing.sm,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  featureText: {
    ...typography.body,
    color: colors.text.secondary,
    marginLeft: spacing.sm,
  },
  restoreButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
    marginBottom: spacing.md,
  },
  restoreButtonText: {
    ...typography.body,
    color: colors.text.tertiary,
    marginLeft: spacing.sm,
    textDecorationLine: 'underline',
  },
  helpText: {
    ...typography.caption,
    color: colors.text.disabled,
    textAlign: 'center',
    lineHeight: 18,
  },
});
