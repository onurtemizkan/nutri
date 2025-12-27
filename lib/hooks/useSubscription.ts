/**
 * useSubscription Hook
 *
 * Provides subscription status and actions throughout the app.
 * Wrapper around SubscriptionContext for convenience.
 */

import { useCallback, useMemo } from 'react';
import { useRouter } from 'expo-router';

import { useSubscriptionContext } from '@/lib/context/SubscriptionContext';
import type { FeatureFlags } from '@/lib/services/purchases/types';

interface UseSubscriptionReturn {
  // Status
  isLoading: boolean;
  isPro: boolean;
  isTrial: boolean;
  expiresAt: Date | null;

  // Trial info
  trialDaysRemaining: number | null;
  isInTrial: boolean;

  // Expiration
  isExpiring: boolean;
  daysRemaining: number | null;

  // Auto-renewal
  autoRenewEnabled: boolean;

  // Feature flags
  hasFeature: (feature: keyof FeatureFlags) => boolean;
  featureFlags: FeatureFlags;

  // Actions
  syncEntitlements: () => Promise<void>;
  refreshStatus: () => Promise<void>;
  goToPaywall: () => void;
  goToSubscription: () => void;

  // Helpers
  requirePro: (onNotPro?: () => void) => boolean;
}

/**
 * Hook to access subscription status and actions
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { isPro, hasFeature, goToPaywall, requirePro } = useSubscription();
 *
 *   const handlePremiumAction = () => {
 *     if (!requirePro()) return;
 *     // Do premium action
 *   };
 *
 *   return (
 *     <View>
 *       {hasFeature('mlInsights') && <MLInsightsPanel />}
 *       <Button onPress={handlePremiumAction}>Premium Action</Button>
 *       {!isPro && <Button onPress={goToPaywall}>Upgrade to Pro</Button>}
 *     </View>
 *   );
 * }
 * ```
 */
export function useSubscription(): UseSubscriptionReturn {
  const context = useSubscriptionContext();
  const router = useRouter();

  /**
   * Navigate to paywall
   */
  const goToPaywall = useCallback(() => {
    router.push('/paywall');
  }, [router]);

  /**
   * Navigate to subscription management
   */
  const goToSubscription = useCallback(() => {
    router.push('/subscription');
  }, [router]);

  /**
   * Check if user has a specific feature
   */
  const hasFeature = useCallback(
    (feature: keyof FeatureFlags): boolean => {
      return context.checkFeature(feature);
    },
    [context]
  );

  /**
   * Guard for pro-only actions
   * Returns true if user is Pro, otherwise navigates to paywall
   *
   * @param onNotPro Optional callback instead of paywall navigation
   */
  const requirePro = useCallback(
    (onNotPro?: () => void): boolean => {
      if (context.isPro) {
        return true;
      }

      if (onNotPro) {
        onNotPro();
      } else {
        goToPaywall();
      }
      return false;
    },
    [context.isPro, goToPaywall]
  );

  return useMemo(
    () => ({
      // Status
      isLoading: context.isLoading,
      isPro: context.isPro,
      isTrial: context.isTrial,
      expiresAt: context.expiresAt,

      // Trial info
      trialDaysRemaining: context.trialDaysRemaining,
      isInTrial: context.isTrial,

      // Expiration
      isExpiring: context.isExpiring,
      daysRemaining: context.daysRemaining,

      // Auto-renewal
      autoRenewEnabled: context.autoRenewEnabled,

      // Feature flags
      hasFeature,
      featureFlags: context.featureFlags,

      // Actions
      syncEntitlements: context.syncSubscription,
      refreshStatus: context.refreshStatus,
      goToPaywall,
      goToSubscription,

      // Helpers
      requirePro,
    }),
    [context, hasFeature, goToPaywall, goToSubscription, requirePro]
  );
}

export default useSubscription;
