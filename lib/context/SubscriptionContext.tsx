/**
 * Subscription Context
 *
 * Provides subscription status and entitlements throughout the app.
 * Handles sync on app launch and after purchases.
 */

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  ReactNode,
} from 'react';
import { AppState, AppStateStatus } from 'react-native';

import {
  checkEntitlement,
  syncEntitlements,
  getFeatureFlags,
  checkIsInTrialPeriod,
  getTrialDaysRemaining,
  getExpirationInfo,
  clearEntitlementCache,
  type EntitlementResult,
  type FeatureFlags,
} from '@/lib/services/purchases/entitlements';
import { getSubscriptionInfo } from '@/lib/services/purchases';
import type { SubscriptionTier } from '@/lib/services/purchases/types';

interface SubscriptionState {
  // Core status
  isLoading: boolean;
  isPro: boolean;
  isTrial: boolean;
  tier: SubscriptionTier;
  expiresAt: Date | null;

  // Trial info
  trialDaysRemaining: number | null;

  // Expiration warnings
  isExpiring: boolean;
  daysRemaining: number | null;

  // Feature flags
  featureFlags: FeatureFlags;

  // Auto-renewal status
  autoRenewEnabled: boolean;

  // Source of data
  source: EntitlementResult['source'] | 'initial';

  // Last sync time
  lastSyncedAt: Date | null;
}

interface SubscriptionContextValue extends SubscriptionState {
  // Actions
  syncSubscription: () => Promise<void>;
  checkFeature: (feature: keyof FeatureFlags) => boolean;
  clearSubscription: () => Promise<void>;
  refreshStatus: () => Promise<void>;
}

const defaultState: SubscriptionState = {
  isLoading: true,
  isPro: false,
  isTrial: false,
  tier: 'free',
  expiresAt: null,
  trialDaysRemaining: null,
  isExpiring: false,
  daysRemaining: null,
  featureFlags: {
    unlimitedHistory: false,
    mlInsights: false,
    advancedAnalytics: false,
    customGoals: false,
    exportData: false,
    prioritySupport: false,
  },
  autoRenewEnabled: true,
  source: 'initial',
  lastSyncedAt: null,
};

const SubscriptionContext = createContext<SubscriptionContextValue | null>(null);

interface SubscriptionProviderProps {
  children: ReactNode;
}

export function SubscriptionProvider({ children }: SubscriptionProviderProps) {
  const [state, setState] = useState<SubscriptionState>(defaultState);

  /**
   * Load current subscription status from cache/server
   */
  const loadSubscriptionStatus = useCallback(async () => {
    try {
      setState((prev) => ({ ...prev, isLoading: true }));

      // Get entitlement from cache or server
      const entitlement = await checkEntitlement();

      // Get additional info
      const [isTrial, trialDays, expiration, flags, subInfo] = await Promise.all([
        checkIsInTrialPeriod(),
        getTrialDaysRemaining(),
        getExpirationInfo(),
        getFeatureFlags(),
        getSubscriptionInfo(),
      ]);

      setState({
        isLoading: false,
        isPro: entitlement.tier === 'pro',
        isTrial,
        tier: entitlement.tier,
        expiresAt: entitlement.expiresAt ? new Date(entitlement.expiresAt) : null,
        trialDaysRemaining: trialDays,
        isExpiring: expiration.isExpiring,
        daysRemaining: expiration.daysRemaining,
        featureFlags: flags,
        autoRenewEnabled: subInfo.autoRenewEnabled,
        source: entitlement.source,
        lastSyncedAt: subInfo.lastSyncedAt ? new Date(subInfo.lastSyncedAt) : null,
      });
    } catch (error) {
      console.error('[SubscriptionContext] Error loading status:', error);
      setState((prev) => ({
        ...prev,
        isLoading: false,
        isPro: false,
        tier: 'free',
        source: 'cache',
      }));
    }
  }, []);

  /**
   * Sync subscription with server
   */
  const syncSubscription = useCallback(async () => {
    try {
      console.log('[SubscriptionContext] Syncing subscription...');
      setState((prev) => ({ ...prev, isLoading: true }));

      await syncEntitlements();
      await loadSubscriptionStatus();

      console.log('[SubscriptionContext] Sync complete');
    } catch (error) {
      console.error('[SubscriptionContext] Sync failed:', error);
      setState((prev) => ({ ...prev, isLoading: false }));
    }
  }, [loadSubscriptionStatus]);

  /**
   * Refresh status (alias for loadSubscriptionStatus)
   */
  const refreshStatus = useCallback(async () => {
    await loadSubscriptionStatus();
  }, [loadSubscriptionStatus]);

  /**
   * Check if a feature is available
   */
  const checkFeature = useCallback(
    (feature: keyof FeatureFlags): boolean => {
      return state.featureFlags[feature];
    },
    [state.featureFlags]
  );

  /**
   * Clear subscription data (for sign out)
   */
  const clearSubscription = useCallback(async () => {
    await clearEntitlementCache();
    setState(defaultState);
  }, []);

  // Load status on mount
  useEffect(() => {
    loadSubscriptionStatus();
  }, [loadSubscriptionStatus]);

  // Sync when app comes to foreground
  useEffect(() => {
    const handleAppStateChange = (nextState: AppStateStatus) => {
      if (nextState === 'active') {
        // Only sync if last sync was more than 1 hour ago
        const oneHourAgo = Date.now() - 60 * 60 * 1000;
        if (!state.lastSyncedAt || state.lastSyncedAt.getTime() < oneHourAgo) {
          syncSubscription();
        }
      }
    };

    const subscription = AppState.addEventListener('change', handleAppStateChange);
    return () => subscription.remove();
  }, [state.lastSyncedAt, syncSubscription]);

  const value = useMemo<SubscriptionContextValue>(
    () => ({
      ...state,
      syncSubscription,
      checkFeature,
      clearSubscription,
      refreshStatus,
    }),
    [state, syncSubscription, checkFeature, clearSubscription, refreshStatus]
  );

  return <SubscriptionContext.Provider value={value}>{children}</SubscriptionContext.Provider>;
}

/**
 * Hook to access subscription context
 */
export function useSubscriptionContext(): SubscriptionContextValue {
  const context = useContext(SubscriptionContext);
  if (!context) {
    throw new Error('useSubscriptionContext must be used within a SubscriptionProvider');
  }
  return context;
}

export default SubscriptionContext;
