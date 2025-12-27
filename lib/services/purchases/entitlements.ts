/**
 * Entitlement Service
 *
 * Handles checking user entitlements based on subscription status.
 * Provides caching, offline support, and feature flag resolution.
 */

import * as SecureStore from 'expo-secure-store';

import { getSubscriptionInfo, saveSubscriptionInfo } from './index';
import {
  SUBSCRIPTION_CACHE_KEY,
  ENTITLEMENT_CACHE_TTL_MS,
  OFFLINE_CACHE_MAX_AGE_MS,
  FREE_TIER_FEATURES,
  PRO_TIER_FEATURES,
  FEATURE_ACCESS,
} from './products';
import {
  checkAllProductEligibility,
  isInTrialPeriod,
  isInIntroOfferPeriod,
  getTrialInfo,
  getRemainingTrialDays,
  type EligibilityStatus,
  type ProductEligibility,
} from './eligibility';
import type {
  SubscriptionTier,
  FeatureFlags,
  EntitlementResult,
  LocalSubscriptionInfo,
} from './types';

/**
 * Check if user has access to a specific feature
 */
export async function checkFeatureAccess(feature: keyof FeatureFlags): Promise<boolean> {
  const tier = await getUserTier();
  const requiredTier = FEATURE_ACCESS[feature];

  if (requiredTier === 'free') {
    return true;
  }

  return tier === 'pro';
}

/**
 * Get current user's subscription tier
 */
export async function getUserTier(): Promise<SubscriptionTier> {
  const entitlement = await checkEntitlement();
  return entitlement.tier;
}

/**
 * Check user's current entitlement status
 * Uses cached data when available, with offline fallback
 */
export async function checkEntitlement(): Promise<EntitlementResult> {
  try {
    const subscriptionInfo = await getSubscriptionInfo();

    // Check if cache is valid
    const lastSynced = new Date(subscriptionInfo.lastSyncedAt).getTime();
    const now = Date.now();
    const age = now - lastSynced;

    // If cache is fresh enough, use it
    if (age < ENTITLEMENT_CACHE_TTL_MS) {
      return {
        hasAccess: subscriptionInfo.tier === 'pro' && subscriptionInfo.status === 'active',
        tier: subscriptionInfo.tier,
        expiresAt: subscriptionInfo.expiresAt,
        source: 'cache',
      };
    }

    // Cache is stale, try to refresh from server
    // TODO: Implement server sync (subtask 38.7)
    // For now, if within offline grace period, use cache
    if (age < OFFLINE_CACHE_MAX_AGE_MS) {
      console.log('[Entitlements] Cache stale but within offline period');
      return {
        hasAccess: subscriptionInfo.tier === 'pro' && subscriptionInfo.status === 'active',
        tier: subscriptionInfo.tier,
        expiresAt: subscriptionInfo.expiresAt,
        source: 'offline',
      };
    }

    // Cache too old, treat as free tier
    console.log('[Entitlements] Cache too old, defaulting to free');
    return {
      hasAccess: false,
      tier: 'free',
      expiresAt: null,
      source: 'cache',
    };
  } catch (error) {
    console.error('[Entitlements] Error checking entitlement:', error);
    return {
      hasAccess: false,
      tier: 'free',
      expiresAt: null,
      source: 'cache',
    };
  }
}

/**
 * Get all feature flags for current user
 */
export async function getFeatureFlags(): Promise<FeatureFlags> {
  const tier = await getUserTier();
  return tier === 'pro' ? PRO_TIER_FEATURES : FREE_TIER_FEATURES;
}

/**
 * Sync entitlements with backend
 * Should be called on app launch and after purchases
 */
export async function syncEntitlements(): Promise<EntitlementResult> {
  try {
    console.log('[Entitlements] Syncing with server...');

    // Import subscription API dynamically to avoid circular dependencies
    const subscriptionApi = await import('@/lib/api/subscription');

    // Call backend API to get subscription status
    const serverStatus = await subscriptionApi.getSubscriptionStatus();

    // Map server status to local format
    const tierMap: Record<string, SubscriptionTier> = {
      FREE: 'free',
      PRO_TRIAL: 'pro',
      PRO: 'pro',
    };

    const statusMap: Record<string, LocalSubscriptionInfo['status']> = {
      ACTIVE: 'active',
      EXPIRED: 'expired',
      IN_GRACE_PERIOD: 'in_grace_period',
      IN_BILLING_RETRY: 'in_billing_retry',
      REVOKED: 'revoked',
      REFUNDED: 'refunded',
    };

    const updated: LocalSubscriptionInfo = {
      tier: tierMap[serverStatus.tier] || 'free',
      status: serverStatus.status ? statusMap[serverStatus.status] || null : null,
      expiresAt: serverStatus.expiresAt,
      productId: serverStatus.productId,
      originalTransactionId: null, // Not exposed by backend
      autoRenewEnabled: serverStatus.autoRenewEnabled,
      lastSyncedAt: new Date().toISOString(),
    };

    await saveSubscriptionInfo(updated);

    console.log('[Entitlements] Sync complete:', { tier: updated.tier, status: updated.status });

    return {
      hasAccess: serverStatus.isActive,
      tier: updated.tier,
      expiresAt: updated.expiresAt,
      source: 'server',
    };
  } catch (error) {
    console.error('[Entitlements] Sync failed:', error);
    // Fallback to cached data
    return checkEntitlement();
  }
}

/**
 * Check if user is eligible for free trial
 * Uses StoreKit 2 eligibility APIs for accurate determination
 */
export async function isEligibleForFreeTrial(): Promise<boolean> {
  try {
    const eligibility = await checkAllProductEligibility();
    return eligibility.hasAnyTrial;
  } catch (error) {
    console.warn('[Entitlements] Trial eligibility check failed:', error);
    // Fallback to local check
    const info = await getSubscriptionInfo();
    return info.originalTransactionId === null;
  }
}

/**
 * Check if user is eligible for promotional offer
 * Uses StoreKit 2 eligibility APIs for lapsed subscriber detection
 */
export async function isEligibleForPromoOffer(): Promise<boolean> {
  try {
    const eligibility = await checkAllProductEligibility();
    return eligibility.hasAnyPromo;
  } catch (error) {
    console.warn('[Entitlements] Promo eligibility check failed:', error);
    // Fallback to local check
    const info = await getSubscriptionInfo();
    return info.originalTransactionId !== null && info.status === 'expired' && info.tier === 'free';
  }
}

/**
 * Get detailed eligibility status for all products
 */
export async function getProductEligibility(): Promise<EligibilityStatus> {
  return checkAllProductEligibility();
}

/**
 * Check if user is currently in a trial period
 */
export async function checkIsInTrialPeriod(): Promise<boolean> {
  return isInTrialPeriod();
}

/**
 * Check if user is in an intro offer period (not trial)
 */
export async function checkIsInIntroOfferPeriod(): Promise<boolean> {
  return isInIntroOfferPeriod();
}

/**
 * Get trial information for display
 */
export async function getTrialDisplayInfo(): Promise<{
  isInTrial: boolean;
  trialDuration: number;
  daysRemaining: number | null;
  expiresAt: Date | null;
}> {
  return getTrialInfo();
}

/**
 * Get remaining trial days
 */
export async function getTrialDaysRemaining(): Promise<number | null> {
  return getRemainingTrialDays();
}

/**
 * Get subscription expiration info for UI
 */
export async function getExpirationInfo(): Promise<{
  isExpiring: boolean;
  expiresAt: Date | null;
  daysRemaining: number | null;
}> {
  const info = await getSubscriptionInfo();

  if (!info.expiresAt) {
    return {
      isExpiring: false,
      expiresAt: null,
      daysRemaining: null,
    };
  }

  const expiresAt = new Date(info.expiresAt);
  const now = new Date();
  const diffMs = expiresAt.getTime() - now.getTime();
  const daysRemaining = Math.ceil(diffMs / (1000 * 60 * 60 * 24));

  return {
    isExpiring: daysRemaining <= 7 && !info.autoRenewEnabled,
    expiresAt,
    daysRemaining: Math.max(0, daysRemaining),
  };
}

/**
 * Clear entitlement cache (for sign out)
 */
export async function clearEntitlementCache(): Promise<void> {
  try {
    await SecureStore.deleteItemAsync(SUBSCRIPTION_CACHE_KEY);
    console.log('[Entitlements] Cache cleared');
  } catch (error) {
    console.error('[Entitlements] Error clearing cache:', error);
  }
}

export { type EntitlementResult, type FeatureFlags } from './types';

export { type EligibilityStatus, type ProductEligibility } from './eligibility';
