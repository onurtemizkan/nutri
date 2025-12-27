/**
 * Product IDs and subscription configuration constants
 *
 * These IDs must match exactly what's configured in App Store Connect.
 * Do not modify these without also updating App Store Connect.
 */

import type { SubscriptionPeriod, FeatureAccess, FeatureFlags } from './types';

/** Bundle identifier prefix for products */
export const BUNDLE_ID_PREFIX = 'com.anonymous.nutri';

/** Subscription product IDs */
export const PRODUCT_IDS = {
  /** Monthly Pro subscription */
  PRO_MONTHLY: `${BUNDLE_ID_PREFIX}.pro.monthly`,
  /** Yearly Pro subscription (with discount) */
  PRO_YEARLY: `${BUNDLE_ID_PREFIX}.pro.yearly`,
} as const;

/** Array of all subscription product IDs */
export const SUBSCRIPTION_PRODUCT_IDS = [PRODUCT_IDS.PRO_MONTHLY, PRODUCT_IDS.PRO_YEARLY] as const;

/** Map product ID to period */
export const PRODUCT_PERIOD_MAP: Record<string, SubscriptionPeriod> = {
  [PRODUCT_IDS.PRO_MONTHLY]: 'monthly',
  [PRODUCT_IDS.PRO_YEARLY]: 'yearly',
};

/** Subscription group ID in App Store Connect */
export const SUBSCRIPTION_GROUP_ID = 'nutri_pro_subscriptions';

/** Free trial duration in days */
export const FREE_TRIAL_DAYS = 7;

/** Yearly subscription savings percentage */
export const YEARLY_SAVINGS_PERCENT = 17;

/** Grace period duration in days (Apple default is 16 days) */
export const GRACE_PERIOD_DAYS = 16;

/** Billing retry period in days (Apple retries for 60 days) */
export const BILLING_RETRY_DAYS = 60;

/** Cache duration for entitlements in milliseconds (24 hours) */
export const ENTITLEMENT_CACHE_TTL_MS = 24 * 60 * 60 * 1000;

/** Maximum cache age for offline use in milliseconds (7 days) */
export const OFFLINE_CACHE_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;

/** Feature flags for free tier */
export const FREE_TIER_FEATURES: FeatureFlags = {
  unlimitedHistory: false,
  mlInsights: false,
  advancedAnalytics: false,
  prioritySupport: false,
  exportData: false,
  customGoals: false,
};

/** Feature flags for Pro tier */
export const PRO_TIER_FEATURES: FeatureFlags = {
  unlimitedHistory: true,
  mlInsights: true,
  advancedAnalytics: true,
  prioritySupport: true,
  exportData: true,
  customGoals: true,
};

/** Which tier is required for each feature */
export const FEATURE_ACCESS: FeatureAccess = {
  unlimitedHistory: 'pro',
  mlInsights: 'pro',
  advancedAnalytics: 'pro',
  prioritySupport: 'pro',
  exportData: 'pro',
  customGoals: 'pro',
};

/** SecureStore key for cached subscription info */
export const SUBSCRIPTION_CACHE_KEY = 'nutri_subscription_info';

/** App Store subscription management URL */
export const APP_STORE_SUBSCRIPTION_URL = 'https://apps.apple.com/account/subscriptions';

/** Human-readable product names */
export const PRODUCT_DISPLAY_NAMES: Record<string, string> = {
  [PRODUCT_IDS.PRO_MONTHLY]: 'Nutri Pro (Monthly)',
  [PRODUCT_IDS.PRO_YEARLY]: 'Nutri Pro (Yearly)',
};

/** Product descriptions */
export const PRODUCT_DESCRIPTIONS: Record<string, string> = {
  [PRODUCT_IDS.PRO_MONTHLY]: 'Full access to all Pro features, billed monthly.',
  [PRODUCT_IDS.PRO_YEARLY]: 'Full access to all Pro features, billed annually. Save 17%!',
};
