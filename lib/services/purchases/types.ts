/**
 * Types for In-App Purchase system
 * Uses react-native-iap v14+ with StoreKit 2 support
 */

import type { Purchase, PurchaseError as IAPError } from 'react-native-iap';

// Re-export types from react-native-iap for convenience
export type { Purchase, PurchaseError as IAPError } from 'react-native-iap';

/** Subscription tier levels */
export type SubscriptionTier = 'free' | 'pro';

/** Subscription period options */
export type SubscriptionPeriod = 'monthly' | 'yearly';

/** Subscription status from App Store */
export type SubscriptionStatus =
  | 'active'
  | 'expired'
  | 'in_grace_period'
  | 'in_billing_retry'
  | 'revoked'
  | 'refunded';

/** Purchase state during a transaction */
export type PurchaseState =
  | 'idle'
  | 'loading_products'
  | 'purchasing'
  | 'restoring'
  | 'verifying'
  | 'success'
  | 'error';

/** Local subscription info stored on device */
export interface LocalSubscriptionInfo {
  tier: SubscriptionTier;
  productId: string | null;
  status: SubscriptionStatus | null;
  expiresAt: string | null;
  isTrialPeriod: boolean;
  isIntroOfferPeriod: boolean;
  autoRenewEnabled: boolean;
  lastSyncedAt: string;
  originalTransactionId: string | null;
}

/** Product information with display details */
export interface SubscriptionProduct {
  productId: string;
  title: string;
  description: string;
  price: string;
  priceValue: number;
  currency: string;
  period: SubscriptionPeriod;
  /** Formatted price with period (e.g., "$9.99/month") */
  displayPrice: string;
  /** Intro offer if available */
  introOffer?: {
    type: 'free_trial' | 'pay_up_front' | 'pay_as_you_go';
    price: string;
    priceValue: number;
    period: string;
    cycles: number;
  };
  /** Whether user is eligible for intro offer */
  isEligibleForIntroOffer: boolean;
}

/** Purchase result from StoreKit */
export interface PurchaseResult {
  success: boolean;
  transactionId?: string;
  originalTransactionId?: string;
  productId?: string;
  error?: PurchaseError;
}

/** Purchase error details */
export interface PurchaseError {
  code: PurchaseErrorCode;
  message: string;
  userMessage: string;
}

/** Purchase error codes */
export type PurchaseErrorCode =
  | 'E_USER_CANCELLED'
  | 'E_NETWORK_ERROR'
  | 'E_PRODUCT_NOT_FOUND'
  | 'E_PURCHASE_FAILED'
  | 'E_VERIFICATION_FAILED'
  | 'E_ALREADY_OWNED'
  | 'E_DEFERRED'
  | 'E_UNKNOWN';

/** Transaction listener callback */
export type TransactionListener = (purchase: Purchase) => Promise<void>;

/** Purchase service configuration */
export interface PurchaseServiceConfig {
  /** Whether to run in sandbox mode */
  sandbox: boolean;
  /** Backend API URL for verification */
  apiUrl: string;
}

/** Backend verification response */
export interface VerificationResponse {
  valid: boolean;
  subscriptionInfo: {
    status: SubscriptionStatus;
    productId: string;
    expiresAt: string;
    isTrialPeriod: boolean;
    isIntroOfferPeriod: boolean;
    autoRenewEnabled: boolean;
    originalTransactionId: string;
  };
}

/** Entitlement check result */
export interface EntitlementResult {
  hasAccess: boolean;
  tier: SubscriptionTier;
  expiresAt: string | null;
  source: 'cache' | 'server' | 'offline';
}

/** Feature flags based on subscription tier */
export interface FeatureFlags {
  unlimitedHistory: boolean;
  mlInsights: boolean;
  advancedAnalytics: boolean;
  prioritySupport: boolean;
  exportData: boolean;
  customGoals: boolean;
}

/** Map of features to required tier */
export type FeatureAccess = {
  [K in keyof FeatureFlags]: SubscriptionTier;
};
