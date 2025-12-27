/**
 * Eligibility Service
 *
 * Handles checking user eligibility for:
 * - Free trials (introductory offers)
 * - Promotional offers (for lapsed subscribers)
 * - Win-back offers (for re-subscription discounts)
 *
 * Uses StoreKit 2 eligibility APIs when available.
 */

import { Platform } from 'react-native';
import {
  fetchProducts,
  isEligibleForIntroOfferIOS,
  subscriptionStatusIOS,
  type ProductSubscriptionIOS,
} from 'react-native-iap';

import { getSubscriptionInfo } from './index';
import { SUBSCRIPTION_PRODUCT_IDS, FREE_TRIAL_DAYS } from './products';
import type { SubscriptionProduct, LocalSubscriptionInfo } from './types';

/** Eligibility result for a product */
export interface ProductEligibility {
  productId: string;
  isEligibleForFreeTrial: boolean;
  isEligibleForIntroOffer: boolean;
  isEligibleForPromoOffer: boolean;
  introOfferDetails?: IntroOfferDetails;
  promoOfferDetails?: PromoOfferDetails;
}

/** Intro offer details */
export interface IntroOfferDetails {
  type: 'free_trial' | 'pay_up_front' | 'pay_as_you_go';
  price: string;
  priceValue: number;
  periodUnit: string;
  periodCount: number;
  totalCycles: number;
}

/** Promotional offer details */
export interface PromoOfferDetails {
  offerId: string;
  price: string;
  priceValue: number;
  periodUnit: string;
  periodCount: number;
  paymentMode: 'pay_up_front' | 'pay_as_you_go' | 'free_trial';
}

/** Overall eligibility status for all products */
export interface EligibilityStatus {
  hasAnyTrial: boolean;
  hasAnyPromo: boolean;
  products: ProductEligibility[];
  lastCheckedAt: string;
}

/** Cached eligibility status */
let cachedEligibility: EligibilityStatus | null = null;
let eligibilityCacheTime = 0;
const ELIGIBILITY_CACHE_TTL = 5 * 60 * 1000; // 5 minutes

/**
 * Check eligibility for all subscription products
 * Uses StoreKit 2 APIs for accurate eligibility determination
 */
export async function checkAllProductEligibility(): Promise<EligibilityStatus> {
  // Return cached result if still valid
  if (cachedEligibility && Date.now() - eligibilityCacheTime < ELIGIBILITY_CACHE_TTL) {
    return cachedEligibility;
  }

  const eligibilities: ProductEligibility[] = [];

  for (const productId of SUBSCRIPTION_PRODUCT_IDS) {
    const eligibility = await checkProductEligibility(productId);
    eligibilities.push(eligibility);
  }

  cachedEligibility = {
    hasAnyTrial: eligibilities.some((e) => e.isEligibleForFreeTrial),
    hasAnyPromo: eligibilities.some((e) => e.isEligibleForPromoOffer),
    products: eligibilities,
    lastCheckedAt: new Date().toISOString(),
  };
  eligibilityCacheTime = Date.now();

  return cachedEligibility;
}

/**
 * Check eligibility for a specific product
 */
export async function checkProductEligibility(productId: string): Promise<ProductEligibility> {
  const result: ProductEligibility = {
    productId,
    isEligibleForFreeTrial: false,
    isEligibleForIntroOffer: false,
    isEligibleForPromoOffer: false,
  };

  try {
    if (Platform.OS === 'ios') {
      // Use StoreKit 2 eligibility check for iOS
      result.isEligibleForIntroOffer = await checkIOSIntroOfferEligibility(productId);
      result.isEligibleForFreeTrial = result.isEligibleForIntroOffer;

      // Check promotional offer eligibility
      result.isEligibleForPromoOffer = await checkIOSPromoOfferEligibility(productId);

      // Get offer details if eligible
      if (result.isEligibleForIntroOffer) {
        result.introOfferDetails = await getIntroOfferDetails(productId);
      }
    } else if (Platform.OS === 'android') {
      // Android handles offers differently - check subscription info
      result.isEligibleForFreeTrial = await checkAndroidTrialEligibility();
      result.isEligibleForIntroOffer = result.isEligibleForFreeTrial;
      // Android promo offers are handled through Google Play Console
      result.isEligibleForPromoOffer = await checkAndroidPromoEligibility();
    }
  } catch (error) {
    console.error('[Eligibility] Error checking eligibility:', error);
    // Default to true on error to not block purchases
    result.isEligibleForFreeTrial = await fallbackEligibilityCheck();
    result.isEligibleForIntroOffer = result.isEligibleForFreeTrial;
  }

  return result;
}

/**
 * Check iOS introductory offer eligibility using StoreKit 2
 * This is the most accurate method for iOS
 */
async function checkIOSIntroOfferEligibility(productId: string): Promise<boolean> {
  try {
    // StoreKit 2 provides a direct eligibility check
    const result = await isEligibleForIntroOfferIOS(productId);
    const isEligible = result ?? false;
    console.log('[Eligibility] iOS intro offer eligibility for', productId, ':', isEligible);
    return isEligible;
  } catch (error) {
    console.warn('[Eligibility] iOS intro offer check failed:', error);
    // Fall back to local check
    return fallbackEligibilityCheck();
  }
}

/**
 * Check iOS promotional offer eligibility
 * Promotional offers require:
 * 1. User has previously subscribed (originalTransactionId exists)
 * 2. Subscription is not currently active
 */
async function checkIOSPromoOfferEligibility(productId: string): Promise<boolean> {
  try {
    // First check local subscription status
    const subscriptionInfo = await getSubscriptionInfo();

    // Must have previous subscription history
    if (!subscriptionInfo.originalTransactionId) {
      return false;
    }

    // Must not be currently active
    if (subscriptionInfo.status === 'active') {
      return false;
    }

    // Check with StoreKit if offer is available
    // Note: Promotional offers require signature from your server
    // For now, we return eligibility based on subscription status
    const isLapsedSubscriber =
      subscriptionInfo.status === 'expired' ||
      subscriptionInfo.status === 'refunded' ||
      subscriptionInfo.tier === 'free';

    console.log('[Eligibility] iOS promo offer eligibility:', isLapsedSubscriber);
    return isLapsedSubscriber;
  } catch (error) {
    console.warn('[Eligibility] iOS promo offer check failed:', error);
    return false;
  }
}

/**
 * Check subscription offer eligibility (for custom promotional offers)
 * Uses subscription status to determine if user is eligible for promotional offers
 */
export async function checkSubscriptionOfferEligibility(
  productId: string,
  _offerId: string
): Promise<boolean> {
  if (Platform.OS !== 'ios') {
    return false;
  }

  try {
    // Check subscription status to determine promo offer eligibility
    // Users must have previously subscribed but currently lapsed
    const status = await subscriptionStatusIOS(productId);
    const isLapsed = !status || (status as { state?: string })?.state !== 'active';

    // Check if user has subscription history
    const subscriptionInfo = await getSubscriptionInfo();
    const hasHistory = subscriptionInfo.originalTransactionId !== null;

    const isEligible = isLapsed && hasHistory;
    console.log('[Eligibility] Subscription offer eligibility:', isEligible);
    return isEligible;
  } catch (error) {
    console.warn('[Eligibility] Subscription offer check failed:', error);
    return false;
  }
}

/**
 * Check Android trial eligibility
 * Android uses the subscription basePlanId to determine trials
 */
async function checkAndroidTrialEligibility(): Promise<boolean> {
  try {
    const subscriptionInfo = await getSubscriptionInfo();

    // If user has never had a subscription, they're eligible for trial
    if (!subscriptionInfo.originalTransactionId) {
      return true;
    }

    // If they've used trial before, check subscription status
    // Note: Android tracks trial usage through purchase token
    return !subscriptionInfo.isTrialPeriod;
  } catch (error) {
    console.warn('[Eligibility] Android trial check failed:', error);
    return true;
  }
}

/**
 * Check Android promotional offer eligibility
 */
async function checkAndroidPromoEligibility(): Promise<boolean> {
  try {
    const subscriptionInfo = await getSubscriptionInfo();

    // Must have previous subscription
    if (!subscriptionInfo.originalTransactionId) {
      return false;
    }

    // Must not be active
    return subscriptionInfo.status !== 'active' && subscriptionInfo.tier === 'free';
  } catch (error) {
    console.warn('[Eligibility] Android promo check failed:', error);
    return false;
  }
}

/**
 * Fallback eligibility check when StoreKit API fails
 * Uses local subscription info as best guess
 */
async function fallbackEligibilityCheck(): Promise<boolean> {
  try {
    const subscriptionInfo = await getSubscriptionInfo();

    // If no previous transaction, user is likely eligible
    if (!subscriptionInfo.originalTransactionId) {
      console.log('[Eligibility] Fallback: No previous transaction, assuming eligible');
      return true;
    }

    // If they've had a trial before, not eligible for another
    if (subscriptionInfo.isTrialPeriod) {
      return false;
    }

    // Conservative approach: if we can't determine, assume not eligible
    return false;
  } catch (error) {
    // If all else fails, assume eligible to not block purchase
    return true;
  }
}

/**
 * Get detailed intro offer information for a product
 */
async function getIntroOfferDetails(productId: string): Promise<IntroOfferDetails | undefined> {
  try {
    const products = await fetchProducts({
      skus: [productId],
      type: 'subs',
    });

    if (!products || products.length === 0) {
      return undefined;
    }

    const product = products[0] as ProductSubscriptionIOS;

    if (
      !product.introductoryPricePaymentModeIOS ||
      product.introductoryPricePaymentModeIOS === 'empty'
    ) {
      return undefined;
    }

    return {
      type: mapPaymentMode(product.introductoryPricePaymentModeIOS),
      price: product.introductoryPriceIOS ?? '0',
      priceValue: parseFloat(product.introductoryPriceAsAmountIOS ?? '0'),
      periodUnit: product.introductoryPriceSubscriptionPeriodIOS ?? 'day',
      periodCount: parseInt(product.introductoryPriceNumberOfPeriodsIOS ?? '1', 10),
      totalCycles: parseInt(product.introductoryPriceNumberOfPeriodsIOS ?? '1', 10),
    };
  } catch (error) {
    console.warn('[Eligibility] Error getting intro offer details:', error);
    return undefined;
  }
}

/**
 * Map payment mode string to typed value
 */
function mapPaymentMode(mode: string): 'free_trial' | 'pay_up_front' | 'pay_as_you_go' {
  const normalizedMode = mode.toLowerCase().replace(/[^a-z]/g, '');

  switch (normalizedMode) {
    case 'freetrial':
      return 'free_trial';
    case 'payupfront':
      return 'pay_up_front';
    case 'payasyougo':
      return 'pay_as_you_go';
    default:
      return 'free_trial';
  }
}

/**
 * Check if user is in trial period
 */
export async function isInTrialPeriod(): Promise<boolean> {
  const subscriptionInfo = await getSubscriptionInfo();
  return subscriptionInfo.isTrialPeriod && subscriptionInfo.status === 'active';
}

/**
 * Check if user is in intro offer period (but not free trial)
 */
export async function isInIntroOfferPeriod(): Promise<boolean> {
  const subscriptionInfo = await getSubscriptionInfo();
  return (
    subscriptionInfo.isIntroOfferPeriod &&
    !subscriptionInfo.isTrialPeriod &&
    subscriptionInfo.status === 'active'
  );
}

/**
 * Get remaining trial days
 */
export async function getRemainingTrialDays(): Promise<number | null> {
  const subscriptionInfo = await getSubscriptionInfo();

  if (!subscriptionInfo.isTrialPeriod || !subscriptionInfo.expiresAt) {
    return null;
  }

  const expiresAt = new Date(subscriptionInfo.expiresAt);
  const now = new Date();
  const diffMs = expiresAt.getTime() - now.getTime();
  const daysRemaining = Math.ceil(diffMs / (1000 * 60 * 60 * 24));

  return Math.max(0, daysRemaining);
}

/**
 * Get trial info for display
 */
export async function getTrialInfo(): Promise<{
  isInTrial: boolean;
  trialDuration: number;
  daysRemaining: number | null;
  expiresAt: Date | null;
}> {
  const subscriptionInfo = await getSubscriptionInfo();
  const remainingDays = await getRemainingTrialDays();

  return {
    isInTrial: subscriptionInfo.isTrialPeriod && subscriptionInfo.status === 'active',
    trialDuration: FREE_TRIAL_DAYS,
    daysRemaining: remainingDays,
    expiresAt: subscriptionInfo.expiresAt ? new Date(subscriptionInfo.expiresAt) : null,
  };
}

/**
 * Clear eligibility cache (call when subscription status changes)
 */
export function clearEligibilityCache(): void {
  cachedEligibility = null;
  eligibilityCacheTime = 0;
}

/**
 * Update eligibility status after purchase/restore
 */
export async function updateEligibilityAfterPurchase(): Promise<void> {
  clearEligibilityCache();
  // Refresh eligibility
  await checkAllProductEligibility();
}
