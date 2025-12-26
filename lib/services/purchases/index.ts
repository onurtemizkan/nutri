/**
 * Purchase Service
 *
 * Main service for handling In-App Purchases using react-native-iap v14+ with StoreKit 2.
 * Implements transaction listener, purchase flow, and proper transaction finishing.
 */

import { Platform } from 'react-native';
import {
  initConnection,
  endConnection,
  fetchProducts,
  requestPurchase,
  purchaseUpdatedListener,
  purchaseErrorListener,
  finishTransaction,
  getAvailablePurchases,
  restorePurchases as iapRestorePurchases,
  type Purchase,
  type PurchaseError as IAPPurchaseError,
  type ProductSubscription,
  type ProductSubscriptionIOS,
  type EventSubscription,
} from 'react-native-iap';
import * as SecureStore from 'expo-secure-store';

import {
  SUBSCRIPTION_PRODUCT_IDS,
  PRODUCT_PERIOD_MAP,
  SUBSCRIPTION_CACHE_KEY,
  ENTITLEMENT_CACHE_TTL_MS,
  PRODUCT_DISPLAY_NAMES,
  PRODUCT_DESCRIPTIONS,
} from './products';
import {
  checkProductEligibility,
  clearEligibilityCache,
  updateEligibilityAfterPurchase,
} from './eligibility';
import type {
  SubscriptionProduct,
  PurchaseResult,
  PurchaseError,
  PurchaseErrorCode,
  PurchaseState,
  LocalSubscriptionInfo,
  SubscriptionTier,
} from './types';

/** Default subscription info for free tier */
const DEFAULT_SUBSCRIPTION_INFO: LocalSubscriptionInfo = {
  tier: 'free',
  productId: null,
  status: null,
  expiresAt: null,
  isTrialPeriod: false,
  isIntroOfferPeriod: false,
  autoRenewEnabled: false,
  lastSyncedAt: new Date().toISOString(),
  originalTransactionId: null,
};

/** Purchase service singleton state */
let isInitialized = false;
let purchaseUpdateSubscription: EventSubscription | null = null;
let purchaseErrorSubscription: EventSubscription | null = null;
let currentPurchaseState: PurchaseState = 'idle';
let cachedProducts: SubscriptionProduct[] = [];

/**
 * Initialize the purchase service
 * Must be called before any other purchase operations
 */
export async function initializePurchaseService(): Promise<boolean> {
  if (isInitialized) {
    console.log('[Purchases] Already initialized');
    return true;
  }

  try {
    console.log('[Purchases] Initializing connection...');
    const result = await initConnection();
    console.log('[Purchases] Connection initialized:', result);

    // Set up transaction listeners
    setupTransactionListeners();

    isInitialized = true;
    currentPurchaseState = 'idle';
    console.log('[Purchases] Service initialized successfully');
    return true;
  } catch (error) {
    console.error('[Purchases] Failed to initialize:', error);
    return false;
  }
}

/**
 * Clean up the purchase service
 * Call when the app is about to close
 */
export async function cleanupPurchaseService(): Promise<void> {
  console.log('[Purchases] Cleaning up...');

  // Remove listeners
  if (purchaseUpdateSubscription) {
    purchaseUpdateSubscription.remove();
    purchaseUpdateSubscription = null;
  }
  if (purchaseErrorSubscription) {
    purchaseErrorSubscription.remove();
    purchaseErrorSubscription = null;
  }

  // End connection
  try {
    await endConnection();
  } catch (error) {
    console.error('[Purchases] Error ending connection:', error);
  }

  isInitialized = false;
  currentPurchaseState = 'idle';
  cachedProducts = [];
  console.log('[Purchases] Cleanup complete');
}

/**
 * Set up transaction listeners for real-time purchase updates
 */
function setupTransactionListeners(): void {
  // Listen for successful purchases
  purchaseUpdateSubscription = purchaseUpdatedListener(async (purchase: Purchase) => {
    console.log('[Purchases] Purchase update received:', purchase.productId);
    await handlePurchaseUpdate(purchase);
  });

  // Listen for purchase errors
  purchaseErrorSubscription = purchaseErrorListener((error: IAPPurchaseError) => {
    console.warn('[Purchases] Purchase error:', error.code, error.message);
    currentPurchaseState = 'error';
  });

  console.log('[Purchases] Transaction listeners set up');
}

/**
 * Handle incoming purchase updates from StoreKit
 * This includes new purchases and restored purchases
 */
async function handlePurchaseUpdate(purchase: Purchase): Promise<void> {
  const { productId, purchaseToken } = purchase;
  // For iOS, transactionId is always present; for Android use purchaseToken
  const transactionId = 'transactionId' in purchase ? purchase.transactionId : purchaseToken;

  if (!purchaseToken && !transactionId) {
    console.warn('[Purchases] No token/transactionId in purchase update');
    return;
  }

  try {
    currentPurchaseState = 'verifying';

    // TODO: Verify with backend (subtask 38.7)
    // For now, we trust the local purchase and update cache
    console.log('[Purchases] Processing purchase for:', productId);

    // Get original transaction ID for iOS
    const originalTransactionId =
      'originalTransactionIdentifierIOS' in purchase
        ? purchase.originalTransactionIdentifierIOS
        : transactionId;

    // Update local subscription info
    const subscriptionInfo: LocalSubscriptionInfo = {
      tier: 'pro',
      productId,
      status: 'active',
      expiresAt: null, // Will be set by backend
      isTrialPeriod: false, // Will be determined by backend
      isIntroOfferPeriod: false,
      autoRenewEnabled: true,
      lastSyncedAt: new Date().toISOString(),
      originalTransactionId: originalTransactionId ?? null,
    };

    await saveSubscriptionInfo(subscriptionInfo);

    // IMPORTANT: Finish the transaction to prevent duplicate charges
    // This must be called after we've processed and saved the purchase
    await finishTransaction({ purchase, isConsumable: false });
    console.log('[Purchases] Transaction finished:', transactionId);

    // Clear eligibility cache since subscription status changed
    clearEligibilityCache();

    currentPurchaseState = 'success';
  } catch (error) {
    console.error('[Purchases] Error processing purchase:', error);
    currentPurchaseState = 'error';
  }
}

/**
 * Fetch available subscription products from the App Store
 */
export async function fetchSubscriptionProducts(): Promise<SubscriptionProduct[]> {
  if (!isInitialized) {
    throw new Error('Purchase service not initialized');
  }

  try {
    currentPurchaseState = 'loading_products';
    console.log('[Purchases] Fetching products:', SUBSCRIPTION_PRODUCT_IDS);

    const products = await fetchProducts({
      skus: [...SUBSCRIPTION_PRODUCT_IDS],
      type: 'subs',
    });
    console.log('[Purchases] Fetched products:', products?.length ?? 0);

    if (!products || products.length === 0) {
      console.warn('[Purchases] No products found');
      currentPurchaseState = 'idle';
      return [];
    }

    // Filter to subscription products only and map with eligibility
    const subscriptions = (products as ProductSubscription[]).filter((p) => p.type === 'subs');

    // Map products with eligibility checking
    cachedProducts = await Promise.all(
      subscriptions.map((sub) => mapSubscriptionToProductWithEligibility(sub))
    );

    currentPurchaseState = 'idle';
    return cachedProducts;
  } catch (error) {
    console.error('[Purchases] Error fetching products:', error);
    currentPurchaseState = 'error';
    throw error;
  }
}

/**
 * Map StoreKit subscription to our product format with eligibility checking
 */
async function mapSubscriptionToProductWithEligibility(
  subscription: ProductSubscription
): Promise<SubscriptionProduct> {
  const { id, title, description, displayPrice, price, currency } = subscription;
  const productId = id;
  const period = PRODUCT_PERIOD_MAP[productId] ?? 'monthly';
  const periodLabel = period === 'yearly' ? 'year' : 'month';

  // Check for intro offer availability (iOS only)
  let introOffer: SubscriptionProduct['introOffer'];
  if (subscription.platform === 'ios') {
    const iosSub = subscription as ProductSubscriptionIOS;
    if (
      iosSub.introductoryPricePaymentModeIOS &&
      iosSub.introductoryPricePaymentModeIOS !== 'empty'
    ) {
      introOffer = {
        type: mapIntroOfferType(iosSub.introductoryPricePaymentModeIOS),
        price: iosSub.introductoryPriceIOS ?? '0',
        priceValue: parseFloat(iosSub.introductoryPriceAsAmountIOS ?? '0'),
        period: iosSub.introductoryPriceSubscriptionPeriodIOS ?? '',
        cycles: parseInt(iosSub.introductoryPriceNumberOfPeriodsIOS ?? '1', 10),
      };
    }
  }

  // Check user's eligibility for this product
  let isEligibleForIntroOffer = false;
  try {
    const eligibility = await checkProductEligibility(productId);
    isEligibleForIntroOffer = eligibility.isEligibleForIntroOffer;
    console.log('[Purchases] Eligibility for', productId, ':', isEligibleForIntroOffer);
  } catch (error) {
    console.warn('[Purchases] Eligibility check failed, defaulting to available:', error);
    // Default to showing offer if check fails (to not block UI)
    isEligibleForIntroOffer = !!introOffer;
  }

  return {
    productId,
    title: PRODUCT_DISPLAY_NAMES[productId] ?? title,
    description: PRODUCT_DESCRIPTIONS[productId] ?? description,
    price: displayPrice,
    priceValue: price ?? 0,
    currency: currency ?? 'USD',
    period,
    displayPrice: `${displayPrice}/${periodLabel}`,
    introOffer,
    isEligibleForIntroOffer,
  };
}

/**
 * Map iOS intro offer payment mode to our type
 */
function mapIntroOfferType(mode: string): 'free_trial' | 'pay_up_front' | 'pay_as_you_go' {
  switch (mode.toLowerCase()) {
    case 'free-trial':
    case 'freetrial':
      return 'free_trial';
    case 'pay-up-front':
    case 'payupfront':
      return 'pay_up_front';
    case 'pay-as-you-go':
    case 'payasyougo':
      return 'pay_as_you_go';
    default:
      return 'free_trial';
  }
}

/**
 * Request a subscription purchase
 */
export async function purchaseSubscription(productId: string): Promise<PurchaseResult> {
  if (!isInitialized) {
    return {
      success: false,
      error: createPurchaseError('E_UNKNOWN', 'Purchase service not initialized'),
    };
  }

  try {
    currentPurchaseState = 'purchasing';
    console.log('[Purchases] Requesting subscription:', productId);

    // Request the subscription using the new API
    if (Platform.OS === 'ios') {
      await requestPurchase({
        type: 'subs',
        request: {
          apple: {
            sku: productId,
            // For StoreKit 2, appAccountToken can be used for server-side validation
            // appAccountToken: userId, // Uncomment when backend is ready
          },
        },
      });
    } else {
      await requestPurchase({
        type: 'subs',
        request: {
          google: {
            skus: [productId],
          },
        },
      });
    }

    // The actual result comes through the purchaseUpdatedListener
    // We wait a bit for the listener to process
    await new Promise((resolve) => setTimeout(resolve, 500));

    // Get current state (may have been updated by listener)
    const state = getPurchaseState();
    if (state === 'success') {
      const info = await getSubscriptionInfo();
      return {
        success: true,
        productId,
        originalTransactionId: info.originalTransactionId ?? undefined,
      };
    } else if (state === 'error') {
      return {
        success: false,
        error: createPurchaseError('E_PURCHASE_FAILED', 'Purchase failed'),
      };
    }

    // Purchase might be pending (deferred transaction)
    return {
      success: false,
      error: createPurchaseError('E_DEFERRED', 'Purchase is pending approval'),
    };
  } catch (error) {
    console.error('[Purchases] Purchase error:', error);
    currentPurchaseState = 'error';
    return {
      success: false,
      error: mapPurchaseError(error),
    };
  }
}

/**
 * Restore previous purchases
 * Uses Transaction.currentEntitlements for efficient StoreKit 2 restoration
 */
export async function restorePurchases(): Promise<PurchaseResult> {
  if (!isInitialized) {
    return {
      success: false,
      error: createPurchaseError('E_UNKNOWN', 'Purchase service not initialized'),
    };
  }

  try {
    currentPurchaseState = 'restoring';
    console.log('[Purchases] Restoring purchases...');

    // First try to restore through the IAP API
    await iapRestorePurchases();

    // Then get available purchases
    const purchases = await getAvailablePurchases();
    console.log('[Purchases] Found purchases:', purchases.length);

    if (purchases.length === 0) {
      currentPurchaseState = 'idle';
      return {
        success: false,
        error: createPurchaseError(
          'E_PURCHASE_FAILED',
          'No purchases to restore',
          'No previous purchases were found.'
        ),
      };
    }

    // Find the most recent subscription purchase
    const subscriptionPurchase = purchases.find((p) =>
      SUBSCRIPTION_PRODUCT_IDS.includes(p.productId as (typeof SUBSCRIPTION_PRODUCT_IDS)[number])
    );

    if (!subscriptionPurchase) {
      currentPurchaseState = 'idle';
      return {
        success: false,
        error: createPurchaseError(
          'E_PURCHASE_FAILED',
          'No subscription purchases found',
          'No subscription purchases were found.'
        ),
      };
    }

    // Process the restored purchase
    await handlePurchaseUpdate(subscriptionPurchase);

    currentPurchaseState = 'idle';
    // Get transaction ID: for iOS use transactionId, for Android use purchaseToken
    let transactionId: string | undefined;
    if ('transactionId' in subscriptionPurchase && subscriptionPurchase.transactionId) {
      transactionId = subscriptionPurchase.transactionId;
    } else if (subscriptionPurchase.purchaseToken) {
      transactionId = subscriptionPurchase.purchaseToken;
    }

    return {
      success: true,
      productId: subscriptionPurchase.productId,
      originalTransactionId: transactionId,
    };
  } catch (error) {
    console.error('[Purchases] Restore error:', error);
    currentPurchaseState = 'error';
    return {
      success: false,
      error: mapPurchaseError(error),
    };
  }
}

/**
 * Get cached subscription info from secure storage
 */
export async function getSubscriptionInfo(): Promise<LocalSubscriptionInfo> {
  try {
    const cached = await SecureStore.getItemAsync(SUBSCRIPTION_CACHE_KEY);
    if (!cached) {
      return DEFAULT_SUBSCRIPTION_INFO;
    }

    const info: LocalSubscriptionInfo = JSON.parse(cached);

    // Check if cache is still valid
    const lastSynced = new Date(info.lastSyncedAt).getTime();
    const now = Date.now();
    const isStale = now - lastSynced > ENTITLEMENT_CACHE_TTL_MS;

    if (isStale) {
      console.log('[Purchases] Cache is stale, should sync with backend');
      // Still return cached info but mark for refresh
    }

    return info;
  } catch (error) {
    console.error('[Purchases] Error reading subscription info:', error);
    return DEFAULT_SUBSCRIPTION_INFO;
  }
}

/**
 * Save subscription info to secure storage
 */
export async function saveSubscriptionInfo(info: LocalSubscriptionInfo): Promise<void> {
  try {
    await SecureStore.setItemAsync(SUBSCRIPTION_CACHE_KEY, JSON.stringify(info));
    console.log('[Purchases] Subscription info saved');
  } catch (error) {
    console.error('[Purchases] Error saving subscription info:', error);
  }
}

/**
 * Clear subscription info from secure storage
 */
export async function clearSubscriptionInfo(): Promise<void> {
  try {
    await SecureStore.deleteItemAsync(SUBSCRIPTION_CACHE_KEY);
    console.log('[Purchases] Subscription info cleared');
  } catch (error) {
    console.error('[Purchases] Error clearing subscription info:', error);
  }
}

/**
 * Check if user has Pro access
 */
export async function hasProAccess(): Promise<boolean> {
  const info = await getSubscriptionInfo();
  return info.tier === 'pro' && info.status === 'active';
}

/**
 * Get current purchase state
 */
export function getPurchaseState(): PurchaseState {
  return currentPurchaseState;
}

/**
 * Get cached products
 */
export function getCachedProducts(): SubscriptionProduct[] {
  return cachedProducts;
}

/**
 * Check if purchase service is initialized
 */
export function isServiceInitialized(): boolean {
  return isInitialized;
}

/**
 * Get user's subscription tier
 */
export async function getSubscriptionTier(): Promise<SubscriptionTier> {
  const info = await getSubscriptionInfo();
  return info.tier;
}

/**
 * Check if running in sandbox environment
 */
export function isSandboxEnvironment(): boolean {
  // In debug mode, assume sandbox
  if (__DEV__) {
    return true;
  }
  // TODO: Detect based on receipt or backend response
  return false;
}

/**
 * Create a purchase error object
 */
function createPurchaseError(
  code: PurchaseErrorCode,
  message: string,
  userMessage?: string
): PurchaseError {
  return {
    code,
    message,
    userMessage: userMessage ?? getUserFriendlyMessage(code),
  };
}

/**
 * Map raw error to purchase error
 */
function mapPurchaseError(error: unknown): PurchaseError {
  if (error instanceof Error) {
    const message = error.message.toLowerCase();

    if (message.includes('cancelled') || message.includes('canceled')) {
      return createPurchaseError('E_USER_CANCELLED', error.message);
    }
    if (message.includes('network')) {
      return createPurchaseError('E_NETWORK_ERROR', error.message);
    }
    if (message.includes('not found') || message.includes('invalid product')) {
      return createPurchaseError('E_PRODUCT_NOT_FOUND', error.message);
    }
    if (message.includes('already owned')) {
      return createPurchaseError('E_ALREADY_OWNED', error.message);
    }
    if (message.includes('deferred')) {
      return createPurchaseError('E_DEFERRED', error.message);
    }

    return createPurchaseError('E_PURCHASE_FAILED', error.message);
  }

  return createPurchaseError('E_UNKNOWN', 'An unknown error occurred');
}

/**
 * Get user-friendly error message
 */
function getUserFriendlyMessage(code: PurchaseErrorCode): string {
  switch (code) {
    case 'E_USER_CANCELLED':
      return 'Purchase was cancelled.';
    case 'E_NETWORK_ERROR':
      return 'Network error. Please check your connection and try again.';
    case 'E_PRODUCT_NOT_FOUND':
      return 'Product not available. Please try again later.';
    case 'E_PURCHASE_FAILED':
      return 'Purchase failed. Please try again.';
    case 'E_VERIFICATION_FAILED':
      return 'Could not verify purchase. Please contact support.';
    case 'E_ALREADY_OWNED':
      return 'You already own this subscription.';
    case 'E_DEFERRED':
      return 'Purchase requires approval. It will be completed once approved.';
    case 'E_UNKNOWN':
    default:
      return 'An error occurred. Please try again.';
  }
}

export {
  type SubscriptionProduct,
  type PurchaseResult,
  type PurchaseError,
  type LocalSubscriptionInfo,
} from './types';
