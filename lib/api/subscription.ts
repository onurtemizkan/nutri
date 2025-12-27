/**
 * Subscription API Client
 *
 * Handles communication with the subscription backend endpoints
 * for purchase verification, restoration, and status checks.
 */

import api from './client';

// ============================================================================
// Subscription Tier Type (matching backend)
// ============================================================================

export type SubscriptionTier = 'FREE' | 'PRO_TRIAL' | 'PRO';

// ============================================================================
// Types
// ============================================================================

export interface SubscriptionProduct {
  id: string;
  name: string;
  description: string;
  tier: SubscriptionTier;
  billingCycle: 'MONTHLY' | 'ANNUAL';
  features: string[];
}

export interface SubscriptionStatus {
  isActive: boolean;
  tier: SubscriptionTier;
  status:
    | 'ACTIVE'
    | 'EXPIRED'
    | 'IN_GRACE_PERIOD'
    | 'IN_BILLING_RETRY'
    | 'REVOKED'
    | 'REFUNDED'
    | null;
  expiresAt: string | null;
  daysRemaining: number | null;
  isTrialPeriod: boolean;
  autoRenewEnabled: boolean;
  productId: string | null;
}

export interface VerifyPurchaseRequest {
  transactionId: string;
  originalTransactionId?: string;
  productId: string;
  purchaseDate?: string;
  environment?: 'Production' | 'Sandbox';
}

export interface VerifyPurchaseResponse {
  success: boolean;
  subscription: {
    id: string;
    productId: string;
    status: string;
    expiresAt: string | null;
    isTrialPeriod: boolean;
    autoRenewEnabled: boolean;
  };
  userStatus: SubscriptionStatus;
}

export interface RestorePurchasesRequest {
  transactionIds: string[];
}

export interface RestorePurchasesResponse {
  success: boolean;
  restored: number;
  alreadyActive: number;
  errors: string[];
  userStatus: SubscriptionStatus;
}

export interface ProductsResponse {
  success: boolean;
  products: SubscriptionProduct[];
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Get available subscription products
 */
export async function getProducts(): Promise<SubscriptionProduct[]> {
  try {
    const response = await api.get<ProductsResponse>('/subscription/products');
    return response.data.products;
  } catch (error) {
    console.error('[Subscription API] Failed to get products:', error);
    throw error;
  }
}

/**
 * Verify a purchase with the backend
 */
export async function verifyPurchase(
  request: VerifyPurchaseRequest
): Promise<VerifyPurchaseResponse> {
  try {
    const response = await api.post<VerifyPurchaseResponse>('/subscription/verify', request);
    return response.data;
  } catch (error) {
    console.error('[Subscription API] Failed to verify purchase:', error);
    throw error;
  }
}

/**
 * Restore purchases for the current user
 */
export async function restorePurchases(
  transactionIds: string[]
): Promise<RestorePurchasesResponse> {
  try {
    const response = await api.post<RestorePurchasesResponse>('/subscription/restore', {
      transactionIds,
    });
    return response.data;
  } catch (error) {
    console.error('[Subscription API] Failed to restore purchases:', error);
    throw error;
  }
}

/**
 * Get the current user's subscription status
 */
export async function getSubscriptionStatus(): Promise<SubscriptionStatus> {
  try {
    const response = await api.get<{ success: boolean } & SubscriptionStatus>(
      '/subscription/status'
    );
    const { success: _, ...status } = response.data;
    return status;
  } catch (error) {
    console.error('[Subscription API] Failed to get subscription status:', error);
    // Return free tier status on error
    return {
      isActive: false,
      tier: 'FREE',
      status: null,
      expiresAt: null,
      daysRemaining: null,
      isTrialPeriod: false,
      autoRenewEnabled: false,
      productId: null,
    };
  }
}

export default {
  getProducts,
  verifyPurchase,
  restorePurchases,
  getSubscriptionStatus,
};
