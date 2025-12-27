/**
 * App Store Server API Service
 *
 * Implements the modern App Store Server API for receipt validation
 * and subscription management. Uses JWT authentication.
 *
 * @see https://developer.apple.com/documentation/appstoreserverapi
 */

import jwt from 'jsonwebtoken';
import { logger } from '../config/logger';

// App Store Server API Endpoints
const PRODUCTION_URL = 'https://api.storekit.itunes.apple.com';
const SANDBOX_URL = 'https://api.storekit-sandbox.itunes.apple.com';

// JWT algorithm for App Store Connect
const JWT_ALGORITHM = 'ES256';
const JWT_AUDIENCE = 'appstoreconnect-v1';
const JWT_EXPIRY_SECONDS = 3600; // 1 hour

/**
 * App Store Server API configuration
 */
export interface AppStoreConfig {
  /** Key ID from App Store Connect */
  keyId: string;
  /** Issuer ID from App Store Connect */
  issuerId: string;
  /** Private key content (PEM format) */
  privateKey: string;
  /** Bundle ID of the app */
  bundleId: string;
  /** Whether to use sandbox environment */
  sandbox: boolean;
}

/**
 * Transaction info from the App Store
 */
export interface TransactionInfo {
  transactionId: string;
  originalTransactionId: string;
  bundleId: string;
  productId: string;
  purchaseDate: Date;
  expiresDate: Date | null;
  type:
    | 'Auto-Renewable Subscription'
    | 'Non-Consumable'
    | 'Consumable'
    | 'Non-Renewing Subscription';
  inAppOwnershipType: 'PURCHASED' | 'FAMILY_SHARED';
  signedDate: Date;
  environment: 'Production' | 'Sandbox';
  offerType?: number;
  offerIdentifier?: string;
}

/**
 * Subscription status from the App Store
 */
export interface SubscriptionStatus {
  state: 'active' | 'expired' | 'billing_retry' | 'billing_grace_period' | 'revoked';
  renewalInfo: {
    autoRenewStatus: boolean;
    autoRenewProductId: string;
    expirationIntent?: number;
    gracePeriodExpiresDate?: Date;
    isInBillingRetryPeriod?: boolean;
    offerIdentifier?: string;
    priceIncreaseStatus?: number;
  } | null;
  transactionInfo: TransactionInfo | null;
}

/**
 * Subscription group status response
 */
export interface SubscriptionGroupStatus {
  subscriptionGroupIdentifier: string;
  lastTransactions: Array<{
    status: number;
    originalTransactionId: string;
    signedTransactionInfo: string;
    signedRenewalInfo: string;
  }>;
}

/**
 * App Store Server API client
 */
export class AppStoreServerApiClient {
  private config: AppStoreConfig;
  private cachedToken: string | null = null;
  private tokenExpiry: number = 0;

  constructor(config: AppStoreConfig) {
    this.config = config;
  }

  /**
   * Get the API base URL based on environment
   */
  private getBaseUrl(): string {
    return this.config.sandbox ? SANDBOX_URL : PRODUCTION_URL;
  }

  /**
   * Generate a JWT for App Store Server API authentication
   */
  private generateToken(): string {
    const now = Math.floor(Date.now() / 1000);

    // Check if cached token is still valid (with 5 min buffer)
    if (this.cachedToken && this.tokenExpiry > now + 300) {
      return this.cachedToken;
    }

    const payload = {
      iss: this.config.issuerId,
      iat: now,
      exp: now + JWT_EXPIRY_SECONDS,
      aud: JWT_AUDIENCE,
      bid: this.config.bundleId,
    };

    const header = {
      alg: JWT_ALGORITHM,
      kid: this.config.keyId,
      typ: 'JWT',
    };

    this.cachedToken = jwt.sign(payload, this.config.privateKey, {
      algorithm: JWT_ALGORITHM as jwt.Algorithm,
      header,
    });
    this.tokenExpiry = now + JWT_EXPIRY_SECONDS;

    return this.cachedToken;
  }

  /**
   * Make an authenticated request to the App Store Server API
   */
  private async makeRequest<T>(
    endpoint: string,
    method: 'GET' | 'POST' | 'PUT' = 'GET',
    body?: unknown
  ): Promise<T> {
    const url = `${this.getBaseUrl()}${endpoint}`;
    const token = this.generateToken();

    logger.debug({ endpoint, method }, 'Making App Store Server API request');

    const response = await fetch(url, {
      method,
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const errorBody = await response.text();
      logger.error(
        { status: response.status, body: errorBody, endpoint },
        'App Store Server API error'
      );
      throw new AppStoreApiError(
        `App Store API request failed: ${response.status}`,
        response.status,
        errorBody
      );
    }

    return response.json() as Promise<T>;
  }

  /**
   * Get transaction info by transaction ID
   */
  async getTransactionInfo(transactionId: string): Promise<TransactionInfo | null> {
    try {
      const response = await this.makeRequest<{ signedTransactionInfo: string }>(
        `/inApps/v1/transactions/${transactionId}`
      );

      if (!response.signedTransactionInfo) {
        return null;
      }

      return this.decodeSignedTransaction(response.signedTransactionInfo);
    } catch (error) {
      if (error instanceof AppStoreApiError && error.statusCode === 404) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Get subscription status by original transaction ID
   */
  async getSubscriptionStatus(originalTransactionId: string): Promise<SubscriptionStatus | null> {
    try {
      const response = await this.makeRequest<{
        data: SubscriptionGroupStatus[];
        bundleId: string;
        appAppleId: number;
        environment: string;
      }>(`/inApps/v1/subscriptions/${originalTransactionId}`);

      if (!response.data || response.data.length === 0) {
        return null;
      }

      // Find the latest transaction across all subscription groups
      let latestTransaction: SubscriptionGroupStatus['lastTransactions'][0] | null = null;
      for (const group of response.data) {
        for (const txn of group.lastTransactions) {
          if (!latestTransaction || txn.status > 0) {
            latestTransaction = txn;
          }
        }
      }

      if (!latestTransaction) {
        return null;
      }

      // Decode transaction and renewal info
      const transactionInfo = this.decodeSignedTransaction(latestTransaction.signedTransactionInfo);
      const renewalInfo = this.decodeSignedRenewalInfo(latestTransaction.signedRenewalInfo);

      // Determine subscription state
      const state = this.determineSubscriptionState(latestTransaction.status, transactionInfo);

      return {
        state,
        renewalInfo,
        transactionInfo,
      };
    } catch (error) {
      if (error instanceof AppStoreApiError && error.statusCode === 404) {
        return null;
      }
      throw error;
    }
  }

  /**
   * Get transaction history for a customer
   */
  async getTransactionHistory(
    originalTransactionId: string,
    revision?: string
  ): Promise<{
    transactions: TransactionInfo[];
    hasMore: boolean;
    revision: string;
  }> {
    let endpoint = `/inApps/v1/history/${originalTransactionId}`;
    if (revision) {
      endpoint += `?revision=${revision}`;
    }

    const response = await this.makeRequest<{
      signedTransactions: string[];
      hasMore: boolean;
      revision: string;
    }>(endpoint);

    const transactions = response.signedTransactions.map((signed) =>
      this.decodeSignedTransaction(signed)
    );

    return {
      transactions,
      hasMore: response.hasMore,
      revision: response.revision,
    };
  }

  /**
   * Request a test notification (for testing webhooks)
   */
  async requestTestNotification(): Promise<{ testNotificationToken: string }> {
    return this.makeRequest<{ testNotificationToken: string }>(
      '/inApps/v1/notifications/test',
      'POST'
    );
  }

  /**
   * Get notification history
   */
  async getNotificationHistory(
    paginationToken?: string,
    startDate?: Date,
    endDate?: Date
  ): Promise<{
    notificationHistory: Array<{
      signedPayload: string;
      sendAttempts: Array<{
        attemptDate: number;
        sendAttemptResult: string;
      }>;
    }>;
    hasMore: boolean;
    paginationToken: string;
  }> {
    const body: Record<string, unknown> = {};
    if (paginationToken) body.paginationToken = paginationToken;
    if (startDate) body.startDate = startDate.getTime();
    if (endDate) body.endDate = endDate.getTime();

    return this.makeRequest('/inApps/v1/notifications/history', 'POST', body);
  }

  /**
   * Decode a signed transaction JWS
   */
  private decodeSignedTransaction(signedTransaction: string): TransactionInfo {
    // JWS format: header.payload.signature
    const parts = signedTransaction.split('.');
    if (parts.length !== 3) {
      throw new Error('Invalid JWS format');
    }

    // Decode payload (base64url)
    const payload = JSON.parse(Buffer.from(parts[1], 'base64url').toString('utf-8'));

    return {
      transactionId: payload.transactionId,
      originalTransactionId: payload.originalTransactionId,
      bundleId: payload.bundleId,
      productId: payload.productId,
      purchaseDate: new Date(payload.purchaseDate),
      expiresDate: payload.expiresDate ? new Date(payload.expiresDate) : null,
      type: payload.type,
      inAppOwnershipType: payload.inAppOwnershipType,
      signedDate: new Date(payload.signedDate),
      environment: payload.environment,
      offerType: payload.offerType,
      offerIdentifier: payload.offerIdentifier,
    };
  }

  /**
   * Decode signed renewal info JWS
   */
  private decodeSignedRenewalInfo(signedRenewalInfo: string): SubscriptionStatus['renewalInfo'] {
    const parts = signedRenewalInfo.split('.');
    if (parts.length !== 3) {
      return null;
    }

    const payload = JSON.parse(Buffer.from(parts[1], 'base64url').toString('utf-8'));

    return {
      autoRenewStatus: payload.autoRenewStatus === 1,
      autoRenewProductId: payload.autoRenewProductId,
      expirationIntent: payload.expirationIntent,
      gracePeriodExpiresDate: payload.gracePeriodExpiresDate
        ? new Date(payload.gracePeriodExpiresDate)
        : undefined,
      isInBillingRetryPeriod: payload.isInBillingRetryPeriod,
      offerIdentifier: payload.offerIdentifier,
      priceIncreaseStatus: payload.priceIncreaseStatus,
    };
  }

  /**
   * Determine subscription state from status code
   */
  private determineSubscriptionState(
    status: number,
    transactionInfo: TransactionInfo | null
  ): SubscriptionStatus['state'] {
    // Status codes from App Store Server API
    // 1 = Active
    // 2 = Expired
    // 3 = Billing Retry
    // 4 = Billing Grace Period
    // 5 = Revoked
    switch (status) {
      case 1:
        return 'active';
      case 2:
        return 'expired';
      case 3:
        return 'billing_retry';
      case 4:
        return 'billing_grace_period';
      case 5:
        return 'revoked';
      default:
        // If we have transaction info with future expiry, consider it active
        if (transactionInfo?.expiresDate && transactionInfo.expiresDate > new Date()) {
          return 'active';
        }
        return 'expired';
    }
  }
}

/**
 * Custom error for App Store API errors
 */
export class AppStoreApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public responseBody: string
  ) {
    super(message);
    this.name = 'AppStoreApiError';
  }
}

/**
 * Create an App Store Server API client from environment variables
 */
export function createAppStoreClient(): AppStoreServerApiClient | null {
  const keyId = process.env.APPLE_KEY_ID;
  const issuerId = process.env.APPLE_ISSUER_ID;
  const privateKey = process.env.APPLE_PRIVATE_KEY;
  const bundleId = process.env.APPLE_BUNDLE_ID || 'com.anonymous.nutri';
  const sandbox = process.env.NODE_ENV !== 'production';

  if (!keyId || !issuerId || !privateKey) {
    logger.warn('App Store Server API credentials not configured');
    return null;
  }

  return new AppStoreServerApiClient({
    keyId,
    issuerId,
    privateKey: privateKey.replace(/\\n/g, '\n'), // Handle escaped newlines
    bundleId,
    sandbox,
  });
}

/**
 * Singleton instance
 */
let appStoreClient: AppStoreServerApiClient | null = null;

/**
 * Get the App Store Server API client instance
 */
export function getAppStoreClient(): AppStoreServerApiClient | null {
  if (!appStoreClient) {
    appStoreClient = createAppStoreClient();
  }
  return appStoreClient;
}

export default AppStoreServerApiClient;
