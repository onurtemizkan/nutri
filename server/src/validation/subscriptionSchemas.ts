/**
 * Subscription Validation Schemas
 *
 * Zod schemas for validating subscription-related API requests
 * and App Store Server API responses.
 */

import { z } from 'zod';

// ============================================================================
// Request Schemas
// ============================================================================

/**
 * Schema for validating a purchase receipt from the client
 */
export const validateReceiptSchema = z.object({
  transactionId: z.string().min(1, 'Transaction ID is required'),
  originalTransactionId: z.string().optional(),
  productId: z.string().min(1, 'Product ID is required'),
  purchaseDate: z.string().datetime().optional(),
  environment: z.enum(['Production', 'Sandbox']).optional(),
});

/**
 * Schema for restoring purchases
 */
export const restorePurchasesSchema = z.object({
  transactionIds: z.array(z.string().min(1)).min(1, 'At least one transaction ID is required'),
});

/**
 * Schema for checking subscription status
 */
export const checkSubscriptionSchema = z.object({
  originalTransactionId: z.string().min(1, 'Original transaction ID is required'),
});

/**
 * Schema for App Store webhook notification
 */
export const appStoreNotificationSchema = z.object({
  signedPayload: z.string().min(1, 'Signed payload is required'),
});

// ============================================================================
// Response/Internal Schemas
// ============================================================================

/**
 * Subscription state enum
 */
export const subscriptionStateSchema = z.enum([
  'active',
  'expired',
  'billing_retry',
  'billing_grace_period',
  'revoked',
]);

/**
 * Transaction type enum
 */
export const transactionTypeSchema = z.enum([
  'Auto-Renewable Subscription',
  'Non-Consumable',
  'Consumable',
  'Non-Renewing Subscription',
]);

/**
 * Ownership type enum
 */
export const ownershipTypeSchema = z.enum(['PURCHASED', 'FAMILY_SHARED']);

/**
 * Environment enum
 */
export const environmentSchema = z.enum(['Production', 'Sandbox']);

/**
 * Transaction info schema (from decoded JWS)
 */
export const transactionInfoSchema = z.object({
  transactionId: z.string(),
  originalTransactionId: z.string(),
  bundleId: z.string(),
  productId: z.string(),
  purchaseDate: z.coerce.date(),
  expiresDate: z.coerce.date().nullable(),
  type: transactionTypeSchema,
  inAppOwnershipType: ownershipTypeSchema,
  signedDate: z.coerce.date(),
  environment: environmentSchema,
  offerType: z.number().optional(),
  offerIdentifier: z.string().optional(),
});

/**
 * Renewal info schema (from decoded JWS)
 */
export const renewalInfoSchema = z
  .object({
    autoRenewStatus: z.boolean(),
    autoRenewProductId: z.string(),
    expirationIntent: z.number().optional(),
    gracePeriodExpiresDate: z.coerce.date().optional(),
    isInBillingRetryPeriod: z.boolean().optional(),
    offerIdentifier: z.string().optional(),
    priceIncreaseStatus: z.number().optional(),
  })
  .nullable();

/**
 * Full subscription status schema
 */
export const subscriptionStatusSchema = z.object({
  state: subscriptionStateSchema,
  renewalInfo: renewalInfoSchema,
  transactionInfo: transactionInfoSchema.nullable(),
});

// ============================================================================
// App Store Server Notification Schemas
// ============================================================================

/**
 * Notification type enum (App Store Server Notifications V2)
 */
export const notificationTypeSchema = z.enum([
  'CONSUMPTION_REQUEST',
  'DID_CHANGE_RENEWAL_PREF',
  'DID_CHANGE_RENEWAL_STATUS',
  'DID_FAIL_TO_RENEW',
  'DID_RENEW',
  'EXPIRED',
  'GRACE_PERIOD_EXPIRED',
  'OFFER_REDEEMED',
  'PRICE_INCREASE',
  'REFUND',
  'REFUND_DECLINED',
  'REFUND_REVERSED',
  'RENEWAL_EXTENDED',
  'RENEWAL_EXTENSION',
  'REVOKE',
  'SUBSCRIBED',
  'TEST',
]);

/**
 * Notification subtype enum
 */
export const notificationSubtypeSchema = z
  .enum([
    'INITIAL_BUY',
    'RESUBSCRIBE',
    'DOWNGRADE',
    'UPGRADE',
    'AUTO_RENEW_ENABLED',
    'AUTO_RENEW_DISABLED',
    'VOLUNTARY',
    'BILLING_RETRY',
    'PRICE_INCREASE',
    'GRACE_PERIOD',
    'BILLING_RECOVERY',
    'PENDING',
    'ACCEPTED',
    'SUMMARY',
    'FAILURE',
  ])
  .optional();

/**
 * Decoded notification payload schema
 */
export const notificationPayloadSchema = z.object({
  notificationType: notificationTypeSchema,
  subtype: notificationSubtypeSchema,
  notificationUUID: z.string(),
  version: z.string(),
  signedDate: z.number(),
  data: z.object({
    appAppleId: z.number().optional(),
    bundleId: z.string(),
    bundleVersion: z.string().optional(),
    environment: environmentSchema,
    signedTransactionInfo: z.string().optional(),
    signedRenewalInfo: z.string().optional(),
  }),
});

// ============================================================================
// Database Record Schemas
// ============================================================================

/**
 * Subscription tier enum
 */
export const subscriptionTierSchema = z.enum(['free', 'pro_monthly', 'pro_yearly']);

/**
 * Schema for creating a subscription record
 */
export const createSubscriptionRecordSchema = z.object({
  userId: z.string().uuid(),
  originalTransactionId: z.string(),
  productId: z.string(),
  tier: subscriptionTierSchema,
  status: subscriptionStateSchema,
  purchaseDate: z.coerce.date(),
  expiresDate: z.coerce.date().nullable(),
  environment: environmentSchema,
  isTrialPeriod: z.boolean().default(false),
  autoRenewEnabled: z.boolean().default(true),
});

/**
 * Schema for updating a subscription record
 */
export const updateSubscriptionRecordSchema = z.object({
  status: subscriptionStateSchema.optional(),
  expiresDate: z.coerce.date().nullable().optional(),
  autoRenewEnabled: z.boolean().optional(),
  cancellationDate: z.coerce.date().nullable().optional(),
  gracePeriodExpiresDate: z.coerce.date().nullable().optional(),
});

// ============================================================================
// Type Exports
// ============================================================================

export type ValidateReceiptInput = z.infer<typeof validateReceiptSchema>;
export type RestorePurchasesInput = z.infer<typeof restorePurchasesSchema>;
export type CheckSubscriptionInput = z.infer<typeof checkSubscriptionSchema>;
export type AppStoreNotificationInput = z.infer<typeof appStoreNotificationSchema>;
export type SubscriptionState = z.infer<typeof subscriptionStateSchema>;
export type TransactionType = z.infer<typeof transactionTypeSchema>;
export type TransactionInfo = z.infer<typeof transactionInfoSchema>;
export type RenewalInfo = z.infer<typeof renewalInfoSchema>;
export type SubscriptionStatus = z.infer<typeof subscriptionStatusSchema>;
export type NotificationType = z.infer<typeof notificationTypeSchema>;
export type NotificationSubtype = z.infer<typeof notificationSubtypeSchema>;
export type NotificationPayload = z.infer<typeof notificationPayloadSchema>;
export type SubscriptionTier = z.infer<typeof subscriptionTierSchema>;
export type CreateSubscriptionRecord = z.infer<typeof createSubscriptionRecordSchema>;
export type UpdateSubscriptionRecord = z.infer<typeof updateSubscriptionRecordSchema>;
