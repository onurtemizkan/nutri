/**
 * Webhook Controller
 *
 * Handles App Store Server Notifications V2 webhooks.
 * Processes subscription events and updates the database.
 */

import { Request, Response } from 'express';
import { logger } from '../config/logger';
import { decodeSignedData } from '../middleware/webhookAuth';
import { HTTP_STATUS } from '../config/constants';
import * as subscriptionService from '../services/subscriptionService';

// In-memory set for idempotency (in production, use Redis or database)
const processedNotifications = new Set<string>();
const NOTIFICATION_CACHE_MAX_SIZE = 10000;

/**
 * Notification types from App Store Server Notifications V2
 */
type NotificationType =
  | 'CONSUMPTION_REQUEST'
  | 'DID_CHANGE_RENEWAL_PREF'
  | 'DID_CHANGE_RENEWAL_STATUS'
  | 'DID_FAIL_TO_RENEW'
  | 'DID_RENEW'
  | 'EXPIRED'
  | 'GRACE_PERIOD_EXPIRED'
  | 'OFFER_REDEEMED'
  | 'PRICE_INCREASE'
  | 'REFUND'
  | 'REFUND_DECLINED'
  | 'REFUND_REVERSED'
  | 'RENEWAL_EXTENDED'
  | 'RENEWAL_EXTENSION'
  | 'REVOKE'
  | 'SUBSCRIBED'
  | 'TEST';

/**
 * Decoded notification payload structure
 */
interface NotificationPayload {
  notificationType: NotificationType;
  subtype?: string;
  notificationUUID: string;
  version: string;
  signedDate: number;
  data: {
    appAppleId?: number;
    bundleId: string;
    bundleVersion?: string;
    environment: 'Production' | 'Sandbox';
    signedTransactionInfo?: string;
    signedRenewalInfo?: string;
  };
}

/**
 * Transaction info structure
 */
interface TransactionInfo {
  transactionId: string;
  originalTransactionId: string;
  bundleId: string;
  productId: string;
  purchaseDate: number;
  expiresDate?: number;
  type: string;
  inAppOwnershipType: string;
  signedDate: number;
  environment: string;
  offerType?: number;
  offerIdentifier?: string;
  revocationDate?: number;
  revocationReason?: number;
}

/**
 * Renewal info structure
 */
interface RenewalInfo {
  autoRenewStatus: number;
  autoRenewProductId: string;
  expirationIntent?: number;
  gracePeriodExpiresDate?: number;
  isInBillingRetryPeriod?: boolean;
  offerIdentifier?: string;
  priceIncreaseStatus?: number;
  signedDate: number;
}

/**
 * Check if notification was already processed (idempotency)
 */
function isNotificationProcessed(notificationUUID: string): boolean {
  return processedNotifications.has(notificationUUID);
}

/**
 * Mark notification as processed
 */
function markNotificationProcessed(notificationUUID: string): void {
  // Simple LRU-like cleanup when cache gets too large
  if (processedNotifications.size >= NOTIFICATION_CACHE_MAX_SIZE) {
    const iterator = processedNotifications.values();
    const firstItem = iterator.next().value;
    if (firstItem) {
      processedNotifications.delete(firstItem);
    }
  }
  processedNotifications.add(notificationUUID);
}

/**
 * Handle SUBSCRIBED notification (new subscription)
 * Note: This requires the user ID from the app_account_token in the transaction.
 * For webhooks, we may need to look up the user by originalTransactionId from a previous purchase.
 */
async function handleSubscribed(
  payload: NotificationPayload,
  transactionInfo: TransactionInfo,
  renewalInfo: RenewalInfo | null
): Promise<void> {
  logger.info(
    {
      type: 'SUBSCRIBED',
      subtype: payload.subtype,
      originalTransactionId: transactionInfo.originalTransactionId,
      productId: transactionInfo.productId,
    },
    'Processing new subscription'
  );

  // Look up existing subscription by transaction ID to get user ID
  const existingSubscription = await subscriptionService.findByTransactionId(
    transactionInfo.originalTransactionId
  );

  if (existingSubscription) {
    // Update existing subscription (renewal from webhook)
    await subscriptionService.createOrUpdateSubscription({
      userId: existingSubscription.userId,
      originalTransactionId: transactionInfo.originalTransactionId,
      productId: transactionInfo.productId,
      purchaseDate: new Date(transactionInfo.purchaseDate),
      expiresDate: transactionInfo.expiresDate ? new Date(transactionInfo.expiresDate) : null,
      environment: payload.data.environment,
      autoRenewEnabled: renewalInfo?.autoRenewStatus === 1,
      autoRenewProductId: renewalInfo?.autoRenewProductId,
    });
  } else {
    // New subscription from webhook - log warning as we need app-side verification first
    logger.warn(
      {
        originalTransactionId: transactionInfo.originalTransactionId,
        productId: transactionInfo.productId,
      },
      'Received SUBSCRIBED webhook for unknown transaction - subscription should be created via app verification first'
    );
  }
}

/**
 * Handle DID_RENEW notification (subscription renewed)
 */
async function handleDidRenew(
  _payload: NotificationPayload,
  transactionInfo: TransactionInfo,
  _renewalInfo: RenewalInfo | null
): Promise<void> {
  logger.info(
    {
      type: 'DID_RENEW',
      originalTransactionId: transactionInfo.originalTransactionId,
      newExpiresDate: transactionInfo.expiresDate,
    },
    'Processing subscription renewal'
  );

  // Update subscription with new expiry date
  const newExpiresDate = transactionInfo.expiresDate ? new Date(transactionInfo.expiresDate) : null;

  if (newExpiresDate) {
    await subscriptionService.renewSubscription(
      transactionInfo.originalTransactionId,
      newExpiresDate
    );
  }
}

/**
 * Handle DID_CHANGE_RENEWAL_STATUS notification
 */
async function handleDidChangeRenewalStatus(
  payload: NotificationPayload,
  transactionInfo: TransactionInfo,
  renewalInfo: RenewalInfo | null
): Promise<void> {
  const autoRenewEnabled = renewalInfo?.autoRenewStatus === 1;

  logger.info(
    {
      type: 'DID_CHANGE_RENEWAL_STATUS',
      subtype: payload.subtype,
      originalTransactionId: transactionInfo.originalTransactionId,
      autoRenewEnabled,
    },
    'Processing renewal status change'
  );

  // Update auto-renew status in database
  await subscriptionService.updateSubscription(transactionInfo.originalTransactionId, {
    autoRenewEnabled,
    // If user turned off auto-renew, record cancellation intent
    cancelledAt: !autoRenewEnabled ? new Date() : undefined,
  });
}

/**
 * Handle DID_FAIL_TO_RENEW notification
 */
async function handleDidFailToRenew(
  payload: NotificationPayload,
  transactionInfo: TransactionInfo,
  renewalInfo: RenewalInfo | null
): Promise<void> {
  logger.info(
    {
      type: 'DID_FAIL_TO_RENEW',
      subtype: payload.subtype,
      originalTransactionId: transactionInfo.originalTransactionId,
      isInBillingRetry: renewalInfo?.isInBillingRetryPeriod,
    },
    'Processing renewal failure'
  );

  // Check if in grace period or billing retry
  if (renewalInfo?.gracePeriodExpiresDate) {
    // User is in grace period - still has access
    await subscriptionService.enterGracePeriod(
      transactionInfo.originalTransactionId,
      new Date(renewalInfo.gracePeriodExpiresDate)
    );
  } else {
    // User is in billing retry period
    await subscriptionService.enterBillingRetry(transactionInfo.originalTransactionId);
  }
}

/**
 * Handle EXPIRED notification
 */
async function handleExpired(
  _payload: NotificationPayload,
  transactionInfo: TransactionInfo,
  _renewalInfo: RenewalInfo | null
): Promise<void> {
  logger.info(
    {
      type: 'EXPIRED',
      originalTransactionId: transactionInfo.originalTransactionId,
    },
    'Processing subscription expiration'
  );

  // Mark subscription as expired
  await subscriptionService.expireSubscription(transactionInfo.originalTransactionId);
}

/**
 * Handle GRACE_PERIOD_EXPIRED notification
 */
async function handleGracePeriodExpired(
  _payload: NotificationPayload,
  transactionInfo: TransactionInfo,
  _renewalInfo: RenewalInfo | null
): Promise<void> {
  logger.info(
    {
      type: 'GRACE_PERIOD_EXPIRED',
      originalTransactionId: transactionInfo.originalTransactionId,
    },
    'Processing grace period expiration'
  );

  // Mark subscription as expired after grace period
  await subscriptionService.updateSubscription(transactionInfo.originalTransactionId, {
    status: 'EXPIRED',
    gracePeriodExpiresAt: null,
  });
}

/**
 * Handle REFUND notification
 */
async function handleRefund(
  _payload: NotificationPayload,
  transactionInfo: TransactionInfo
): Promise<void> {
  logger.info(
    {
      type: 'REFUND',
      originalTransactionId: transactionInfo.originalTransactionId,
      transactionId: transactionInfo.transactionId,
    },
    'Processing refund'
  );

  // Mark subscription as revoked due to refund
  await subscriptionService.revokeSubscription(transactionInfo.originalTransactionId, 'REFUND');
}

/**
 * Handle REVOKE notification
 */
async function handleRevoke(
  _payload: NotificationPayload,
  transactionInfo: TransactionInfo
): Promise<void> {
  logger.info(
    {
      type: 'REVOKE',
      originalTransactionId: transactionInfo.originalTransactionId,
      revocationReason: transactionInfo.revocationReason,
    },
    'Processing revocation'
  );

  // Mark subscription as revoked
  await subscriptionService.revokeSubscription(transactionInfo.originalTransactionId, 'OTHER');
}

/**
 * Handle OFFER_REDEEMED notification
 */
async function handleOfferRedeemed(
  payload: NotificationPayload,
  transactionInfo: TransactionInfo
): Promise<void> {
  logger.info(
    {
      type: 'OFFER_REDEEMED',
      originalTransactionId: transactionInfo.originalTransactionId,
      offerType: transactionInfo.offerType,
      offerIdentifier: transactionInfo.offerIdentifier,
    },
    'Processing offer redemption'
  );

  // Find subscription to log the event
  const subscription = await subscriptionService.findByTransactionId(
    transactionInfo.originalTransactionId
  );

  if (subscription) {
    await subscriptionService.createSubscriptionEvent({
      subscriptionId: subscription.id,
      notificationType: 'OFFER_REDEEMED',
      originalTransactionId: transactionInfo.originalTransactionId,
      notificationUUID: payload.notificationUUID,
      eventData: {
        offerType: transactionInfo.offerType,
        offerIdentifier: transactionInfo.offerIdentifier,
      },
    });
  }
}

/**
 * Handle DID_CHANGE_RENEWAL_PREF notification
 */
async function handleDidChangeRenewalPref(
  payload: NotificationPayload,
  transactionInfo: TransactionInfo,
  renewalInfo: RenewalInfo | null
): Promise<void> {
  logger.info(
    {
      type: 'DID_CHANGE_RENEWAL_PREF',
      originalTransactionId: transactionInfo.originalTransactionId,
      newProductId: renewalInfo?.autoRenewProductId,
    },
    'Processing renewal preference change'
  );

  // Update the auto-renew product ID (plan change takes effect at next renewal)
  if (renewalInfo?.autoRenewProductId) {
    await subscriptionService.updateSubscription(transactionInfo.originalTransactionId, {
      autoRenewProductId: renewalInfo.autoRenewProductId,
    });
  }

  // Log the plan change event
  const subscription = await subscriptionService.findByTransactionId(
    transactionInfo.originalTransactionId
  );

  if (subscription) {
    await subscriptionService.createSubscriptionEvent({
      subscriptionId: subscription.id,
      notificationType: 'DID_CHANGE_RENEWAL_PREF',
      originalTransactionId: transactionInfo.originalTransactionId,
      notificationUUID: payload.notificationUUID,
      eventData: {
        newProductId: renewalInfo?.autoRenewProductId,
      },
    });
  }
}

/**
 * Create subscription event audit log
 */
async function createAuditLog(
  payload: NotificationPayload,
  transactionInfo: TransactionInfo | null
): Promise<void> {
  logger.info(
    {
      notificationType: payload.notificationType,
      notificationUUID: payload.notificationUUID,
      originalTransactionId: transactionInfo?.originalTransactionId,
      environment: payload.data.environment,
    },
    'Creating subscription event audit log'
  );

  // Find subscription to create audit log
  if (transactionInfo?.originalTransactionId) {
    const subscription = await subscriptionService.findByTransactionId(
      transactionInfo.originalTransactionId
    );

    if (subscription) {
      await subscriptionService.createSubscriptionEvent({
        subscriptionId: subscription.id,
        notificationType: payload.notificationType,
        subtype: payload.subtype,
        transactionId: transactionInfo.transactionId,
        originalTransactionId: transactionInfo.originalTransactionId,
        notificationUUID: payload.notificationUUID,
        eventData: {
          environment: payload.data.environment,
          bundleId: payload.data.bundleId,
          productId: transactionInfo.productId,
        },
      });
    } else {
      logger.warn(
        {
          originalTransactionId: transactionInfo.originalTransactionId,
          notificationType: payload.notificationType,
        },
        'Cannot create audit log - subscription not found'
      );
    }
  }
}

/**
 * Main webhook handler for App Store notifications
 */
export async function handleAppStoreWebhook(req: Request, res: Response): Promise<void> {
  try {
    const payload = req.body.decodedPayload as NotificationPayload;

    // Check idempotency
    if (isNotificationProcessed(payload.notificationUUID)) {
      logger.info(
        { notificationUUID: payload.notificationUUID },
        'Duplicate notification, skipping'
      );
      res.status(HTTP_STATUS.OK).json({ success: true, message: 'Already processed' });
      return;
    }

    logger.info(
      {
        notificationType: payload.notificationType,
        subtype: payload.subtype,
        notificationUUID: payload.notificationUUID,
        environment: payload.data.environment,
      },
      'Received App Store notification'
    );

    // Decode transaction and renewal info if present
    let transactionInfo: TransactionInfo | null = null;
    let renewalInfo: RenewalInfo | null = null;

    if (payload.data.signedTransactionInfo) {
      transactionInfo = decodeSignedData<TransactionInfo>(payload.data.signedTransactionInfo);
    }

    if (payload.data.signedRenewalInfo) {
      renewalInfo = decodeSignedData<RenewalInfo>(payload.data.signedRenewalInfo);
    }

    // Process based on notification type
    switch (payload.notificationType) {
      case 'SUBSCRIBED':
        if (transactionInfo) {
          await handleSubscribed(payload, transactionInfo, renewalInfo);
        }
        break;

      case 'DID_RENEW':
        if (transactionInfo) {
          await handleDidRenew(payload, transactionInfo, renewalInfo);
        }
        break;

      case 'DID_CHANGE_RENEWAL_STATUS':
        if (transactionInfo) {
          await handleDidChangeRenewalStatus(payload, transactionInfo, renewalInfo);
        }
        break;

      case 'DID_CHANGE_RENEWAL_PREF':
        if (transactionInfo) {
          await handleDidChangeRenewalPref(payload, transactionInfo, renewalInfo);
        }
        break;

      case 'DID_FAIL_TO_RENEW':
        if (transactionInfo) {
          await handleDidFailToRenew(payload, transactionInfo, renewalInfo);
        }
        break;

      case 'EXPIRED':
        if (transactionInfo) {
          await handleExpired(payload, transactionInfo, renewalInfo);
        }
        break;

      case 'GRACE_PERIOD_EXPIRED':
        if (transactionInfo) {
          await handleGracePeriodExpired(payload, transactionInfo, renewalInfo);
        }
        break;

      case 'REFUND':
        if (transactionInfo) {
          await handleRefund(payload, transactionInfo);
        }
        break;

      case 'REVOKE':
        if (transactionInfo) {
          await handleRevoke(payload, transactionInfo);
        }
        break;

      case 'OFFER_REDEEMED':
        if (transactionInfo) {
          await handleOfferRedeemed(payload, transactionInfo);
        }
        break;

      case 'TEST':
        logger.info({ notificationUUID: payload.notificationUUID }, 'Received test notification');
        break;

      default:
        logger.warn({ notificationType: payload.notificationType }, 'Unhandled notification type');
    }

    // Create audit log for all notifications
    await createAuditLog(payload, transactionInfo);

    // Mark as processed
    markNotificationProcessed(payload.notificationUUID);

    res.status(HTTP_STATUS.OK).json({ success: true });
  } catch (error) {
    logger.error({ error }, 'Failed to process App Store webhook');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({ error: 'Failed to process notification' });
  }
}

export default { handleAppStoreWebhook };
