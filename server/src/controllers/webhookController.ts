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
 */
async function handleSubscribed(
  payload: NotificationPayload,
  transactionInfo: TransactionInfo,
  _renewalInfo: RenewalInfo | null
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

  // TODO: Create or update Subscription record in database
  // await subscriptionService.createOrUpdate({
  //   originalTransactionId: transactionInfo.originalTransactionId,
  //   productId: transactionInfo.productId,
  //   status: 'active',
  //   purchaseDate: new Date(transactionInfo.purchaseDate),
  //   expiresDate: transactionInfo.expiresDate ? new Date(transactionInfo.expiresDate) : null,
  //   environment: transactionInfo.environment,
  //   autoRenewEnabled: renewalInfo?.autoRenewStatus === 1,
  // });
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

  // TODO: Update Subscription record with new expiry date
  // await subscriptionService.update(transactionInfo.originalTransactionId, {
  //   status: 'active',
  //   expiresDate: transactionInfo.expiresDate ? new Date(transactionInfo.expiresDate) : null,
  // });
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

  // TODO: Update auto-renew status in database
  // await subscriptionService.update(transactionInfo.originalTransactionId, {
  //   autoRenewEnabled,
  // });
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

  // TODO: Update subscription status
  // const status = renewalInfo?.gracePeriodExpiresDate ? 'billing_grace_period' : 'billing_retry';
  // await subscriptionService.update(transactionInfo.originalTransactionId, {
  //   status,
  //   gracePeriodExpiresDate: renewalInfo?.gracePeriodExpiresDate
  //     ? new Date(renewalInfo.gracePeriodExpiresDate)
  //     : null,
  // });
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

  // TODO: Mark subscription as expired
  // await subscriptionService.update(transactionInfo.originalTransactionId, {
  //   status: 'expired',
  // });
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

  // TODO: Mark subscription as expired after grace period
  // await subscriptionService.update(transactionInfo.originalTransactionId, {
  //   status: 'expired',
  //   gracePeriodExpiresDate: null,
  // });
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

  // TODO: Mark subscription as revoked due to refund
  // await subscriptionService.update(transactionInfo.originalTransactionId, {
  //   status: 'revoked',
  //   cancellationDate: new Date(),
  // });
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

  // TODO: Mark subscription as revoked
  // await subscriptionService.update(transactionInfo.originalTransactionId, {
  //   status: 'revoked',
  //   cancellationDate: transactionInfo.revocationDate
  //     ? new Date(transactionInfo.revocationDate)
  //     : new Date(),
  // });
}

/**
 * Handle OFFER_REDEEMED notification
 */
async function handleOfferRedeemed(
  _payload: NotificationPayload,
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

  // TODO: Log offer redemption event
  // await subscriptionEventService.create({
  //   originalTransactionId: transactionInfo.originalTransactionId,
  //   eventType: 'OFFER_REDEEMED',
  //   offerType: transactionInfo.offerType,
  //   offerIdentifier: transactionInfo.offerIdentifier,
  // });
}

/**
 * Handle DID_CHANGE_RENEWAL_PREF notification
 */
async function handleDidChangeRenewalPref(
  _payload: NotificationPayload,
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

  // TODO: Log the plan change (upgrade/downgrade takes effect at next renewal)
  // await subscriptionEventService.create({
  //   originalTransactionId: transactionInfo.originalTransactionId,
  //   eventType: 'RENEWAL_PREF_CHANGED',
  //   newProductId: renewalInfo?.autoRenewProductId,
  // });
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

  // TODO: Create SubscriptionEvent audit record
  // await prisma.subscriptionEvent.create({
  //   data: {
  //     notificationUUID: payload.notificationUUID,
  //     notificationType: payload.notificationType,
  //     subtype: payload.subtype,
  //     originalTransactionId: transactionInfo?.originalTransactionId,
  //     environment: payload.data.environment,
  //     rawPayload: JSON.stringify(payload),
  //     processedAt: new Date(),
  //   },
  // });
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
