/**
 * Subscription Controller
 *
 * Handles client-side subscription operations:
 * - Verify purchases from StoreKit 2
 * - Restore purchases
 * - Get subscription status
 */

import { Request, Response } from 'express';
import { HTTP_STATUS } from '../config/constants';
import { logger } from '../config/logger';
import * as subscriptionService from '../services/subscriptionService';
import { validateReceiptSchema, restorePurchasesSchema } from '../validation/subscriptionSchemas';
import { AuthenticatedRequest } from '../types';

/**
 * Verify a purchase from StoreKit 2 and create/update subscription
 * POST /api/subscription/verify
 */
export async function verifyPurchase(req: AuthenticatedRequest, res: Response): Promise<void> {
  try {
    const userId = req.userId;
    if (!userId) {
      res.status(HTTP_STATUS.UNAUTHORIZED).json({ error: 'Authentication required' });
      return;
    }

    const validation = validateReceiptSchema.safeParse(req.body);
    if (!validation.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid request',
        details: validation.error.errors,
      });
      return;
    }

    const { transactionId, originalTransactionId, productId, purchaseDate, environment } =
      validation.data;

    logger.info(
      {
        userId,
        transactionId,
        originalTransactionId,
        productId,
      },
      'Verifying purchase'
    );

    // Create or update the subscription
    const { subscription, isNew } = await subscriptionService.createOrUpdateSubscription({
      userId,
      originalTransactionId: originalTransactionId || transactionId,
      productId,
      purchaseDate: purchaseDate ? new Date(purchaseDate) : new Date(),
      expiresDate: null, // Will be updated by webhook or determined by product
      environment: environment || 'Production',
      autoRenewEnabled: true,
    });

    // Get the updated subscription status
    const status = await subscriptionService.getSubscriptionStatus(userId);

    res.status(isNew ? HTTP_STATUS.CREATED : HTTP_STATUS.OK).json({
      success: true,
      subscription: {
        id: subscription.id,
        productId: subscription.productId,
        status: subscription.status,
        expiresAt: subscription.expiresAt,
        isTrialPeriod: subscription.isTrialPeriod,
        autoRenewEnabled: subscription.autoRenewEnabled,
      },
      userStatus: status,
    });
  } catch (error) {
    logger.error({ error }, 'Failed to verify purchase');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to verify purchase',
    });
  }
}

/**
 * Restore purchases for a user
 * POST /api/subscription/restore
 */
export async function restorePurchases(req: AuthenticatedRequest, res: Response): Promise<void> {
  try {
    const userId = req.userId;
    if (!userId) {
      res.status(HTTP_STATUS.UNAUTHORIZED).json({ error: 'Authentication required' });
      return;
    }

    const validation = restorePurchasesSchema.safeParse(req.body);
    if (!validation.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid request',
        details: validation.error.errors,
      });
      return;
    }

    const { transactionIds } = validation.data;

    logger.info(
      {
        userId,
        transactionCount: transactionIds.length,
      },
      'Restoring purchases'
    );

    const result = await subscriptionService.restorePurchases(userId, transactionIds);

    // Get the updated subscription status
    const status = await subscriptionService.getSubscriptionStatus(userId);

    res.status(HTTP_STATUS.OK).json({
      success: true,
      restored: result.restored,
      alreadyActive: result.alreadyActive,
      errors: result.errors,
      userStatus: status,
    });
  } catch (error) {
    logger.error({ error }, 'Failed to restore purchases');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to restore purchases',
    });
  }
}

/**
 * Get subscription status for the authenticated user
 * GET /api/subscription/status
 */
export async function getSubscriptionStatus(
  req: AuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    const userId = req.userId;
    if (!userId) {
      res.status(HTTP_STATUS.UNAUTHORIZED).json({ error: 'Authentication required' });
      return;
    }

    const status = await subscriptionService.getSubscriptionStatus(userId);

    res.status(HTTP_STATUS.OK).json({
      success: true,
      ...status,
    });
  } catch (error) {
    logger.error({ error }, 'Failed to get subscription status');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to get subscription status',
    });
  }
}

/**
 * Get available products (static for now, could be fetched from App Store Connect API)
 * GET /api/subscription/products
 */
export async function getProducts(_req: Request, res: Response): Promise<void> {
  try {
    const products = [
      {
        id: subscriptionService.PRODUCT_IDS.PRO_MONTHLY,
        name: 'Nutri Pro Monthly',
        description: 'Unlock all premium features with a monthly subscription',
        tier: 'PRO',
        billingCycle: 'MONTHLY',
        features: [
          'Unlimited meal logging',
          'Advanced nutrition insights',
          'AI-powered predictions',
          'Priority support',
        ],
      },
      {
        id: subscriptionService.PRODUCT_IDS.PRO_YEARLY,
        name: 'Nutri Pro Yearly',
        description: 'Unlock all premium features with an annual subscription (save 17%)',
        tier: 'PRO',
        billingCycle: 'ANNUAL',
        features: [
          'Unlimited meal logging',
          'Advanced nutrition insights',
          'AI-powered predictions',
          'Priority support',
          '2 months free compared to monthly',
        ],
      },
    ];

    res.status(HTTP_STATUS.OK).json({
      success: true,
      products,
    });
  } catch (error) {
    logger.error({ error }, 'Failed to get products');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to get products',
    });
  }
}

export default {
  verifyPurchase,
  restorePurchases,
  getSubscriptionStatus,
  getProducts,
};
