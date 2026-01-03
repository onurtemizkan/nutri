/**
 * Subscription Guard Middleware
 *
 * Middleware for protecting routes that require an active Pro subscription.
 * Works with the authentication middleware to check subscription status.
 */

import { Response, NextFunction } from 'express';
import { AuthenticatedRequest } from '../types';
import { HTTP_STATUS } from '../config/constants';
import { logger } from '../config/logger';
import * as subscriptionService from '../services/subscriptionService';

/** Subscription tier types */
type SubscriptionTier = 'FREE' | 'PRO_TRIAL' | 'PRO';

/**
 * Create subscription guard middleware
 *
 * @param options Configuration options
 * @returns Express middleware function
 *
 * @example
 * ```ts
 * // Require any Pro subscription
 * router.get('/premium-feature', authenticate, requireSubscription(), handler);
 *
 * // Require specific tier
 * router.get('/pro-only', authenticate, requireSubscription({ tier: 'PRO' }), handler);
 *
 * // Allow trial users
 * router.get('/with-trial', authenticate, requireSubscription({ allowTrial: true }), handler);
 * ```
 */
export function requireSubscription(options?: {
  /** Required subscription tier (defaults to PRO or PRO_TRIAL) */
  tier?: SubscriptionTier;
  /** Allow trial subscriptions (defaults to true) */
  allowTrial?: boolean;
  /** Custom error message */
  message?: string;
}) {
  const { tier, allowTrial = true, message } = options ?? {};

  return async (req: AuthenticatedRequest, res: Response, next: NextFunction): Promise<void> => {
    try {
      const userId = req.userId;

      if (!userId) {
        res.status(HTTP_STATUS.UNAUTHORIZED).json({
          error: 'Authentication required',
          code: 'AUTH_REQUIRED',
        });
        return;
      }

      // Get user's subscription status
      const status = await subscriptionService.getSubscriptionStatus(userId);

      // Check if subscription is active
      if (!status.isActive) {
        logger.info(
          { userId, tier: status.tier, status: status.status },
          'Subscription required but not active'
        );

        res.status(HTTP_STATUS.FORBIDDEN).json({
          error: message ?? 'This feature requires an active Pro subscription',
          code: 'SUBSCRIPTION_REQUIRED',
          currentTier: status.tier,
          upgradeUrl: '/paywall',
        });
        return;
      }

      // Check tier requirements
      const userTier = status.tier;

      // If specific tier required
      if (tier) {
        if (userTier !== tier) {
          logger.info(
            { userId, requiredTier: tier, currentTier: userTier },
            'Subscription tier insufficient'
          );

          res.status(HTTP_STATUS.FORBIDDEN).json({
            error: message ?? `This feature requires ${tier} subscription`,
            code: 'TIER_INSUFFICIENT',
            requiredTier: tier,
            currentTier: userTier,
            upgradeUrl: '/paywall',
          });
          return;
        }
      } else {
        // Default: require PRO or PRO_TRIAL
        const isPro = userTier === 'PRO';
        const isTrial = userTier === 'PRO_TRIAL';

        if (!isPro && !(allowTrial && isTrial)) {
          logger.info(
            { userId, currentTier: userTier, allowTrial },
            'Subscription tier insufficient'
          );

          res.status(HTTP_STATUS.FORBIDDEN).json({
            error: message ?? 'This feature requires a Pro subscription',
            code: 'SUBSCRIPTION_REQUIRED',
            currentTier: userTier,
            upgradeUrl: '/paywall',
          });
          return;
        }
      }

      // Subscription is valid, proceed
      logger.debug({ userId, tier: userTier }, 'Subscription check passed');

      next();
    } catch (error) {
      logger.error({ error, userId: req.userId }, 'Error checking subscription');

      // On error, deny access (fail closed)
      res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
        error: 'Unable to verify subscription status',
        code: 'SUBSCRIPTION_CHECK_FAILED',
      });
    }
  };
}

/**
 * Middleware to attach subscription info to request
 * Useful for routes that need to know subscription status but don't require it
 */
export async function attachSubscriptionInfo(
  req: AuthenticatedRequest,
  _res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const userId = req.userId;

    if (userId) {
      const status = await subscriptionService.getSubscriptionStatus(userId);
      (req as AuthenticatedRequest & { subscriptionStatus?: typeof status }).subscriptionStatus =
        status;
    }

    next();
  } catch (error) {
    logger.warn({ error, userId: req.userId }, 'Error attaching subscription info');
    // Don't fail the request, just continue without subscription info
    next();
  }
}

/**
 * Check if a feature is accessible based on tier
 */
export function checkFeatureAccess(
  userTier: SubscriptionTier,
  requiredTier: SubscriptionTier
): boolean {
  const tierOrder: Record<SubscriptionTier, number> = {
    FREE: 0,
    PRO_TRIAL: 1,
    PRO: 2,
  };

  return tierOrder[userTier] >= tierOrder[requiredTier];
}

export default {
  requireSubscription,
  attachSubscriptionInfo,
  checkFeatureAccess,
};
