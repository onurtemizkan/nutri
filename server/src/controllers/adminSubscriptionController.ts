import { Response } from 'express';
import {
  getSubscriptionList,
  getSubscriptionDetail,
  lookupByTransactionId,
  grantSubscription,
  extendSubscription,
  revokeSubscription,
} from '../services/adminSubscriptionService';
import {
  userIdParamSchema,
  grantSubscriptionSchema,
  extendSubscriptionSchema,
  revokeSubscriptionSchema,
  lookupSubscriptionQuerySchema,
} from '../validation/adminSchemas';
import { z } from 'zod';
import { logger } from '../config/logger';
import { HTTP_STATUS } from '../config/constants';
import { AdminAuthenticatedRequest } from '../types';

// Query schema for listing subscriptions
const listSubscriptionsQuerySchema = z.object({
  status: z.enum(['active', 'trial', 'expired', 'none']).optional(),
  page: z.coerce.number().int().positive().default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
});

/**
 * GET /api/admin/subscriptions
 * List users with subscriptions
 */
export async function listSubscriptions(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    const parseResult = listSubscriptionsQuerySchema.safeParse(req.query);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid query parameters',
        details: parseResult.error.errors,
      });
      return;
    }

    const result = await getSubscriptionList(parseResult.data);

    res.status(HTTP_STATUS.OK).json(result);
  } catch (error) {
    logger.error({ error }, 'Error listing subscriptions');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to list subscriptions',
    });
  }
}

/**
 * GET /api/admin/subscriptions/:id
 * Get detailed subscription information for a user
 */
export async function getSubscription(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    const parseResult = userIdParamSchema.safeParse(req.params);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid user ID',
        details: parseResult.error.errors,
      });
      return;
    }

    const { id } = parseResult.data;
    const subscription = await getSubscriptionDetail(id);

    if (!subscription) {
      res.status(HTTP_STATUS.NOT_FOUND).json({
        error: 'User not found',
      });
      return;
    }

    res.status(HTTP_STATUS.OK).json(subscription);
  } catch (error) {
    logger.error({ error }, 'Error getting subscription detail');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to get subscription details',
    });
  }
}

/**
 * GET /api/admin/subscriptions/lookup
 * Lookup subscription by Apple transaction ID
 */
export async function lookupSubscription(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    const parseResult = lookupSubscriptionQuerySchema.safeParse(req.query);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Transaction ID required',
        details: parseResult.error.errors,
      });
      return;
    }

    const { txn } = parseResult.data;
    const subscription = await lookupByTransactionId(txn);

    if (!subscription) {
      res.status(HTTP_STATUS.NOT_FOUND).json({
        error: 'No subscription found for this transaction ID',
      });
      return;
    }

    res.status(HTTP_STATUS.OK).json(subscription);
  } catch (error) {
    logger.error({ error }, 'Error looking up subscription');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to lookup subscription',
    });
  }
}

/**
 * POST /api/admin/subscriptions/:id/grant
 * Manually grant Pro subscription to a user
 * Requires SUPER_ADMIN role
 */
export async function grantUserSubscription(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    // Validate ID parameter
    const idParseResult = userIdParamSchema.safeParse(req.params);
    if (!idParseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid user ID',
        details: idParseResult.error.errors,
      });
      return;
    }

    // Validate body
    const bodyParseResult = grantSubscriptionSchema.safeParse(req.body);
    if (!bodyParseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid request body',
        details: bodyParseResult.error.errors,
      });
      return;
    }

    const { id } = idParseResult.data;
    const { duration, reason } = bodyParseResult.data;
    const adminUserId = req.adminUser?.id || 'unknown';

    const result = await grantSubscription(id, duration, reason, adminUserId);

    if (!result) {
      res.status(HTTP_STATUS.NOT_FOUND).json({
        error: 'User not found',
      });
      return;
    }

    res.status(HTTP_STATUS.OK).json({
      message: 'Subscription granted successfully',
      subscription: result,
    });
  } catch (error) {
    logger.error({ error }, 'Error granting subscription');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to grant subscription',
    });
  }
}

/**
 * POST /api/admin/subscriptions/:id/extend
 * Extend a user's subscription
 * Requires SUPER_ADMIN role
 */
export async function extendUserSubscription(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    // Validate ID parameter
    const idParseResult = userIdParamSchema.safeParse(req.params);
    if (!idParseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid user ID',
        details: idParseResult.error.errors,
      });
      return;
    }

    // Validate body
    const bodyParseResult = extendSubscriptionSchema.safeParse(req.body);
    if (!bodyParseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid request body',
        details: bodyParseResult.error.errors,
      });
      return;
    }

    const { id } = idParseResult.data;
    const { days, reason } = bodyParseResult.data;
    const adminUserId = req.adminUser?.id || 'unknown';

    const result = await extendSubscription(id, days, reason, adminUserId);

    if (!result) {
      res.status(HTTP_STATUS.NOT_FOUND).json({
        error: 'User not found',
      });
      return;
    }

    res.status(HTTP_STATUS.OK).json({
      message: 'Subscription extended successfully',
      subscription: result,
    });
  } catch (error) {
    logger.error({ error }, 'Error extending subscription');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to extend subscription',
    });
  }
}

/**
 * POST /api/admin/subscriptions/:id/revoke
 * Revoke a user's subscription
 * Requires SUPER_ADMIN role
 */
export async function revokeUserSubscription(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    // Validate ID parameter
    const idParseResult = userIdParamSchema.safeParse(req.params);
    if (!idParseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid user ID',
        details: idParseResult.error.errors,
      });
      return;
    }

    // Validate body
    const bodyParseResult = revokeSubscriptionSchema.safeParse(req.body);
    if (!bodyParseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid request body',
        details: bodyParseResult.error.errors,
      });
      return;
    }

    const { id } = idParseResult.data;
    const { reason } = bodyParseResult.data;
    const adminUserId = req.adminUser?.id || 'unknown';

    const result = await revokeSubscription(id, reason, adminUserId);

    if (!result) {
      res.status(HTTP_STATUS.NOT_FOUND).json({
        error: 'User not found',
      });
      return;
    }

    res.status(HTTP_STATUS.OK).json({
      message: 'Subscription revoked successfully',
      subscription: result,
    });
  } catch (error) {
    logger.error({ error }, 'Error revoking subscription');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to revoke subscription',
    });
  }
}
