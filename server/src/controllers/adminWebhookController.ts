import { Response } from 'express';
import {
  getWebhookList,
  getWebhookDetail,
  searchWebhooksByTransactionId,
  retryWebhookEvent,
  getWebhookStats,
} from '../services/adminWebhookService';
import { z } from 'zod';
import { logger } from '../config/logger';
import { HTTP_STATUS } from '../config/constants';
import { AdminAuthenticatedRequest } from '../types';

// Query schemas
const listWebhooksQuerySchema = z.object({
  notificationType: z.string().optional(),
  status: z.enum(['PENDING', 'SUCCESS', 'FAILED']).optional(),
  startDate: z
    .string()
    .optional()
    .transform((val) => (val ? new Date(val) : undefined)),
  endDate: z
    .string()
    .optional()
    .transform((val) => (val ? new Date(val) : undefined)),
  originalTransactionId: z.string().optional(),
  userId: z.string().optional(),
  page: z.coerce.number().int().positive().default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
});

const webhookIdParamSchema = z.object({
  id: z.string().min(1),
});

const searchWebhooksQuerySchema = z.object({
  txn: z.string().min(1),
});

/**
 * GET /api/admin/webhooks
 * List webhook events with filters and pagination
 */
export async function listWebhooks(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    const parseResult = listWebhooksQuerySchema.safeParse(req.query);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid query parameters',
        details: parseResult.error.errors,
      });
      return;
    }

    const result = await getWebhookList(parseResult.data);

    res.status(HTTP_STATUS.OK).json(result);
  } catch (error) {
    logger.error({ error }, 'Error listing webhooks');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to list webhook events',
    });
  }
}

/**
 * GET /api/admin/webhooks/stats
 * Get webhook event statistics
 */
export async function getWebhookStatistics(
  _req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    const stats = await getWebhookStats();

    res.status(HTTP_STATUS.OK).json(stats);
  } catch (error) {
    logger.error({ error }, 'Error getting webhook stats');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to get webhook statistics',
    });
  }
}

/**
 * GET /api/admin/webhooks/search
 * Search webhooks by transaction ID
 */
export async function searchWebhooks(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    const parseResult = searchWebhooksQuerySchema.safeParse(req.query);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Transaction ID required',
        details: parseResult.error.errors,
      });
      return;
    }

    const { txn } = parseResult.data;
    const events = await searchWebhooksByTransactionId(txn);

    res.status(HTTP_STATUS.OK).json({ events });
  } catch (error) {
    logger.error({ error }, 'Error searching webhooks');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to search webhook events',
    });
  }
}

/**
 * GET /api/admin/webhooks/:id
 * Get detailed webhook event information
 */
export async function getWebhook(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    const parseResult = webhookIdParamSchema.safeParse(req.params);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid webhook ID',
        details: parseResult.error.errors,
      });
      return;
    }

    const { id } = parseResult.data;
    const event = await getWebhookDetail(id);

    if (!event) {
      res.status(HTTP_STATUS.NOT_FOUND).json({
        error: 'Webhook event not found',
      });
      return;
    }

    res.status(HTTP_STATUS.OK).json(event);
  } catch (error) {
    logger.error({ error }, 'Error getting webhook detail');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to get webhook event details',
    });
  }
}

/**
 * POST /api/admin/webhooks/:id/retry
 * Retry processing a failed webhook event
 * Requires SUPER_ADMIN role
 */
export async function retryWebhook(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    const parseResult = webhookIdParamSchema.safeParse(req.params);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid webhook ID',
        details: parseResult.error.errors,
      });
      return;
    }

    const { id } = parseResult.data;
    const adminUserId = req.adminUser?.id || 'unknown';

    const result = await retryWebhookEvent(id, adminUserId);

    if (!result) {
      res.status(HTTP_STATUS.NOT_FOUND).json({
        error: 'Webhook event not found',
      });
      return;
    }

    res.status(HTTP_STATUS.OK).json({
      message: 'Webhook retry initiated',
      event: result,
    });
  } catch (error) {
    logger.error({ error }, 'Error retrying webhook');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to retry webhook event',
    });
  }
}
