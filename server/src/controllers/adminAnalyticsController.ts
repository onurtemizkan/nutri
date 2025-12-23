import { Response } from 'express';
import {
  getSubscriptionMetrics,
  getSubscribersOverTime,
  getRevenueOverTime,
} from '../services/adminAnalyticsService';
import { z } from 'zod';
import { logger } from '../config/logger';
import { HTTP_STATUS } from '../config/constants';
import { AdminAuthenticatedRequest } from '../types';

// Query schemas
const subscribersTimeSeriesSchema = z.object({
  days: z.coerce.number().int().min(7).max(365).default(30),
});

const revenueTimeSeriesSchema = z.object({
  months: z.coerce.number().int().min(1).max(24).default(12),
});

/**
 * GET /api/admin/analytics/overview
 * Get comprehensive subscription analytics
 */
export async function getAnalyticsOverview(
  _req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    const metrics = await getSubscriptionMetrics();

    res.status(HTTP_STATUS.OK).json(metrics);
  } catch (error) {
    logger.error({ error }, 'Error getting analytics overview');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to get analytics',
    });
  }
}

/**
 * GET /api/admin/analytics/subscribers-over-time
 * Get daily subscriber counts for the last N days
 */
export async function getSubscribersTimeSeries(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    const parseResult = subscribersTimeSeriesSchema.safeParse(req.query);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid query parameters',
        details: parseResult.error.errors,
      });
      return;
    }

    const { days } = parseResult.data;
    const data = await getSubscribersOverTime(days);

    res.status(HTTP_STATUS.OK).json(data);
  } catch (error) {
    logger.error({ error }, 'Error getting subscribers time series');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to get subscriber data',
    });
  }
}

/**
 * GET /api/admin/analytics/revenue-over-time
 * Get monthly revenue for the last N months
 */
export async function getRevenueTimeSeries(
  req: AdminAuthenticatedRequest,
  res: Response
): Promise<void> {
  try {
    const parseResult = revenueTimeSeriesSchema.safeParse(req.query);
    if (!parseResult.success) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid query parameters',
        details: parseResult.error.errors,
      });
      return;
    }

    const { months } = parseResult.data;
    const data = await getRevenueOverTime(months);

    res.status(HTTP_STATUS.OK).json(data);
  } catch (error) {
    logger.error({ error }, 'Error getting revenue time series');
    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to get revenue data',
    });
  }
}
