/**
 * ML Insights Controller
 *
 * Handles HTTP requests for ML-generated insights.
 */

import { Response, NextFunction } from 'express';
import { z } from 'zod';
import { insightsService } from '../services/insightsService';
import { AuthenticatedRequest } from '../types';
import { HTTP_STATUS } from '../config/constants';
import { logger } from '../config/logger';

// ============================================================================
// Validation Schemas
// ============================================================================

const insightTypeSchema = z.enum([
  'CORRELATION',
  'PREDICTION',
  'ANOMALY',
  'RECOMMENDATION',
  'GOAL_PROGRESS',
  'PATTERN_DETECTED',
]);

const insightPrioritySchema = z.enum(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']);

export const getInsightsQuerySchema = z.object({
  insightType: insightTypeSchema.optional(),
  priority: insightPrioritySchema.optional(),
  viewed: z
    .string()
    .transform((val) => val === 'true')
    .optional(),
  dismissed: z
    .string()
    .transform((val) => val === 'true')
    .optional(),
  limit: z
    .string()
    .transform((val) => parseInt(val, 10))
    .pipe(z.number().int().positive().max(100))
    .optional(),
  offset: z
    .string()
    .transform((val) => parseInt(val, 10))
    .pipe(z.number().int().nonnegative())
    .optional(),
});

export const generateInsightsSchema = z.object({
  targetMetrics: z.array(z.string()).optional(),
  lookbackDays: z.number().int().min(7).max(180).optional(),
  regenerate: z.boolean().optional(),
});

export const updateInsightSchema = z.object({
  viewed: z.boolean().optional(),
  dismissed: z.boolean().optional(),
  helpful: z.boolean().optional(),
});

export const feedbackSchema = z.object({
  helpful: z.boolean(),
  feedbackText: z.string().max(500).optional(),
});

// ============================================================================
// Controller Methods
// ============================================================================

/**
 * Get user's insights with filtering
 * GET /api/insights
 */
export const getInsights = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const userId = req.userId!;
    const query = getInsightsQuerySchema.parse(req.query);

    const result = await insightsService.getInsights(userId, query);

    res.status(HTTP_STATUS.OK).json(result);
  } catch (error) {
    next(error);
  }
};

/**
 * Get insights summary (counts, by type, etc.)
 * GET /api/insights/summary
 */
export const getInsightsSummary = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const userId = req.userId!;
    const summary = await insightsService.getInsightsSummary(userId);

    res.status(HTTP_STATUS.OK).json(summary);
  } catch (error) {
    next(error);
  }
};

/**
 * Get a single insight by ID
 * GET /api/insights/:id
 */
export const getInsightById = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const userId = req.userId!;
    const { id } = req.params;

    const insight = await insightsService.getInsightById(userId, id);

    res.status(HTTP_STATUS.OK).json(insight);
  } catch (error) {
    if (error instanceof Error && error.message === 'Insight not found') {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Insight not found' });
      return;
    }
    next(error);
  }
};

/**
 * Generate new insights for the user
 * POST /api/insights/generate
 */
export const generateInsights = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const userId = req.userId!;
    const options = generateInsightsSchema.parse(req.body);

    logger.info({ userId, options }, 'Starting insight generation');

    const result = await insightsService.generateInsights(userId, options);

    res.status(HTTP_STATUS.OK).json({
      message: 'Insight generation completed',
      ...result,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Update an insight (mark as viewed, dismissed, or helpful)
 * PATCH /api/insights/:id
 */
export const updateInsight = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const userId = req.userId!;
    const { id } = req.params;
    const data = updateInsightSchema.parse(req.body);

    const insight = await insightsService.updateInsight(userId, id, data);

    res.status(HTTP_STATUS.OK).json(insight);
  } catch (error) {
    if (error instanceof Error && error.message === 'Insight not found') {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Insight not found' });
      return;
    }
    next(error);
  }
};

/**
 * Mark an insight as viewed
 * POST /api/insights/:id/view
 */
export const markInsightViewed = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const userId = req.userId!;
    const { id } = req.params;

    const insight = await insightsService.markAsViewed(userId, id);

    res.status(HTTP_STATUS.OK).json(insight);
  } catch (error) {
    if (error instanceof Error && error.message === 'Insight not found') {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Insight not found' });
      return;
    }
    next(error);
  }
};

/**
 * Dismiss an insight
 * POST /api/insights/:id/dismiss
 */
export const dismissInsight = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const userId = req.userId!;
    const { id } = req.params;

    const insight = await insightsService.dismissInsight(userId, id);

    res.status(HTTP_STATUS.OK).json(insight);
  } catch (error) {
    if (error instanceof Error && error.message === 'Insight not found') {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Insight not found' });
      return;
    }
    next(error);
  }
};

/**
 * Provide feedback on an insight
 * POST /api/insights/:id/feedback
 */
export const provideInsightFeedback = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const userId = req.userId!;
    const { id } = req.params;
    const { helpful, feedbackText } = feedbackSchema.parse(req.body);

    logger.info(
      { userId, insightId: id, helpful, hasFeedbackText: !!feedbackText },
      'Insight feedback received'
    );

    const insight = await insightsService.provideFeedback(userId, id, helpful);

    res.status(HTTP_STATUS.OK).json(insight);
  } catch (error) {
    if (error instanceof Error && error.message === 'Insight not found') {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Insight not found' });
      return;
    }
    next(error);
  }
};

/**
 * Cleanup old insights
 * DELETE /api/insights/cleanup
 */
export const cleanupInsights = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const userId = req.userId!;
    const daysOld = req.query.daysOld ? parseInt(req.query.daysOld as string, 10) : 30;

    const result = await insightsService.cleanupOldInsights(userId, daysOld);

    res.status(HTTP_STATUS.OK).json({
      message: `Cleaned up ${result.deleted} old insights`,
      ...result,
    });
  } catch (error) {
    next(error);
  }
};
