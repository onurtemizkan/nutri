/**
 * Food Feedback Controller
 *
 * Handles food classification feedback endpoints:
 * - Submit feedback when classification is incorrect
 * - Get feedback statistics
 * - Get feedback list (admin)
 * - Update feedback status (admin)
 */

import { Request } from 'express';
import { foodFeedbackService } from '../services/foodFeedbackService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import { withErrorHandling, ErrorHandlers } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import {
  submitFoodFeedbackSchema,
  feedbackListQuerySchema,
  updateFeedbackStatusSchema,
  feedbackIdParamSchema,
} from '../validation/schemas';

export class FoodFeedbackController {
  /**
   * Submit feedback for food classification
   * POST /api/foods/feedback
   */
  submitFeedback = withErrorHandling<Request>(async (req, res) => {
    const validated = submitFoodFeedbackSchema.parse(req.body);

    // Get userId if authenticated (optional)
    let userId: string | undefined;
    const authReq = req as AuthenticatedRequest;
    if (authReq.userId) {
      userId = authReq.userId;
    }

    const result = await foodFeedbackService.submitFeedback({
      userId,
      imageHash: validated.imageHash,
      classificationId: validated.classificationId,
      originalPrediction: validated.originalPrediction,
      originalConfidence: validated.originalConfidence,
      originalCategory: validated.originalCategory,
      selectedFdcId: validated.selectedFdcId,
      selectedFoodName: validated.selectedFoodName,
      wasCorrect: validated.wasCorrect,
      classificationHints: validated.classificationHints as Record<string, unknown>,
      userDescription: validated.userDescription,
    });

    if (result.isDuplicate) {
      res.status(HTTP_STATUS.OK).json({
        success: true,
        message: 'Feedback already recorded',
        isDuplicate: true,
      });
      return;
    }

    res.status(HTTP_STATUS.CREATED).json({
      success: true,
      feedbackId: result.feedbackId,
      patternFlagged: result.patternFlagged,
      message: result.patternFlagged
        ? 'Feedback recorded. This pattern has been flagged for review.'
        : 'Feedback recorded successfully.',
    });
  });

  /**
   * Get feedback statistics
   * GET /api/foods/feedback/stats
   */
  getStats = withErrorHandling<Request>(async (_req, res) => {
    const stats = await foodFeedbackService.getStats();
    res.status(HTTP_STATUS.OK).json(stats);
  });

  /**
   * Get patterns needing review (for learning loop)
   * GET /api/foods/feedback/patterns-needing-review
   */
  getPatternsNeedingReview = withErrorHandling<Request>(async (_req, res) => {
    const patterns = await foodFeedbackService.getPatternsNeedingReview();
    res.status(HTTP_STATUS.OK).json({
      patterns,
      count: patterns.length,
    });
  });

  /**
   * Get misclassifications for a specific food
   * GET /api/foods/feedback/misclassifications/:foodKey
   */
  getMisclassificationsForFood = withErrorHandling<Request>(async (req, res) => {
    const { foodKey } = req.params;

    if (!foodKey || foodKey.length < 2) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Food key must be at least 2 characters',
      });
      return;
    }

    const patterns = await foodFeedbackService.getMisclassificationsForFood(
      foodKey
    );

    res.status(HTTP_STATUS.OK).json({
      foodKey,
      patterns,
      count: patterns.length,
    });
  });

  /**
   * Get feedback list with pagination (admin)
   * GET /api/foods/feedback/list?page=1&limit=20&status=pending
   */
  getFeedbackList = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validated = feedbackListQuerySchema.parse(req.query);

    const result = await foodFeedbackService.getFeedbackList({
      page: validated.page,
      limit: validated.limit,
      status: validated.status,
    });

    res.status(HTTP_STATUS.OK).json(result);
  });

  /**
   * Update feedback status (admin)
   * PATCH /api/foods/feedback/:feedbackId/status
   */
  updateFeedbackStatus = ErrorHandlers.withNotFound<AuthenticatedRequest>(
    async (req, res) => {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const { feedbackId } = feedbackIdParamSchema.parse(req.params);
      const { status } = updateFeedbackStatusSchema.parse(req.body);

      await foodFeedbackService.updateFeedbackStatus(feedbackId, status);

      res.status(HTTP_STATUS.OK).json({
        success: true,
        message: `Feedback status updated to ${status}`,
      });
    }
  );

  /**
   * Mark pattern as reviewed
   * POST /api/foods/feedback/patterns/:patternId/review
   */
  markPatternReviewed = ErrorHandlers.withNotFound<AuthenticatedRequest>(
    async (req, res) => {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const { patternId } = req.params;

      if (!patternId) {
        res.status(HTTP_STATUS.BAD_REQUEST).json({
          error: 'Pattern ID is required',
        });
        return;
      }

      await foodFeedbackService.markPatternReviewed(patternId);

      res.status(HTTP_STATUS.OK).json({
        success: true,
        message: 'Pattern marked as reviewed',
      });
    }
  );

  /**
   * Sync approved feedback to ML service
   * POST /api/foods/feedback/sync
   */
  syncToMLService = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const result = await foodFeedbackService.syncFeedbackToMLService();

    res.status(HTTP_STATUS.OK).json({
      success: true,
      synced: result.synced,
      failed: result.failed,
      message: `Synced ${result.synced} feedback items to ML service`,
    });
  });
}

export const foodFeedbackController = new FoodFeedbackController();
