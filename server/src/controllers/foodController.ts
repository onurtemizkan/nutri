/**
 * Food Controller
 *
 * Handles USDA FoodData Central API endpoints:
 * - Food search
 * - Get food by ID
 * - Get nutrients
 * - Popular foods
 */

import { Request } from 'express';
import { foodDatabaseService } from '../services/foodDatabaseService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import {
  withErrorHandling,
  ErrorHandlers,
} from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import {
  foodSearchQuerySchema,
  fdcIdParamSchema,
  bulkFdcIdsSchema,
} from '../validation/schemas';
import { USDAApiError } from '../types/usda';

export class FoodController {

  /**
   * Search for foods
   * GET /api/foods/search?q=query&page=1&limit=25&dataType=Foundation,Branded
   */
  searchFoods = withErrorHandling<Request>(async (req, res) => {
    const validated = foodSearchQuerySchema.parse(req.query);

    const result = await foodDatabaseService.searchFoods({
      query: validated.q,
      page: validated.page,
      limit: validated.limit,
      dataType: validated.dataType,
      sortBy: validated.sortBy,
      sortOrder: validated.sortOrder,
      brandOwner: validated.brandOwner,
    });

    res.status(HTTP_STATUS.OK).json(result);
  });

  /**
   * Get a single food by FDC ID
   * GET /api/foods/:fdcId
   */
  getFoodById = ErrorHandlers.withNotFound<Request>(async (req, res) => {
    const { fdcId } = fdcIdParamSchema.parse(req.params);

    try {
      const food = await foodDatabaseService.getFoodById(fdcId);
      res.status(HTTP_STATUS.OK).json(food);
    } catch (error) {
      if (error instanceof USDAApiError && error.statusCode === 404) {
        res.status(HTTP_STATUS.NOT_FOUND).json({
          error: `Food with FDC ID ${fdcId} not found`,
        });
        return;
      }
      throw error;
    }
  });

  /**
   * Get nutrients for a food
   * GET /api/foods/:fdcId/nutrients
   */
  getFoodNutrients = ErrorHandlers.withNotFound<Request>(async (req, res) => {
    const { fdcId } = fdcIdParamSchema.parse(req.params);

    try {
      const nutrients = await foodDatabaseService.getFoodNutrients(fdcId);

      res.status(HTTP_STATUS.OK).json({
        fdcId,
        nutrients,
        per: '100g',
      });
    } catch (error) {
      if (error instanceof USDAApiError && error.statusCode === 404) {
        res.status(HTTP_STATUS.NOT_FOUND).json({
          error: `Food with FDC ID ${fdcId} not found`,
        });
        return;
      }
      throw error;
    }
  });

  /**
   * Get scaled nutrients for a food (for specific serving size)
   * GET /api/foods/:fdcId/nutrients?grams=150
   */
  getScaledNutrients = ErrorHandlers.withNotFound<Request>(async (req, res) => {
    const { fdcId } = fdcIdParamSchema.parse(req.params);
    const grams = parseFloat(req.query.grams as string) || 100;

    if (grams <= 0 || grams > 10000) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Grams must be between 0 and 10000',
      });
      return;
    }

    try {
      const nutrients = await foodDatabaseService.getScaledNutrients(
        fdcId,
        grams
      );

      res.status(HTTP_STATUS.OK).json({
        fdcId,
        nutrients,
        servingSize: grams,
        servingSizeUnit: 'g',
      });
    } catch (error) {
      if (error instanceof USDAApiError && error.statusCode === 404) {
        res.status(HTTP_STATUS.NOT_FOUND).json({
          error: `Food with FDC ID ${fdcId} not found`,
        });
        return;
      }
      throw error;
    }
  });

  /**
   * Get multiple foods by IDs
   * POST /api/foods/bulk
   */
  getBulkFoods = withErrorHandling<Request>(async (req, res) => {
    const { fdcIds } = bulkFdcIdsSchema.parse(req.body);

    const foods = await foodDatabaseService.getFoodsByIds(fdcIds);

    res.status(HTTP_STATUS.OK).json({
      foods,
      requested: fdcIds.length,
      returned: foods.length,
    });
  });

  /**
   * Get popular foods
   * GET /api/foods/popular
   */
  getPopularFoods = withErrorHandling<Request>(async (_req, res) => {
    const foods = await foodDatabaseService.getPopularFoods();

    res.status(HTTP_STATUS.OK).json({
      foods,
      count: foods.length,
    });
  });

  /**
   * Get recent foods for a user (requires auth)
   * GET /api/foods/recent
   */
  getRecentFoods = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { foodCacheService } = await import('../services/foodCacheService');

    // Get recent food IDs from cache
    const recentFdcIds = await foodCacheService.getUserRecentFoods(userId);

    if (recentFdcIds.length === 0) {
      res.status(HTTP_STATUS.OK).json({
        foods: [],
        count: 0,
      });
      return;
    }

    // Fetch full food details
    const foods = await foodDatabaseService.getFoodsByIds(recentFdcIds);

    res.status(HTTP_STATUS.OK).json({
      foods,
      count: foods.length,
    });
  });

  /**
   * Health check for USDA API
   * GET /api/foods/health
   */
  healthCheck = withErrorHandling<Request>(async (_req, res) => {
    const { redisHealthCheck } = await import('../config/redis');
    const { foodCacheService } = await import('../services/foodCacheService');

    const [usdaHealth, redisHealth] = await Promise.all([
      foodDatabaseService.checkHealth(),
      redisHealthCheck(),
    ]);

    const rateLimit = foodDatabaseService.getRateLimitStatus();
    const cacheStats = foodCacheService.getStats();

    const overallHealthy = usdaHealth.healthy;

    res.status(overallHealthy ? HTTP_STATUS.OK : 503).json({
      usda: usdaHealth,
      redis: redisHealth,
      rateLimit,
      cacheStats,
    });
  });

  /**
   * Record a food selection for the user (adds to recent foods)
   * POST /api/foods/:fdcId/select
   */
  recordFoodSelection = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { fdcId } = fdcIdParamSchema.parse(req.params);

    const { foodCacheService } = await import('../services/foodCacheService');

    // Add to user's recent foods
    await foodCacheService.addUserRecentFood(userId, fdcId);

    res.status(HTTP_STATUS.OK).json({
      success: true,
      message: 'Food selection recorded',
    });
  });
}

// Export singleton instance
export const foodController = new FoodController();
