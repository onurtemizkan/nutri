/**
 * Food Feedback Service
 *
 * Handles:
 * 1. Storing user corrections when food classification is wrong
 * 2. Aggregating feedback patterns for learning loop
 * 3. Providing stats for analytics
 * 4. Integration with ML service feedback_service.py
 */

import prisma from '../config/database';
import { foodCacheService } from './foodCacheService';
import { Prisma, FoodFeedbackAggregation } from '@prisma/client';

// Threshold for flagging patterns that need model review
const CORRECTION_THRESHOLD = 10;

// Redis cache keys
const FEEDBACK_STATS_KEY = 'food:feedback:stats';
const FEEDBACK_STATS_TTL = 3600; // 1 hour

export interface FeedbackSubmission {
  userId?: string;
  imageHash: string;
  classificationId?: string;
  originalPrediction: string;
  originalConfidence: number;
  originalCategory?: string;
  selectedFdcId: number;
  selectedFoodName: string;
  wasCorrect: boolean;
  classificationHints?: Record<string, unknown>;
  userDescription?: string;
}

export interface FeedbackStats {
  totalFeedback: number;
  pendingFeedback: number;
  approvedFeedback: number;
  rejectedFeedback: number;
  topMisclassifications: Array<{
    original: string;
    corrected: string;
    count: number;
  }>;
  problemFoods: Array<{
    food: string;
    correctionCount: number;
    avgConfidence: number;
  }>;
  patternsNeedingReview: number;
}

export interface AggregatedPattern {
  id: string;
  originalPrediction: string;
  correctedFood: string;
  correctionCount: number;
  avgConfidence: number;
  needsReview: boolean;
  firstOccurrence: Date;
  lastOccurrence: Date;
}

class FoodFeedbackService {
  /**
   * Submit feedback when a food classification is incorrect
   */
  async submitFeedback(data: FeedbackSubmission): Promise<{
    feedbackId: string;
    isDuplicate: boolean;
    patternFlagged: boolean;
  }> {
    // Normalize the prediction string
    const normalizedPrediction = data.originalPrediction
      .toLowerCase()
      .trim()
      .replace(/\s+/g, '_');
    const normalizedFoodName = data.selectedFoodName
      .toLowerCase()
      .trim()
      .replace(/\s+/g, '_');

    try {
      // Create feedback record
      const feedback = await prisma.foodFeedback.create({
        data: {
          userId: data.userId,
          imageHash: data.imageHash,
          classificationId: data.classificationId,
          originalPrediction: normalizedPrediction,
          originalConfidence: data.originalConfidence,
          originalCategory: data.originalCategory,
          selectedFdcId: data.selectedFdcId,
          selectedFoodName: normalizedFoodName,
          wasCorrect: data.wasCorrect,
          classificationHints: data.classificationHints as Prisma.InputJsonValue,
          userDescription: data.userDescription,
          status: 'pending',
        },
      });

      // Update aggregation only if it was a correction (not a confirmation)
      let patternFlagged = false;
      if (!data.wasCorrect) {
        patternFlagged = await this.updateAggregation(
          normalizedPrediction,
          normalizedFoodName,
          data.originalConfidence
        );
      }

      // Invalidate stats cache
      await this.invalidateStatsCache();

      return {
        feedbackId: feedback.id,
        isDuplicate: false,
        patternFlagged,
      };
    } catch (error) {
      // Check for unique constraint violation (duplicate)
      if (
        error instanceof Prisma.PrismaClientKnownRequestError &&
        error.code === 'P2002'
      ) {
        return {
          feedbackId: '',
          isDuplicate: true,
          patternFlagged: false,
        };
      }
      throw error;
    }
  }

  /**
   * Update aggregation record for a correction pattern
   */
  private async updateAggregation(
    originalPrediction: string,
    correctedFood: string,
    confidence: number
  ): Promise<boolean> {
    const existing = await prisma.foodFeedbackAggregation.findUnique({
      where: {
        originalPrediction_correctedFood: {
          originalPrediction,
          correctedFood,
        },
      },
    });

    if (existing) {
      // Update existing aggregation
      const newCount = existing.correctionCount + 1;
      const newAvgConfidence =
        (existing.avgConfidence * existing.correctionCount + confidence) /
        newCount;
      const needsReview = newCount >= CORRECTION_THRESHOLD && !existing.needsReview;

      await prisma.foodFeedbackAggregation.update({
        where: { id: existing.id },
        data: {
          correctionCount: newCount,
          avgConfidence: newAvgConfidence,
          lastOccurrence: new Date(),
          needsReview: needsReview || existing.needsReview,
        },
      });

      return needsReview;
    } else {
      // Create new aggregation
      await prisma.foodFeedbackAggregation.create({
        data: {
          originalPrediction,
          correctedFood,
          correctionCount: 1,
          avgConfidence: confidence,
          firstOccurrence: new Date(),
          lastOccurrence: new Date(),
          needsReview: false,
        },
      });

      return false;
    }
  }

  /**
   * Get feedback statistics
   */
  async getStats(): Promise<FeedbackStats> {
    // Try cache first
    const cached = await foodCacheService.getCachedValue<FeedbackStats>(
      FEEDBACK_STATS_KEY
    );
    if (cached) {
      return cached;
    }

    // Get status counts
    const statusCounts = await prisma.foodFeedback.groupBy({
      by: ['status'],
      _count: { id: true },
    });

    const countsByStatus: Record<string, number> = {};
    for (const item of statusCounts) {
      countsByStatus[item.status] = item._count.id;
    }

    const totalFeedback = Object.values(countsByStatus).reduce(
      (sum, count) => sum + count,
      0
    );

    // Get top misclassifications from aggregation table
    const topMisclassifications = await prisma.foodFeedbackAggregation.findMany(
      {
        where: { correctionCount: { gt: 0 } },
        orderBy: { correctionCount: 'desc' },
        take: 10,
      }
    );

    // Get problem foods (most corrections needed)
    const problemFoods = await prisma.foodFeedback.groupBy({
      by: ['originalPrediction'],
      where: { wasCorrect: false },
      _count: { id: true },
      _avg: { originalConfidence: true },
      orderBy: { _count: { id: 'desc' } },
      take: 10,
    });

    // Get count of patterns needing review
    const patternsNeedingReview = await prisma.foodFeedbackAggregation.count({
      where: { needsReview: true, reviewedAt: null },
    });

    const stats: FeedbackStats = {
      totalFeedback,
      pendingFeedback: countsByStatus['pending'] || 0,
      approvedFeedback: countsByStatus['approved'] || 0,
      rejectedFeedback: countsByStatus['rejected'] || 0,
      topMisclassifications: topMisclassifications.map((item: FoodFeedbackAggregation) => ({
        original: item.originalPrediction,
        corrected: item.correctedFood,
        count: item.correctionCount,
      })),
      problemFoods: problemFoods.map((item: { originalPrediction: string; _count: { id: number }; _avg: { originalConfidence: number | null } }) => ({
        food: item.originalPrediction,
        correctionCount: item._count.id,
        avgConfidence: Math.round((item._avg.originalConfidence || 0) * 100) / 100,
      })),
      patternsNeedingReview,
    };

    // Cache the stats
    await foodCacheService.cacheValue(FEEDBACK_STATS_KEY, stats, FEEDBACK_STATS_TTL);

    return stats;
  }

  /**
   * Get patterns that need review (threshold exceeded)
   */
  async getPatternsNeedingReview(limit = 20): Promise<AggregatedPattern[]> {
    const patterns = await prisma.foodFeedbackAggregation.findMany({
      where: {
        needsReview: true,
        reviewedAt: null,
      },
      orderBy: { correctionCount: 'desc' },
      take: limit,
    });

    return patterns;
  }

  /**
   * Get top misclassifications for a specific food
   */
  async getMisclassificationsForFood(
    foodKey: string,
    limit = 5
  ): Promise<AggregatedPattern[]> {
    const normalizedKey = foodKey.toLowerCase().trim().replace(/\s+/g, '_');

    const patterns = await prisma.foodFeedbackAggregation.findMany({
      where: {
        OR: [
          { originalPrediction: normalizedKey },
          { correctedFood: normalizedKey },
        ],
      },
      orderBy: { correctionCount: 'desc' },
      take: limit,
    });

    return patterns;
  }

  /**
   * Mark a pattern as reviewed
   */
  async markPatternReviewed(patternId: string): Promise<void> {
    await prisma.foodFeedbackAggregation.update({
      where: { id: patternId },
      data: {
        reviewedAt: new Date(),
      },
    });
  }

  /**
   * Update feedback status (for admin approval workflow)
   */
  async updateFeedbackStatus(
    feedbackId: string,
    status: 'pending' | 'approved' | 'rejected' | 'applied'
  ): Promise<void> {
    await prisma.foodFeedback.update({
      where: { id: feedbackId },
      data: { status },
    });

    // Invalidate stats cache
    await this.invalidateStatsCache();
  }

  /**
   * Get feedback list with pagination
   */
  async getFeedbackList(options: {
    page?: number;
    limit?: number;
    status?: string;
    userId?: string;
  }): Promise<{
    items: Array<{
      id: string;
      imageHash: string;
      originalPrediction: string;
      originalConfidence: number;
      selectedFdcId: number;
      selectedFoodName: string;
      wasCorrect: boolean;
      status: string;
      createdAt: Date;
    }>;
    total: number;
    page: number;
    totalPages: number;
  }> {
    const page = options.page || 1;
    const limit = Math.min(options.limit || 20, 100);
    const skip = (page - 1) * limit;

    const where: Prisma.FoodFeedbackWhereInput = {};
    if (options.status) {
      where.status = options.status;
    }
    if (options.userId) {
      where.userId = options.userId;
    }

    const [items, total] = await Promise.all([
      prisma.foodFeedback.findMany({
        where,
        orderBy: { createdAt: 'desc' },
        skip,
        take: limit,
        select: {
          id: true,
          imageHash: true,
          originalPrediction: true,
          originalConfidence: true,
          selectedFdcId: true,
          selectedFoodName: true,
          wasCorrect: true,
          status: true,
          createdAt: true,
        },
      }),
      prisma.foodFeedback.count({ where }),
    ]);

    return {
      items,
      total,
      page,
      totalPages: Math.ceil(total / limit),
    };
  }

  /**
   * Sync feedback to ML service (for learning loop)
   */
  async syncFeedbackToMLService(): Promise<{
    synced: number;
    failed: number;
  }> {
    // Get pending feedback that hasn't been synced
    const pendingFeedback = await prisma.foodFeedback.findMany({
      where: {
        status: 'approved',
      },
      take: 100, // Batch size
    });

    let synced = 0;
    let failed = 0;

    for (const feedback of pendingFeedback) {
      try {
        // Call ML service feedback endpoint
        const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://localhost:8000';
        const response = await fetch(`${mlServiceUrl}/api/food/feedback`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image_hash: feedback.imageHash,
            original_prediction: feedback.originalPrediction,
            original_confidence: feedback.originalConfidence,
            corrected_label: feedback.selectedFoodName,
            user_description: feedback.userDescription,
            user_id: feedback.userId,
          }),
        });

        if (response.ok) {
          // Mark as applied
          await this.updateFeedbackStatus(feedback.id, 'applied');
          synced++;
        } else {
          failed++;
        }
      } catch {
        failed++;
      }
    }

    return { synced, failed };
  }

  /**
   * Invalidate stats cache
   */
  private async invalidateStatsCache(): Promise<void> {
    try {
      await foodCacheService.deleteCache(FEEDBACK_STATS_KEY);
    } catch {
      // Ignore cache errors
    }
  }
}

export const foodFeedbackService = new FoodFeedbackService();
