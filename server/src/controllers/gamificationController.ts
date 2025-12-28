import { gamificationService } from '../services/gamificationService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import { withErrorHandling } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { z } from 'zod';
import { AchievementCategory } from '@prisma/client';

// Validation schemas
const markSeenSchema = z.object({
  achievementIds: z.array(z.string().min(1)),
});

const leaderboardQuerySchema = z.object({
  type: z.enum(['weekly', 'monthly', 'allTime']).optional().default('weekly'),
  limit: z.coerce.number().int().min(1).max(100).optional().default(10),
});

const achievementsQuerySchema = z.object({
  category: z.nativeEnum(AchievementCategory).optional(),
  includeHidden: z.coerce.boolean().optional().default(false),
});

export class GamificationController {
  // ============================================================================
  // SUMMARY
  // ============================================================================

  getSummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const summary = await gamificationService.getGamificationSummary(userId);

    res.status(HTTP_STATUS.OK).json(summary);
  });

  // ============================================================================
  // STREAKS
  // ============================================================================

  getStreak = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const streak = await gamificationService.getStreakStats(userId);

    res.status(HTTP_STATUS.OK).json(streak);
  });

  // ============================================================================
  // XP & LEVELS
  // ============================================================================

  getXP = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const xp = await gamificationService.getXPStats(userId);

    res.status(HTTP_STATUS.OK).json(xp);
  });

  getXPHistory = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const limit = req.query.limit ? parseInt(req.query.limit as string, 10) : 20;
    const history = await gamificationService.getXPHistory(userId, limit);

    res.status(HTTP_STATUS.OK).json(history);
  });

  // ============================================================================
  // ACHIEVEMENTS
  // ============================================================================

  getAchievements = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { category, includeHidden } = achievementsQuerySchema.parse(req.query);
    const achievements = await gamificationService.getAchievements(userId, {
      category,
      includeHidden,
    });

    res.status(HTTP_STATUS.OK).json(achievements);
  });

  getUnseenAchievements = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const unseen = await gamificationService.getUnseenAchievements(userId);

    res.status(HTTP_STATUS.OK).json(unseen);
  });

  markAchievementsSeen = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { achievementIds } = markSeenSchema.parse(req.body);
    await gamificationService.markAchievementsSeen(userId, achievementIds);

    res.status(HTTP_STATUS.OK).json({ message: 'Achievements marked as seen' });
  });

  // ============================================================================
  // LEADERBOARDS
  // ============================================================================

  getLeaderboard = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { type, limit } = leaderboardQuerySchema.parse(req.query);
    const leaderboard = await gamificationService.getLeaderboard(type, limit);

    res.status(HTTP_STATUS.OK).json(leaderboard);
  });

  getUserRank = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { type } = leaderboardQuerySchema.parse(req.query);
    const rank = await gamificationService.getUserRank(userId, type);

    res.status(HTTP_STATUS.OK).json(rank);
  });
}

export const gamificationController = new GamificationController();
