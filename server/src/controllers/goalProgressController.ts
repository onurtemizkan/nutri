import { goalProgressService } from '../services/goalProgressService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import { withErrorHandling } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { parseOptionalDate } from '../utils/dateHelpers';

export class GoalProgressController {
  /**
   * Get today's goal progress
   * GET /api/goals/progress/today
   */
  getTodayProgress = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const progress = await goalProgressService.getDailyProgress(userId);
    res.status(HTTP_STATUS.OK).json(progress);
  });

  /**
   * Get goal progress for a specific date
   * GET /api/goals/progress/daily?date=2024-01-15
   */
  getDailyProgress = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const date = parseOptionalDate(req.query.date as string | undefined);
    const progress = await goalProgressService.getDailyProgress(userId, date);

    res.status(HTTP_STATUS.OK).json(progress);
  });

  /**
   * Get goal progress history for a date range
   * GET /api/goals/progress/history?startDate=2024-01-01&endDate=2024-01-31&days=30
   */
  getProgressHistory = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const days = req.query.days ? parseInt(req.query.days as string, 10) : 30;
    const startDate = req.query.startDate
      ? new Date(req.query.startDate as string)
      : new Date(Date.now() - days * 24 * 60 * 60 * 1000);
    const endDate = req.query.endDate ? new Date(req.query.endDate as string) : new Date();

    const progress = await goalProgressService.getProgressHistory(userId, startDate, endDate);

    res.status(HTTP_STATUS.OK).json(progress);
  });

  /**
   * Get current streak information
   * GET /api/goals/streak
   */
  getStreak = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const streak = await goalProgressService.getStreak(userId);
    res.status(HTTP_STATUS.OK).json(streak);
  });

  /**
   * Get weekly summary with trend
   * GET /api/goals/summary/weekly?offset=0
   */
  getWeeklySummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const offset = req.query.offset ? parseInt(req.query.offset as string, 10) : 0;
    const summary = await goalProgressService.getWeeklySummary(userId, offset);

    res.status(HTTP_STATUS.OK).json(summary);
  });

  /**
   * Get monthly summary
   * GET /api/goals/summary/monthly?offset=0
   */
  getMonthlySummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const offset = req.query.offset ? parseInt(req.query.offset as string, 10) : 0;
    const summary = await goalProgressService.getMonthlySummary(userId, offset);

    res.status(HTTP_STATUS.OK).json(summary);
  });

  /**
   * Get dashboard progress data (today + streak + weekly trends)
   * GET /api/goals/dashboard
   */
  getDashboardProgress = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const dashboardData = await goalProgressService.getDashboardProgress(userId);
    res.status(HTTP_STATUS.OK).json(dashboardData);
  });

  /**
   * Get full historical progress with all summaries
   * GET /api/goals/historical?days=30
   */
  getHistoricalProgress = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const days = req.query.days ? parseInt(req.query.days as string, 10) : 30;
    const historicalData = await goalProgressService.getHistoricalProgress(userId, days);

    res.status(HTTP_STATUS.OK).json(historicalData);
  });
}

export const goalProgressController = new GoalProgressController();
