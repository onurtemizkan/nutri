import { Router } from 'express';
import { goalProgressController } from '../controllers/goalProgressController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All goal routes are protected
router.use(authenticate);

// Dashboard endpoint (optimized for home screen)
router.get('/dashboard', (req, res) => goalProgressController.getDashboardProgress(req, res));

// Today's progress
router.get('/progress/today', (req, res) => goalProgressController.getTodayProgress(req, res));

// Daily progress for a specific date
router.get('/progress/daily', (req, res) => goalProgressController.getDailyProgress(req, res));

// Historical progress for date range
router.get('/progress/history', (req, res) => goalProgressController.getProgressHistory(req, res));

// Streak information
router.get('/streak', (req, res) => goalProgressController.getStreak(req, res));

// Weekly summary
router.get('/summary/weekly', (req, res) => goalProgressController.getWeeklySummary(req, res));

// Monthly summary
router.get('/summary/monthly', (req, res) => goalProgressController.getMonthlySummary(req, res));

// Full historical data with summaries
router.get('/historical', (req, res) => goalProgressController.getHistoricalProgress(req, res));

export default router;
