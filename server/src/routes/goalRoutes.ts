import { Router } from 'express';
import { goalProgressController } from '../controllers/goalProgressController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All goal routes are protected
router.use(authenticate);

/**
 * @swagger
 * /goals/dashboard:
 *   get:
 *     summary: Get dashboard progress (optimized for home screen)
 *     tags: [Goals]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Dashboard progress data
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 nutrition:
 *                   type: object
 *                   properties:
 *                     calories:
 *                       type: object
 *                       properties:
 *                         current:
 *                           type: number
 *                         goal:
 *                           type: number
 *                         progress:
 *                           type: number
 *                     protein:
 *                       type: object
 *                     carbs:
 *                       type: object
 *                     fat:
 *                       type: object
 *                 water:
 *                   type: object
 *                   properties:
 *                     current:
 *                       type: integer
 *                     goal:
 *                       type: integer
 *                     progress:
 *                       type: number
 *                 streak:
 *                   type: object
 *                   properties:
 *                     current:
 *                       type: integer
 *                     longest:
 *                       type: integer
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/dashboard', (req, res) => goalProgressController.getDashboardProgress(req, res));

/**
 * @swagger
 * /goals/progress/today:
 *   get:
 *     summary: Get today's goal progress
 *     tags: [Goals]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Today's progress
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 date:
 *                   type: string
 *                   format: date
 *                 calories:
 *                   type: object
 *                   properties:
 *                     consumed:
 *                       type: number
 *                     goal:
 *                       type: number
 *                     remaining:
 *                       type: number
 *                     progress:
 *                       type: number
 *                 macros:
 *                   type: object
 *                   properties:
 *                     protein:
 *                       type: object
 *                     carbs:
 *                       type: object
 *                     fat:
 *                       type: object
 *                     fiber:
 *                       type: object
 *                 water:
 *                   type: object
 *                 goalsMet:
 *                   type: array
 *                   items:
 *                     type: string
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/progress/today', (req, res) => goalProgressController.getTodayProgress(req, res));

/**
 * @swagger
 * /goals/progress/daily:
 *   get:
 *     summary: Get daily progress for a specific date
 *     tags: [Goals]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - $ref: '#/components/parameters/dateParam'
 *     responses:
 *       200:
 *         description: Daily progress
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/progress/daily', (req, res) => goalProgressController.getDailyProgress(req, res));

/**
 * @swagger
 * /goals/progress/history:
 *   get:
 *     summary: Get historical progress for date range
 *     tags: [Goals]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - name: startDate
 *         in: query
 *         required: true
 *         schema:
 *           type: string
 *           format: date
 *       - name: endDate
 *         in: query
 *         required: true
 *         schema:
 *           type: string
 *           format: date
 *     responses:
 *       200:
 *         description: Historical progress data
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 startDate:
 *                   type: string
 *                   format: date
 *                 endDate:
 *                   type: string
 *                   format: date
 *                 days:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       date:
 *                         type: string
 *                         format: date
 *                       calories:
 *                         type: number
 *                       goalMet:
 *                         type: boolean
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/progress/history', (req, res) => goalProgressController.getProgressHistory(req, res));

/**
 * @swagger
 * /goals/streak:
 *   get:
 *     summary: Get streak information
 *     tags: [Goals]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Streak data
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 currentStreak:
 *                   type: integer
 *                   description: Current consecutive days meeting goals
 *                 longestStreak:
 *                   type: integer
 *                   description: Longest streak ever
 *                 lastActiveDate:
 *                   type: string
 *                   format: date
 *                 totalDaysActive:
 *                   type: integer
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/streak', (req, res) => goalProgressController.getStreak(req, res));

/**
 * @swagger
 * /goals/summary/weekly:
 *   get:
 *     summary: Get weekly goal summary
 *     tags: [Goals]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - name: startDate
 *         in: query
 *         schema:
 *           type: string
 *           format: date
 *     responses:
 *       200:
 *         description: Weekly summary
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 week:
 *                   type: string
 *                 daysTracked:
 *                   type: integer
 *                 daysMetCalorieGoal:
 *                   type: integer
 *                 averageCalories:
 *                   type: number
 *                 averageProtein:
 *                   type: number
 *                 averageCarbs:
 *                   type: number
 *                 averageFat:
 *                   type: number
 *                 totalWater:
 *                   type: integer
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/summary/weekly', (req, res) => goalProgressController.getWeeklySummary(req, res));

/**
 * @swagger
 * /goals/summary/monthly:
 *   get:
 *     summary: Get monthly goal summary
 *     tags: [Goals]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - name: month
 *         in: query
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 12
 *       - name: year
 *         in: query
 *         schema:
 *           type: integer
 *     responses:
 *       200:
 *         description: Monthly summary
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 month:
 *                   type: integer
 *                 year:
 *                   type: integer
 *                 daysTracked:
 *                   type: integer
 *                 daysMetGoal:
 *                   type: integer
 *                 successRate:
 *                   type: number
 *                 averageCalories:
 *                   type: number
 *                 weeklyBreakdown:
 *                   type: array
 *                   items:
 *                     type: object
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/summary/monthly', (req, res) => goalProgressController.getMonthlySummary(req, res));

/**
 * @swagger
 * /goals/historical:
 *   get:
 *     summary: Get full historical data with summaries
 *     tags: [Goals]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - name: period
 *         in: query
 *         schema:
 *           type: string
 *           enum: [week, month, quarter, year, all]
 *           default: month
 *     responses:
 *       200:
 *         description: Historical data with summaries
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 period:
 *                   type: string
 *                 summary:
 *                   type: object
 *                 data:
 *                   type: array
 *                   items:
 *                     type: object
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/historical', (req, res) => goalProgressController.getHistoricalProgress(req, res));

export default router;
