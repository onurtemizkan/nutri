import { Router } from 'express';
import { gamificationController } from '../controllers/gamificationController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All gamification routes are protected
router.use(authenticate);

// ============================================================================
// SUMMARY
// ============================================================================

/**
 * @swagger
 * /api/gamification/summary:
 *   get:
 *     summary: Get gamification summary (streak, XP, recent achievements)
 *     tags: [Gamification]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Gamification summary
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 streak:
 *                   type: object
 *                   properties:
 *                     currentStreak:
 *                       type: integer
 *                     longestStreak:
 *                       type: integer
 *                     isActive:
 *                       type: boolean
 *                 xp:
 *                   type: object
 *                   properties:
 *                     totalXP:
 *                       type: integer
 *                     currentLevel:
 *                       type: integer
 *                     progressPercent:
 *                       type: integer
 *                 recentAchievements:
 *                   type: array
 *                   items:
 *                     type: object
 *                 unseenCount:
 *                   type: integer
 */
router.get('/summary', (req, res) => gamificationController.getSummary(req, res));

// ============================================================================
// STREAKS
// ============================================================================

/**
 * @swagger
 * /api/gamification/streak:
 *   get:
 *     summary: Get user's streak statistics
 *     tags: [Gamification]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Streak statistics
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 currentStreak:
 *                   type: integer
 *                 longestStreak:
 *                   type: integer
 *                 mealStreak:
 *                   type: integer
 *                 waterStreak:
 *                   type: integer
 *                 exerciseStreak:
 *                   type: integer
 *                 freezeTokens:
 *                   type: integer
 *                 isActive:
 *                   type: boolean
 */
router.get('/streak', (req, res) => gamificationController.getStreak(req, res));

// ============================================================================
// XP & LEVELS
// ============================================================================

/**
 * @swagger
 * /api/gamification/xp:
 *   get:
 *     summary: Get user's XP and level statistics
 *     tags: [Gamification]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: XP statistics
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 totalXP:
 *                   type: integer
 *                 currentLevel:
 *                   type: integer
 *                 xpToNextLevel:
 *                   type: integer
 *                 progressPercent:
 *                   type: integer
 *                 weeklyXP:
 *                   type: integer
 *                 monthlyXP:
 *                   type: integer
 */
router.get('/xp', (req, res) => gamificationController.getXP(req, res));

/**
 * @swagger
 * /api/gamification/xp/history:
 *   get:
 *     summary: Get user's XP transaction history
 *     tags: [Gamification]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 20
 *     responses:
 *       200:
 *         description: XP transaction history
 */
router.get('/xp/history', (req, res) => gamificationController.getXPHistory(req, res));

// ============================================================================
// ACHIEVEMENTS
// ============================================================================

/**
 * @swagger
 * /api/gamification/achievements:
 *   get:
 *     summary: Get all achievements with user progress
 *     tags: [Gamification]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: category
 *         schema:
 *           type: string
 *           enum: [STREAK, LOGGING, NUTRITION, EXERCISE, MILESTONE, SOCIAL, SPECIAL]
 *       - in: query
 *         name: includeHidden
 *         schema:
 *           type: boolean
 *           default: false
 *     responses:
 *       200:
 *         description: List of achievements with progress
 */
router.get('/achievements', (req, res) => gamificationController.getAchievements(req, res));

/**
 * @swagger
 * /api/gamification/achievements/unseen:
 *   get:
 *     summary: Get newly earned achievements that haven't been seen
 *     tags: [Gamification]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: List of unseen achievements
 */
router.get('/achievements/unseen', (req, res) =>
  gamificationController.getUnseenAchievements(req, res)
);

/**
 * @swagger
 * /api/gamification/achievements/seen:
 *   post:
 *     summary: Mark achievements as seen
 *     tags: [Gamification]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - achievementIds
 *             properties:
 *               achievementIds:
 *                 type: array
 *                 items:
 *                   type: string
 *     responses:
 *       200:
 *         description: Achievements marked as seen
 */
router.post('/achievements/seen', (req, res) =>
  gamificationController.markAchievementsSeen(req, res)
);

// ============================================================================
// LEADERBOARDS
// ============================================================================

/**
 * @swagger
 * /api/gamification/leaderboard:
 *   get:
 *     summary: Get the leaderboard
 *     tags: [Gamification]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: type
 *         schema:
 *           type: string
 *           enum: [weekly, monthly, allTime]
 *           default: weekly
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 10
 *           maximum: 100
 *     responses:
 *       200:
 *         description: Leaderboard entries
 */
router.get('/leaderboard', (req, res) => gamificationController.getLeaderboard(req, res));

/**
 * @swagger
 * /api/gamification/rank:
 *   get:
 *     summary: Get the current user's rank
 *     tags: [Gamification]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: type
 *         schema:
 *           type: string
 *           enum: [weekly, monthly, allTime]
 *           default: weekly
 *     responses:
 *       200:
 *         description: User's rank and XP
 */
router.get('/rank', (req, res) => gamificationController.getUserRank(req, res));

export default router;
