/**
 * @swagger
 * components:
 *   schemas:
 *     MLInsight:
 *       type: object
 *       properties:
 *         id:
 *           type: string
 *           example: "clx123abc"
 *         userId:
 *           type: string
 *         insightType:
 *           type: string
 *           enum: [CORRELATION, PREDICTION, ANOMALY, RECOMMENDATION, GOAL_PROGRESS, PATTERN_DETECTED]
 *         priority:
 *           type: string
 *           enum: [LOW, MEDIUM, HIGH, CRITICAL]
 *         title:
 *           type: string
 *           example: "Higher protein correlates with better HRV"
 *         description:
 *           type: string
 *           example: "Your data shows a moderate positive correlation..."
 *         recommendation:
 *           type: string
 *           example: "Consider maintaining your protein intake..."
 *         correlation:
 *           type: number
 *           nullable: true
 *           example: 0.68
 *         confidence:
 *           type: number
 *           example: 0.85
 *         dataPoints:
 *           type: integer
 *           example: 30
 *         metadata:
 *           type: object
 *         viewed:
 *           type: boolean
 *         viewedAt:
 *           type: string
 *           format: date-time
 *           nullable: true
 *         dismissed:
 *           type: boolean
 *         dismissedAt:
 *           type: string
 *           format: date-time
 *           nullable: true
 *         helpful:
 *           type: boolean
 *           nullable: true
 *         createdAt:
 *           type: string
 *           format: date-time
 *         expiresAt:
 *           type: string
 *           format: date-time
 *           nullable: true
 *
 *     InsightSummary:
 *       type: object
 *       properties:
 *         totalInsights:
 *           type: integer
 *           example: 12
 *         unviewedCount:
 *           type: integer
 *           example: 5
 *         highPriorityCount:
 *           type: integer
 *           example: 2
 *         byType:
 *           type: object
 *           properties:
 *             CORRELATION:
 *               type: integer
 *             PREDICTION:
 *               type: integer
 *             ANOMALY:
 *               type: integer
 *             RECOMMENDATION:
 *               type: integer
 *             GOAL_PROGRESS:
 *               type: integer
 *             PATTERN_DETECTED:
 *               type: integer
 *         lastGeneratedAt:
 *           type: string
 *           format: date-time
 *           nullable: true
 *
 *     GenerateInsightsRequest:
 *       type: object
 *       properties:
 *         targetMetrics:
 *           type: array
 *           items:
 *             type: string
 *           description: Health metrics to analyze (defaults to all)
 *           example: ["RESTING_HEART_RATE", "HEART_RATE_VARIABILITY_SDNN"]
 *         lookbackDays:
 *           type: integer
 *           minimum: 7
 *           maximum: 180
 *           default: 30
 *           description: Days of historical data to analyze
 *         regenerate:
 *           type: boolean
 *           default: false
 *           description: Clear existing non-dismissed insights before generating
 *
 *     GenerateInsightsResponse:
 *       type: object
 *       properties:
 *         message:
 *           type: string
 *           example: "Insight generation completed"
 *         generated:
 *           type: integer
 *           example: 8
 *         skipped:
 *           type: integer
 *           example: 2
 *         errors:
 *           type: array
 *           items:
 *             type: string
 *
 *     UpdateInsightRequest:
 *       type: object
 *       properties:
 *         viewed:
 *           type: boolean
 *         dismissed:
 *           type: boolean
 *         helpful:
 *           type: boolean
 *
 *     FeedbackRequest:
 *       type: object
 *       required:
 *         - helpful
 *       properties:
 *         helpful:
 *           type: boolean
 *           description: Whether the insight was helpful
 *         feedbackText:
 *           type: string
 *           maxLength: 500
 *           description: Optional feedback text
 */

import { Router } from 'express';
import { authenticate } from '../middleware/auth';
import { rateLimiters } from '../middleware/rateLimiter';
import {
  getInsights,
  getInsightsSummary,
  getInsightById,
  generateInsights,
  updateInsight,
  markInsightViewed,
  dismissInsight,
  provideInsightFeedback,
  cleanupInsights,
} from '../controllers/insightsController';

const router = Router();

// All routes require authentication
router.use(authenticate);

/**
 * @swagger
 * /api/insights:
 *   get:
 *     summary: Get user's ML insights
 *     description: Retrieves ML-generated insights with optional filtering by type, priority, and status
 *     tags: [Insights]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: insightType
 *         schema:
 *           type: string
 *           enum: [CORRELATION, PREDICTION, ANOMALY, RECOMMENDATION, GOAL_PROGRESS, PATTERN_DETECTED]
 *         description: Filter by insight type
 *       - in: query
 *         name: priority
 *         schema:
 *           type: string
 *           enum: [LOW, MEDIUM, HIGH, CRITICAL]
 *         description: Filter by priority level
 *       - in: query
 *         name: viewed
 *         schema:
 *           type: boolean
 *         description: Filter by viewed status
 *       - in: query
 *         name: dismissed
 *         schema:
 *           type: boolean
 *         description: Include dismissed insights (default false)
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 20
 *           maximum: 100
 *         description: Number of insights to return
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           default: 0
 *         description: Number of insights to skip
 *     responses:
 *       200:
 *         description: List of insights
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 insights:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/MLInsight'
 *                 total:
 *                   type: integer
 *                 limit:
 *                   type: integer
 *                 offset:
 *                   type: integer
 *                 hasMore:
 *                   type: boolean
 *       401:
 *         description: Unauthorized
 */
router.get('/', getInsights);

/**
 * @swagger
 * /api/insights/summary:
 *   get:
 *     summary: Get insights summary
 *     description: Get a summary of user's insights including counts by type and priority
 *     tags: [Insights]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Insights summary
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/InsightSummary'
 *       401:
 *         description: Unauthorized
 */
router.get('/summary', getInsightsSummary);

/**
 * @swagger
 * /api/insights/generate:
 *   post:
 *     summary: Generate new insights
 *     description: |
 *       Trigger ML insight generation by analyzing correlations between
 *       nutrition/activity patterns and health metrics.
 *
 *       This process:
 *       1. Analyzes correlations for specified health metrics
 *       2. Identifies significant patterns and relationships
 *       3. Generates human-readable insights with recommendations
 *       4. Stores insights in the database
 *
 *       Note: Requires at least 7 days of meal and health metric data.
 *     tags: [Insights]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/GenerateInsightsRequest'
 *     responses:
 *       200:
 *         description: Insight generation completed
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/GenerateInsightsResponse'
 *       401:
 *         description: Unauthorized
 */
router.post('/generate', rateLimiters.insightGeneration, generateInsights);

/**
 * @swagger
 * /api/insights/cleanup:
 *   delete:
 *     summary: Cleanup old insights
 *     description: Delete old or expired insights. Keeps insights marked as helpful.
 *     tags: [Insights]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: daysOld
 *         schema:
 *           type: integer
 *           default: 30
 *         description: Delete insights older than this many days
 *     responses:
 *       200:
 *         description: Cleanup completed
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 deleted:
 *                   type: integer
 *       401:
 *         description: Unauthorized
 */
router.delete('/cleanup', cleanupInsights);

/**
 * @swagger
 * /api/insights/{id}:
 *   get:
 *     summary: Get insight by ID
 *     description: Retrieve a specific insight by its ID
 *     tags: [Insights]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *         description: Insight ID
 *     responses:
 *       200:
 *         description: Insight details
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/MLInsight'
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Insight not found
 */
router.get('/:id', getInsightById);

/**
 * @swagger
 * /api/insights/{id}:
 *   patch:
 *     summary: Update insight
 *     description: Update insight properties (viewed, dismissed, helpful)
 *     tags: [Insights]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *         description: Insight ID
 *     requestBody:
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/UpdateInsightRequest'
 *     responses:
 *       200:
 *         description: Updated insight
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/MLInsight'
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Insight not found
 */
router.patch('/:id', updateInsight);

/**
 * @swagger
 * /api/insights/{id}/view:
 *   post:
 *     summary: Mark insight as viewed
 *     description: Mark an insight as viewed (updates viewedAt timestamp)
 *     tags: [Insights]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *         description: Insight ID
 *     responses:
 *       200:
 *         description: Insight marked as viewed
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/MLInsight'
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Insight not found
 */
router.post('/:id/view', markInsightViewed);

/**
 * @swagger
 * /api/insights/{id}/dismiss:
 *   post:
 *     summary: Dismiss insight
 *     description: Dismiss an insight so it no longer appears in the feed
 *     tags: [Insights]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *         description: Insight ID
 *     responses:
 *       200:
 *         description: Insight dismissed
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/MLInsight'
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Insight not found
 */
router.post('/:id/dismiss', dismissInsight);

/**
 * @swagger
 * /api/insights/{id}/feedback:
 *   post:
 *     summary: Provide feedback on insight
 *     description: Submit feedback about whether an insight was helpful
 *     tags: [Insights]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *         description: Insight ID
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/FeedbackRequest'
 *     responses:
 *       200:
 *         description: Feedback recorded
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/MLInsight'
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Insight not found
 */
router.post('/:id/feedback', provideInsightFeedback);

export default router;
