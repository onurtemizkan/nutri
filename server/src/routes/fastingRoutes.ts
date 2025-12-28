import { Router } from 'express';
import { fastingController } from '../controllers/fastingController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All fasting routes are protected
router.use(authenticate);

// ============================================================================
// PROTOCOLS
// ============================================================================

/**
 * @swagger
 * /api/fasting/protocols:
 *   get:
 *     summary: Get available fasting protocols
 *     tags: [Fasting]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: List of system and custom protocols
 */
router.get('/protocols', (req, res) => fastingController.getProtocols(req, res));

/**
 * @swagger
 * /api/fasting/protocols:
 *   post:
 *     summary: Create a custom fasting protocol
 *     tags: [Fasting]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - name
 *               - fastingHours
 *               - eatingHours
 *             properties:
 *               name:
 *                 type: string
 *               description:
 *                 type: string
 *               fastingHours:
 *                 type: integer
 *                 minimum: 1
 *                 maximum: 48
 *               eatingHours:
 *                 type: integer
 *                 minimum: 0
 *                 maximum: 24
 *               icon:
 *                 type: string
 *               color:
 *                 type: string
 *     responses:
 *       201:
 *         description: Protocol created successfully
 */
router.post('/protocols', (req, res) => fastingController.createProtocol(req, res));

/**
 * @swagger
 * /api/fasting/protocols/{id}:
 *   delete:
 *     summary: Delete a custom fasting protocol
 *     tags: [Fasting]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Protocol deleted
 */
router.delete('/protocols/:id', (req, res) => fastingController.deleteProtocol(req, res));

// ============================================================================
// SETTINGS
// ============================================================================

/**
 * @swagger
 * /api/fasting/settings:
 *   get:
 *     summary: Get user's fasting settings
 *     tags: [Fasting]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: User's fasting settings
 */
router.get('/settings', (req, res) => fastingController.getSettings(req, res));

/**
 * @swagger
 * /api/fasting/settings:
 *   put:
 *     summary: Update user's fasting settings
 *     tags: [Fasting]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               activeProtocolId:
 *                 type: string
 *               preferredStartTime:
 *                 type: string
 *                 pattern: '^([01]\d|2[0-3]):([0-5]\d)$'
 *               preferredEndTime:
 *                 type: string
 *                 pattern: '^([01]\d|2[0-3]):([0-5]\d)$'
 *               notifyOnFastStart:
 *                 type: boolean
 *               notifyOnFastEnd:
 *                 type: boolean
 *               notifyBeforeEnd:
 *                 type: integer
 *               weeklyFastingGoal:
 *                 type: integer
 *               monthlyFastingGoal:
 *                 type: integer
 *               showOnDashboard:
 *                 type: boolean
 *     responses:
 *       200:
 *         description: Settings updated
 */
router.put('/settings', (req, res) => fastingController.updateSettings(req, res));

// ============================================================================
// TIMER
// ============================================================================

/**
 * @swagger
 * /api/fasting/timer:
 *   get:
 *     summary: Get current fasting timer status
 *     tags: [Fasting]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Timer status with progress
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 isActive:
 *                   type: boolean
 *                 session:
 *                   type: object
 *                 progress:
 *                   type: object
 *                   properties:
 *                     elapsedMinutes:
 *                       type: integer
 *                     remainingMinutes:
 *                       type: integer
 *                     progressPercent:
 *                       type: integer
 */
router.get('/timer', (req, res) => fastingController.getTimerStatus(req, res));

// ============================================================================
// SESSIONS
// ============================================================================

/**
 * @swagger
 * /api/fasting/sessions:
 *   post:
 *     summary: Start a new fasting session
 *     tags: [Fasting]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               protocolId:
 *                 type: string
 *               customDuration:
 *                 type: integer
 *                 description: Duration in minutes (60-2880)
 *               startTime:
 *                 type: string
 *                 format: date-time
 *               notes:
 *                 type: string
 *     responses:
 *       201:
 *         description: Fasting session started
 */
router.post('/sessions', (req, res) => fastingController.startFast(req, res));

/**
 * @swagger
 * /api/fasting/sessions:
 *   get:
 *     summary: Get fasting history
 *     tags: [Fasting]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 20
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           default: 0
 *       - in: query
 *         name: status
 *         schema:
 *           type: string
 *           enum: [ACTIVE, COMPLETED, BROKEN, CANCELLED]
 *     responses:
 *       200:
 *         description: Paginated fasting history
 */
router.get('/sessions', (req, res) => fastingController.getHistory(req, res));

/**
 * @swagger
 * /api/fasting/sessions/{id}/end:
 *   post:
 *     summary: End a fasting session
 *     tags: [Fasting]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *     requestBody:
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               moodRating:
 *                 type: integer
 *                 minimum: 1
 *                 maximum: 5
 *               energyLevel:
 *                 type: integer
 *                 minimum: 1
 *                 maximum: 5
 *               hungerLevel:
 *                 type: integer
 *                 minimum: 1
 *                 maximum: 5
 *               notes:
 *                 type: string
 *               breakFastReason:
 *                 type: string
 *               endWeight:
 *                 type: number
 *     responses:
 *       200:
 *         description: Fasting session ended
 */
router.post('/sessions/:id/end', (req, res) => fastingController.endFast(req, res));

/**
 * @swagger
 * /api/fasting/sessions/{id}/checkpoint:
 *   post:
 *     summary: Add a checkpoint during fasting
 *     tags: [Fasting]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *     requestBody:
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               moodRating:
 *                 type: integer
 *                 minimum: 1
 *                 maximum: 5
 *               energyLevel:
 *                 type: integer
 *                 minimum: 1
 *                 maximum: 5
 *               hungerLevel:
 *                 type: integer
 *                 minimum: 1
 *                 maximum: 5
 *               notes:
 *                 type: string
 *     responses:
 *       201:
 *         description: Checkpoint added
 */
router.post('/sessions/:id/checkpoint', (req, res) => fastingController.addCheckpoint(req, res));

// ============================================================================
// STREAKS & ANALYTICS
// ============================================================================

/**
 * @swagger
 * /api/fasting/streak:
 *   get:
 *     summary: Get fasting streak statistics
 *     tags: [Fasting]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Fasting streak stats
 */
router.get('/streak', (req, res) => fastingController.getStreak(req, res));

/**
 * @swagger
 * /api/fasting/analytics:
 *   get:
 *     summary: Get fasting analytics
 *     tags: [Fasting]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: days
 *         schema:
 *           type: integer
 *           default: 30
 *           maximum: 365
 *     responses:
 *       200:
 *         description: Fasting analytics
 */
router.get('/analytics', (req, res) => fastingController.getAnalytics(req, res));

export default router;
