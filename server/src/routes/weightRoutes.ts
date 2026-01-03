import { Router } from 'express';
import { weightController } from '../controllers/weightController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All weight routes are protected
router.use(authenticate);

/**
 * @swagger
 * /weight:
 *   post:
 *     summary: Record weight
 *     tags: [Weight]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - weight
 *             properties:
 *               weight:
 *                 type: number
 *                 description: Weight in kg
 *                 example: 75.5
 *               notes:
 *                 type: string
 *               recordedAt:
 *                 type: string
 *                 format: date-time
 *     responses:
 *       201:
 *         description: Weight recorded
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/WeightRecord'
 *       400:
 *         $ref: '#/components/responses/ValidationError'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.post('/', (req, res) => weightController.createWeightRecord(req, res));

/**
 * @swagger
 * /weight:
 *   get:
 *     summary: Get weight records
 *     tags: [Weight]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - $ref: '#/components/parameters/limitParam'
 *       - $ref: '#/components/parameters/offsetParam'
 *       - name: startDate
 *         in: query
 *         schema:
 *           type: string
 *           format: date
 *       - name: endDate
 *         in: query
 *         schema:
 *           type: string
 *           format: date
 *     responses:
 *       200:
 *         description: List of weight records
 *         content:
 *           application/json:
 *             schema:
 *               allOf:
 *                 - $ref: '#/components/schemas/PaginatedResponse'
 *                 - type: object
 *                   properties:
 *                     data:
 *                       type: array
 *                       items:
 *                         $ref: '#/components/schemas/WeightRecord'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/', (req, res) => weightController.getWeightRecords(req, res));

/**
 * @swagger
 * /weight/day:
 *   get:
 *     summary: Get weight records for a specific day
 *     tags: [Weight]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - $ref: '#/components/parameters/dateParam'
 *     responses:
 *       200:
 *         description: Weight records for the day
 *         content:
 *           application/json:
 *             schema:
 *               type: array
 *               items:
 *                 $ref: '#/components/schemas/WeightRecord'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/day', (req, res) => weightController.getWeightRecordsForDay(req, res));

/**
 * @swagger
 * /weight/trends:
 *   get:
 *     summary: Get weight trends over time
 *     tags: [Weight]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - name: period
 *         in: query
 *         schema:
 *           type: string
 *           enum: [week, month, quarter, year]
 *           default: month
 *     responses:
 *       200:
 *         description: Weight trends
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 period:
 *                   type: string
 *                 startWeight:
 *                   type: number
 *                 endWeight:
 *                   type: number
 *                 change:
 *                   type: number
 *                 changePercent:
 *                   type: number
 *                 minWeight:
 *                   type: number
 *                 maxWeight:
 *                   type: number
 *                 averageWeight:
 *                   type: number
 *                 dataPoints:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       date:
 *                         type: string
 *                         format: date
 *                       weight:
 *                         type: number
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/trends', (req, res) => weightController.getWeightTrends(req, res));

/**
 * @swagger
 * /weight/progress:
 *   get:
 *     summary: Get progress towards weight goal
 *     tags: [Weight]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Weight goal progress
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 currentWeight:
 *                   type: number
 *                 goalWeight:
 *                   type: number
 *                 startWeight:
 *                   type: number
 *                 remainingToGoal:
 *                   type: number
 *                 progressPercent:
 *                   type: number
 *                 trend:
 *                   type: string
 *                   enum: [gaining, losing, maintaining]
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/progress', (req, res) => weightController.getWeightProgress(req, res));

/**
 * @swagger
 * /weight/summary:
 *   get:
 *     summary: Get weight summary statistics
 *     tags: [Weight]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Weight summary
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 totalRecords:
 *                   type: integer
 *                 latestWeight:
 *                   type: number
 *                 earliestWeight:
 *                   type: number
 *                 averageWeight:
 *                   type: number
 *                 minWeight:
 *                   type: number
 *                 maxWeight:
 *                   type: number
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/summary', (req, res) => weightController.getWeightSummary(req, res));

/**
 * @swagger
 * /weight/goal:
 *   put:
 *     summary: Update goal weight
 *     tags: [Weight]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - goalWeight
 *             properties:
 *               goalWeight:
 *                 type: number
 *                 description: Target weight in kg
 *                 example: 70.0
 *     responses:
 *       200:
 *         description: Goal weight updated
 *       400:
 *         $ref: '#/components/responses/ValidationError'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.put('/goal', (req, res) => weightController.updateGoalWeight(req, res));

/**
 * @swagger
 * /weight/{id}:
 *   get:
 *     summary: Get weight record by ID
 *     tags: [Weight]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - name: id
 *         in: path
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Weight record
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/WeightRecord'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 *       404:
 *         $ref: '#/components/responses/NotFoundError'
 */
router.get('/:id', (req, res) => weightController.getWeightRecordById(req, res));

/**
 * @swagger
 * /weight/{id}:
 *   put:
 *     summary: Update weight record
 *     tags: [Weight]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - name: id
 *         in: path
 *         required: true
 *         schema:
 *           type: string
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               weight:
 *                 type: number
 *               notes:
 *                 type: string
 *               recordedAt:
 *                 type: string
 *                 format: date-time
 *     responses:
 *       200:
 *         description: Weight record updated
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/WeightRecord'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 *       404:
 *         $ref: '#/components/responses/NotFoundError'
 */
router.put('/:id', (req, res) => weightController.updateWeightRecord(req, res));

/**
 * @swagger
 * /weight/{id}:
 *   delete:
 *     summary: Delete weight record
 *     tags: [Weight]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - name: id
 *         in: path
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Weight record deleted
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 *       404:
 *         $ref: '#/components/responses/NotFoundError'
 */
router.delete('/:id', (req, res) => weightController.deleteWeightRecord(req, res));

export default router;
