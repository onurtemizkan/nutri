import { Router } from 'express';
import { waterController } from '../controllers/waterController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All water routes are protected
router.use(authenticate);

/**
 * @swagger
 * /water/quick-add:
 *   post:
 *     summary: Quick add water intake with preset amounts
 *     tags: [Water]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - amount
 *             properties:
 *               amount:
 *                 type: integer
 *                 description: Water amount in ml
 *                 enum: [250, 500, 750, 1000]
 *                 example: 250
 *     responses:
 *       201:
 *         description: Water intake recorded
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/WaterIntake'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.post('/quick-add', (req, res) => waterController.quickAddWater(req, res));

/**
 * @swagger
 * /water/summary/daily:
 *   get:
 *     summary: Get daily water intake summary
 *     tags: [Water]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - $ref: '#/components/parameters/dateParam'
 *     responses:
 *       200:
 *         description: Daily water summary
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 date:
 *                   type: string
 *                   format: date
 *                 totalAmount:
 *                   type: integer
 *                   description: Total water in ml
 *                 goal:
 *                   type: integer
 *                 progress:
 *                   type: number
 *                   description: Percentage of goal achieved
 *                 intakeCount:
 *                   type: integer
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/summary/daily', (req, res) => waterController.getDailySummary(req, res));

/**
 * @swagger
 * /water/summary/weekly:
 *   get:
 *     summary: Get weekly water intake summary
 *     tags: [Water]
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
 *         description: Weekly water summary
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
 *                 totalAmount:
 *                   type: integer
 *                 averageDaily:
 *                   type: integer
 *                 daysMetGoal:
 *                   type: integer
 *                 dailyBreakdown:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       date:
 *                         type: string
 *                         format: date
 *                       amount:
 *                         type: integer
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/summary/weekly', (req, res) => waterController.getWeeklySummary(req, res));

/**
 * @swagger
 * /water/goal:
 *   get:
 *     summary: Get user's water goal
 *     tags: [Water]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Water goal
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 goal:
 *                   type: integer
 *                   description: Daily water goal in ml
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/goal', (req, res) => waterController.getWaterGoal(req, res));

/**
 * @swagger
 * /water/goal:
 *   put:
 *     summary: Update user's water goal
 *     tags: [Water]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - goal
 *             properties:
 *               goal:
 *                 type: integer
 *                 minimum: 500
 *                 maximum: 10000
 *                 description: Daily water goal in ml
 *                 example: 2500
 *     responses:
 *       200:
 *         description: Water goal updated
 *       400:
 *         $ref: '#/components/responses/ValidationError'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.put('/goal', (req, res) => waterController.updateWaterGoal(req, res));

/**
 * @swagger
 * /water:
 *   post:
 *     summary: Record water intake
 *     tags: [Water]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - amount
 *             properties:
 *               amount:
 *                 type: integer
 *                 description: Water amount in ml
 *                 example: 250
 *               recordedAt:
 *                 type: string
 *                 format: date-time
 *     responses:
 *       201:
 *         description: Water intake recorded
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/WaterIntake'
 *       400:
 *         $ref: '#/components/responses/ValidationError'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.post('/', (req, res) => waterController.createWaterIntake(req, res));

/**
 * @swagger
 * /water:
 *   get:
 *     summary: Get water intake records
 *     tags: [Water]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - $ref: '#/components/parameters/limitParam'
 *       - $ref: '#/components/parameters/offsetParam'
 *       - $ref: '#/components/parameters/dateParam'
 *     responses:
 *       200:
 *         description: List of water intake records
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
 *                         $ref: '#/components/schemas/WaterIntake'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/', (req, res) => waterController.getWaterIntakes(req, res));

/**
 * @swagger
 * /water/{id}:
 *   get:
 *     summary: Get water intake by ID
 *     tags: [Water]
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
 *         description: Water intake record
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/WaterIntake'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 *       404:
 *         $ref: '#/components/responses/NotFoundError'
 */
router.get('/:id', (req, res) => waterController.getWaterIntakeById(req, res));

/**
 * @swagger
 * /water/{id}:
 *   put:
 *     summary: Update water intake record
 *     tags: [Water]
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
 *               amount:
 *                 type: integer
 *               recordedAt:
 *                 type: string
 *                 format: date-time
 *     responses:
 *       200:
 *         description: Water intake updated
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/WaterIntake'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 *       404:
 *         $ref: '#/components/responses/NotFoundError'
 */
router.put('/:id', (req, res) => waterController.updateWaterIntake(req, res));

/**
 * @swagger
 * /water/{id}:
 *   delete:
 *     summary: Delete water intake record
 *     tags: [Water]
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
 *         description: Water intake deleted
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 *       404:
 *         $ref: '#/components/responses/NotFoundError'
 */
router.delete('/:id', (req, res) => waterController.deleteWaterIntake(req, res));

export default router;
