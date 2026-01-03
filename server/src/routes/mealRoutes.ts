import { Router } from 'express';
import { mealController } from '../controllers/mealController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All meal routes are protected
router.use(authenticate);

/**
 * @swagger
 * /meals:
 *   post:
 *     summary: Create a new meal
 *     tags: [Meals]
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
 *               - mealType
 *               - calories
 *               - protein
 *               - carbs
 *               - fat
 *             properties:
 *               name:
 *                 type: string
 *                 example: Grilled Chicken Salad
 *               mealType:
 *                 type: string
 *                 enum: [breakfast, lunch, dinner, snack]
 *               calories:
 *                 type: number
 *                 example: 450
 *               protein:
 *                 type: number
 *                 example: 35
 *               carbs:
 *                 type: number
 *                 example: 20
 *               fat:
 *                 type: number
 *                 example: 15
 *               fiber:
 *                 type: number
 *                 example: 5
 *               sugar:
 *                 type: number
 *                 example: 3
 *               sodium:
 *                 type: number
 *                 example: 500
 *               notes:
 *                 type: string
 *               consumedAt:
 *                 type: string
 *                 format: date-time
 *     responses:
 *       201:
 *         description: Meal created successfully
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Meal'
 *       400:
 *         $ref: '#/components/responses/ValidationError'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.post('/', (req, res) => mealController.createMeal(req, res));

/**
 * @swagger
 * /meals:
 *   get:
 *     summary: Get all meals for the authenticated user
 *     tags: [Meals]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - $ref: '#/components/parameters/limitParam'
 *       - $ref: '#/components/parameters/offsetParam'
 *       - $ref: '#/components/parameters/dateParam'
 *       - name: mealType
 *         in: query
 *         schema:
 *           type: string
 *           enum: [breakfast, lunch, dinner, snack]
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
 *         description: List of meals
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
 *                         $ref: '#/components/schemas/Meal'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/', (req, res) => mealController.getMeals(req, res));

/**
 * @swagger
 * /meals/summary/daily:
 *   get:
 *     summary: Get daily nutrition summary
 *     tags: [Meals]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - $ref: '#/components/parameters/dateParam'
 *     responses:
 *       200:
 *         description: Daily nutrition summary
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 date:
 *                   type: string
 *                   format: date
 *                 totalCalories:
 *                   type: number
 *                 totalProtein:
 *                   type: number
 *                 totalCarbs:
 *                   type: number
 *                 totalFat:
 *                   type: number
 *                 totalFiber:
 *                   type: number
 *                 mealCount:
 *                   type: integer
 *                 goalProgress:
 *                   type: object
 *                   properties:
 *                     calories:
 *                       type: number
 *                     protein:
 *                       type: number
 *                     carbs:
 *                       type: number
 *                     fat:
 *                       type: number
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/summary/daily', (req, res) => mealController.getDailySummary(req, res));

/**
 * @swagger
 * /meals/summary/weekly:
 *   get:
 *     summary: Get weekly nutrition summary
 *     tags: [Meals]
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
 *         description: Weekly nutrition summary with daily breakdown
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
 *                 totalCalories:
 *                   type: number
 *                 averageCalories:
 *                   type: number
 *                 dailyBreakdown:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       date:
 *                         type: string
 *                         format: date
 *                       calories:
 *                         type: number
 *                       protein:
 *                         type: number
 *                       carbs:
 *                         type: number
 *                       fat:
 *                         type: number
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 */
router.get('/summary/weekly', (req, res) => mealController.getWeeklySummary(req, res));

/**
 * @swagger
 * /meals/{id}:
 *   get:
 *     summary: Get a specific meal by ID
 *     tags: [Meals]
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
 *         description: Meal details
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Meal'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 *       404:
 *         $ref: '#/components/responses/NotFoundError'
 */
router.get('/:id', (req, res) => mealController.getMealById(req, res));

/**
 * @swagger
 * /meals/{id}:
 *   put:
 *     summary: Update a meal
 *     tags: [Meals]
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
 *               name:
 *                 type: string
 *               mealType:
 *                 type: string
 *                 enum: [breakfast, lunch, dinner, snack]
 *               calories:
 *                 type: number
 *               protein:
 *                 type: number
 *               carbs:
 *                 type: number
 *               fat:
 *                 type: number
 *               fiber:
 *                 type: number
 *               sugar:
 *                 type: number
 *               sodium:
 *                 type: number
 *               notes:
 *                 type: string
 *               consumedAt:
 *                 type: string
 *                 format: date-time
 *     responses:
 *       200:
 *         description: Meal updated
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Meal'
 *       400:
 *         $ref: '#/components/responses/ValidationError'
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 *       404:
 *         $ref: '#/components/responses/NotFoundError'
 */
router.put('/:id', (req, res) => mealController.updateMeal(req, res));

/**
 * @swagger
 * /meals/{id}:
 *   delete:
 *     summary: Delete a meal
 *     tags: [Meals]
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
 *         description: Meal deleted successfully
 *       401:
 *         $ref: '#/components/responses/UnauthorizedError'
 *       404:
 *         $ref: '#/components/responses/NotFoundError'
 */
router.delete('/:id', (req, res) => mealController.deleteMeal(req, res));

export default router;
