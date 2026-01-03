import { Router } from 'express';
import { favoritesController } from '../controllers/favoritesController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All favorites routes are protected
router.use(authenticate);

// ============================================================================
// FAVORITE MEALS
// ============================================================================

/**
 * @swagger
 * /api/favorites:
 *   post:
 *     summary: Create a new favorite meal
 *     tags: [Favorites]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/CreateFavoriteMeal'
 *     responses:
 *       201:
 *         description: Favorite created successfully
 */
router.post('/', (req, res) => favoritesController.createFavorite(req, res));

/**
 * @swagger
 * /api/favorites/from-meal:
 *   post:
 *     summary: Create a favorite from an existing meal
 *     tags: [Favorites]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - mealId
 *             properties:
 *               mealId:
 *                 type: string
 *               customName:
 *                 type: string
 *     responses:
 *       201:
 *         description: Favorite created from meal successfully
 */
router.post('/from-meal', (req, res) => favoritesController.createFavoriteFromMeal(req, res));

/**
 * @swagger
 * /api/favorites:
 *   get:
 *     summary: Get user's favorite meals
 *     tags: [Favorites]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 50
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           default: 0
 *       - in: query
 *         name: sortBy
 *         schema:
 *           type: string
 *           enum: [usageCount, lastUsedAt, sortOrder, createdAt]
 *       - in: query
 *         name: sortOrder
 *         schema:
 *           type: string
 *           enum: [asc, desc]
 *     responses:
 *       200:
 *         description: List of favorites with pagination
 */
router.get('/', (req, res) => favoritesController.getFavorites(req, res));

/**
 * @swagger
 * /api/favorites/reorder:
 *   post:
 *     summary: Reorder favorite meals
 *     tags: [Favorites]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - items
 *             properties:
 *               items:
 *                 type: array
 *                 items:
 *                   type: object
 *                   properties:
 *                     id:
 *                       type: string
 *                     sortOrder:
 *                       type: integer
 *     responses:
 *       200:
 *         description: Favorites reordered successfully
 */
router.post('/reorder', (req, res) => favoritesController.reorderFavorites(req, res));

/**
 * @swagger
 * /api/favorites/{id}:
 *   get:
 *     summary: Get a specific favorite meal
 *     tags: [Favorites]
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
 *         description: Favorite meal details
 *       404:
 *         description: Favorite not found
 */
router.get('/:id', (req, res) => favoritesController.getFavoriteById(req, res));

/**
 * @swagger
 * /api/favorites/{id}:
 *   put:
 *     summary: Update a favorite meal
 *     tags: [Favorites]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/UpdateFavoriteMeal'
 *     responses:
 *       200:
 *         description: Favorite updated successfully
 */
router.put('/:id', (req, res) => favoritesController.updateFavorite(req, res));

/**
 * @swagger
 * /api/favorites/{id}:
 *   delete:
 *     summary: Delete a favorite meal
 *     tags: [Favorites]
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
 *         description: Favorite deleted successfully
 */
router.delete('/:id', (req, res) => favoritesController.deleteFavorite(req, res));

/**
 * @swagger
 * /api/favorites/{id}/use:
 *   post:
 *     summary: Create a meal from a favorite
 *     tags: [Favorites]
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
 *               mealType:
 *                 type: string
 *                 enum: [breakfast, lunch, dinner, snack]
 *     responses:
 *       201:
 *         description: Meal created from favorite
 */
router.post('/:id/use', (req, res) => favoritesController.useFavorite(req, res));

// ============================================================================
// MEAL TEMPLATES
// ============================================================================

/**
 * @swagger
 * /api/favorites/templates:
 *   post:
 *     summary: Create a new meal template
 *     tags: [Templates]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/CreateMealTemplate'
 *     responses:
 *       201:
 *         description: Template created successfully
 */
router.post('/templates', (req, res) => favoritesController.createTemplate(req, res));

/**
 * @swagger
 * /api/favorites/templates:
 *   get:
 *     summary: Get user's meal templates
 *     tags: [Templates]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 50
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           default: 0
 *     responses:
 *       200:
 *         description: List of templates with pagination
 */
router.get('/templates', (req, res) => favoritesController.getTemplates(req, res));

/**
 * @swagger
 * /api/favorites/templates/{id}:
 *   get:
 *     summary: Get a specific meal template
 *     tags: [Templates]
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
 *         description: Template details
 *       404:
 *         description: Template not found
 */
router.get('/templates/:id', (req, res) => favoritesController.getTemplateById(req, res));

/**
 * @swagger
 * /api/favorites/templates/{id}:
 *   put:
 *     summary: Update a meal template
 *     tags: [Templates]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/UpdateMealTemplate'
 *     responses:
 *       200:
 *         description: Template updated successfully
 */
router.put('/templates/:id', (req, res) => favoritesController.updateTemplate(req, res));

/**
 * @swagger
 * /api/favorites/templates/{id}:
 *   delete:
 *     summary: Delete a meal template
 *     tags: [Templates]
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
 *         description: Template deleted successfully
 */
router.delete('/templates/:id', (req, res) => favoritesController.deleteTemplate(req, res));

/**
 * @swagger
 * /api/favorites/templates/{id}/use:
 *   post:
 *     summary: Create a meal from a template
 *     tags: [Templates]
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
 *               mealType:
 *                 type: string
 *                 enum: [breakfast, lunch, dinner, snack]
 *     responses:
 *       201:
 *         description: Meal created from template
 */
router.post('/templates/:id/use', (req, res) => favoritesController.useTemplate(req, res));

// ============================================================================
// QUICK ADD PRESETS
// ============================================================================

/**
 * @swagger
 * /api/favorites/presets:
 *   post:
 *     summary: Create a new quick add preset
 *     tags: [Presets]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/CreateQuickAddPreset'
 *     responses:
 *       201:
 *         description: Preset created successfully
 */
router.post('/presets', (req, res) => favoritesController.createPreset(req, res));

/**
 * @swagger
 * /api/favorites/presets:
 *   get:
 *     summary: Get user's quick add presets
 *     tags: [Presets]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: activeOnly
 *         schema:
 *           type: boolean
 *           default: true
 *     responses:
 *       200:
 *         description: List of presets
 */
router.get('/presets', (req, res) => favoritesController.getPresets(req, res));

/**
 * @swagger
 * /api/favorites/presets/{id}:
 *   get:
 *     summary: Get a specific quick add preset
 *     tags: [Presets]
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
 *         description: Preset details
 *       404:
 *         description: Preset not found
 */
router.get('/presets/:id', (req, res) => favoritesController.getPresetById(req, res));

/**
 * @swagger
 * /api/favorites/presets/{id}:
 *   put:
 *     summary: Update a quick add preset
 *     tags: [Presets]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/UpdateQuickAddPreset'
 *     responses:
 *       200:
 *         description: Preset updated successfully
 */
router.put('/presets/:id', (req, res) => favoritesController.updatePreset(req, res));

/**
 * @swagger
 * /api/favorites/presets/{id}:
 *   delete:
 *     summary: Delete a quick add preset
 *     tags: [Presets]
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
 *         description: Preset deleted successfully
 */
router.delete('/presets/:id', (req, res) => favoritesController.deletePreset(req, res));

/**
 * @swagger
 * /api/favorites/presets/{id}/execute:
 *   post:
 *     summary: Execute a quick add preset (create meal or water intake)
 *     tags: [Presets]
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
 *               mealType:
 *                 type: string
 *                 enum: [breakfast, lunch, dinner, snack]
 *               consumedAt:
 *                 type: string
 *                 format: date-time
 *     responses:
 *       201:
 *         description: Preset executed (meal or water created)
 */
router.post('/presets/:id/execute', (req, res) => favoritesController.executePreset(req, res));

// ============================================================================
// RECENT FOODS
// ============================================================================

/**
 * @swagger
 * /api/favorites/recent:
 *   get:
 *     summary: Get user's recent foods
 *     tags: [Recent Foods]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 20
 *           maximum: 50
 *     responses:
 *       200:
 *         description: List of recent foods
 */
router.get('/recent', (req, res) => favoritesController.getRecentFoods(req, res));

/**
 * @swagger
 * /api/favorites/recent:
 *   delete:
 *     summary: Clear all recent foods
 *     tags: [Recent Foods]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Recent foods cleared successfully
 */
router.delete('/recent', (req, res) => favoritesController.clearRecentFoods(req, res));

/**
 * @swagger
 * /api/favorites/recent/{id}/use:
 *   post:
 *     summary: Create a meal from a recent food
 *     tags: [Recent Foods]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - mealType
 *             properties:
 *               mealType:
 *                 type: string
 *                 enum: [breakfast, lunch, dinner, snack]
 *     responses:
 *       201:
 *         description: Meal created from recent food
 */
router.post('/recent/:id/use', (req, res) => favoritesController.useRecentFood(req, res));

export default router;
