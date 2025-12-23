/**
 * Food Routes
 *
 * Routes for USDA FoodData Central integration:
 * - Search foods
 * - Get food by ID
 * - Get nutrients
 * - Popular foods
 * - Recent foods (authenticated)
 */

import { Router } from 'express';
import { foodController } from '../controllers/foodController';
import { authenticate } from '../middleware/auth';

const router = Router();

// =============================================================================
// Public Routes (no auth required)
// =============================================================================

// Search foods
// GET /api/foods/search?q=apple&page=1&limit=25&dataType=Foundation,Branded
router.get('/search', (req, res) => foodController.searchFoods(req, res));

// Get popular foods
// GET /api/foods/popular
router.get('/popular', (req, res) => foodController.getPopularFoods(req, res));

// Health check for USDA API
// GET /api/foods/health
router.get('/health', (req, res) => foodController.healthCheck(req, res));

// Get multiple foods by IDs
// POST /api/foods/bulk
router.post('/bulk', (req, res) => foodController.getBulkFoods(req, res));

// Get single food by FDC ID
// GET /api/foods/:fdcId
router.get('/:fdcId', (req, res) => foodController.getFoodById(req, res));

// Get nutrients for a food (optionally scaled)
// GET /api/foods/:fdcId/nutrients?grams=150
router.get('/:fdcId/nutrients', (req, res) => {
  // If grams query param is present, return scaled nutrients
  if (req.query.grams) {
    return foodController.getScaledNutrients(req, res);
  }
  return foodController.getFoodNutrients(req, res);
});

// =============================================================================
// Protected Routes (auth required)
// =============================================================================

// Get user's recent foods
// GET /api/foods/recent
router.get('/recent', authenticate, (req, res) =>
  foodController.getRecentFoods(req, res)
);

// Record food selection (adds to recent foods)
// POST /api/foods/:fdcId/select
router.post('/:fdcId/select', authenticate, (req, res) =>
  foodController.recordFoodSelection(req, res)
);

export default router;
