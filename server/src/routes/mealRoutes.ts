import { Router } from 'express';
import { mealController } from '../controllers/mealController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All meal routes are protected
router.use(authenticate);

router.post('/', (req, res) => mealController.createMeal(req, res));
router.get('/', (req, res) => mealController.getMeals(req, res));
router.get('/summary/daily', (req, res) => mealController.getDailySummary(req, res));
router.get('/summary/weekly', (req, res) => mealController.getWeeklySummary(req, res));
router.get('/:id', (req, res) => mealController.getMealById(req, res));
router.put('/:id', (req, res) => mealController.updateMeal(req, res));
router.delete('/:id', (req, res) => mealController.deleteMeal(req, res));

export default router;
