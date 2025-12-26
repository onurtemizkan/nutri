import { Router } from 'express';
import { waterController } from '../controllers/waterController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All water routes are protected
router.use(authenticate);

// Quick add water (must be before /:id to avoid route conflict)
router.post('/quick-add', (req, res) => waterController.quickAddWater(req, res));

// Summary routes (must be before /:id to avoid route conflict)
router.get('/summary/daily', (req, res) => waterController.getDailySummary(req, res));
router.get('/summary/weekly', (req, res) => waterController.getWeeklySummary(req, res));

// Goal routes
router.get('/goal', (req, res) => waterController.getWaterGoal(req, res));
router.put('/goal', (req, res) => waterController.updateWaterGoal(req, res));

// CRUD routes
router.post('/', (req, res) => waterController.createWaterIntake(req, res));
router.get('/', (req, res) => waterController.getWaterIntakes(req, res));
router.get('/:id', (req, res) => waterController.getWaterIntakeById(req, res));
router.put('/:id', (req, res) => waterController.updateWaterIntake(req, res));
router.delete('/:id', (req, res) => waterController.deleteWaterIntake(req, res));

export default router;
