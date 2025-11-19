import { Router } from 'express';
import { activityController } from '../controllers/activityController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All activity routes are protected
router.use(authenticate);

// Create
router.post('/', (req, res) => activityController.createActivity(req, res));
router.post('/bulk', (req, res) => activityController.createBulkActivities(req, res));

// Read
router.get('/', (req, res) => activityController.getActivities(req, res));
router.get('/summary/daily', (req, res) => activityController.getDailySummary(req, res));
router.get('/summary/weekly', (req, res) => activityController.getWeeklySummary(req, res));
router.get('/recovery', (req, res) => activityController.getRecoveryTime(req, res));
router.get('/stats/:activityType', (req, res) =>
  activityController.getActivityStatsByType(req, res)
);
router.get('/:id', (req, res) => activityController.getActivityById(req, res));

// Update
router.put('/:id', (req, res) => activityController.updateActivity(req, res));

// Delete
router.delete('/:id', (req, res) => activityController.deleteActivity(req, res));

export default router;
