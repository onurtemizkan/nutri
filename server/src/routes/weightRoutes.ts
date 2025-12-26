import { Router } from 'express';
import { weightController } from '../controllers/weightController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All weight routes are protected
router.use(authenticate);

// Weight record CRUD
router.post('/', (req, res) => weightController.createWeightRecord(req, res));
router.get('/', (req, res) => weightController.getWeightRecords(req, res));
router.get('/day', (req, res) => weightController.getWeightRecordsForDay(req, res));
router.get('/trends', (req, res) => weightController.getWeightTrends(req, res));
router.get('/progress', (req, res) => weightController.getWeightProgress(req, res));
router.get('/summary', (req, res) => weightController.getWeightSummary(req, res));
router.get('/:id', (req, res) => weightController.getWeightRecordById(req, res));
router.put('/goal', (req, res) => weightController.updateGoalWeight(req, res));
router.put('/:id', (req, res) => weightController.updateWeightRecord(req, res));
router.delete('/:id', (req, res) => weightController.deleteWeightRecord(req, res));

export default router;
