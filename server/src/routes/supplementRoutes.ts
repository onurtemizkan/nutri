import { Router } from 'express';
import { supplementController } from '../controllers/supplementController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All supplement routes are protected
router.use(authenticate);

// Supplement CRUD
router.post('/', (req, res) => supplementController.createSupplement(req, res));
router.get('/', (req, res) => supplementController.getSupplements(req, res));
router.get('/today', (req, res) => supplementController.getTodayStatus(req, res));
router.get('/:id', (req, res) => supplementController.getSupplementById(req, res));
router.get('/:id/history', (req, res) => supplementController.getHistory(req, res));
router.put('/:id', (req, res) => supplementController.updateSupplement(req, res));
router.delete('/:id', (req, res) => supplementController.deleteSupplement(req, res));

// Supplement logs
router.post('/logs', (req, res) => supplementController.logIntake(req, res));
router.post('/logs/bulk', (req, res) => supplementController.bulkLogIntake(req, res));
router.get('/logs/day', (req, res) => supplementController.getLogsForDay(req, res));
router.delete('/logs/:logId', (req, res) => supplementController.deleteLog(req, res));

export default router;
