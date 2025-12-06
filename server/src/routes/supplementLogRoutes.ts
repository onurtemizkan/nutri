import { Router } from 'express';
import { supplementController } from '../controllers/supplementController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All supplement log routes are protected
router.use(authenticate);

// ==========================================================================
// SUPPLEMENT LOGS (Intake Tracking)
// ==========================================================================

// POST /supplement-logs - Create supplement log entry
router.post('/', (req, res) => supplementController.createSupplementLog(req, res));

// POST /supplement-logs/bulk - Bulk create supplement log entries
router.post('/bulk', (req, res) => supplementController.createBulkSupplementLogs(req, res));

// GET /supplement-logs - Get supplement logs with optional filtering
router.get('/', (req, res) => supplementController.getSupplementLogs(req, res));

// GET /supplement-logs/:id - Get single supplement log
router.get('/:id', (req, res) => supplementController.getSupplementLogById(req, res));

// PUT /supplement-logs/:id - Update supplement log
router.put('/:id', (req, res) => supplementController.updateSupplementLog(req, res));

// DELETE /supplement-logs/:id - Delete supplement log
router.delete('/:id', (req, res) => supplementController.deleteSupplementLog(req, res));

export default router;
