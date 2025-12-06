import { Router } from 'express';
import { supplementController } from '../controllers/supplementController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All supplement routes are protected
router.use(authenticate);

// ==========================================================================
// SUMMARIES (must be before :id routes to avoid conflicts)
// ==========================================================================

router.get('/summary/daily', (req, res) => supplementController.getDailySummary(req, res));
router.get('/summary/weekly', (req, res) => supplementController.getWeeklySummary(req, res));

// ==========================================================================
// MASTER SUPPLEMENT LIST (read-only)
// ==========================================================================

// GET /supplements - Get all supplements with optional filtering
router.get('/', (req, res) => supplementController.getSupplements(req, res));

// GET /supplements/:id/stats - Get usage statistics for a supplement
router.get('/:id/stats', (req, res) => supplementController.getSupplementStats(req, res));

// GET /supplements/:id - Get single supplement by ID
router.get('/:id', (req, res) => supplementController.getSupplementById(req, res));

// ==========================================================================
// USER DATA MANAGEMENT
// ==========================================================================

// DELETE /supplements/user-data - Delete all user supplement data (GDPR)
router.delete('/user-data', (req, res) =>
  supplementController.deleteAllUserSupplementData(req, res)
);

export default router;
