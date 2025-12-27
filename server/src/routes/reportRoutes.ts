import { Router } from 'express';
import { reportController } from '../controllers/reportController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All report routes are protected
router.use(authenticate);

/**
 * GET /api/reports/weekly
 * Generate a weekly nutrition report
 * Query params:
 *   - date (optional): Any date within the desired week (YYYY-MM-DD)
 *                      Defaults to current week if not provided
 */
router.get('/weekly', (req, res) => reportController.getWeeklyReport(req, res));

/**
 * GET /api/reports/monthly
 * Generate a monthly nutrition report
 * Query params:
 *   - month (optional): The desired month (YYYY-MM)
 *                       Defaults to current month if not provided
 */
router.get('/monthly', (req, res) => reportController.getMonthlyReport(req, res));

/**
 * GET /api/reports/weekly/export
 * Export weekly report in specified format
 * Query params:
 *   - date (optional): Any date within the desired week (YYYY-MM-DD)
 *   - format (required): Export format - 'pdf', 'image', or 'json'
 */
router.get('/weekly/export', (req, res) => reportController.exportWeeklyReport(req, res));

/**
 * GET /api/reports/monthly/export
 * Export monthly report in specified format
 * Query params:
 *   - month (optional): The desired month (YYYY-MM)
 *   - format (required): Export format - 'pdf', 'image', or 'json'
 */
router.get('/monthly/export', (req, res) => reportController.exportMonthlyReport(req, res));

export default router;
