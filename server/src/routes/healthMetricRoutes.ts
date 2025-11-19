import { Router } from 'express';
import { healthMetricController } from '../controllers/healthMetricController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All health metric routes are protected
router.use(authenticate);

// Create
router.post('/', (req, res) => healthMetricController.createHealthMetric(req, res));
router.post('/bulk', (req, res) => healthMetricController.createBulkHealthMetrics(req, res));

// Read
router.get('/', (req, res) => healthMetricController.getHealthMetrics(req, res));
router.get('/latest/:metricType', (req, res) => healthMetricController.getLatestMetric(req, res));
router.get('/average/daily/:metricType', (req, res) =>
  healthMetricController.getDailyAverage(req, res)
);
router.get('/average/weekly/:metricType', (req, res) =>
  healthMetricController.getWeeklyAverage(req, res)
);
router.get('/timeseries/:metricType', (req, res) =>
  healthMetricController.getTimeSeries(req, res)
);
router.get('/stats/:metricType', (req, res) => healthMetricController.getMetricStats(req, res));
router.get('/:id', (req, res) => healthMetricController.getHealthMetricById(req, res));

// Delete
router.delete('/:id', (req, res) => healthMetricController.deleteHealthMetric(req, res));

export default router;
