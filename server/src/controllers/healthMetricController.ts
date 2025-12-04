import { healthMetricService } from '../services/healthMetricService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import {
  parseOptionalHealthMetricType,
  parseOptionalHealthMetricSource,
  parseHealthMetricType,
} from '../utils/enumValidation';
import {
  createHealthMetricSchema,
  bulkCreateHealthMetricsSchema,
} from '../validation/schemas';
import {
  withErrorHandling,
  ErrorHandlers,
} from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { parseOptionalDate, getDaysAgo, getEndOfDay } from '../utils/dateHelpers';

export class HealthMetricController {
  createHealthMetric = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = createHealthMetricSchema.parse(req.body);

    const metricData = {
      ...validatedData,
      recordedAt: new Date(validatedData.recordedAt),
    };

    const metric = await healthMetricService.createHealthMetric(userId, metricData);

    res.status(HTTP_STATUS.CREATED).json(metric);
  });

  createBulkHealthMetrics = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = bulkCreateHealthMetricsSchema.parse(req.body);

    const metricsData = validatedData.metrics.map((metric) => ({
      ...metric,
      recordedAt: new Date(metric.recordedAt),
    }));

    const result = await healthMetricService.createBulkHealthMetrics(userId, metricsData);

    res.status(HTTP_STATUS.CREATED).json(result);
  });

  getHealthMetrics = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const query = {
      metricType: parseOptionalHealthMetricType(req.query.metricType),
      startDate: parseOptionalDate(req.query.startDate as string | undefined),
      endDate: parseOptionalDate(req.query.endDate as string | undefined),
      source: parseOptionalHealthMetricSource(req.query.source),
      limit: req.query.limit ? parseInt(req.query.limit as string) : undefined,
    };

    const metrics = await healthMetricService.getHealthMetrics(userId, query);

    res.status(HTTP_STATUS.OK).json(metrics);
  });

  getHealthMetricById = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const metric = await healthMetricService.getHealthMetricById(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(metric);
  });

  getLatestMetric = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const metricType = parseHealthMetricType(req.params.metricType);

    const metric = await healthMetricService.getLatestMetric(userId, metricType);

    if (!metric) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'No metrics found for this type' });
      return;
    }

    res.status(HTTP_STATUS.OK).json(metric);
  });

  getDailyAverage = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const metricType = parseHealthMetricType(req.params.metricType);
    const date = parseOptionalDate(req.query.date as string | undefined);

    const average = await healthMetricService.getDailyAverage(userId, metricType, date);

    if (!average) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'No metrics found for this type and date' });
      return;
    }

    res.status(HTTP_STATUS.OK).json(average);
  });

  getWeeklyAverage = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const metricType = parseHealthMetricType(req.params.metricType);

    const average = await healthMetricService.getWeeklyAverage(userId, metricType);

    if (!average) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'No metrics found for this type' });
      return;
    }

    res.status(HTTP_STATUS.OK).json(average);
  });

  getTimeSeries = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const metricType = parseHealthMetricType(req.params.metricType);
    const startDate = parseOptionalDate(req.query.startDate as string | undefined) || getDaysAgo(30);
    const endDate = parseOptionalDate(req.query.endDate as string | undefined) || getEndOfDay();

    const timeSeries = await healthMetricService.getTimeSeries(
      userId,
      metricType,
      startDate,
      endDate
    );

    res.status(HTTP_STATUS.OK).json(timeSeries);
  });

  getMetricStats = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const metricType = parseHealthMetricType(req.params.metricType);
    const days = req.query.days ? parseInt(req.query.days as string) : 30;

    const stats = await healthMetricService.getMetricStats(userId, metricType, days);

    if (!stats) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'No metrics found for this type' });
      return;
    }

    res.status(HTTP_STATUS.OK).json(stats);
  });

  deleteHealthMetric = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const result = await healthMetricService.deleteHealthMetric(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(result);
  });
}

export const healthMetricController = new HealthMetricController();
