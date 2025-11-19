import { Response } from 'express';
import { z } from 'zod';
import { healthMetricService } from '../services/healthMetricService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import {
  parseOptionalHealthMetricType,
  parseOptionalHealthMetricSource,
  parseHealthMetricType,
} from '../utils/enumValidation';

// Zod schemas for validation
const createHealthMetricSchema = z.object({
  metricType: z.enum([
    'RESTING_HEART_RATE',
    'HEART_RATE_VARIABILITY_SDNN',
    'HEART_RATE_VARIABILITY_RMSSD',
    'BLOOD_PRESSURE_SYSTOLIC',
    'BLOOD_PRESSURE_DIASTOLIC',
    'RESPIRATORY_RATE',
    'OXYGEN_SATURATION',
    'VO2_MAX',
    'SLEEP_DURATION',
    'DEEP_SLEEP_DURATION',
    'REM_SLEEP_DURATION',
    'SLEEP_EFFICIENCY',
    'SLEEP_SCORE',
    'STEPS',
    'ACTIVE_CALORIES',
    'TOTAL_CALORIES',
    'EXERCISE_MINUTES',
    'STANDING_HOURS',
    'RECOVERY_SCORE',
    'STRAIN_SCORE',
    'READINESS_SCORE',
    'BODY_FAT_PERCENTAGE',
    'MUSCLE_MASS',
    'BONE_MASS',
    'WATER_PERCENTAGE',
    'SKIN_TEMPERATURE',
    'BLOOD_GLUCOSE',
    'STRESS_LEVEL',
  ]),
  value: z.number(),
  unit: z.string().min(1, 'Unit is required'),
  recordedAt: z.string().datetime(),
  source: z.enum(['apple_health', 'fitbit', 'garmin', 'oura', 'whoop', 'manual']),
  sourceId: z.string().optional(),
  metadata: z.record(z.any()).optional(),
});

const bulkCreateHealthMetricsSchema = z.object({
  metrics: z.array(createHealthMetricSchema).min(1, 'At least one metric is required'),
});

export class HealthMetricController {
  async createHealthMetric(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const validatedData = createHealthMetricSchema.parse(req.body);

      const metricData = {
        ...validatedData,
        recordedAt: new Date(validatedData.recordedAt),
      };

      const metric = await healthMetricService.createHealthMetric(userId, metricData);

      res.status(201).json(metric);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors[0].message });
        return;
      }

      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async createBulkHealthMetrics(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const validatedData = bulkCreateHealthMetricsSchema.parse(req.body);

      const metricsData = validatedData.metrics.map((metric) => ({
        ...metric,
        recordedAt: new Date(metric.recordedAt),
      }));

      const result = await healthMetricService.createBulkHealthMetrics(userId, metricsData);

      res.status(201).json(result);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors[0].message });
        return;
      }

      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getHealthMetrics(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const query = {
        metricType: parseOptionalHealthMetricType(req.query.metricType),
        startDate: req.query.startDate ? new Date(req.query.startDate as string) : undefined,
        endDate: req.query.endDate ? new Date(req.query.endDate as string) : undefined,
        source: parseOptionalHealthMetricSource(req.query.source),
        limit: req.query.limit ? parseInt(req.query.limit as string) : undefined,
      };

      const metrics = await healthMetricService.getHealthMetrics(userId, query);

      res.status(200).json(metrics);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getHealthMetricById(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const metric = await healthMetricService.getHealthMetricById(userId, req.params.id);

      res.status(200).json(metric);
    } catch (error) {
      if (error instanceof Error) {
        res.status(404).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getLatestMetric(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const metricType = parseHealthMetricType(req.params.metricType);

      const metric = await healthMetricService.getLatestMetric(userId, metricType);

      if (!metric) {
        res.status(404).json({ error: 'No metrics found for this type' });
        return;
      }

      res.status(200).json(metric);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getDailyAverage(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const metricType = parseHealthMetricType(req.params.metricType);
      const date = req.query.date ? new Date(req.query.date as string) : undefined;

      const average = await healthMetricService.getDailyAverage(userId, metricType, date);

      if (!average) {
        res.status(404).json({ error: 'No metrics found for this type and date' });
        return;
      }

      res.status(200).json(average);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getWeeklyAverage(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const metricType = parseHealthMetricType(req.params.metricType);

      const average = await healthMetricService.getWeeklyAverage(userId, metricType);

      if (!average) {
        res.status(404).json({ error: 'No metrics found for this type' });
        return;
      }

      res.status(200).json(average);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getTimeSeries(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const metricType = parseHealthMetricType(req.params.metricType);
      const startDate = req.query.startDate ? new Date(req.query.startDate as string) : new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
      const endDate = req.query.endDate ? new Date(req.query.endDate as string) : new Date();

      const timeSeries = await healthMetricService.getTimeSeries(
        userId,
        metricType,
        startDate,
        endDate
      );

      res.status(200).json(timeSeries);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getMetricStats(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const metricType = parseHealthMetricType(req.params.metricType);
      const days = req.query.days ? parseInt(req.query.days as string) : 30;

      const stats = await healthMetricService.getMetricStats(userId, metricType, days);

      if (!stats) {
        res.status(404).json({ error: 'No metrics found for this type' });
        return;
      }

      res.status(200).json(stats);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async deleteHealthMetric(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const result = await healthMetricService.deleteHealthMetric(userId, req.params.id);

      res.status(200).json(result);
    } catch (error) {
      if (error instanceof Error) {
        res.status(404).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }
}

export const healthMetricController = new HealthMetricController();
