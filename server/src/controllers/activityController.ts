import { Response } from 'express';
import { z } from 'zod';
import { activityService } from '../services/activityService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';

// Zod schemas for validation
const createActivitySchema = z.object({
  activityType: z.enum([
    'RUNNING',
    'CYCLING',
    'SWIMMING',
    'WALKING',
    'HIKING',
    'ROWING',
    'ELLIPTICAL',
    'WEIGHT_TRAINING',
    'BODYWEIGHT',
    'CROSSFIT',
    'POWERLIFTING',
    'BASKETBALL',
    'SOCCER',
    'TENNIS',
    'GOLF',
    'YOGA',
    'PILATES',
    'STRETCHING',
    'MARTIAL_ARTS',
    'DANCE',
    'OTHER',
  ]),
  intensity: z.enum(['LOW', 'MODERATE', 'HIGH', 'MAXIMUM']),
  startedAt: z.string().datetime(),
  endedAt: z.string().datetime(),
  duration: z.number().int().min(1, 'Duration must be at least 1 minute'),
  caloriesBurned: z.number().min(0).optional(),
  averageHeartRate: z.number().min(0).optional(),
  maxHeartRate: z.number().min(0).optional(),
  distance: z.number().min(0).optional(),
  steps: z.number().int().min(0).optional(),
  source: z.enum(['apple_health', 'strava', 'garmin', 'manual']),
  sourceId: z.string().optional(),
  notes: z.string().optional(),
});

const updateActivitySchema = z.object({
  activityType: z.enum([
    'RUNNING',
    'CYCLING',
    'SWIMMING',
    'WALKING',
    'HIKING',
    'ROWING',
    'ELLIPTICAL',
    'WEIGHT_TRAINING',
    'BODYWEIGHT',
    'CROSSFIT',
    'POWERLIFTING',
    'BASKETBALL',
    'SOCCER',
    'TENNIS',
    'GOLF',
    'YOGA',
    'PILATES',
    'STRETCHING',
    'MARTIAL_ARTS',
    'DANCE',
    'OTHER',
  ]).optional(),
  intensity: z.enum(['LOW', 'MODERATE', 'HIGH', 'MAXIMUM']).optional(),
  startedAt: z.string().datetime().optional(),
  endedAt: z.string().datetime().optional(),
  duration: z.number().int().min(1).optional(),
  caloriesBurned: z.number().min(0).optional(),
  averageHeartRate: z.number().min(0).optional(),
  maxHeartRate: z.number().min(0).optional(),
  distance: z.number().min(0).optional(),
  steps: z.number().int().min(0).optional(),
  notes: z.string().optional(),
});

const bulkCreateActivitiesSchema = z.object({
  activities: z.array(createActivitySchema).min(1, 'At least one activity is required'),
});

export class ActivityController {
  async createActivity(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const validatedData = createActivitySchema.parse(req.body);

      const activityData = {
        ...validatedData,
        startedAt: new Date(validatedData.startedAt),
        endedAt: new Date(validatedData.endedAt),
      };

      const activity = await activityService.createActivity(userId, activityData);

      res.status(201).json(activity);
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

  async createBulkActivities(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const validatedData = bulkCreateActivitiesSchema.parse(req.body);

      const activitiesData = validatedData.activities.map((activity) => ({
        ...activity,
        startedAt: new Date(activity.startedAt),
        endedAt: new Date(activity.endedAt),
      }));

      const result = await activityService.createBulkActivities(userId, activitiesData);

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

  async getActivities(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const query = {
        activityType: req.query.activityType as any,
        intensity: req.query.intensity as any,
        startDate: req.query.startDate ? new Date(req.query.startDate as string) : undefined,
        endDate: req.query.endDate ? new Date(req.query.endDate as string) : undefined,
        source: req.query.source as any,
        limit: req.query.limit ? parseInt(req.query.limit as string) : undefined,
      };

      const activities = await activityService.getActivities(userId, query);

      res.status(200).json(activities);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getActivityById(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const activity = await activityService.getActivityById(userId, req.params.id);

      res.status(200).json(activity);
    } catch (error) {
      if (error instanceof Error) {
        res.status(404).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async updateActivity(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const validatedData = updateActivitySchema.parse(req.body);

      const activityData = {
        ...validatedData,
        startedAt: validatedData.startedAt ? new Date(validatedData.startedAt) : undefined,
        endedAt: validatedData.endedAt ? new Date(validatedData.endedAt) : undefined,
      };

      const activity = await activityService.updateActivity(userId, req.params.id, activityData);

      res.status(200).json(activity);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ error: error.errors[0].message });
        return;
      }

      if (error instanceof Error) {
        res.status(404).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async deleteActivity(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const result = await activityService.deleteActivity(userId, req.params.id);

      res.status(200).json(result);
    } catch (error) {
      if (error instanceof Error) {
        res.status(404).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getDailySummary(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const date = req.query.date ? new Date(req.query.date as string) : undefined;

      const summary = await activityService.getDailySummary(userId, date);

      res.status(200).json(summary);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getWeeklySummary(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const summary = await activityService.getWeeklySummary(userId);

      res.status(200).json(summary);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }

  async getActivityStatsByType(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const activityType = req.params.activityType as any;
      const days = req.query.days ? parseInt(req.query.days as string) : 30;

      const stats = await activityService.getActivityStatsByType(userId, activityType, days);

      if (!stats) {
        res.status(404).json({ error: 'No activities found for this type' });
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

  async getRecoveryTime(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = requireAuth(req, res);
      if (!userId) return;

      const recovery = await activityService.getRecoveryTime(userId);

      if (!recovery) {
        res.status(404).json({ error: 'No high-intensity activities found' });
        return;
      }

      res.status(200).json(recovery);
    } catch (error) {
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
        return;
      }

      res.status(500).json({ error: 'Internal server error' });
    }
  }
}

export const activityController = new ActivityController();
