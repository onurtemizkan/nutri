import { activityService } from '../services/activityService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import {
  parseOptionalActivityType,
  parseOptionalActivityIntensity,
  parseOptionalActivitySource,
  parseActivityType,
} from '../utils/enumValidation';
import {
  createActivitySchema,
  updateActivitySchema,
  bulkCreateActivitiesSchema,
} from '../validation/schemas';
import {
  withErrorHandling,
  ErrorHandlers,
} from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { parseOptionalDate } from '../utils/dateHelpers';

export class ActivityController {
  createActivity = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = createActivitySchema.parse(req.body);

    const activityData = {
      ...validatedData,
      startedAt: new Date(validatedData.startedAt),
      endedAt: new Date(validatedData.endedAt),
    };

    const activity = await activityService.createActivity(userId, activityData);

    res.status(HTTP_STATUS.CREATED).json(activity);
  });

  createBulkActivities = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = bulkCreateActivitiesSchema.parse(req.body);

    const activitiesData = validatedData.activities.map((activity) => ({
      ...activity,
      startedAt: new Date(activity.startedAt),
      endedAt: new Date(activity.endedAt),
    }));

    const result = await activityService.createBulkActivities(userId, activitiesData);

    res.status(HTTP_STATUS.CREATED).json(result);
  });

  getActivities = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const query = {
      activityType: parseOptionalActivityType(req.query.activityType),
      intensity: parseOptionalActivityIntensity(req.query.intensity),
      startDate: parseOptionalDate(req.query.startDate as string | undefined),
      endDate: parseOptionalDate(req.query.endDate as string | undefined),
      source: parseOptionalActivitySource(req.query.source),
      limit: req.query.limit ? parseInt(req.query.limit as string) : undefined,
    };

    const activities = await activityService.getActivities(userId, query);

    res.status(HTTP_STATUS.OK).json(activities);
  });

  getActivityById = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const activity = await activityService.getActivityById(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(activity);
  });

  updateActivity = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = updateActivitySchema.parse(req.body);

    const activityData = {
      ...validatedData,
      startedAt: validatedData.startedAt ? new Date(validatedData.startedAt) : undefined,
      endedAt: validatedData.endedAt ? new Date(validatedData.endedAt) : undefined,
    };

    const activity = await activityService.updateActivity(userId, req.params.id, activityData);

    res.status(HTTP_STATUS.OK).json(activity);
  });

  deleteActivity = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const result = await activityService.deleteActivity(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(result);
  });

  getDailySummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const date = parseOptionalDate(req.query.date as string | undefined);

    const summary = await activityService.getDailySummary(userId, date);

    res.status(HTTP_STATUS.OK).json(summary);
  });

  getWeeklySummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const summary = await activityService.getWeeklySummary(userId);

    res.status(HTTP_STATUS.OK).json(summary);
  });

  getActivityStatsByType = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const activityType = parseActivityType(req.params.activityType);
    const days = req.query.days ? parseInt(req.query.days as string) : 30;

    const stats = await activityService.getActivityStatsByType(userId, activityType, days);

    if (!stats) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'No activities found for this type' });
      return;
    }

    res.status(HTTP_STATUS.OK).json(stats);
  });

  getRecoveryTime = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const recovery = await activityService.getRecoveryTime(userId);

    if (!recovery) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'No high-intensity activities found' });
      return;
    }

    res.status(HTTP_STATUS.OK).json(recovery);
  });
}

export const activityController = new ActivityController();
