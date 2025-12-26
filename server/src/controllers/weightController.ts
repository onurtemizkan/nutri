import { weightService } from '../services/weightService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import {
  createWeightRecordSchema,
  updateWeightRecordSchema,
  updateWeightGoalSchema,
  weightTrendsQuerySchema,
  weightRecordsQuerySchema,
} from '../validation/schemas';
import { withErrorHandling, ErrorHandlers } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { parseOptionalDate } from '../utils/dateHelpers';

export class WeightController {
  /**
   * Create a new weight record
   * POST /api/weight
   */
  createWeightRecord = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = createWeightRecordSchema.parse(req.body);

    const weightRecord = await weightService.createWeightRecord(userId, {
      weight: validatedData.weight,
      recordedAt: validatedData.recordedAt,
    });

    res.status(HTTP_STATUS.CREATED).json(weightRecord);
  });

  /**
   * Get all weight records for the user
   * GET /api/weight
   */
  getWeightRecords = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const queryData = weightRecordsQuerySchema.parse(req.query);

    const records = await weightService.getWeightRecords(userId, {
      startDate: queryData.startDate ? new Date(queryData.startDate) : undefined,
      endDate: queryData.endDate ? new Date(queryData.endDate) : undefined,
      limit: queryData.limit,
    });

    res.status(HTTP_STATUS.OK).json(records);
  });

  /**
   * Get a specific weight record by ID
   * GET /api/weight/:id
   */
  getWeightRecordById = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const record = await weightService.getWeightRecordById(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(record);
  });

  /**
   * Get weight records for a specific day
   * GET /api/weight/day
   */
  getWeightRecordsForDay = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const date = parseOptionalDate(req.query.date as string | undefined);

    const records = await weightService.getWeightRecordsForDay(userId, date);

    res.status(HTTP_STATUS.OK).json(records);
  });

  /**
   * Update a weight record
   * PUT /api/weight/:id
   */
  updateWeightRecord = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = updateWeightRecordSchema.parse(req.body);

    const record = await weightService.updateWeightRecord(userId, req.params.id, validatedData);

    res.status(HTTP_STATUS.OK).json(record);
  });

  /**
   * Delete a weight record
   * DELETE /api/weight/:id
   */
  deleteWeightRecord = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const result = await weightService.deleteWeightRecord(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(result);
  });

  /**
   * Get weight trends with moving averages
   * GET /api/weight/trends
   */
  getWeightTrends = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const queryData = weightTrendsQuerySchema.parse(req.query);

    const trends = await weightService.getWeightTrends(userId, queryData.days);

    res.status(HTTP_STATUS.OK).json(trends);
  });

  /**
   * Get weight progress towards goal
   * GET /api/weight/progress
   */
  getWeightProgress = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const progress = await weightService.getWeightProgress(userId);

    res.status(HTTP_STATUS.OK).json(progress);
  });

  /**
   * Get weight summary for dashboard widget
   * GET /api/weight/summary
   */
  getWeightSummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const summary = await weightService.getWeightSummary(userId);

    res.status(HTTP_STATUS.OK).json(summary);
  });

  /**
   * Update user's goal weight
   * PUT /api/weight/goal
   */
  updateGoalWeight = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = updateWeightGoalSchema.parse(req.body);

    const result = await weightService.updateGoalWeight(userId, validatedData.goalWeight);

    res.status(HTTP_STATUS.OK).json(result);
  });
}

export const weightController = new WeightController();
