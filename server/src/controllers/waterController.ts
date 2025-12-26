import { waterService } from '../services/waterService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import {
  createWaterIntakeSchema,
  updateWaterIntakeSchema,
  updateWaterGoalSchema,
  quickAddWaterSchema,
} from '../validation/schemas';
import { withErrorHandling, ErrorHandlers } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { parseOptionalDate } from '../utils/dateHelpers';

export class WaterController {
  /**
   * Create a new water intake record
   * POST /api/water
   */
  createWaterIntake = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = createWaterIntakeSchema.parse(req.body);

    const intakeData = {
      ...validatedData,
      recordedAt: validatedData.recordedAt ? new Date(validatedData.recordedAt) : undefined,
    };

    const intake = await waterService.createWaterIntake(userId, intakeData);

    res.status(HTTP_STATUS.CREATED).json(intake);
  });

  /**
   * Quick add water using preset amounts
   * POST /api/water/quick-add
   */
  quickAddWater = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = quickAddWaterSchema.parse(req.body);

    const intake = await waterService.quickAddWater(
      userId,
      validatedData.preset,
      validatedData.customAmount
    );

    res.status(HTTP_STATUS.CREATED).json(intake);
  });

  /**
   * Get water intakes for a specific day
   * GET /api/water
   */
  getWaterIntakes = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const date = parseOptionalDate(req.query.date as string | undefined);

    const intakes = await waterService.getWaterIntakes(userId, date);

    res.status(HTTP_STATUS.OK).json(intakes);
  });

  /**
   * Get a single water intake by ID
   * GET /api/water/:id
   */
  getWaterIntakeById = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const intake = await waterService.getWaterIntakeById(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(intake);
  });

  /**
   * Update a water intake record
   * PUT /api/water/:id
   */
  updateWaterIntake = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = updateWaterIntakeSchema.parse(req.body);

    const intakeData = {
      ...validatedData,
      recordedAt: validatedData.recordedAt ? new Date(validatedData.recordedAt) : undefined,
    };

    const intake = await waterService.updateWaterIntake(userId, req.params.id, intakeData);

    res.status(HTTP_STATUS.OK).json(intake);
  });

  /**
   * Delete a water intake record
   * DELETE /api/water/:id
   */
  deleteWaterIntake = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const result = await waterService.deleteWaterIntake(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(result);
  });

  /**
   * Get daily water summary
   * GET /api/water/summary/daily
   */
  getDailySummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const date = parseOptionalDate(req.query.date as string | undefined);

    const summary = await waterService.getDailySummary(userId, date);

    res.status(HTTP_STATUS.OK).json(summary);
  });

  /**
   * Get weekly water summary
   * GET /api/water/summary/weekly
   */
  getWeeklySummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const summary = await waterService.getWeeklySummary(userId);

    res.status(HTTP_STATUS.OK).json(summary);
  });

  /**
   * Get user's water goal
   * GET /api/water/goal
   */
  getWaterGoal = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const goal = await waterService.getWaterGoal(userId);

    res.status(HTTP_STATUS.OK).json(goal);
  });

  /**
   * Update user's water goal
   * PUT /api/water/goal
   */
  updateWaterGoal = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = updateWaterGoalSchema.parse(req.body);

    const goal = await waterService.updateWaterGoal(userId, validatedData.goalWater);

    res.status(HTTP_STATUS.OK).json(goal);
  });
}

export const waterController = new WaterController();
