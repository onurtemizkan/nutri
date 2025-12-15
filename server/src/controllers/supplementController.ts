import { supplementService } from '../services/supplementService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import {
  createSupplementSchema,
  updateSupplementSchema,
  createSupplementLogSchema,
  bulkCreateSupplementLogsSchema,
} from '../validation/schemas';
import { withErrorHandling, ErrorHandlers } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { parseOptionalDate } from '../utils/dateHelpers';

export class SupplementController {
  /**
   * Create a new supplement
   */
  createSupplement = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = createSupplementSchema.parse(req.body);

    const supplementData = {
      ...validatedData,
      startDate: validatedData.startDate ? new Date(validatedData.startDate) : undefined,
      endDate: validatedData.endDate ? new Date(validatedData.endDate) : undefined,
    };

    const supplement = await supplementService.createSupplement(userId, supplementData);

    res.status(HTTP_STATUS.CREATED).json(supplement);
  });

  /**
   * Get all supplements for the user
   */
  getSupplements = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const activeOnly = req.query.activeOnly === 'true';
    const supplements = await supplementService.getSupplements(userId, activeOnly);

    res.status(HTTP_STATUS.OK).json(supplements);
  });

  /**
   * Get a single supplement by ID
   */
  getSupplementById = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const supplement = await supplementService.getSupplementById(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(supplement);
  });

  /**
   * Update a supplement
   */
  updateSupplement = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = updateSupplementSchema.parse(req.body);

    const supplementData = {
      ...validatedData,
      startDate: validatedData.startDate ? new Date(validatedData.startDate) : undefined,
      endDate: validatedData.endDate === null ? null : validatedData.endDate ? new Date(validatedData.endDate) : undefined,
    };

    const supplement = await supplementService.updateSupplement(userId, req.params.id, supplementData);

    res.status(HTTP_STATUS.OK).json(supplement);
  });

  /**
   * Delete a supplement
   */
  deleteSupplement = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const result = await supplementService.deleteSupplement(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(result);
  });

  /**
   * Log a supplement intake
   */
  logIntake = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = createSupplementLogSchema.parse(req.body);

    const logData = {
      ...validatedData,
      takenAt: validatedData.takenAt ? new Date(validatedData.takenAt) : undefined,
    };

    const log = await supplementService.logSupplementIntake(userId, logData);

    res.status(HTTP_STATUS.CREATED).json(log);
  });

  /**
   * Bulk log supplement intakes
   */
  bulkLogIntake = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = bulkCreateSupplementLogsSchema.parse(req.body);

    const logs = validatedData.logs.map(log => ({
      ...log,
      takenAt: log.takenAt ? new Date(log.takenAt) : undefined,
    }));

    const result = await supplementService.bulkLogSupplementIntake(userId, logs);

    res.status(HTTP_STATUS.CREATED).json(result);
  });

  /**
   * Get logs for a specific day
   */
  getLogsForDay = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const date = parseOptionalDate(req.query.date as string | undefined);
    const logs = await supplementService.getLogsForDay(userId, date);

    res.status(HTTP_STATUS.OK).json(logs);
  });

  /**
   * Get today's supplement status
   */
  getTodayStatus = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const status = await supplementService.getTodayStatus(userId);

    res.status(HTTP_STATUS.OK).json(status);
  });

  /**
   * Delete a supplement log
   */
  deleteLog = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const result = await supplementService.deleteLog(userId, req.params.logId);

    res.status(HTTP_STATUS.OK).json(result);
  });

  /**
   * Get supplement history/streak
   */
  getHistory = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const days = parseInt(req.query.days as string) || 30;
    const history = await supplementService.getSupplementHistory(userId, req.params.id, days);

    res.status(HTTP_STATUS.OK).json(history);
  });
}

export const supplementController = new SupplementController();
