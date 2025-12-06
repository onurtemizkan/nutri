import { supplementService } from '../services/supplementService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import { parseOptionalSupplementCategory } from '../utils/enumValidation';
import {
  createUserSupplementSchema,
  updateUserSupplementSchema,
  createSupplementLogSchema,
  updateSupplementLogSchema,
  bulkCreateSupplementLogsSchema,
  getSupplementsQuerySchema,
  getSupplementLogsQuerySchema,
  getScheduledSupplementsQuerySchema,
} from '../validation/schemas';
import { withErrorHandling, ErrorHandlers } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { parseOptionalDate } from '../utils/dateHelpers';

/**
 * SupplementController handles all HTTP requests for supplement tracking:
 * - Master supplement list (read-only)
 * - User supplement schedules (CRUD)
 * - Supplement logs (intake tracking)
 * - Summaries and analytics
 */
export class SupplementController {
  // ==========================================================================
  // MASTER SUPPLEMENT LIST (read-only for users)
  // ==========================================================================

  /**
   * GET /supplements
   * Get all supplements with optional filtering by category or search
   */
  getSupplements = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedQuery = getSupplementsQuerySchema.parse(req.query);

    const query = {
      category: parseOptionalSupplementCategory(validatedQuery.category),
      search: validatedQuery.search,
    };

    const supplements = await supplementService.getSupplements(query);

    res.status(HTTP_STATUS.OK).json(supplements);
  });

  /**
   * GET /supplements/:id
   * Get a specific supplement by ID
   */
  getSupplementById = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const supplement = await supplementService.getSupplementById(req.params.id);

    res.status(HTTP_STATUS.OK).json(supplement);
  });

  // ==========================================================================
  // USER SUPPLEMENT SCHEDULES
  // ==========================================================================

  /**
   * POST /user-supplements
   * Create a new user supplement schedule
   */
  createUserSupplement = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = createUserSupplementSchema.parse(req.body);

    const supplementData = {
      supplementId: validatedData.supplementId,
      dosage: validatedData.dosage,
      unit: validatedData.unit,
      scheduleType: validatedData.scheduleType,
      scheduleTimes: validatedData.scheduleTimes,
      weeklySchedule: validatedData.weeklySchedule,
      intervalDays: validatedData.intervalDays,
      startDate: new Date(validatedData.startDate),
      endDate: validatedData.endDate ? new Date(validatedData.endDate) : undefined,
      notes: validatedData.notes,
    };

    const userSupplement = await supplementService.createUserSupplement(userId, supplementData);

    res.status(HTTP_STATUS.CREATED).json(userSupplement);
  });

  /**
   * GET /user-supplements
   * Get all user supplement schedules
   */
  getUserSupplements = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const includeInactive = req.query.includeInactive === 'true';

    const userSupplements = await supplementService.getUserSupplements(userId, includeInactive);

    res.status(HTTP_STATUS.OK).json(userSupplements);
  });

  /**
   * GET /user-supplements/:id
   * Get a specific user supplement by ID
   */
  getUserSupplementById = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const userSupplement = await supplementService.getUserSupplementById(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(userSupplement);
  });

  /**
   * PUT /user-supplements/:id
   * Update a user supplement schedule
   */
  updateUserSupplement = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = updateUserSupplementSchema.parse(req.body);

    const supplementData = {
      ...validatedData,
      startDate: validatedData.startDate ? new Date(validatedData.startDate) : undefined,
      endDate: validatedData.endDate !== undefined
        ? (validatedData.endDate ? new Date(validatedData.endDate) : null)
        : undefined,
    };

    const userSupplement = await supplementService.updateUserSupplement(
      userId,
      req.params.id,
      supplementData
    );

    res.status(HTTP_STATUS.OK).json(userSupplement);
  });

  /**
   * DELETE /user-supplements/:id
   * Deactivate a user supplement (soft delete)
   */
  deleteUserSupplement = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const result = await supplementService.deleteUserSupplement(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(result);
  });

  /**
   * GET /user-supplements/scheduled
   * Get supplements scheduled for a specific date (defaults to today)
   */
  getScheduledSupplements = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedQuery = getScheduledSupplementsQuerySchema.parse(req.query);
    const date = validatedQuery.date ? new Date(validatedQuery.date) : new Date();

    const scheduled = await supplementService.getScheduledSupplements(userId, date);

    res.status(HTTP_STATUS.OK).json(scheduled);
  });

  // ==========================================================================
  // SUPPLEMENT LOGS (Intake Tracking)
  // ==========================================================================

  /**
   * POST /supplement-logs
   * Create a supplement log entry
   */
  createSupplementLog = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = createSupplementLogSchema.parse(req.body);

    const logData = {
      userSupplementId: validatedData.userSupplementId,
      supplementId: validatedData.supplementId,
      dosage: validatedData.dosage,
      unit: validatedData.unit,
      takenAt: new Date(validatedData.takenAt),
      scheduledFor: validatedData.scheduledFor ? new Date(validatedData.scheduledFor) : undefined,
      source: validatedData.source,
      notes: validatedData.notes,
    };

    const log = await supplementService.createSupplementLog(userId, logData);

    res.status(HTTP_STATUS.CREATED).json(log);
  });

  /**
   * POST /supplement-logs/bulk
   * Create multiple supplement log entries at once
   */
  createBulkSupplementLogs = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = bulkCreateSupplementLogsSchema.parse(req.body);

    const logsData = validatedData.logs.map((log) => ({
      userSupplementId: log.userSupplementId,
      supplementId: log.supplementId,
      dosage: log.dosage,
      unit: log.unit,
      takenAt: new Date(log.takenAt),
      scheduledFor: log.scheduledFor ? new Date(log.scheduledFor) : undefined,
      source: log.source,
      notes: log.notes,
    }));

    const logs = await supplementService.bulkCreateSupplementLogs(userId, logsData);

    res.status(HTTP_STATUS.CREATED).json(logs);
  });

  /**
   * GET /supplement-logs
   * Get supplement logs with optional filtering
   */
  getSupplementLogs = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedQuery = getSupplementLogsQuerySchema.parse(req.query);

    const query = {
      startDate: parseOptionalDate(validatedQuery.startDate),
      endDate: parseOptionalDate(validatedQuery.endDate),
      supplementId: validatedQuery.supplementId,
      userSupplementId: validatedQuery.userSupplementId,
    };

    const logs = await supplementService.getSupplementLogs(userId, query);

    res.status(HTTP_STATUS.OK).json(logs);
  });

  /**
   * GET /supplement-logs/:id
   * Get a specific supplement log by ID
   */
  getSupplementLogById = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const log = await supplementService.getSupplementLogById(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(log);
  });

  /**
   * PUT /supplement-logs/:id
   * Update a supplement log
   */
  updateSupplementLog = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = updateSupplementLogSchema.parse(req.body);

    const logData = {
      ...validatedData,
      takenAt: validatedData.takenAt ? new Date(validatedData.takenAt) : undefined,
    };

    const log = await supplementService.updateSupplementLog(userId, req.params.id, logData);

    res.status(HTTP_STATUS.OK).json(log);
  });

  /**
   * DELETE /supplement-logs/:id
   * Delete a supplement log
   */
  deleteSupplementLog = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const result = await supplementService.deleteSupplementLog(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json(result);
  });

  // ==========================================================================
  // SUMMARIES AND ANALYTICS
  // ==========================================================================

  /**
   * GET /supplements/summary/daily
   * Get daily supplement summary (scheduled vs taken)
   */
  getDailySummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const date = parseOptionalDate(req.query.date as string | undefined);

    const summary = await supplementService.getDailySummary(userId, date ?? new Date());

    res.status(HTTP_STATUS.OK).json(summary);
  });

  /**
   * GET /supplements/summary/weekly
   * Get weekly supplement adherence summary
   */
  getWeeklySummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const summary = await supplementService.getWeeklySummary(userId);

    res.status(HTTP_STATUS.OK).json(summary);
  });

  /**
   * GET /supplements/:id/stats
   * Get usage statistics for a specific supplement
   */
  getSupplementStats = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const days = req.query.days ? parseInt(req.query.days as string, 10) : 30;

    const stats = await supplementService.getSupplementStats(userId, req.params.id, days);

    if (!stats) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'No logs found for this supplement' });
      return;
    }

    res.status(HTTP_STATUS.OK).json(stats);
  });

  /**
   * DELETE /supplements/user-data
   * Delete all supplement data for a user (GDPR/privacy)
   */
  deleteAllUserSupplementData = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const result = await supplementService.deleteAllUserSupplementData(userId);

    res.status(HTTP_STATUS.OK).json(result);
  });
}

export const supplementController = new SupplementController();
