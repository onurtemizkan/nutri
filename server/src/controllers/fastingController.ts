import { fastingService } from '../services/fastingService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import { withErrorHandling, ErrorHandlers } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { z } from 'zod';
import { FastingStatus } from '@prisma/client';

// Validation schemas
const createProtocolSchema = z
  .object({
    name: z.string().min(1).max(50),
    description: z.string().max(500).optional(),
    fastingHours: z.number().int().min(1).max(48),
    eatingHours: z.number().int().min(0).max(24),
    icon: z.string().max(50).optional(),
    color: z
      .string()
      .regex(/^#[0-9A-Fa-f]{6}$/)
      .optional(),
  })
  .refine(
    (data) => {
      // For daily protocols with an eating window, hours must sum to 24
      // Extended fasts (no eating window) don't have this constraint
      if (data.eatingHours > 0) {
        return data.fastingHours + data.eatingHours === 24;
      }
      return true;
    },
    {
      message: 'For daily protocols, fasting hours + eating hours must equal 24',
      path: ['eatingHours'],
    }
  );

const updateSettingsSchema = z.object({
  activeProtocolId: z.string().optional(),
  preferredStartTime: z
    .string()
    .regex(/^([01]\d|2[0-3]):([0-5]\d)$/)
    .optional(),
  preferredEndTime: z
    .string()
    .regex(/^([01]\d|2[0-3]):([0-5]\d)$/)
    .optional(),
  notifyOnFastStart: z.boolean().optional(),
  notifyOnFastEnd: z.boolean().optional(),
  notifyBeforeEnd: z.number().int().min(0).max(120).optional(),
  weeklyFastingGoal: z.number().int().min(0).max(168).optional(),
  monthlyFastingGoal: z.number().int().min(0).max(720).optional(),
  showOnDashboard: z.boolean().optional(),
});

const startFastSchema = z.object({
  protocolId: z.string().optional(),
  customDuration: z.number().int().min(60).max(2880).optional(), // 1 hour to 48 hours
  startTime: z.string().datetime().optional(),
  notes: z.string().max(500).optional(),
});

const endFastSchema = z.object({
  moodRating: z.number().int().min(1).max(5).optional(),
  energyLevel: z.number().int().min(1).max(5).optional(),
  hungerLevel: z.number().int().min(1).max(5).optional(),
  notes: z.string().max(500).optional(),
  earlyEndReason: z.string().max(200).optional(),
  endWeight: z.number().positive().optional(),
});

const checkpointSchema = z.object({
  moodRating: z.number().int().min(1).max(5).optional(),
  energyLevel: z.number().int().min(1).max(5).optional(),
  hungerLevel: z.number().int().min(1).max(5).optional(),
  notes: z.string().max(500).optional(),
});

const historyQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(100).optional().default(20),
  offset: z.coerce.number().int().min(0).optional().default(0),
  status: z.nativeEnum(FastingStatus).optional(),
});

const analyticsQuerySchema = z.object({
  days: z.coerce.number().int().min(1).max(365).optional().default(30),
});

export class FastingController {
  // ============================================================================
  // PROTOCOLS
  // ============================================================================

  getProtocols = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const protocols = await fastingService.getProtocols(userId);

    res.status(HTTP_STATUS.OK).json(protocols);
  });

  createProtocol = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const data = createProtocolSchema.parse(req.body);
    const protocol = await fastingService.createCustomProtocol(userId, data);

    res.status(HTTP_STATUS.CREATED).json(protocol);
  });

  deleteProtocol = ErrorHandlers.withNotFound<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    await fastingService.deleteCustomProtocol(userId, req.params.id);

    res.status(HTTP_STATUS.OK).json({ message: 'Protocol deleted successfully' });
  });

  // ============================================================================
  // SETTINGS
  // ============================================================================

  getSettings = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const settings = await fastingService.getOrCreateSettings(userId);

    res.status(HTTP_STATUS.OK).json(settings);
  });

  updateSettings = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const data = updateSettingsSchema.parse(req.body);
    const settings = await fastingService.updateSettings(userId, data);

    res.status(HTTP_STATUS.OK).json(settings);
  });

  // ============================================================================
  // TIMER / SESSIONS
  // ============================================================================

  getTimerStatus = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const status = await fastingService.getTimerStatus(userId);

    res.status(HTTP_STATUS.OK).json(status);
  });

  startFast = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const data = startFastSchema.parse(req.body);
    const session = await fastingService.startFast(userId, {
      ...data,
      startTime: data.startTime ? new Date(data.startTime) : undefined,
    });

    res.status(HTTP_STATUS.CREATED).json(session);
  });

  endFast = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const data = endFastSchema.parse(req.body);
    const session = await fastingService.endFast(userId, req.params.id, data);

    res.status(HTTP_STATUS.OK).json(session);
  });

  addCheckpoint = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const data = checkpointSchema.parse(req.body);
    const checkpoint = await fastingService.addCheckpoint(userId, req.params.id, data);

    res.status(HTTP_STATUS.CREATED).json(checkpoint);
  });

  // ============================================================================
  // HISTORY
  // ============================================================================

  getHistory = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const query = historyQuerySchema.parse(req.query);
    const history = await fastingService.getFastingHistory(userId, query);

    res.status(HTTP_STATUS.OK).json(history);
  });

  // ============================================================================
  // STREAKS & ANALYTICS
  // ============================================================================

  getStreak = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const streak = await fastingService.getStreakStats(userId);

    res.status(HTTP_STATUS.OK).json(streak);
  });

  getAnalytics = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { days } = analyticsQuerySchema.parse(req.query);
    const analytics = await fastingService.getAnalytics(userId, days);

    res.status(HTTP_STATUS.OK).json(analytics);
  });
}

export const fastingController = new FastingController();
