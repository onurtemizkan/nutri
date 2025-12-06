import prisma from '../config/database';
import {
  CreateUserSupplementInput,
  UpdateUserSupplementInput,
  CreateSupplementLogInput,
  UpdateSupplementLogInput,
  GetSupplementsQuery,
  GetSupplementLogsQuery,
  WeeklySchedule,
  DayOfWeek,
} from '../types';
import { Prisma, SupplementCategory } from '@prisma/client';
import { DEFAULT_PAGE_LIMIT, WEEK_IN_DAYS } from '../config/constants';
import { getDayBoundaries, getDaysAgo, getStartOfDay } from '../utils/dateHelpers';

/**
 * Map day index (0=Sunday) to day name
 */
const DAY_INDEX_TO_NAME: DayOfWeek[] = [
  'sunday',
  'monday',
  'tuesday',
  'wednesday',
  'thursday',
  'friday',
  'saturday',
];

/**
 * SupplementService handles all business logic for supplement tracking:
 * - Master supplement list queries
 * - User supplement schedule management
 * - Supplement logging (intake tracking)
 * - Scheduled supplement calculations
 * - Adherence summaries
 */
export class SupplementService {
  // ==========================================================================
  // MASTER SUPPLEMENT LIST (read-only for users)
  // ==========================================================================

  /**
   * Get all supplements with optional filtering
   */
  async getSupplements(query: GetSupplementsQuery = {}) {
    const { category, search } = query;

    const where: Prisma.SupplementWhereInput = {};

    if (category) {
      where.category = category as SupplementCategory;
    }

    if (search) {
      where.name = {
        contains: search,
        mode: 'insensitive',
      };
    }

    const supplements = await prisma.supplement.findMany({
      where,
      orderBy: [{ category: 'asc' }, { name: 'asc' }],
    });

    return supplements;
  }

  /**
   * Get a specific supplement by ID
   */
  async getSupplementById(supplementId: string) {
    const supplement = await prisma.supplement.findUnique({
      where: { id: supplementId },
    });

    if (!supplement) {
      throw new Error('Supplement not found');
    }

    return supplement;
  }

  // ==========================================================================
  // USER SUPPLEMENT SCHEDULES
  // ==========================================================================

  /**
   * Create a new user supplement schedule
   */
  async createUserSupplement(userId: string, data: CreateUserSupplementInput) {
    // Verify the supplement exists
    await this.getSupplementById(data.supplementId);

    const userSupplement = await prisma.userSupplement.create({
      data: {
        userId,
        supplementId: data.supplementId,
        dosage: data.dosage,
        unit: data.unit,
        scheduleType: data.scheduleType,
        scheduleTimes: data.scheduleTimes
          ? (data.scheduleTimes as Prisma.InputJsonValue)
          : Prisma.JsonNull,
        weeklySchedule: data.weeklySchedule
          ? (data.weeklySchedule as Prisma.InputJsonValue)
          : Prisma.JsonNull,
        intervalDays: data.intervalDays || null,
        startDate: data.startDate,
        endDate: data.endDate || null,
        notes: data.notes || null,
      },
      include: {
        supplement: true,
      },
    });

    return userSupplement;
  }

  /**
   * Get all user supplements (schedules)
   */
  async getUserSupplements(userId: string, includeInactive = false) {
    const where: Prisma.UserSupplementWhereInput = { userId };

    if (!includeInactive) {
      where.isActive = true;
    }

    const userSupplements = await prisma.userSupplement.findMany({
      where,
      include: {
        supplement: true,
      },
      orderBy: {
        createdAt: 'desc',
      },
    });

    return userSupplements;
  }

  /**
   * Get a specific user supplement by ID
   */
  async getUserSupplementById(userId: string, userSupplementId: string) {
    const userSupplement = await prisma.userSupplement.findFirst({
      where: {
        id: userSupplementId,
        userId,
      },
      include: {
        supplement: true,
      },
    });

    if (!userSupplement) {
      throw new Error('User supplement not found');
    }

    return userSupplement;
  }

  /**
   * Update a user supplement schedule
   */
  async updateUserSupplement(
    userId: string,
    userSupplementId: string,
    data: UpdateUserSupplementInput
  ) {
    // Verify ownership
    await this.getUserSupplementById(userId, userSupplementId);

    const userSupplement = await prisma.userSupplement.update({
      where: { id: userSupplementId },
      data: {
        dosage: data.dosage,
        unit: data.unit,
        scheduleType: data.scheduleType,
        scheduleTimes: data.scheduleTimes,
        weeklySchedule: data.weeklySchedule as Prisma.InputJsonValue,
        intervalDays: data.intervalDays,
        startDate: data.startDate,
        endDate: data.endDate,
        isActive: data.isActive,
        notes: data.notes,
      },
      include: {
        supplement: true,
      },
    });

    return userSupplement;
  }

  /**
   * Deactivate a user supplement (soft delete)
   */
  async deleteUserSupplement(userId: string, userSupplementId: string) {
    // Verify ownership
    await this.getUserSupplementById(userId, userSupplementId);

    await prisma.userSupplement.update({
      where: { id: userSupplementId },
      data: { isActive: false },
    });

    return { message: 'Supplement schedule deactivated successfully' };
  }

  /**
   * Get scheduled supplements for a specific date
   * Calculates which supplements should be taken on a given day based on schedule rules
   */
  async getScheduledSupplements(userId: string, date: Date = new Date()) {
    const targetDate = getStartOfDay(date);
    const dayOfWeek = DAY_INDEX_TO_NAME[targetDate.getDay()];

    // Get active user supplements that are within their date range
    const userSupplements = await prisma.userSupplement.findMany({
      where: {
        userId,
        isActive: true,
        startDate: { lte: targetDate },
        OR: [{ endDate: null }, { endDate: { gte: targetDate } }],
      },
      include: {
        supplement: true,
      },
    });

    // Get today's logs to check what's already taken
    const { startOfDay, endOfDay } = getDayBoundaries(date);
    const todaysLogs = await prisma.supplementLog.findMany({
      where: {
        userId,
        takenAt: {
          gte: startOfDay,
          lte: endOfDay,
        },
      },
    });

    const scheduledSupplements = userSupplements
      .map((us) => {
        const scheduledTimes = this.calculateScheduledTimes(us, targetDate, dayOfWeek);

        if (scheduledTimes.length === 0) {
          return null;
        }

        // Check if already taken
        const logsForSupplement = todaysLogs.filter(
          (log) => log.userSupplementId === us.id || log.supplementId === us.supplementId
        );

        return {
          userSupplement: us,
          scheduledTimes,
          takenCount: logsForSupplement.length,
          taken: logsForSupplement.length >= scheduledTimes.length,
          logs: logsForSupplement,
        };
      })
      .filter((s) => s !== null);

    return scheduledSupplements;
  }

  /**
   * Calculate scheduled times for a user supplement on a given day
   */
  private calculateScheduledTimes(
    userSupplement: {
      scheduleType: string;
      scheduleTimes: Prisma.JsonValue;
      weeklySchedule: Prisma.JsonValue;
      intervalDays: number | null;
      startDate: Date;
    },
    targetDate: Date,
    dayOfWeek: DayOfWeek
  ): string[] {
    switch (userSupplement.scheduleType) {
      case 'DAILY':
        return ['00:00']; // Default time for daily supplements

      case 'DAILY_MULTIPLE':
        return (userSupplement.scheduleTimes as string[]) || [];

      case 'WEEKLY': {
        const weeklySchedule = userSupplement.weeklySchedule as WeeklySchedule | null;
        if (!weeklySchedule) return [];
        return weeklySchedule[dayOfWeek] || [];
      }

      case 'INTERVAL': {
        if (!userSupplement.intervalDays) return [];
        const daysSinceStart = Math.floor(
          (targetDate.getTime() - userSupplement.startDate.getTime()) / (1000 * 60 * 60 * 24)
        );
        if (daysSinceStart % userSupplement.intervalDays === 0) {
          return ['00:00']; // Default time for interval supplements
        }
        return [];
      }

      case 'ONE_TIME':
        // One-time supplements are not scheduled, they're logged manually
        return [];

      default:
        return [];
    }
  }

  // ==========================================================================
  // SUPPLEMENT LOGS (Intake Tracking)
  // ==========================================================================

  /**
   * Create a supplement log entry
   */
  async createSupplementLog(userId: string, data: CreateSupplementLogInput) {
    // Verify the supplement exists
    await this.getSupplementById(data.supplementId);

    // If userSupplementId provided, verify ownership
    if (data.userSupplementId) {
      await this.getUserSupplementById(userId, data.userSupplementId);
    }

    const log = await prisma.supplementLog.create({
      data: {
        userId,
        userSupplementId: data.userSupplementId || null,
        supplementId: data.supplementId,
        dosage: data.dosage,
        unit: data.unit,
        takenAt: data.takenAt,
        scheduledFor: data.scheduledFor || null,
        source: data.source,
        notes: data.notes || null,
      },
      include: {
        supplement: true,
        userSupplement: {
          include: {
            supplement: true,
          },
        },
      },
    });

    return log;
  }

  /**
   * Bulk create supplement logs
   */
  async bulkCreateSupplementLogs(userId: string, logs: CreateSupplementLogInput[]) {
    // Validate all supplements and user supplements exist
    for (const log of logs) {
      await this.getSupplementById(log.supplementId);
      if (log.userSupplementId) {
        await this.getUserSupplementById(userId, log.userSupplementId);
      }
    }

    const createdLogs = await prisma.$transaction(
      logs.map((log) =>
        prisma.supplementLog.create({
          data: {
            userId,
            userSupplementId: log.userSupplementId || null,
            supplementId: log.supplementId,
            dosage: log.dosage,
            unit: log.unit,
            takenAt: log.takenAt,
            scheduledFor: log.scheduledFor || null,
            source: log.source,
            notes: log.notes || null,
          },
          include: {
            supplement: true,
          },
        })
      )
    );

    return createdLogs;
  }

  /**
   * Get supplement logs with filtering
   */
  async getSupplementLogs(userId: string, query: GetSupplementLogsQuery = {}) {
    const { startDate, endDate, supplementId, userSupplementId } = query;

    const where: Prisma.SupplementLogWhereInput = { userId };

    if (supplementId) {
      where.supplementId = supplementId;
    }

    if (userSupplementId) {
      where.userSupplementId = userSupplementId;
    }

    if (startDate || endDate) {
      where.takenAt = {};
      if (startDate) {
        where.takenAt.gte = startDate;
      }
      if (endDate) {
        where.takenAt.lte = endDate;
      }
    }

    const logs = await prisma.supplementLog.findMany({
      where,
      include: {
        supplement: true,
        userSupplement: {
          include: {
            supplement: true,
          },
        },
      },
      orderBy: {
        takenAt: 'desc',
      },
      take: DEFAULT_PAGE_LIMIT,
    });

    return logs;
  }

  /**
   * Get a specific supplement log by ID
   */
  async getSupplementLogById(userId: string, logId: string) {
    const log = await prisma.supplementLog.findFirst({
      where: {
        id: logId,
        userId,
      },
      include: {
        supplement: true,
        userSupplement: {
          include: {
            supplement: true,
          },
        },
      },
    });

    if (!log) {
      throw new Error('Supplement log not found');
    }

    return log;
  }

  /**
   * Update a supplement log
   */
  async updateSupplementLog(userId: string, logId: string, data: UpdateSupplementLogInput) {
    // Verify ownership
    await this.getSupplementLogById(userId, logId);

    const log = await prisma.supplementLog.update({
      where: { id: logId },
      data: {
        dosage: data.dosage,
        unit: data.unit,
        takenAt: data.takenAt,
        notes: data.notes,
      },
      include: {
        supplement: true,
        userSupplement: {
          include: {
            supplement: true,
          },
        },
      },
    });

    return log;
  }

  /**
   * Delete a supplement log
   */
  async deleteSupplementLog(userId: string, logId: string) {
    // Verify ownership
    await this.getSupplementLogById(userId, logId);

    await prisma.supplementLog.delete({
      where: { id: logId },
    });

    return { message: 'Supplement log deleted successfully' };
  }

  // ==========================================================================
  // SUMMARIES AND ANALYTICS
  // ==========================================================================

  /**
   * Get daily supplement summary
   */
  async getDailySummary(userId: string, date: Date = new Date()) {
    const { startOfDay, endOfDay } = getDayBoundaries(date);

    // Get scheduled supplements for the day
    const scheduled = await this.getScheduledSupplements(userId, date);

    // Get all logs for the day
    const logs = await prisma.supplementLog.findMany({
      where: {
        userId,
        takenAt: {
          gte: startOfDay,
          lte: endOfDay,
        },
      },
      include: {
        supplement: true,
      },
      orderBy: {
        takenAt: 'asc',
      },
    });

    // Calculate totals
    const scheduledCount = scheduled.reduce((acc, s) => acc + (s?.scheduledTimes.length || 0), 0);
    const takenCount = logs.length;
    const adherencePercentage =
      scheduledCount > 0 ? Math.round((takenCount / scheduledCount) * 100) : 100;

    return {
      date: startOfDay.toISOString().split('T')[0],
      scheduledCount,
      takenCount,
      adherencePercentage,
      scheduled,
      logs,
    };
  }

  /**
   * Get weekly supplement summary
   */
  async getWeeklySummary(userId: string) {
    const days: Array<{
      date: string;
      scheduledCount: number;
      takenCount: number;
      adherencePercentage: number;
    }> = [];

    let totalScheduled = 0;
    let totalTaken = 0;

    // Get summary for each of the last 7 days
    for (let i = WEEK_IN_DAYS - 1; i >= 0; i--) {
      const date = getDaysAgo(i, true);
      const { startOfDay, endOfDay } = getDayBoundaries(date);

      // Get scheduled count for this day
      const scheduled = await this.getScheduledSupplements(userId, date);
      const scheduledCount = scheduled.reduce(
        (acc, s) => acc + (s?.scheduledTimes.length || 0),
        0
      );

      // Get taken count for this day
      const takenCount = await prisma.supplementLog.count({
        where: {
          userId,
          takenAt: {
            gte: startOfDay,
            lte: endOfDay,
          },
        },
      });

      const adherencePercentage =
        scheduledCount > 0 ? Math.round((takenCount / scheduledCount) * 100) : 100;

      days.push({
        date: startOfDay.toISOString().split('T')[0],
        scheduledCount,
        takenCount,
        adherencePercentage,
      });

      totalScheduled += scheduledCount;
      totalTaken += takenCount;
    }

    const averageAdherence =
      totalScheduled > 0 ? Math.round((totalTaken / totalScheduled) * 100) : 100;

    return {
      days,
      totalScheduled,
      totalTaken,
      averageAdherence,
    };
  }

  /**
   * Get supplement usage statistics by supplement
   */
  async getSupplementStats(userId: string, supplementId: string, days: number = 30) {
    const startDate = getDaysAgo(days);

    const logs = await prisma.supplementLog.findMany({
      where: {
        userId,
        supplementId,
        takenAt: {
          gte: startDate,
        },
      },
      orderBy: {
        takenAt: 'desc',
      },
    });

    if (logs.length === 0) {
      return null;
    }

    return {
      supplementId,
      days,
      totalLogs: logs.length,
      averagePerDay: Math.round((logs.length / days) * 10) / 10,
      lastTaken: logs[0].takenAt,
      logs,
    };
  }

  /**
   * Delete all supplement data for a user (for privacy/GDPR)
   */
  async deleteAllUserSupplementData(userId: string) {
    // Delete logs first (foreign key constraint)
    const logsResult = await prisma.supplementLog.deleteMany({
      where: { userId },
    });

    // Delete user supplements
    const supplementsResult = await prisma.userSupplement.deleteMany({
      where: { userId },
    });

    return {
      message: `Deleted ${logsResult.count} logs and ${supplementsResult.count} supplement schedules`,
    };
  }
}

export const supplementService = new SupplementService();
