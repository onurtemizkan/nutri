import prisma from '../config/database';
import { SupplementFrequency, SupplementTimeOfDay } from '@prisma/client';
import { getDayBoundaries } from '../utils/dateHelpers';

export interface CreateSupplementInput {
  name: string;
  brand?: string;
  dosageAmount: number;
  dosageUnit: string;
  frequency?: SupplementFrequency;
  timesPerDay?: number;
  timeOfDay?: SupplementTimeOfDay[];
  withFood?: boolean;
  isActive?: boolean;
  startDate?: Date;
  endDate?: Date | null;
  notes?: string;
  color?: string;
}

export interface UpdateSupplementInput {
  name?: string;
  brand?: string;
  dosageAmount?: number;
  dosageUnit?: string;
  frequency?: SupplementFrequency;
  timesPerDay?: number;
  timeOfDay?: SupplementTimeOfDay[];
  withFood?: boolean;
  isActive?: boolean;
  startDate?: Date;
  endDate?: Date | null;
  notes?: string;
  color?: string;
}

export interface CreateSupplementLogInput {
  supplementId: string;
  takenAt?: Date;
  dosageAmount?: number;
  notes?: string;
  skipped?: boolean;
}

export class SupplementService {
  /**
   * Create a new supplement
   */
  async createSupplement(userId: string, data: CreateSupplementInput) {
    const supplement = await prisma.supplement.create({
      data: {
        userId,
        name: data.name,
        brand: data.brand,
        dosageAmount: data.dosageAmount,
        dosageUnit: data.dosageUnit,
        frequency: data.frequency || 'DAILY',
        timesPerDay: data.timesPerDay || 1,
        timeOfDay: data.timeOfDay || [],
        withFood: data.withFood || false,
        isActive: data.isActive ?? true,
        startDate: data.startDate || new Date(),
        endDate: data.endDate,
        notes: data.notes,
        color: data.color,
      },
    });

    return supplement;
  }

  /**
   * Get all supplements for a user
   */
  async getSupplements(userId: string, activeOnly: boolean = false) {
    const supplements = await prisma.supplement.findMany({
      where: {
        userId,
        ...(activeOnly ? { isActive: true } : {}),
      },
      orderBy: [
        { isActive: 'desc' },
        { name: 'asc' },
      ],
    });

    return supplements;
  }

  /**
   * Get a single supplement by ID
   */
  async getSupplementById(userId: string, supplementId: string) {
    const supplement = await prisma.supplement.findFirst({
      where: {
        id: supplementId,
        userId,
      },
      include: {
        logs: {
          orderBy: { takenAt: 'desc' },
          take: 10,
        },
      },
    });

    if (!supplement) {
      throw new Error('Supplement not found');
    }

    return supplement;
  }

  /**
   * Update a supplement
   */
  async updateSupplement(userId: string, supplementId: string, data: UpdateSupplementInput) {
    // Verify supplement belongs to user
    await this.getSupplementById(userId, supplementId);

    const supplement = await prisma.supplement.update({
      where: { id: supplementId },
      data,
    });

    return supplement;
  }

  /**
   * Delete a supplement
   */
  async deleteSupplement(userId: string, supplementId: string) {
    // Verify supplement belongs to user
    await this.getSupplementById(userId, supplementId);

    await prisma.supplement.delete({
      where: { id: supplementId },
    });

    return { message: 'Supplement deleted successfully' };
  }

  /**
   * Log a supplement intake
   */
  async logSupplementIntake(userId: string, data: CreateSupplementLogInput) {
    // Verify supplement belongs to user
    const supplement = await prisma.supplement.findFirst({
      where: {
        id: data.supplementId,
        userId,
      },
    });

    if (!supplement) {
      throw new Error('Supplement not found');
    }

    const log = await prisma.supplementLog.create({
      data: {
        userId,
        supplementId: data.supplementId,
        takenAt: data.takenAt || new Date(),
        dosageAmount: data.dosageAmount,
        notes: data.notes,
        skipped: data.skipped || false,
      },
      include: {
        supplement: true,
      },
    });

    return log;
  }

  /**
   * Bulk log supplement intakes
   */
  async bulkLogSupplementIntake(userId: string, logs: CreateSupplementLogInput[]) {
    // Verify all supplements belong to user
    const supplementIds = [...new Set(logs.map(log => log.supplementId))];
    const supplements = await prisma.supplement.findMany({
      where: {
        id: { in: supplementIds },
        userId,
      },
    });

    if (supplements.length !== supplementIds.length) {
      throw new Error('One or more supplements not found');
    }

    const createdLogs = await prisma.supplementLog.createMany({
      data: logs.map(log => ({
        userId,
        supplementId: log.supplementId,
        takenAt: log.takenAt || new Date(),
        dosageAmount: log.dosageAmount,
        notes: log.notes,
        skipped: log.skipped || false,
      })),
    });

    return { count: createdLogs.count };
  }

  /**
   * Get supplement logs for a specific day
   */
  async getLogsForDay(userId: string, date?: Date) {
    const { startOfDay, endOfDay } = getDayBoundaries(date);

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
      orderBy: { takenAt: 'asc' },
    });

    return logs;
  }

  /**
   * Get today's supplement status (which have been taken, which are pending)
   */
  async getTodayStatus(userId: string) {
    const { startOfDay, endOfDay } = getDayBoundaries();

    // Get all active supplements
    const supplements = await prisma.supplement.findMany({
      where: {
        userId,
        isActive: true,
      },
      orderBy: { name: 'asc' },
    });

    // Get today's logs
    const todayLogs = await prisma.supplementLog.findMany({
      where: {
        userId,
        takenAt: {
          gte: startOfDay,
          lte: endOfDay,
        },
      },
    });

    // Build status for each supplement
    const status = supplements.map(supplement => {
      const logsForSupplement = todayLogs.filter(
        log => log.supplementId === supplement.id
      );
      const takenCount = logsForSupplement.filter(log => !log.skipped).length;
      const skippedCount = logsForSupplement.filter(log => log.skipped).length;
      const targetCount = supplement.timesPerDay;

      return {
        supplement,
        takenCount,
        skippedCount,
        targetCount,
        isComplete: takenCount >= targetCount,
        logs: logsForSupplement,
      };
    });

    const totalSupplements = supplements.length;
    const completedSupplements = status.filter(s => s.isComplete).length;

    return {
      date: startOfDay,
      totalSupplements,
      completedSupplements,
      completionRate: totalSupplements > 0 ? (completedSupplements / totalSupplements) * 100 : 0,
      supplements: status,
    };
  }

  /**
   * Delete a supplement log
   */
  async deleteLog(userId: string, logId: string) {
    const log = await prisma.supplementLog.findFirst({
      where: {
        id: logId,
        userId,
      },
    });

    if (!log) {
      throw new Error('Log not found');
    }

    await prisma.supplementLog.delete({
      where: { id: logId },
    });

    return { message: 'Log deleted successfully' };
  }

  /**
   * Get supplement history/streak
   */
  async getSupplementHistory(userId: string, supplementId: string, days: number = 30) {
    // Verify supplement belongs to user
    const supplement = await this.getSupplementById(userId, supplementId);

    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    startDate.setHours(0, 0, 0, 0);

    const logs = await prisma.supplementLog.findMany({
      where: {
        supplementId,
        userId,
        takenAt: {
          gte: startDate,
        },
      },
      orderBy: { takenAt: 'desc' },
    });

    // Calculate daily stats
    const dailyStats: Record<string, { taken: number; skipped: number }> = {};

    logs.forEach(log => {
      const dateKey = log.takenAt.toISOString().split('T')[0];
      if (!dailyStats[dateKey]) {
        dailyStats[dateKey] = { taken: 0, skipped: 0 };
      }
      if (log.skipped) {
        dailyStats[dateKey].skipped++;
      } else {
        dailyStats[dateKey].taken++;
      }
    });

    // Calculate streak
    let currentStreak = 0;
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    for (let i = 0; i < days; i++) {
      const checkDate = new Date(today);
      checkDate.setDate(checkDate.getDate() - i);
      const dateKey = checkDate.toISOString().split('T')[0];

      const stats = dailyStats[dateKey];
      if (stats && stats.taken >= supplement.timesPerDay) {
        currentStreak++;
      } else if (i > 0) {
        // Don't break streak if today hasn't been completed yet
        break;
      }
    }

    return {
      supplement,
      days,
      logs,
      dailyStats,
      currentStreak,
      totalTaken: logs.filter(l => !l.skipped).length,
      totalSkipped: logs.filter(l => l.skipped).length,
    };
  }
}

export const supplementService = new SupplementService();
