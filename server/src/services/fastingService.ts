import prisma from '../config/database';
import { logger } from '../config/logger';
import { FastingStatus } from '@prisma/client';

// Default system protocols
const SYSTEM_PROTOCOLS = [
  {
    name: '16:8',
    description:
      'Fast for 16 hours, eat within 8 hours. The most popular intermittent fasting method.',
    fastingHours: 16,
    eatingHours: 8,
    icon: 'clock',
    color: '#4CAF50',
  },
  {
    name: '18:6',
    description: 'Fast for 18 hours, eat within 6 hours. A more advanced fasting window.',
    fastingHours: 18,
    eatingHours: 6,
    icon: 'clock',
    color: '#2196F3',
  },
  {
    name: '20:4',
    description: 'Fast for 20 hours, eat within 4 hours. Also known as the Warrior Diet.',
    fastingHours: 20,
    eatingHours: 4,
    icon: 'sword',
    color: '#FF9800',
  },
  {
    name: 'OMAD',
    description: 'One Meal A Day - Fast for 23 hours, eat within 1 hour.',
    fastingHours: 23,
    eatingHours: 1,
    icon: 'utensils',
    color: '#9C27B0',
  },
  {
    name: '5:2',
    description: 'Eat normally 5 days, restrict to 500-600 calories 2 days per week.',
    fastingHours: 0,
    eatingHours: 24,
    isWeeklyProtocol: true,
    fastingDaysPerWeek: 2,
    caloriesOnFastDays: 500,
    icon: 'calendar',
    color: '#E91E63',
  },
  {
    name: '14:10',
    description: 'Fast for 14 hours, eat within 10 hours. A gentle introduction to fasting.',
    fastingHours: 14,
    eatingHours: 10,
    icon: 'clock',
    color: '#00BCD4',
  },
];

interface StartFastingOptions {
  protocolId?: string;
  customDuration?: number; // In minutes
  startTime?: Date;
  notes?: string;
}

interface EndFastingOptions {
  moodRating?: number;
  energyLevel?: number;
  hungerLevel?: number;
  notes?: string;
  breakFastReason?: string;
  endWeight?: number;
}

interface CheckpointData {
  moodRating?: number;
  energyLevel?: number;
  hungerLevel?: number;
  notes?: string;
}

/**
 * Fasting Service
 * Handles fasting sessions, protocols, streaks, and analytics
 */
class FastingService {
  // ============================================================================
  // PROTOCOL MANAGEMENT
  // ============================================================================

  async initializeSystemProtocols() {
    for (const protocol of SYSTEM_PROTOCOLS) {
      const existing = await prisma.fastingProtocol.findFirst({
        where: { name: protocol.name, isSystem: true },
      });

      if (!existing) {
        await prisma.fastingProtocol.create({
          data: {
            ...protocol,
            isSystem: true,
            userId: null,
          },
        });
      }
    }
    logger.info('System fasting protocols initialized');
  }

  async getProtocols(userId: string) {
    const protocols = await prisma.fastingProtocol.findMany({
      where: {
        OR: [{ isSystem: true }, { userId }],
        isActive: true,
      },
      orderBy: [{ isSystem: 'desc' }, { name: 'asc' }],
    });

    return protocols;
  }

  async createCustomProtocol(
    userId: string,
    data: {
      name: string;
      description?: string;
      fastingHours: number;
      eatingHours: number;
      icon?: string;
      color?: string;
    }
  ) {
    const protocol = await prisma.fastingProtocol.create({
      data: {
        userId,
        name: data.name,
        description: data.description,
        fastingHours: data.fastingHours,
        eatingHours: data.eatingHours,
        icon: data.icon,
        color: data.color,
        isSystem: false,
      },
    });

    logger.info({ userId, protocolId: protocol.id }, 'Created custom fasting protocol');
    return protocol;
  }

  async deleteCustomProtocol(userId: string, protocolId: string) {
    const protocol = await prisma.fastingProtocol.findFirst({
      where: { id: protocolId, userId, isSystem: false },
    });

    if (!protocol) {
      throw new Error('Protocol not found or cannot be deleted');
    }

    await prisma.fastingProtocol.update({
      where: { id: protocolId },
      data: { isActive: false },
    });

    logger.info({ userId, protocolId }, 'Deleted custom fasting protocol');
  }

  // ============================================================================
  // USER SETTINGS
  // ============================================================================

  async getOrCreateSettings(userId: string) {
    let settings = await prisma.userFastingSettings.findUnique({
      where: { userId },
      include: { activeProtocol: true },
    });

    if (!settings) {
      // Default to 16:8 protocol
      const defaultProtocol = await prisma.fastingProtocol.findFirst({
        where: { name: '16:8', isSystem: true },
      });

      settings = await prisma.userFastingSettings.create({
        data: {
          userId,
          activeProtocolId: defaultProtocol?.id,
        },
        include: { activeProtocol: true },
      });
    }

    return settings;
  }

  async updateSettings(
    userId: string,
    data: {
      activeProtocolId?: string;
      preferredStartTime?: string;
      preferredEndTime?: string;
      notifyOnFastStart?: boolean;
      notifyOnFastEnd?: boolean;
      notifyBeforeEnd?: number;
      weeklyFastingGoal?: number;
      monthlyFastingGoal?: number;
      showOnDashboard?: boolean;
    }
  ) {
    const settings = await prisma.userFastingSettings.upsert({
      where: { userId },
      create: { userId, ...data },
      update: data,
      include: { activeProtocol: true },
    });

    logger.info({ userId }, 'Updated fasting settings');
    return settings;
  }

  // ============================================================================
  // FASTING SESSIONS
  // ============================================================================

  async getActiveFast(userId: string) {
    return prisma.fastingSession.findFirst({
      where: { userId, status: FastingStatus.ACTIVE },
      include: {
        protocol: true,
        checkpoints: { orderBy: { hoursIntoFast: 'asc' } },
      },
    });
  }

  async startFast(userId: string, options: StartFastingOptions = {}) {
    // Check for active fast
    const activeFast = await this.getActiveFast(userId);
    if (activeFast) {
      throw new Error('You already have an active fast');
    }

    const settings = await this.getOrCreateSettings(userId);
    const startTime = options.startTime || new Date();

    let plannedDuration: number;
    let protocolId: string | null = null;

    if (options.customDuration) {
      plannedDuration = options.customDuration;
    } else if (options.protocolId) {
      const protocol = await prisma.fastingProtocol.findUnique({
        where: { id: options.protocolId },
      });
      if (!protocol) {
        throw new Error('Protocol not found');
      }
      plannedDuration = protocol.fastingHours * 60;
      protocolId = protocol.id;
    } else if (settings.activeProtocol) {
      plannedDuration = settings.activeProtocol.fastingHours * 60;
      protocolId = settings.activeProtocol.id;
    } else {
      // Default to 16 hours
      plannedDuration = 16 * 60;
    }

    const plannedEndAt = new Date(startTime.getTime() + plannedDuration * 60 * 1000);

    // Get start weight if available
    const latestWeight = await prisma.weightRecord.findFirst({
      where: { userId },
      orderBy: { recordedAt: 'desc' },
    });

    const session = await prisma.fastingSession.create({
      data: {
        userId,
        protocolId,
        startedAt: startTime,
        plannedEndAt,
        plannedDuration,
        notes: options.notes,
        startWeight: latestWeight?.weight,
      },
      include: { protocol: true },
    });

    logger.info({ userId, sessionId: session.id, plannedDuration }, 'Started fasting session');

    return session;
  }

  async endFast(userId: string, sessionId: string, options: EndFastingOptions = {}) {
    const session = await prisma.fastingSession.findFirst({
      where: { id: sessionId, userId, status: FastingStatus.ACTIVE },
    });

    if (!session) {
      throw new Error('Active fasting session not found');
    }

    const actualEndAt = new Date();
    const actualDuration = Math.round(
      (actualEndAt.getTime() - session.startedAt.getTime()) / (60 * 1000)
    );

    // Determine status
    let status: FastingStatus;
    if (actualDuration >= session.plannedDuration * 0.9) {
      status = FastingStatus.COMPLETED;
    } else if (actualDuration < 60) {
      status = FastingStatus.CANCELLED;
    } else {
      status = FastingStatus.BROKEN;
    }

    const updatedSession = await prisma.fastingSession.update({
      where: { id: sessionId },
      data: {
        actualEndAt,
        actualDuration,
        status,
        moodRating: options.moodRating,
        energyLevel: options.energyLevel,
        hungerLevel: options.hungerLevel,
        notes: options.notes || session.notes,
        breakFastReason: options.breakFastReason,
        endWeight: options.endWeight,
      },
      include: { protocol: true },
    });

    // Update streak
    await this.updateStreak(userId, status, actualDuration);

    logger.info({ userId, sessionId, status, actualDuration }, 'Ended fasting session');

    return updatedSession;
  }

  async addCheckpoint(userId: string, sessionId: string, data: CheckpointData) {
    const session = await prisma.fastingSession.findFirst({
      where: { id: sessionId, userId, status: FastingStatus.ACTIVE },
    });

    if (!session) {
      throw new Error('Active fasting session not found');
    }

    const hoursIntoFast = Math.floor((Date.now() - session.startedAt.getTime()) / (60 * 60 * 1000));

    const checkpoint = await prisma.fastingCheckpoint.create({
      data: {
        sessionId,
        hoursIntoFast,
        moodRating: data.moodRating,
        energyLevel: data.energyLevel,
        hungerLevel: data.hungerLevel,
        notes: data.notes,
      },
    });

    logger.info({ userId, sessionId, hoursIntoFast }, 'Added fasting checkpoint');
    return checkpoint;
  }

  async getFastingHistory(
    userId: string,
    options: { limit?: number; offset?: number; status?: FastingStatus } = {}
  ) {
    const { limit = 20, offset = 0, status } = options;

    const [sessions, total] = await Promise.all([
      prisma.fastingSession.findMany({
        where: {
          userId,
          ...(status && { status }),
        },
        include: {
          protocol: true,
          checkpoints: { orderBy: { hoursIntoFast: 'asc' } },
        },
        orderBy: { startedAt: 'desc' },
        skip: offset,
        take: limit,
      }),
      prisma.fastingSession.count({
        where: {
          userId,
          ...(status && { status }),
        },
      }),
    ]);

    return {
      data: sessions,
      total,
      limit,
      offset,
      hasMore: offset + sessions.length < total,
    };
  }

  // ============================================================================
  // STREAK MANAGEMENT
  // ============================================================================

  async getOrCreateStreak(userId: string) {
    let streak = await prisma.fastingStreak.findUnique({
      where: { userId },
    });

    if (!streak) {
      streak = await prisma.fastingStreak.create({
        data: { userId },
      });
    }

    return streak;
  }

  private async updateStreak(userId: string, status: FastingStatus, durationMinutes: number) {
    const streak = await this.getOrCreateStreak(userId);
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const updates: {
      totalFasts: { increment: number };
      totalMinutes: { increment: number };
      successfulFasts?: { increment: number };
      brokenFasts?: { increment: number };
      currentStreak?: number;
      longestStreak?: number;
      lastFastDate?: Date;
      weeklyMinutes?: { increment: number } | number;
      monthlyMinutes?: { increment: number } | number;
      weekStartDate?: Date;
      monthStartDate?: Date;
    } = {
      totalFasts: { increment: 1 },
      totalMinutes: { increment: durationMinutes },
    };

    // Check for weekly/monthly reset
    const weekAgo = new Date(today);
    weekAgo.setDate(weekAgo.getDate() - 7);
    const monthAgo = new Date(today);
    monthAgo.setMonth(monthAgo.getMonth() - 1);

    if (new Date(streak.weekStartDate) < weekAgo) {
      updates.weeklyMinutes = durationMinutes;
      updates.weekStartDate = today;
    } else {
      updates.weeklyMinutes = { increment: durationMinutes };
    }

    if (new Date(streak.monthStartDate) < monthAgo) {
      updates.monthlyMinutes = durationMinutes;
      updates.monthStartDate = today;
    } else {
      updates.monthlyMinutes = { increment: durationMinutes };
    }

    if (status === FastingStatus.COMPLETED) {
      updates.successfulFasts = { increment: 1 };

      // Update streak
      const lastFastDate = streak.lastFastDate ? new Date(streak.lastFastDate) : null;
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);

      if (!lastFastDate) {
        updates.currentStreak = 1;
      } else if (
        lastFastDate.getTime() === yesterday.getTime() ||
        lastFastDate.getTime() === today.getTime()
      ) {
        updates.currentStreak = streak.currentStreak + 1;
      } else {
        updates.currentStreak = 1;
      }

      updates.longestStreak = Math.max(updates.currentStreak, streak.longestStreak);
      updates.lastFastDate = today;
    } else if (status === FastingStatus.BROKEN) {
      updates.brokenFasts = { increment: 1 };
      // Don't reset streak for broken fasts, but don't increment either
    }

    await prisma.fastingStreak.update({
      where: { userId },
      data: updates,
    });
  }

  async getStreakStats(userId: string) {
    const streak = await this.getOrCreateStreak(userId);

    // Check if streak is still active
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    let isActive = false;
    if (streak.lastFastDate) {
      const lastDate = new Date(streak.lastFastDate);
      lastDate.setHours(0, 0, 0, 0);
      isActive =
        lastDate.getTime() === today.getTime() || lastDate.getTime() === yesterday.getTime();
    }

    return {
      currentStreak: isActive ? streak.currentStreak : 0,
      longestStreak: streak.longestStreak,
      totalFasts: streak.totalFasts,
      totalMinutes: streak.totalMinutes,
      totalHours: Math.round(streak.totalMinutes / 60),
      successfulFasts: streak.successfulFasts,
      brokenFasts: streak.brokenFasts,
      successRate:
        streak.totalFasts > 0 ? Math.round((streak.successfulFasts / streak.totalFasts) * 100) : 0,
      weeklyMinutes: streak.weeklyMinutes,
      weeklyHours: Math.round(streak.weeklyMinutes / 60),
      monthlyMinutes: streak.monthlyMinutes,
      monthlyHours: Math.round(streak.monthlyMinutes / 60),
      isActive,
      lastFastDate: streak.lastFastDate,
    };
  }

  // ============================================================================
  // ANALYTICS
  // ============================================================================

  async getAnalytics(userId: string, days = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    const sessions = await prisma.fastingSession.findMany({
      where: {
        userId,
        startedAt: { gte: startDate },
        status: { not: FastingStatus.CANCELLED },
      },
      include: { checkpoints: true },
      orderBy: { startedAt: 'asc' },
    });

    // Calculate averages
    const completedSessions = sessions.filter((s) => s.status === FastingStatus.COMPLETED);
    const avgDuration =
      completedSessions.length > 0
        ? Math.round(
            completedSessions.reduce((sum, s) => sum + (s.actualDuration || 0), 0) /
              completedSessions.length
          )
        : 0;

    // Mood/energy/hunger trends
    const checkpoints = sessions.flatMap((s) => s.checkpoints);
    const avgMood =
      checkpoints.filter((c) => c.moodRating).length > 0
        ? Math.round(
            checkpoints.reduce((sum, c) => sum + (c.moodRating || 0), 0) /
              checkpoints.filter((c) => c.moodRating).length
          )
        : null;

    const avgEnergy =
      checkpoints.filter((c) => c.energyLevel).length > 0
        ? Math.round(
            checkpoints.reduce((sum, c) => sum + (c.energyLevel || 0), 0) /
              checkpoints.filter((c) => c.energyLevel).length
          )
        : null;

    const avgHunger =
      checkpoints.filter((c) => c.hungerLevel).length > 0
        ? Math.round(
            checkpoints.reduce((sum, c) => sum + (c.hungerLevel || 0), 0) /
              checkpoints.filter((c) => c.hungerLevel).length
          )
        : null;

    // Weight change correlation
    const sessionsWithWeight = sessions.filter((s) => s.startWeight && s.endWeight);
    const avgWeightChange =
      sessionsWithWeight.length > 0
        ? sessionsWithWeight.reduce((sum, s) => sum + (s.endWeight! - s.startWeight!), 0) /
          sessionsWithWeight.length
        : null;

    // Fasting by day of week
    const byDayOfWeek: Record<string, number> = {
      Sun: 0,
      Mon: 0,
      Tue: 0,
      Wed: 0,
      Thu: 0,
      Fri: 0,
      Sat: 0,
    };
    const days_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    for (const session of sessions) {
      const day = days_names[session.startedAt.getDay()];
      byDayOfWeek[day]++;
    }

    // Protocol usage
    const protocolUsage = new Map<string, number>();
    for (const session of sessions) {
      const name = session.protocolId ? session.protocolId : 'Custom';
      protocolUsage.set(name, (protocolUsage.get(name) || 0) + 1);
    }

    return {
      period: { days, startDate, endDate: new Date() },
      totals: {
        sessions: sessions.length,
        completed: completedSessions.length,
        broken: sessions.filter((s) => s.status === FastingStatus.BROKEN).length,
        totalMinutes: sessions.reduce((sum, s) => sum + (s.actualDuration || 0), 0),
        totalHours: Math.round(sessions.reduce((sum, s) => sum + (s.actualDuration || 0), 0) / 60),
      },
      averages: {
        durationMinutes: avgDuration,
        durationHours: Math.round((avgDuration / 60) * 10) / 10,
        mood: avgMood,
        energy: avgEnergy,
        hunger: avgHunger,
        weightChange: avgWeightChange ? Math.round(avgWeightChange * 100) / 100 : null,
      },
      distribution: {
        byDayOfWeek,
        completionRate:
          sessions.length > 0 ? Math.round((completedSessions.length / sessions.length) * 100) : 0,
      },
    };
  }

  // ============================================================================
  // TIMER STATUS
  // ============================================================================

  async getTimerStatus(userId: string) {
    const activeFast = await this.getActiveFast(userId);

    if (!activeFast) {
      return {
        isActive: false,
        session: null,
        progress: null,
      };
    }

    const now = Date.now();
    const startTime = activeFast.startedAt.getTime();
    const endTime = activeFast.plannedEndAt.getTime();
    const elapsed = now - startTime;
    const remaining = Math.max(0, endTime - now);
    const progressPercent = Math.min(
      100,
      Math.round((elapsed / (activeFast.plannedDuration * 60 * 1000)) * 100)
    );

    return {
      isActive: true,
      session: activeFast,
      progress: {
        elapsedMinutes: Math.round(elapsed / (60 * 1000)),
        elapsedHours: Math.round((elapsed / (60 * 60 * 1000)) * 10) / 10,
        remainingMinutes: Math.round(remaining / (60 * 1000)),
        remainingHours: Math.round((remaining / (60 * 60 * 1000)) * 10) / 10,
        progressPercent,
        plannedDurationMinutes: activeFast.plannedDuration,
        plannedDurationHours: Math.round(activeFast.plannedDuration / 60),
        isOvertime: remaining === 0,
      },
    };
  }
}

export const fastingService = new FastingService();
