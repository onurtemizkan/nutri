/**
 * Notification Scheduler Service
 *
 * Job queue system for scheduling notifications using Bull and Redis.
 * Supports meal reminders, weekly summaries, campaign sends, and health insights.
 */

import Bull, { Job, Queue } from 'bull';
import moment from 'moment-timezone';
import prisma from '../config/database';
import { logger } from '../config/logger';
import { pushNotificationService } from './pushNotificationService';
import { NotificationPayload, NotificationCategory, MealType } from '../types';

// ============================================================================
// JOB TYPES AND INTERFACES
// ============================================================================

export type NotificationJobType =
  | 'MEAL_REMINDER'
  | 'WEEKLY_SUMMARY'
  | 'CAMPAIGN_SEND'
  | 'HEALTH_INSIGHT'
  | 'STREAK_ALERT'
  | 'SUPPLEMENT_REMINDER'
  | 'CLEANUP_TOKENS'
  | 'CHECK_RECEIPTS';

interface MealReminderJobData {
  type: 'MEAL_REMINDER';
  userId: string;
  mealType: MealType;
}

interface WeeklySummaryJobData {
  type: 'WEEKLY_SUMMARY';
  userId: string;
}

interface CampaignSendJobData {
  type: 'CAMPAIGN_SEND';
  campaignId: string;
  batchIndex?: number;
  batchSize?: number;
}

interface HealthInsightJobData {
  type: 'HEALTH_INSIGHT';
  userId: string;
  insightType: string;
  metricType?: string;
  data?: Record<string, unknown>;
}

interface StreakAlertJobData {
  type: 'STREAK_ALERT';
  userId: string;
  currentStreak: number;
}

interface SupplementReminderJobData {
  type: 'SUPPLEMENT_REMINDER';
  userId: string;
  supplementId: string;
  supplementName: string;
}

interface CleanupTokensJobData {
  type: 'CLEANUP_TOKENS';
}

interface CheckReceiptsJobData {
  type: 'CHECK_RECEIPTS';
  receiptIds: string[];
}

type NotificationJobData =
  | MealReminderJobData
  | WeeklySummaryJobData
  | CampaignSendJobData
  | HealthInsightJobData
  | StreakAlertJobData
  | SupplementReminderJobData
  | CleanupTokensJobData
  | CheckReceiptsJobData;

// Rate limiting: max notifications per user per day
const MAX_NOTIFICATIONS_PER_DAY = 10;

// ============================================================================
// NOTIFICATION SCHEDULER CLASS
// ============================================================================

export class NotificationScheduler {
  private queue: Queue<NotificationJobData>;
  private isInitialized = false;

  constructor() {
    const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';
    const concurrency = parseInt(process.env.NOTIFICATION_QUEUE_CONCURRENCY || '5', 10);

    this.queue = new Bull<NotificationJobData>('notifications', redisUrl, {
      defaultJobOptions: {
        attempts: 3,
        backoff: {
          type: 'exponential',
          delay: 1000,
        },
        removeOnComplete: 100, // Keep last 100 completed jobs
        removeOnFail: 50, // Keep last 50 failed jobs
      },
    });

    // Setup job processors
    this.setupProcessors(concurrency);

    // Setup event handlers
    this.setupEventHandlers();

    this.isInitialized = true;
    logger.info({ concurrency }, 'Notification scheduler initialized');
  }

  // ============================================================================
  // QUEUE SETUP
  // ============================================================================

  private setupProcessors(concurrency: number): void {
    this.queue.process(concurrency, async (job: Job<NotificationJobData>) => {
      const { data } = job;

      switch (data.type) {
        case 'MEAL_REMINDER':
          return this.processMealReminder(job as Job<MealReminderJobData>);
        case 'WEEKLY_SUMMARY':
          return this.processWeeklySummary(job as Job<WeeklySummaryJobData>);
        case 'CAMPAIGN_SEND':
          return this.processCampaignSend(job as Job<CampaignSendJobData>);
        case 'HEALTH_INSIGHT':
          return this.processHealthInsight(job as Job<HealthInsightJobData>);
        case 'STREAK_ALERT':
          return this.processStreakAlert(job as Job<StreakAlertJobData>);
        case 'SUPPLEMENT_REMINDER':
          return this.processSupplementReminder(job as Job<SupplementReminderJobData>);
        case 'CLEANUP_TOKENS':
          return this.processCleanupTokens();
        case 'CHECK_RECEIPTS':
          return this.processCheckReceipts(job as Job<CheckReceiptsJobData>);
        default:
          throw new Error(`Unknown job type: ${(data as NotificationJobData).type}`);
      }
    });
  }

  private setupEventHandlers(): void {
    this.queue.on('completed', (job: Job<NotificationJobData>) => {
      logger.debug({
        jobId: job.id,
        jobType: job.data.type,
      }, 'Notification job completed');
    });

    this.queue.on('failed', (job: Job<NotificationJobData>, err: Error) => {
      logger.error({
        jobId: job.id,
        jobType: job.data.type,
        error: err.message,
        attempts: job.attemptsMade,
      }, 'Notification job failed');
    });

    this.queue.on('stalled', (job: Job<NotificationJobData>) => {
      logger.warn({
        jobId: job.id,
        jobType: job.data.type,
      }, 'Notification job stalled');
    });
  }

  // ============================================================================
  // JOB PROCESSORS
  // ============================================================================

  /**
   * Process meal reminder notification
   */
  private async processMealReminder(job: Job<MealReminderJobData>): Promise<void> {
    const { userId, mealType } = job.data;

    // Check rate limit
    if (await this.isRateLimited(userId)) {
      logger.info({ userId, mealType }, 'Skipping meal reminder - rate limited');
      return;
    }

    // Get user to check if they've already logged a meal of this type today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const existingMeal = await prisma.meal.findFirst({
      where: {
        userId,
        mealType,
        consumedAt: {
          gte: today,
          lt: tomorrow,
        },
      },
    });

    if (existingMeal) {
      logger.info({ userId, mealType }, 'Skipping meal reminder - already logged');
      return;
    }

    // Get user for personalization
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { name: true },
    });

    const mealName = mealType.charAt(0).toUpperCase() + mealType.slice(1);
    const greeting = user?.name ? `Hey ${user.name}!` : 'Hey there!';

    const notification: NotificationPayload = {
      title: `Time for ${mealName}! 🍽️`,
      body: `${greeting} Don't forget to log your ${mealType.toLowerCase()}.`,
      category: 'MEAL_REMINDER',
      data: {
        screen: '/add-meal',
        mealType,
      },
    };

    const results = await pushNotificationService.sendToUser(userId, notification);
    await this.incrementNotificationCount(userId);

    logger.info({
      userId,
      mealType,
      sentCount: results.filter((r) => r.success).length,
    }, 'Meal reminder sent');
  }

  /**
   * Process weekly summary notification
   */
  private async processWeeklySummary(job: Job<WeeklySummaryJobData>): Promise<void> {
    const { userId } = job.data;

    // Check rate limit
    if (await this.isRateLimited(userId)) {
      logger.info({ userId }, 'Skipping weekly summary - rate limited');
      return;
    }

    // Get user's weekly stats
    const weekStart = moment().subtract(7, 'days').startOf('day').toDate();
    const weekEnd = moment().startOf('day').toDate();

    const [mealsCount, avgCalories, user] = await Promise.all([
      prisma.meal.count({
        where: {
          userId,
          consumedAt: { gte: weekStart, lt: weekEnd },
        },
      }),
      prisma.meal.aggregate({
        where: {
          userId,
          consumedAt: { gte: weekStart, lt: weekEnd },
        },
        _avg: { calories: true },
      }),
      prisma.user.findUnique({
        where: { id: userId },
        select: { name: true, goalCalories: true },
      }),
    ]);

    const avgDailyCalories = Math.round((avgCalories._avg.calories || 0) * (mealsCount / 7));
    const goalProgress = user?.goalCalories
      ? Math.round((avgDailyCalories / user.goalCalories) * 100)
      : 0;

    const notification: NotificationPayload = {
      title: 'Your Weekly Summary is Ready! 📊',
      body: `You logged ${mealsCount} meals this week. Average daily intake: ${avgDailyCalories} cal (${goalProgress}% of goal).`,
      category: 'WEEKLY_SUMMARY',
      data: {
        screen: '/(tabs)/health',
        weekStart: weekStart.toISOString(),
      },
    };

    const results = await pushNotificationService.sendToUser(userId, notification);
    await this.incrementNotificationCount(userId);

    logger.info({
      userId,
      mealsCount,
      avgDailyCalories,
      sentCount: results.filter((r) => r.success).length,
    }, 'Weekly summary sent');
  }

  /**
   * Process campaign send notification
   */
  private async processCampaignSend(job: Job<CampaignSendJobData>): Promise<void> {
    const { campaignId, batchIndex = 0, batchSize = 1000 } = job.data;

    // Get campaign
    const campaign = await prisma.notificationCampaign.findUnique({
      where: { id: campaignId },
    });

    if (!campaign) {
      throw new Error(`Campaign not found: ${campaignId}`);
    }

    if (campaign.status === 'CANCELLED') {
      logger.info({ campaignId }, 'Campaign cancelled, skipping send');
      return;
    }

    // Update status to SENDING on first batch
    if (batchIndex === 0) {
      await prisma.notificationCampaign.update({
        where: { id: campaignId },
        data: { status: 'SENDING' },
      });
    }

    // Get target segment users
    const segment = campaign.targetSegment as Record<string, unknown>;
    const whereClause = this.buildSegmentWhereClause(segment);

    const users = await prisma.user.findMany({
      where: whereClause,
      select: { id: true },
      skip: batchIndex * batchSize,
      take: batchSize,
    });

    if (users.length === 0) {
      // No more users, mark campaign as completed
      await prisma.notificationCampaign.update({
        where: { id: campaignId },
        data: {
          status: 'COMPLETED',
          sentAt: new Date(),
        },
      });
      logger.info({ campaignId }, 'Campaign completed');
      return;
    }

    // Build notification from campaign content
    const notification: NotificationPayload = {
      title: campaign.notificationTitle,
      body: campaign.notificationBody,
      category: campaign.category as NotificationCategory,
      data: {
        ...(campaign.data as Record<string, unknown> | undefined),
        campaignId,
      },
    };

    // Send to batch
    const userIds = users.map((u) => u.id);
    const result = await pushNotificationService.sendBatch(userIds, notification, {
      campaignId,
      skipQuietHours: false,
    });

    // Update campaign stats
    await prisma.notificationCampaign.update({
      where: { id: campaignId },
      data: {
        deliveryCount: { increment: result.sent },
      },
    });

    // Schedule next batch if there are more users
    if (users.length === batchSize) {
      await this.queue.add(
        {
          type: 'CAMPAIGN_SEND',
          campaignId,
          batchIndex: batchIndex + 1,
          batchSize,
        },
        { delay: 1000 } // Small delay between batches
      );
    } else {
      // Last batch, mark as completed
      await prisma.notificationCampaign.update({
        where: { id: campaignId },
        data: {
          status: 'COMPLETED',
          sentAt: new Date(),
        },
      });
      logger.info({ campaignId, totalBatches: batchIndex + 1 }, 'Campaign completed');
    }
  }

  /**
   * Process health insight notification
   */
  private async processHealthInsight(job: Job<HealthInsightJobData>): Promise<void> {
    const { userId, insightType, metricType, data } = job.data;

    // Check rate limit
    if (await this.isRateLimited(userId)) {
      logger.info({ userId, insightType }, 'Skipping health insight - rate limited');
      return;
    }

    // Build notification based on insight type
    let notification: NotificationPayload;

    switch (insightType) {
      case 'NUTRITION_CORRELATION':
        notification = {
          title: 'New Health Insight! 🧬',
          body: data?.message as string || `We found a correlation between your nutrition and ${metricType}.`,
          category: 'HEALTH_INSIGHT',
          data: {
            screen: `/health/${metricType}`,
            insightType,
            metricType,
          },
          interruptionLevel: 'timeSensitive',
        };
        break;

      case 'GOAL_ACHIEVED':
        notification = {
          title: 'Goal Achieved! 🎉',
          body: data?.message as string || 'Congratulations on reaching your nutrition goal!',
          category: 'GOAL_PROGRESS',
          data: {
            screen: '/(tabs)/',
            insightType,
          },
        };
        break;

      default:
        notification = {
          title: 'Health Update',
          body: data?.message as string || 'Check out your latest health insights.',
          category: 'HEALTH_INSIGHT',
          data: {
            screen: '/(tabs)/health',
            insightType,
          },
        };
    }

    const results = await pushNotificationService.sendToUser(userId, notification);
    await this.incrementNotificationCount(userId);

    logger.info({
      userId,
      insightType,
      sentCount: results.filter((r) => r.success).length,
    }, 'Health insight sent');
  }

  /**
   * Process streak alert notification
   */
  private async processStreakAlert(job: Job<StreakAlertJobData>): Promise<void> {
    const { userId, currentStreak } = job.data;

    // Check rate limit
    if (await this.isRateLimited(userId)) {
      logger.info({ userId }, 'Skipping streak alert - rate limited');
      return;
    }

    let title: string;
    let body: string;

    if (currentStreak === 0) {
      title = "Don't Break Your Streak! 🔥";
      body = "You haven't logged any meals today. Log now to keep your streak alive!";
    } else if (currentStreak >= 7) {
      title = `${currentStreak} Day Streak! 🔥🔥🔥`;
      body = "Amazing! Keep the momentum going - log your next meal!";
    } else {
      title = `${currentStreak} Day Streak! 🔥`;
      body = "Great progress! Log your meals today to continue your streak.";
    }

    const notification: NotificationPayload = {
      title,
      body,
      category: 'STREAK_ALERT',
      data: {
        screen: '/add-meal',
        currentStreak,
      },
    };

    const results = await pushNotificationService.sendToUser(userId, notification);
    await this.incrementNotificationCount(userId);

    logger.info({
      userId,
      currentStreak,
      sentCount: results.filter((r) => r.success).length,
    }, 'Streak alert sent');
  }

  /**
   * Process supplement reminder notification
   */
  private async processSupplementReminder(job: Job<SupplementReminderJobData>): Promise<void> {
    const { userId, supplementId, supplementName } = job.data;

    // Check rate limit
    if (await this.isRateLimited(userId)) {
      logger.info({ userId, supplementId }, 'Skipping supplement reminder - rate limited');
      return;
    }

    // Check if already taken today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const existingLog = await prisma.supplementLog.findFirst({
      where: {
        userId,
        supplementId,
        takenAt: { gte: today, lt: tomorrow },
        skipped: false,
      },
    });

    if (existingLog) {
      logger.info({ userId, supplementId }, 'Skipping supplement reminder - already taken');
      return;
    }

    const notification: NotificationPayload = {
      title: 'Supplement Reminder 💊',
      body: `Time to take your ${supplementName}!`,
      category: 'SUPPLEMENT_REMINDER',
      data: {
        screen: '/supplements',
        supplementId,
      },
    };

    const results = await pushNotificationService.sendToUser(userId, notification);
    await this.incrementNotificationCount(userId);

    logger.info({
      userId,
      supplementId,
      sentCount: results.filter((r) => r.success).length,
    }, 'Supplement reminder sent');
  }

  /**
   * Process token cleanup job
   */
  private async processCleanupTokens(): Promise<void> {
    const count = await pushNotificationService.cleanupStaleTokens();
    logger.info({ count }, 'Token cleanup completed');
  }

  /**
   * Process receipt check job
   */
  private async processCheckReceipts(job: Job<CheckReceiptsJobData>): Promise<void> {
    const { receiptIds } = job.data;
    await pushNotificationService.checkReceipts(receiptIds);
    logger.info({ count: receiptIds.length }, 'Receipt check completed');
  }

  // ============================================================================
  // SCHEDULING METHODS
  // ============================================================================

  /**
   * Schedule a meal reminder for a user
   */
  async scheduleMealReminder(
    userId: string,
    mealType: MealType,
    time: string, // HH:mm format
    timezone: string = 'UTC'
  ): Promise<string> {
    // Calculate next occurrence
    const [hour, minute] = time.split(':').map(Number);
    const nextRun = this.getNextOccurrence(hour, minute, timezone);

    const jobId = `meal_reminder:${userId}:${mealType}`;

    // Remove existing job if any
    const existingJob = await this.queue.getJob(jobId);
    if (existingJob) {
      await existingJob.remove();
    }

    // Schedule new job
    const job = await this.queue.add(
      { type: 'MEAL_REMINDER', userId, mealType },
      {
        jobId,
        delay: nextRun.getTime() - Date.now(),
        repeat: {
          cron: `${minute} ${hour} * * *`,
          tz: timezone,
        },
      }
    );

    logger.info({
      userId,
      mealType,
      time,
      timezone,
      jobId: job.id,
    }, 'Meal reminder scheduled');

    return job.id as string;
  }

  /**
   * Schedule weekly summary for a user
   */
  async scheduleWeeklySummary(
    userId: string,
    dayOfWeek: number = 0, // 0 = Sunday
    time: string = '09:00',
    timezone: string = 'UTC'
  ): Promise<string> {
    const [hour, minute] = time.split(':').map(Number);
    const jobId = `weekly_summary:${userId}`;

    // Remove existing job if any
    const existingJob = await this.queue.getJob(jobId);
    if (existingJob) {
      await existingJob.remove();
    }

    const job = await this.queue.add(
      { type: 'WEEKLY_SUMMARY', userId },
      {
        jobId,
        repeat: {
          cron: `${minute} ${hour} * * ${dayOfWeek}`,
          tz: timezone,
        },
      }
    );

    logger.info({
      userId,
      dayOfWeek,
      time,
      timezone,
      jobId: job.id,
    }, 'Weekly summary scheduled');

    return job.id as string;
  }

  /**
   * Schedule a campaign send
   */
  async scheduleCampaign(campaignId: string, sendAt: Date): Promise<string> {
    const delay = Math.max(0, sendAt.getTime() - Date.now());

    const job = await this.queue.add(
      { type: 'CAMPAIGN_SEND', campaignId },
      {
        jobId: `campaign:${campaignId}`,
        delay,
      }
    );

    // Update campaign status
    await prisma.notificationCampaign.update({
      where: { id: campaignId },
      data: { status: 'SCHEDULED' },
    });

    logger.info({
      campaignId,
      sendAt: sendAt.toISOString(),
      delay,
      jobId: job.id,
    }, 'Campaign scheduled');

    return job.id as string;
  }

  /**
   * Send a health insight notification
   */
  async sendHealthInsight(
    userId: string,
    insightType: string,
    metricType?: string,
    data?: Record<string, unknown>
  ): Promise<string> {
    const job = await this.queue.add({
      type: 'HEALTH_INSIGHT',
      userId,
      insightType,
      metricType,
      data,
    });

    return job.id as string;
  }

  /**
   * Send a streak alert notification
   */
  async sendStreakAlert(userId: string, currentStreak: number): Promise<string> {
    const job = await this.queue.add({
      type: 'STREAK_ALERT',
      userId,
      currentStreak,
    });

    return job.id as string;
  }

  /**
   * Schedule supplement reminder for a user
   */
  async scheduleSupplementReminder(
    userId: string,
    supplementId: string,
    supplementName: string,
    time: string, // HH:mm format
    timezone: string = 'UTC'
  ): Promise<string> {
    const [hour, minute] = time.split(':').map(Number);
    const jobId = `supplement_reminder:${userId}:${supplementId}`;

    // Remove existing job if any
    const existingJob = await this.queue.getJob(jobId);
    if (existingJob) {
      await existingJob.remove();
    }

    const job = await this.queue.add(
      { type: 'SUPPLEMENT_REMINDER', userId, supplementId, supplementName },
      {
        jobId,
        repeat: {
          cron: `${minute} ${hour} * * *`,
          tz: timezone,
        },
      }
    );

    logger.info({
      userId,
      supplementId,
      time,
      timezone,
      jobId: job.id,
    }, 'Supplement reminder scheduled');

    return job.id as string;
  }

  /**
   * Cancel all scheduled jobs for a user
   */
  async cancelUserJobs(userId: string): Promise<void> {
    const repeatableJobs = await this.queue.getRepeatableJobs();

    for (const job of repeatableJobs) {
      if (job.id?.includes(userId) || job.key?.includes(userId)) {
        await this.queue.removeRepeatableByKey(job.key);
        logger.debug({ jobKey: job.key }, 'Removed repeatable job');
      }
    }

    logger.info({ userId }, 'Cancelled all user notification jobs');
  }

  /**
   * Update meal reminder schedule for a user
   */
  async updateMealReminderSchedule(
    userId: string,
    times: { breakfast?: string; lunch?: string; dinner?: string; snack?: string },
    timezone: string = 'UTC'
  ): Promise<void> {
    // Cancel existing meal reminders
    const mealTypes: MealType[] = ['breakfast', 'lunch', 'dinner', 'snack'];
    for (const mealType of mealTypes) {
      const jobId = `meal_reminder:${userId}:${mealType}`;
      const existingJob = await this.queue.getJob(jobId);
      if (existingJob) {
        await existingJob.remove();
      }

      // Also remove repeatable job
      const repeatableJobs = await this.queue.getRepeatableJobs();
      for (const rJob of repeatableJobs) {
        if (rJob.id === jobId) {
          await this.queue.removeRepeatableByKey(rJob.key);
        }
      }
    }

    // Schedule new meal reminders
    for (const [mealType, time] of Object.entries(times)) {
      if (time) {
        await this.scheduleMealReminder(userId, mealType as MealType, time, timezone);
      }
    }

    logger.info({ userId, times, timezone }, 'Meal reminder schedule updated');
  }

  /**
   * Schedule maintenance jobs (token cleanup, etc.)
   */
  async scheduleMaintenanceJobs(): Promise<void> {
    // Daily token cleanup at 3 AM UTC
    await this.queue.add(
      { type: 'CLEANUP_TOKENS' },
      {
        jobId: 'maintenance:cleanup_tokens',
        repeat: {
          cron: '0 3 * * *',
          tz: 'UTC',
        },
      }
    );

    logger.info('Maintenance jobs scheduled');
  }

  // ============================================================================
  // HELPER METHODS
  // ============================================================================

  /**
   * Check if user is rate limited
   */
  private async isRateLimited(userId: string): Promise<boolean> {
    try {
      // Use in-memory tracking if Redis is not available
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      const tomorrow = new Date(today);
      tomorrow.setDate(tomorrow.getDate() + 1);

      // Count notifications sent to this user today
      const count = await prisma.notificationLog.count({
        where: {
          userId,
          sentAt: { gte: today, lt: tomorrow },
          status: { in: ['SENT', 'DELIVERED'] },
        },
      });

      return count >= MAX_NOTIFICATIONS_PER_DAY;
    } catch (error) {
      logger.warn({ userId, error }, 'Error checking rate limit, allowing notification');
      return false;
    }
  }

  /**
   * Increment notification count for rate limiting
   */
  private async incrementNotificationCount(_userId: string): Promise<void> {
    // Rate limiting is tracked via NotificationLog
    // No additional action needed here
  }

  /**
   * Get next occurrence of a specific time
   */
  private getNextOccurrence(hour: number, minute: number, timezone: string): Date {
    const now = moment().tz(timezone);
    const target = moment()
      .tz(timezone)
      .set({ hour, minute, second: 0, millisecond: 0 });

    if (target.isSameOrBefore(now)) {
      target.add(1, 'day');
    }

    return target.toDate();
  }

  /**
   * Build Prisma where clause from segment filters
   */
  private buildSegmentWhereClause(segment: Record<string, unknown>): Record<string, unknown> {
    const where: Record<string, unknown> = {};

    // Activity level filter
    if (segment.activityLevel) {
      const levels = segment.activityLevel as string[];
      if (levels.length > 0) {
        where.activityLevel = { in: levels };
      }
    }

    // Subscription tier filter
    if (segment.subscriptionTier) {
      const tiers = segment.subscriptionTier as string[];
      if (tiers.length > 0) {
        where.subscriptionTier = { in: tiers };
      }
    }

    // Last active date filter
    if (segment.lastActiveAfter) {
      where.updatedAt = {
        ...((where.updatedAt as Record<string, Date>) || {}),
        gte: new Date(segment.lastActiveAfter as string),
      };
    }

    if (segment.lastActiveBefore) {
      where.updatedAt = {
        ...((where.updatedAt as Record<string, Date>) || {}),
        lt: new Date(segment.lastActiveBefore as string),
      };
    }

    // Must have active device token
    where.deviceTokens = {
      some: {
        isActive: true,
        expoPushToken: { not: null },
      },
    };

    // Must have notifications enabled
    where.notificationPreference = {
      enabled: true,
    };

    return where;
  }

  // ============================================================================
  // QUEUE ACCESS
  // ============================================================================

  /**
   * Get the Bull queue instance (for Bull Board integration)
   */
  getQueue(): Queue<NotificationJobData> {
    return this.queue;
  }

  /**
   * Check if scheduler is initialized
   */
  isReady(): boolean {
    return this.isInitialized;
  }

  /**
   * Graceful shutdown
   */
  async close(): Promise<void> {
    await this.queue.close();
    logger.info('Notification scheduler closed');
  }
}

// Export singleton instance
export const notificationScheduler = new NotificationScheduler();
