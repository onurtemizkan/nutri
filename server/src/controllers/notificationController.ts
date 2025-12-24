import { pushNotificationService } from '../services/pushNotificationService';
import { notificationScheduler } from '../services/notificationScheduler';
import prisma from '../config/database';
import { AuthenticatedRequest, DevicePlatform, NotificationCategory } from '../types';
import { requireAuth } from '../utils/authHelpers';
import {
  registerDeviceSchema,
  unregisterDeviceSchema,
  updateNotificationPreferencesSchema,
  getNotificationHistoryQuerySchema,
  trackNotificationSchema,
  testNotificationSchema,
} from '../validation/schemas';
import { withErrorHandling } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { config } from '../config/env';
import { logger } from '../config/logger';
import { NotificationCategory as PrismaNotificationCategory, Prisma } from '@prisma/client';

export class NotificationController {
  /**
   * Register a device for push notifications
   * POST /api/notifications/register-device
   */
  registerDevice = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = registerDeviceSchema.parse(req.body);

    const result = await pushNotificationService.registerDevice(
      userId,
      validatedData.token,
      validatedData.platform as DevicePlatform,
      {
        expoPushToken: validatedData.expoPushToken,
        deviceModel: validatedData.deviceModel,
        osVersion: validatedData.osVersion,
        appVersion: validatedData.appVersion,
      }
    );

    res.status(HTTP_STATUS.OK).json(result);
  });

  /**
   * Unregister a device from push notifications
   * DELETE /api/notifications/unregister-device
   */
  unregisterDevice = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = unregisterDeviceSchema.parse(req.body);

    const result = await pushNotificationService.unregisterDevice(userId, validatedData.token);

    res.status(HTTP_STATUS.OK).json(result);
  });

  /**
   * Get user's notification preferences
   * GET /api/notifications/preferences
   */
  getPreferences = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    let preferences = await prisma.notificationPreference.findUnique({
      where: { userId },
    });

    // Create default preferences if none exist
    if (!preferences) {
      const defaultCategories: PrismaNotificationCategory[] = [
        'MEAL_REMINDER',
        'GOAL_PROGRESS',
        'HEALTH_INSIGHT',
        'SUPPLEMENT_REMINDER',
        'STREAK_ALERT',
        'WEEKLY_SUMMARY',
        'SYSTEM',
      ];

      preferences = await prisma.notificationPreference.create({
        data: {
          userId,
          enabled: true,
          enabledCategories: defaultCategories,
        },
      });
    }

    res.status(HTTP_STATUS.OK).json({
      enabled: preferences.enabled,
      enabledCategories: preferences.enabledCategories,
      quietHoursEnabled: preferences.quietHoursEnabled,
      quietHoursStart: preferences.quietHoursStart,
      quietHoursEnd: preferences.quietHoursEnd,
      mealReminderTimes: preferences.mealReminderTimes,
      settings: preferences.settings,
    });
  });

  /**
   * Update user's notification preferences
   * PUT /api/notifications/preferences
   */
  updatePreferences = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = updateNotificationPreferencesSchema.parse(req.body);

    // Ensure preferences exist first
    await prisma.notificationPreference.upsert({
      where: { userId },
      update: {},
      create: {
        userId,
        enabled: true,
        enabledCategories: [
          'MEAL_REMINDER',
          'GOAL_PROGRESS',
          'HEALTH_INSIGHT',
          'SUPPLEMENT_REMINDER',
          'STREAK_ALERT',
          'WEEKLY_SUMMARY',
          'SYSTEM',
        ],
      },
    });

    // Get user's timezone
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { timezone: true },
    });
    const timezone = user?.timezone || 'UTC';

    // Now update with the validated data
    const preferences = await prisma.notificationPreference.update({
      where: { userId },
      data: {
        enabled: validatedData.enabled,
        enabledCategories: validatedData.enabledCategories as PrismaNotificationCategory[] | undefined,
        quietHoursEnabled: validatedData.quietHoursEnabled,
        quietHoursStart: validatedData.quietHoursStart,
        quietHoursEnd: validatedData.quietHoursEnd,
        mealReminderTimes: validatedData.mealReminderTimes as Prisma.InputJsonValue | undefined,
        settings: validatedData.settings as Prisma.InputJsonValue | undefined,
      },
    });

    // Update scheduled meal reminders if meal reminder times were provided
    if (validatedData.mealReminderTimes) {
      try {
        const mealTimes = validatedData.mealReminderTimes as {
          breakfast?: string;
          lunch?: string;
          dinner?: string;
          snack?: string;
        };

        // Only schedule if MEAL_REMINDER category is enabled
        if (preferences.enabledCategories.includes('MEAL_REMINDER')) {
          await notificationScheduler.updateMealReminderSchedule(
            userId,
            mealTimes,
            timezone
          );
          logger.info({ userId, mealTimes, timezone }, 'Meal reminder schedule updated');
        } else {
          // Cancel meal reminders if category is disabled
          await notificationScheduler.cancelUserJobs(userId);
          logger.info({ userId }, 'Meal reminders cancelled - category disabled');
        }
      } catch (error) {
        logger.error({ userId, error }, 'Failed to update meal reminder schedule');
        // Don't fail the request if scheduling fails
      }
    }

    // If weekly summary is enabled, schedule it
    if (preferences.enabledCategories.includes('WEEKLY_SUMMARY')) {
      try {
        await notificationScheduler.scheduleWeeklySummary(userId, 0, '09:00', timezone);
        logger.info({ userId, timezone }, 'Weekly summary scheduled');
      } catch (error) {
        logger.error({ userId, error }, 'Failed to schedule weekly summary');
      }
    }

    res.status(HTTP_STATUS.OK).json({
      enabled: preferences.enabled,
      enabledCategories: preferences.enabledCategories,
      quietHoursEnabled: preferences.quietHoursEnabled,
      quietHoursStart: preferences.quietHoursStart,
      quietHoursEnd: preferences.quietHoursEnd,
      mealReminderTimes: preferences.mealReminderTimes,
      settings: preferences.settings,
    });
  });

  /**
   * Get user's notification history
   * GET /api/notifications/history
   */
  getHistory = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedQuery = getNotificationHistoryQuerySchema.parse(req.query);

    const where: {
      userId: string;
      category?: PrismaNotificationCategory;
      status?: 'PENDING' | 'SENT' | 'DELIVERED' | 'OPENED' | 'FAILED';
      sentAt?: { gte?: Date; lte?: Date };
    } = { userId };

    if (validatedQuery.category) {
      where.category = validatedQuery.category as PrismaNotificationCategory;
    }
    if (validatedQuery.status) {
      where.status = validatedQuery.status;
    }
    if (validatedQuery.startDate || validatedQuery.endDate) {
      where.sentAt = {};
      if (validatedQuery.startDate) {
        where.sentAt.gte = new Date(validatedQuery.startDate);
      }
      if (validatedQuery.endDate) {
        where.sentAt.lte = new Date(validatedQuery.endDate);
      }
    }

    const [notifications, total] = await Promise.all([
      prisma.notificationLog.findMany({
        where,
        orderBy: { sentAt: 'desc' },
        skip: (validatedQuery.page - 1) * validatedQuery.limit,
        take: validatedQuery.limit,
        select: {
          id: true,
          category: true,
          title: true,
          body: true,
          platform: true,
          status: true,
          sentAt: true,
          deliveredAt: true,
          openedAt: true,
          actionTaken: true,
        },
      }),
      prisma.notificationLog.count({ where }),
    ]);

    res.status(HTTP_STATUS.OK).json({
      data: notifications,
      pagination: {
        page: validatedQuery.page,
        limit: validatedQuery.limit,
        total,
        totalPages: Math.ceil(total / validatedQuery.limit),
      },
    });
  });

  /**
   * Track notification delivery/open
   * POST /api/notifications/track
   */
  trackNotification = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const validatedData = trackNotificationSchema.parse(req.body);

    // Find and verify the notification belongs to the user
    const notification = await prisma.notificationLog.findFirst({
      where: {
        id: validatedData.notificationLogId,
        userId,
      },
    });

    if (!notification) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Notification not found' });
      return;
    }

    // Update the notification status
    const updateData: {
      status?: 'DELIVERED' | 'OPENED';
      deliveredAt?: Date;
      openedAt?: Date;
      actionTaken?: string;
    } = {};

    if (validatedData.action === 'delivered') {
      updateData.status = 'DELIVERED';
      updateData.deliveredAt = new Date();
    } else if (validatedData.action === 'opened') {
      updateData.status = 'OPENED';
      updateData.openedAt = new Date();
      if (validatedData.actionTaken) {
        updateData.actionTaken = validatedData.actionTaken;
      }
    }

    await prisma.notificationLog.update({
      where: { id: validatedData.notificationLogId },
      data: updateData,
    });

    // Update campaign analytics if applicable
    if (notification.campaignId) {
      if (validatedData.action === 'delivered') {
        await prisma.notificationCampaign.update({
          where: { id: notification.campaignId },
          data: { deliveryCount: { increment: 1 } },
        });
      } else if (validatedData.action === 'opened') {
        await prisma.notificationCampaign.update({
          where: { id: notification.campaignId },
          data: {
            openCount: { increment: 1 },
            actionCount: validatedData.actionTaken ? { increment: 1 } : undefined,
          },
        });
      }
    }

    res.status(HTTP_STATUS.OK).json({ success: true });
  });

  /**
   * Send a test notification (development only)
   * POST /api/notifications/test
   */
  testNotification = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    // Only allow in development
    if (config.nodeEnv !== 'development') {
      res.status(HTTP_STATUS.FORBIDDEN).json({ error: 'Test notifications only available in development' });
      return;
    }

    const validatedData = testNotificationSchema.parse(req.body);

    const results = await pushNotificationService.sendToUser(
      userId,
      {
        title: validatedData.title,
        body: validatedData.body,
        category: validatedData.category as NotificationCategory,
        data: { test: true },
      },
      { skipQuietHours: true }
    );

    const sentCount = results.filter((r) => r.success).length;

    res.status(HTTP_STATUS.OK).json({
      success: sentCount > 0,
      sentTo: sentCount,
      failed: results.length - sentCount,
    });
  });

  /**
   * Get registered devices for the user
   * GET /api/notifications/devices
   */
  getDevices = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const devices = await prisma.deviceToken.findMany({
      where: { userId, isActive: true },
      select: {
        id: true,
        platform: true,
        deviceModel: true,
        osVersion: true,
        appVersion: true,
        lastActiveAt: true,
        createdAt: true,
      },
      orderBy: { lastActiveAt: 'desc' },
    });

    res.status(HTTP_STATUS.OK).json(devices);
  });
}

// Export singleton instance
export const notificationController = new NotificationController();
