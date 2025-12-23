import Expo, {
  ExpoPushMessage,
  ExpoPushTicket,
  ExpoPushReceipt,
  ExpoPushSuccessTicket,
  ExpoPushErrorTicket,
} from 'expo-server-sdk';
import prisma from '../config/database';
import { logger } from '../config/logger';
import {
  DevicePlatform,
  NotificationCategory,
  NotificationPayload,
  NotificationResult,
  SendNotificationOptions,
  InterruptionLevel,
} from '../types';
import {
  NotificationCategory as PrismaNotificationCategory,
  DevicePlatform as PrismaDevicePlatform,
  NotificationStatus,
} from '@prisma/client';

// Create a new Expo SDK client
const expo = new Expo();

// Map interruption levels to relevance scores for priority
const INTERRUPTION_LEVEL_PRIORITY: Record<InterruptionLevel, number> = {
  passive: 0,
  active: 1,
  timeSensitive: 2,
  critical: 3,
};

// Default interruption levels per category
const CATEGORY_INTERRUPTION_LEVELS: Record<NotificationCategory, InterruptionLevel> = {
  MEAL_REMINDER: 'active',
  GOAL_PROGRESS: 'passive',
  HEALTH_INSIGHT: 'timeSensitive',
  SUPPLEMENT_REMINDER: 'active',
  STREAK_ALERT: 'active',
  WEEKLY_SUMMARY: 'passive',
  MARKETING: 'passive',
  SYSTEM: 'active',
};

// Default relevance scores per category (0.0 to 1.0)
const CATEGORY_RELEVANCE_SCORES: Record<NotificationCategory, number> = {
  STREAK_ALERT: 0.9,
  HEALTH_INSIGHT: 0.8,
  MEAL_REMINDER: 0.7,
  SUPPLEMENT_REMINDER: 0.6,
  GOAL_PROGRESS: 0.5,
  WEEKLY_SUMMARY: 0.4,
  SYSTEM: 0.5,
  MARKETING: 0.3,
};

export class PushNotificationService {
  /**
   * Register a device token for push notifications
   */
  async registerDevice(
    userId: string,
    token: string,
    platform: DevicePlatform,
    options?: {
      expoPushToken?: string;
      deviceModel?: string;
      osVersion?: string;
      appVersion?: string;
    }
  ): Promise<{ success: boolean; deviceId: string }> {
    // Validate Expo push token format
    if (options?.expoPushToken && !Expo.isExpoPushToken(options.expoPushToken)) {
      throw new Error('Invalid Expo push token format');
    }

    // Upsert the device token
    const device = await prisma.deviceToken.upsert({
      where: {
        userId_token_platform: {
          userId,
          token,
          platform: platform as PrismaDevicePlatform,
        },
      },
      update: {
        isActive: true,
        lastActiveAt: new Date(),
        expoPushToken: options?.expoPushToken,
        deviceModel: options?.deviceModel,
        osVersion: options?.osVersion,
        appVersion: options?.appVersion,
      },
      create: {
        userId,
        token,
        platform: platform as PrismaDevicePlatform,
        expoPushToken: options?.expoPushToken,
        deviceModel: options?.deviceModel,
        osVersion: options?.osVersion,
        appVersion: options?.appVersion,
        isActive: true,
      },
    });

    // Create default notification preferences if they don't exist
    await this.ensureNotificationPreferences(userId);

    logger.info({ userId, deviceId: device.id, platform }, 'Device registered for push notifications');

    return { success: true, deviceId: device.id };
  }

  /**
   * Unregister a device token
   */
  async unregisterDevice(userId: string, token: string): Promise<{ success: boolean }> {
    await prisma.deviceToken.updateMany({
      where: {
        userId,
        token,
      },
      data: {
        isActive: false,
      },
    });

    logger.info({ userId, token: token.slice(0, 10) + '...' }, 'Device unregistered from push notifications');
    return { success: true };
  }

  /**
   * Send notification to a specific user (all their active devices)
   */
  async sendToUser(
    userId: string,
    notification: NotificationPayload,
    options?: SendNotificationOptions
  ): Promise<NotificationResult[]> {
    // Get user's active devices with Expo push tokens
    const devices = await prisma.deviceToken.findMany({
      where: {
        userId,
        isActive: true,
        expoPushToken: { not: null },
      },
    });

    if (devices.length === 0) {
      logger.info({ userId }, 'No active devices found for user');
      return [];
    }

    // Check quiet hours unless skipped
    if (!options?.skipQuietHours) {
      const isQuietHours = await this.isUserInQuietHours(userId);
      if (isQuietHours) {
        logger.info({ userId }, 'Skipping notification during quiet hours');
        return [];
      }
    }

    // Check if user has enabled this category
    const hasCategory = await this.userHasCategory(userId, notification.category);
    if (!hasCategory) {
      logger.info({ userId, category: notification.category }, 'User has disabled this notification category');
      return [];
    }

    const results: NotificationResult[] = [];

    for (const device of devices) {
      if (!device.expoPushToken) continue;

      const result = await this.sendToDevice(
        device.expoPushToken,
        device.platform as DevicePlatform,
        notification,
        {
          ...options,
          userId,
          deviceToken: device.token,
        }
      );
      results.push(result);
    }

    return results;
  }

  /**
   * Send notification to a specific device
   */
  async sendToDevice(
    expoPushToken: string,
    platform: DevicePlatform,
    notification: NotificationPayload,
    options?: SendNotificationOptions & { userId?: string; deviceToken?: string }
  ): Promise<NotificationResult> {
    // Store token for logging (to avoid type narrowing issues with type guard)
    const tokenForLog = expoPushToken;

    // Validate token
    if (!Expo.isExpoPushToken(expoPushToken)) {
      logger.error({ token: tokenForLog.slice(0, 20) }, 'Invalid Expo push token');
      return {
        success: false,
        expoPushToken: tokenForLog,
        error: 'Invalid Expo push token',
        platform,
      };
    }

    // Build the notification message
    const interruptionLevel = notification.interruptionLevel ?? CATEGORY_INTERRUPTION_LEVELS[notification.category];
    const relevanceScore = notification.relevanceScore ?? CATEGORY_RELEVANCE_SCORES[notification.category];

    const message: ExpoPushMessage = {
      to: expoPushToken,
      title: notification.title,
      body: notification.body,
      data: {
        ...notification.data,
        category: notification.category,
      },
      sound: notification.sound ?? 'default',
      badge: notification.badge,
      channelId: notification.category.toLowerCase(), // Android notification channel
      priority: INTERRUPTION_LEVEL_PRIORITY[interruptionLevel] >= 2 ? 'high' : 'normal',
    };

    // iOS-specific options
    if (platform === 'IOS') {
      message._contentAvailable = true;
      // Note: interruptionLevel and relevanceScore are set via data payload
      // as Expo SDK handles iOS-specific settings
      if (notification.data) {
        (message.data as Record<string, unknown>).interruptionLevel = interruptionLevel;
        (message.data as Record<string, unknown>).relevanceScore = relevanceScore;
      }
    }

    try {
      // Send the notification
      const tickets = await expo.sendPushNotificationsAsync([message]);
      const ticket = tickets[0];

      // Create notification log
      if (options?.userId) {
        await this.createNotificationLog({
          userId: options.userId,
          notification,
          platform,
          deviceToken: options.deviceToken,
          expoPushToken,
          campaignId: options.campaignId,
          interruptionLevel,
          relevanceScore,
          ticket,
        });
      }

      // Check if the ticket indicates success
      if ((ticket as ExpoPushSuccessTicket).id) {
        logger.info({
          expoPushToken: expoPushToken.slice(0, 20),
          ticketId: (ticket as ExpoPushSuccessTicket).id,
        }, 'Push notification sent successfully');
        return {
          success: true,
          expoPushToken,
          platform,
        };
      } else {
        const errorTicket = ticket as ExpoPushErrorTicket;
        logger.error({
          expoPushToken: expoPushToken.slice(0, 20),
          error: errorTicket.message,
          details: errorTicket.details,
        }, 'Push notification failed');

        // Handle invalid tokens
        if (errorTicket.details?.error === 'DeviceNotRegistered') {
          await this.markTokenAsInactive(expoPushToken);
        }

        return {
          success: false,
          expoPushToken,
          error: errorTicket.message,
          platform,
        };
      }
    } catch (error) {
      logger.error({ err: error, expoPushToken: expoPushToken.slice(0, 20) }, 'Error sending push notification');
      return {
        success: false,
        expoPushToken,
        error: error instanceof Error ? error.message : 'Unknown error',
        platform,
      };
    }
  }

  /**
   * Send batch notifications to multiple users
   */
  async sendBatch(
    userIds: string[],
    notification: NotificationPayload,
    options?: SendNotificationOptions
  ): Promise<{ sent: number; failed: number }> {
    let sent = 0;
    let failed = 0;

    // Process in batches of 100 for efficiency
    const batchSize = 100;
    for (let i = 0; i < userIds.length; i += batchSize) {
      const batch = userIds.slice(i, i + batchSize);

      // Get all devices for this batch of users
      const devices = await prisma.deviceToken.findMany({
        where: {
          userId: { in: batch },
          isActive: true,
          expoPushToken: { not: null },
        },
        include: {
          user: {
            include: {
              notificationPreference: true,
            },
          },
        },
      });

      // Filter by preferences and quiet hours
      const eligibleDevices = devices.filter((device) => {
        const prefs = device.user.notificationPreference;
        if (!prefs?.enabled) return false;
        if (!prefs.enabledCategories.includes(notification.category as PrismaNotificationCategory)) return false;
        if (!options?.skipQuietHours && this.checkQuietHours(prefs, device.user.timezone)) return false;
        return true;
      });

      // Build messages
      const messages: ExpoPushMessage[] = eligibleDevices
        .filter((d) => d.expoPushToken && Expo.isExpoPushToken(d.expoPushToken))
        .map((device) => ({
          to: device.expoPushToken!,
          title: notification.title,
          body: notification.body,
          data: {
            ...notification.data,
            category: notification.category,
          },
          sound: 'default',
          channelId: notification.category.toLowerCase(),
        }));

      if (messages.length === 0) continue;

      // Send in chunks (Expo recommends max 100 per request)
      const chunks = expo.chunkPushNotifications(messages);
      for (const chunk of chunks) {
        try {
          const tickets = await expo.sendPushNotificationsAsync(chunk);

          // Log results
          for (let j = 0; j < tickets.length; j++) {
            const ticket = tickets[j];
            const device = eligibleDevices[j];

            await this.createNotificationLog({
              userId: device.userId,
              notification,
              platform: device.platform as DevicePlatform,
              deviceToken: device.token,
              expoPushToken: device.expoPushToken!,
              campaignId: options?.campaignId,
              ticket,
            });

            if ((ticket as ExpoPushSuccessTicket).id) {
              sent++;
            } else {
              failed++;
              const errorTicket = ticket as ExpoPushErrorTicket;
              if (errorTicket.details?.error === 'DeviceNotRegistered') {
                await this.markTokenAsInactive(device.expoPushToken!);
              }
            }
          }
        } catch (error) {
          logger.error({ err: error }, 'Error sending batch notifications');
          failed += chunk.length;
        }
      }
    }

    logger.info({ sent, failed, total: userIds.length }, 'Batch notification complete');
    return { sent, failed };
  }

  /**
   * Check push receipts for delivery status
   */
  async checkReceipts(receiptIds: string[]): Promise<Map<string, ExpoPushReceipt>> {
    const receiptIdChunks = expo.chunkPushNotificationReceiptIds(receiptIds);
    const receipts = new Map<string, ExpoPushReceipt>();

    for (const chunk of receiptIdChunks) {
      try {
        const chunkReceipts = await expo.getPushNotificationReceiptsAsync(chunk);

        for (const [receiptId, receipt] of Object.entries(chunkReceipts)) {
          receipts.set(receiptId, receipt);

          // Update notification log if receipt indicates delivery
          if (receipt.status === 'ok') {
            // Mark as delivered in database
            await prisma.notificationLog.updateMany({
              where: {
                data: {
                  path: ['receiptId'],
                  equals: receiptId,
                },
              },
              data: {
                status: 'DELIVERED',
                deliveredAt: new Date(),
              },
            });
          } else if (receipt.status === 'error') {
            logger.error({
              receiptId,
              message: receipt.message,
              details: receipt.details,
            }, 'Push receipt error');

            // Handle device not registered
            if (receipt.details?.error === 'DeviceNotRegistered') {
              // Token needs to be invalidated - we'd need to track which token this receipt belongs to
              logger.warn({ receiptId }, 'Device no longer registered');
            }
          }
        }
      } catch (error) {
        logger.error({ err: error }, 'Error checking push receipts');
      }
    }

    return receipts;
  }

  /**
   * Clean up stale/invalid tokens
   */
  async cleanupStaleTokens(): Promise<number> {
    // Mark tokens as inactive if they haven't been active in 90 days
    const staleDate = new Date();
    staleDate.setDate(staleDate.getDate() - 90);

    const result = await prisma.deviceToken.updateMany({
      where: {
        lastActiveAt: { lt: staleDate },
        isActive: true,
      },
      data: {
        isActive: false,
      },
    });

    logger.info({ count: result.count }, 'Cleaned up stale device tokens');
    return result.count;
  }

  // ============================================================================
  // PRIVATE HELPER METHODS
  // ============================================================================

  /**
   * Ensure notification preferences exist for a user
   */
  private async ensureNotificationPreferences(userId: string): Promise<void> {
    const existing = await prisma.notificationPreference.findUnique({
      where: { userId },
    });

    if (!existing) {
      // Create with all categories enabled by default (except MARKETING)
      const defaultCategories: PrismaNotificationCategory[] = [
        'MEAL_REMINDER',
        'GOAL_PROGRESS',
        'HEALTH_INSIGHT',
        'SUPPLEMENT_REMINDER',
        'STREAK_ALERT',
        'WEEKLY_SUMMARY',
        'SYSTEM',
      ];

      await prisma.notificationPreference.create({
        data: {
          userId,
          enabled: true,
          enabledCategories: defaultCategories,
        },
      });

      logger.info({ userId }, 'Created default notification preferences');
    }
  }

  /**
   * Check if user is in quiet hours
   */
  private async isUserInQuietHours(userId: string): Promise<boolean> {
    const [prefs, user] = await Promise.all([
      prisma.notificationPreference.findUnique({ where: { userId } }),
      prisma.user.findUnique({ where: { id: userId }, select: { timezone: true } }),
    ]);

    if (!prefs) return false;
    return this.checkQuietHours(prefs, user?.timezone);
  }

  /**
   * Check if current time is within quiet hours
   */
  private checkQuietHours(
    prefs: { quietHoursEnabled: boolean; quietHoursStart: string | null; quietHoursEnd: string | null },
    timezone?: string | null
  ): boolean {
    if (!prefs.quietHoursEnabled || !prefs.quietHoursStart || !prefs.quietHoursEnd) {
      return false;
    }

    // Get current time in user's timezone
    const now = new Date();
    const userTime = timezone
      ? new Date(now.toLocaleString('en-US', { timeZone: timezone }))
      : now;

    const currentMinutes = userTime.getHours() * 60 + userTime.getMinutes();

    // Parse quiet hours
    const [startHour, startMin] = prefs.quietHoursStart.split(':').map(Number);
    const [endHour, endMin] = prefs.quietHoursEnd.split(':').map(Number);

    const startMinutes = startHour * 60 + startMin;
    const endMinutes = endHour * 60 + endMin;

    // Handle overnight quiet hours (e.g., 22:00 - 08:00)
    if (startMinutes > endMinutes) {
      return currentMinutes >= startMinutes || currentMinutes < endMinutes;
    }

    return currentMinutes >= startMinutes && currentMinutes < endMinutes;
  }

  /**
   * Check if user has a category enabled
   */
  private async userHasCategory(userId: string, category: NotificationCategory): Promise<boolean> {
    const prefs = await prisma.notificationPreference.findUnique({
      where: { userId },
    });

    if (!prefs?.enabled) return false;
    return prefs.enabledCategories.includes(category as PrismaNotificationCategory);
  }

  /**
   * Mark a token as inactive
   */
  private async markTokenAsInactive(expoPushToken: string): Promise<void> {
    await prisma.deviceToken.updateMany({
      where: { expoPushToken },
      data: { isActive: false },
    });
    logger.info({ token: expoPushToken.slice(0, 20) }, 'Marked device token as inactive');
  }

  /**
   * Create a notification log entry
   */
  private async createNotificationLog(params: {
    userId: string;
    notification: NotificationPayload;
    platform: DevicePlatform;
    deviceToken?: string;
    expoPushToken: string;
    campaignId?: string;
    interruptionLevel?: InterruptionLevel;
    relevanceScore?: number;
    ticket?: ExpoPushTicket;
  }): Promise<void> {
    const { userId, notification, platform, deviceToken, expoPushToken, campaignId, interruptionLevel, relevanceScore, ticket } = params;

    const status: NotificationStatus = ticket
      ? (ticket as ExpoPushSuccessTicket).id
        ? 'SENT'
        : 'FAILED'
      : 'PENDING';

    const error = ticket && (ticket as ExpoPushErrorTicket).message
      ? (ticket as ExpoPushErrorTicket).message
      : undefined;

    await prisma.notificationLog.create({
      data: {
        userId,
        category: notification.category as PrismaNotificationCategory,
        title: notification.title,
        body: notification.body,
        data: {
          ...notification.data,
          receiptId: (ticket as ExpoPushSuccessTicket)?.id,
        },
        platform: platform as PrismaDevicePlatform,
        deviceToken,
        expoPushToken,
        campaignId,
        interruptionLevel,
        relevanceScore,
        status,
        error,
      },
    });
  }
}

// Export singleton instance
export const pushNotificationService = new PushNotificationService();
