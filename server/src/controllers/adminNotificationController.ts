/**
 * Admin Notification Controller
 *
 * Handles admin operations for notification campaigns, analytics, and user management.
 */

import { Request, Response, NextFunction } from 'express';
import { CampaignStatus, Prisma } from '@prisma/client';
import prisma from '../config/database';
import { HTTP_STATUS } from '../config/constants';
import { notificationScheduler } from '../services/notificationScheduler';
import {
  createCampaignSchema,
  updateCampaignSchema,
  sendTestNotificationSchema,
} from '../validation/adminNotificationSchemas';

// =============================================================================
// CAMPAIGN MANAGEMENT
// =============================================================================

/**
 * List all notification campaigns with pagination
 */
export async function listCampaigns(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 20;
    const status = req.query.status as CampaignStatus | undefined;
    const skip = (page - 1) * limit;

    const where: Prisma.NotificationCampaignWhereInput = status ? { status } : {};

    const [campaigns, total] = await Promise.all([
      prisma.notificationCampaign.findMany({
        where,
        skip,
        take: limit,
        orderBy: { createdAt: 'desc' },
        include: {
          _count: {
            select: {
              notificationLogs: true,
            },
          },
        },
      }),
      prisma.notificationCampaign.count({ where }),
    ]);

    // Calculate stats for each campaign
    const campaignsWithStats = await Promise.all(
      campaigns.map(async (campaign) => {
        const stats = await prisma.notificationLog.groupBy({
          by: ['status'],
          where: { campaignId: campaign.id },
          _count: true,
        });

        const statsMap = stats.reduce(
          (acc, stat) => {
            acc[stat.status] = stat._count;
            return acc;
          },
          {} as Record<string, number>
        );

        return {
          ...campaign,
          stats: {
            total: campaign._count.notificationLogs,
            sent: statsMap.SENT || 0,
            delivered: statsMap.DELIVERED || 0,
            opened: statsMap.OPENED || 0,
            failed: statsMap.FAILED || 0,
          },
        };
      })
    );

    res.json({
      campaigns: campaignsWithStats,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    next(error);
  }
}

/**
 * Get a single campaign with detailed stats
 */
export async function getCampaign(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const { id } = req.params;

    const campaign = await prisma.notificationCampaign.findUnique({
      where: { id },
      include: {
        notificationLogs: {
          take: 100,
          orderBy: { sentAt: 'desc' },
          select: {
            id: true,
            status: true,
            sentAt: true,
            deliveredAt: true,
            openedAt: true,
            error: true,
          },
        },
      },
    });

    if (!campaign) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Campaign not found' });
      return;
    }

    // Get aggregated stats
    const stats = await prisma.notificationLog.groupBy({
      by: ['status'],
      where: { campaignId: id },
      _count: true,
    });

    const statsMap = stats.reduce(
      (acc, stat) => {
        acc[stat.status] = stat._count;
        return acc;
      },
      {} as Record<string, number>
    );

    const totalSent = Object.values(statsMap).reduce((a, b) => a + b, 0);

    res.json({
      ...campaign,
      stats: {
        total: totalSent,
        sent: statsMap.SENT || 0,
        delivered: statsMap.DELIVERED || 0,
        opened: statsMap.OPENED || 0,
        failed: statsMap.FAILED || 0,
        deliveryRate: totalSent > 0 ? ((statsMap.DELIVERED || 0) / totalSent) * 100 : 0,
        openRate: totalSent > 0 ? ((statsMap.OPENED || 0) / totalSent) * 100 : 0,
      },
    });
  } catch (error) {
    next(error);
  }
}

/**
 * Create a new notification campaign
 */
export async function createCampaign(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const validatedData = createCampaignSchema.parse(req.body);
    const adminId = (req as Request & { admin: { id: string } }).admin.id;

    const campaign = await prisma.notificationCampaign.create({
      data: {
        title: validatedData.name,
        notificationTitle: validatedData.title,
        notificationBody: validatedData.body,
        category: 'MARKETING',
        targetSegment: validatedData.targetSegment ? { segment: validatedData.targetSegment } : { segment: 'ALL' },
        scheduledAt: validatedData.scheduledFor ? new Date(validatedData.scheduledFor) : null,
        status: validatedData.scheduledFor ? 'SCHEDULED' : 'DRAFT',
        createdByAdminId: adminId,
        data: (validatedData.metadata ?? {}) as Prisma.InputJsonValue,
      },
    });

    // If scheduled, queue the campaign
    if (campaign.scheduledAt && campaign.status === 'SCHEDULED') {
      await notificationScheduler.scheduleCampaign(campaign.id, campaign.scheduledAt);
    }

    res.status(HTTP_STATUS.CREATED).json(campaign);
  } catch (error) {
    next(error);
  }
}

/**
 * Update a campaign (only allowed for DRAFT or SCHEDULED campaigns)
 */
export async function updateCampaign(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const { id } = req.params;
    const validatedData = updateCampaignSchema.parse(req.body);

    // Check campaign exists and is editable
    const existing = await prisma.notificationCampaign.findUnique({
      where: { id },
    });

    if (!existing) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Campaign not found' });
      return;
    }

    if (!['DRAFT', 'SCHEDULED'].includes(existing.status)) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Cannot update campaign that has already been sent',
      });
      return;
    }

    const campaign = await prisma.notificationCampaign.update({
      where: { id },
      data: {
        title: validatedData.name ?? existing.title,
        notificationTitle: validatedData.title ?? existing.notificationTitle,
        notificationBody: validatedData.body ?? existing.notificationBody,
        targetSegment: validatedData.targetSegment
          ? ({ segment: validatedData.targetSegment } as Prisma.InputJsonValue)
          : (existing.targetSegment as Prisma.InputJsonValue),
        scheduledAt: validatedData.scheduledFor
          ? new Date(validatedData.scheduledFor)
          : existing.scheduledAt,
        status: validatedData.scheduledFor ? 'SCHEDULED' : existing.status,
        data: (validatedData.metadata ?? existing.data ?? {}) as Prisma.InputJsonValue,
      },
    });

    // Schedule if newly scheduled
    if (campaign.scheduledAt && campaign.status === 'SCHEDULED') {
      await notificationScheduler.scheduleCampaign(campaign.id, campaign.scheduledAt);
    }

    res.json(campaign);
  } catch (error) {
    next(error);
  }
}

/**
 * Delete a campaign (only allowed for DRAFT campaigns)
 */
export async function deleteCampaign(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const { id } = req.params;

    const campaign = await prisma.notificationCampaign.findUnique({
      where: { id },
    });

    if (!campaign) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Campaign not found' });
      return;
    }

    if (campaign.status !== 'DRAFT') {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Can only delete draft campaigns. Cancel the campaign first.',
      });
      return;
    }

    await prisma.notificationCampaign.delete({
      where: { id },
    });

    res.status(HTTP_STATUS.OK).json({ success: true });
  } catch (error) {
    next(error);
  }
}

/**
 * Cancel a scheduled campaign
 */
export async function cancelCampaign(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const { id } = req.params;

    const campaign = await prisma.notificationCampaign.findUnique({
      where: { id },
    });

    if (!campaign) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Campaign not found' });
      return;
    }

    if (campaign.status !== 'SCHEDULED') {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Can only cancel scheduled campaigns',
      });
      return;
    }

    const updated = await prisma.notificationCampaign.update({
      where: { id },
      data: { status: 'CANCELLED' },
    });

    res.json(updated);
  } catch (error) {
    next(error);
  }
}

/**
 * Send a campaign immediately
 */
export async function sendCampaign(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const { id } = req.params;

    const campaign = await prisma.notificationCampaign.findUnique({
      where: { id },
    });

    if (!campaign) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Campaign not found' });
      return;
    }

    if (!['DRAFT', 'SCHEDULED'].includes(campaign.status)) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Campaign has already been sent or cancelled',
      });
      return;
    }

    // Update status to SENDING
    await prisma.notificationCampaign.update({
      where: { id },
      data: { status: 'SENDING' },
    });

    // Queue for immediate send
    await notificationScheduler.scheduleCampaign(id, new Date());

    res.json({ success: true, message: 'Campaign send initiated' });
  } catch (error) {
    next(error);
  }
}

// =============================================================================
// NOTIFICATION ANALYTICS
// =============================================================================

/**
 * Get notification analytics overview
 */
export async function getNotificationAnalytics(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const days = parseInt(req.query.days as string) || 30;
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    // Get overall stats
    const [totalStats, dailyStats, categoryStats] = await Promise.all([
      // Overall delivery stats
      prisma.notificationLog.groupBy({
        by: ['status'],
        where: { sentAt: { gte: startDate } },
        _count: true,
      }),

      // Daily breakdown
      prisma.$queryRaw<{ date: Date; status: string; count: bigint }[]>`
        SELECT
          DATE("sentAt") as date,
          status,
          COUNT(*) as count
        FROM "NotificationLog"
        WHERE "sentAt" >= ${startDate}
        GROUP BY DATE("sentAt"), status
        ORDER BY date DESC
      `,

      // Category breakdown
      prisma.notificationLog.groupBy({
        by: ['category'],
        where: { sentAt: { gte: startDate } },
        _count: true,
      }),
    ]);

    // Calculate rates
    const totalSent = totalStats.reduce((acc, stat) => acc + stat._count, 0);
    const delivered = totalStats.find((s) => s.status === 'DELIVERED')?._count || 0;
    const opened = totalStats.find((s) => s.status === 'OPENED')?._count || 0;
    const failed = totalStats.find((s) => s.status === 'FAILED')?._count || 0;

    // Process daily stats
    const dailyData = dailyStats.reduce(
      (acc, stat) => {
        const dateStr = stat.date.toISOString().split('T')[0];
        if (!acc[dateStr]) {
          acc[dateStr] = { date: dateStr, sent: 0, delivered: 0, opened: 0, failed: 0 };
        }
        const count = Number(stat.count);
        if (stat.status === 'SENT') acc[dateStr].sent += count;
        if (stat.status === 'DELIVERED') acc[dateStr].delivered += count;
        if (stat.status === 'OPENED') acc[dateStr].opened += count;
        if (stat.status === 'FAILED') acc[dateStr].failed += count;
        return acc;
      },
      {} as Record<string, { date: string; sent: number; delivered: number; opened: number; failed: number }>
    );

    res.json({
      period: {
        start: startDate.toISOString(),
        end: new Date().toISOString(),
        days,
      },
      totals: {
        sent: totalSent,
        delivered,
        opened,
        failed,
        deliveryRate: totalSent > 0 ? ((delivered / totalSent) * 100).toFixed(2) : '0.00',
        openRate: totalSent > 0 ? ((opened / totalSent) * 100).toFixed(2) : '0.00',
        failureRate: totalSent > 0 ? ((failed / totalSent) * 100).toFixed(2) : '0.00',
      },
      dailyBreakdown: Object.values(dailyData).sort((a, b) => a.date.localeCompare(b.date)),
      categoryBreakdown: categoryStats.map((stat) => ({
        category: stat.category,
        count: stat._count,
      })),
    });
  } catch (error) {
    next(error);
  }
}

/**
 * Get device registration stats
 */
export async function getDeviceStats(
  _req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const [platformStats, activeDevices, totalDevices] = await Promise.all([
      // Platform breakdown
      prisma.deviceToken.groupBy({
        by: ['platform'],
        where: { isActive: true },
        _count: true,
      }),

      // Active devices
      prisma.deviceToken.count({
        where: { isActive: true },
      }),

      // Total devices ever registered
      prisma.deviceToken.count(),
    ]);

    res.json({
      activeDevices,
      totalDevices,
      inactiveDevices: totalDevices - activeDevices,
      platformBreakdown: platformStats.map((stat) => ({
        platform: stat.platform,
        count: stat._count,
      })),
    });
  } catch (error) {
    next(error);
  }
}

// =============================================================================
// USER NOTIFICATION MANAGEMENT
// =============================================================================

/**
 * Get a user's notification preferences and history
 */
export async function getUserNotifications(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const { userId } = req.params;
    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 50;
    const skip = (page - 1) * limit;

    const [preferences, devices, recentNotifications, total] = await Promise.all([
      prisma.notificationPreference.findUnique({
        where: { userId },
      }),
      prisma.deviceToken.findMany({
        where: { userId, isActive: true },
        select: {
          id: true,
          platform: true,
          deviceModel: true,
          osVersion: true,
          appVersion: true,
          createdAt: true,
          lastActiveAt: true,
        },
      }),
      prisma.notificationLog.findMany({
        where: { userId },
        skip,
        take: limit,
        orderBy: { sentAt: 'desc' },
        select: {
          id: true,
          title: true,
          body: true,
          category: true,
          status: true,
          sentAt: true,
          deliveredAt: true,
          openedAt: true,
          error: true,
        },
      }),
      prisma.notificationLog.count({ where: { userId } }),
    ]);

    res.json({
      preferences,
      devices,
      notifications: {
        items: recentNotifications,
        pagination: {
          page,
          limit,
          total,
          pages: Math.ceil(total / limit),
        },
      },
    });
  } catch (error) {
    next(error);
  }
}

/**
 * Send a test notification to a specific user
 */
export async function sendTestNotification(
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> {
  try {
    const { userId } = req.params;
    const validatedData = sendTestNotificationSchema.parse(req.body);

    // Get user's active devices
    const devices = await prisma.deviceToken.findMany({
      where: { userId, isActive: true },
      select: { token: true, expoPushToken: true, platform: true },
    });

    if (devices.length === 0) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'User has no active devices registered for push notifications',
      });
      return;
    }

    // Create notification log entries
    const logs = await Promise.all(
      devices.map((device) =>
        prisma.notificationLog.create({
          data: {
            userId,
            deviceToken: device.token,
            platform: device.platform,
            title: validatedData.title,
            body: validatedData.body,
            category: 'SYSTEM',
            status: 'PENDING',
            data: { test: true },
          },
        })
      )
    );

    res.json({
      success: true,
      message: `Test notification queued for ${devices.length} device(s)`,
      notificationIds: logs.map((l) => l.id),
    });
  } catch (error) {
    next(error);
  }
}

// =============================================================================
// NOTIFICATION TEMPLATES
// =============================================================================

/**
 * Get available notification templates
 */
export async function getNotificationTemplates(
  _req: Request,
  res: Response,
  _next: NextFunction
): Promise<void> {
  // Predefined templates for common notification types
  const templates = [
    {
      id: 'meal_reminder_morning',
      name: 'Breakfast Reminder',
      category: 'MEAL_REMINDER',
      title: "Good morning! Time for breakfast",
      body: "Start your day right! Don't forget to log your breakfast.",
      variables: [],
    },
    {
      id: 'meal_reminder_afternoon',
      name: 'Lunch Reminder',
      category: 'MEAL_REMINDER',
      title: "Lunch time!",
      body: "Take a moment to log what you're eating for lunch.",
      variables: [],
    },
    {
      id: 'meal_reminder_evening',
      name: 'Dinner Reminder',
      category: 'MEAL_REMINDER',
      title: "Dinner time!",
      body: "Don't forget to log your dinner to track your daily nutrition.",
      variables: [],
    },
    {
      id: 'streak_at_risk',
      name: 'Streak At Risk',
      category: 'STREAK_ALERT',
      title: "Keep your streak alive!",
      body: "You haven't logged a meal today. Log now to maintain your {{streak}} day streak!",
      variables: ['streak'],
    },
    {
      id: 'goal_achieved',
      name: 'Goal Achieved',
      category: 'GOAL_PROGRESS',
      title: "Goal achieved!",
      body: "Congratulations! You've hit your {{goal}} goal today!",
      variables: ['goal'],
    },
    {
      id: 'weekly_summary',
      name: 'Weekly Summary',
      category: 'WEEKLY_SUMMARY',
      title: "Your weekly nutrition recap",
      body: "See how you did this week and set new goals!",
      variables: [],
    },
    {
      id: 'health_insight',
      name: 'Health Insight',
      category: 'HEALTH_INSIGHT',
      title: "New health insight available",
      body: "We've noticed a pattern in your data. Tap to learn more.",
      variables: [],
    },
    {
      id: 'supplement_reminder',
      name: 'Supplement Reminder',
      category: 'SUPPLEMENT_REMINDER',
      title: "Time for your supplements",
      body: "Don't forget to take your {{supplement}} today!",
      variables: ['supplement'],
    },
  ];

  res.json({ templates });
}
