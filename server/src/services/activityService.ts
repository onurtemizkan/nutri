import prisma from '../config/database';
import { CreateActivityInput, UpdateActivityInput, GetActivitiesQuery, ActivityType } from '../types';
import { Prisma } from '@prisma/client';

export class ActivityService {
  /**
   * Create a new activity entry
   */
  async createActivity(userId: string, data: CreateActivityInput) {
    const activity = await prisma.activity.create({
      data: {
        userId,
        activityType: data.activityType,
        intensity: data.intensity,
        startedAt: data.startedAt,
        endedAt: data.endedAt,
        duration: data.duration,
        caloriesBurned: data.caloriesBurned,
        averageHeartRate: data.averageHeartRate,
        maxHeartRate: data.maxHeartRate,
        distance: data.distance,
        steps: data.steps,
        source: data.source,
        sourceId: data.sourceId,
        notes: data.notes,
      },
    });

    return activity;
  }

  /**
   * Bulk create activities (for wearable sync)
   */
  async createBulkActivities(userId: string, activities: CreateActivityInput[]) {
    const createdActivities = await prisma.activity.createMany({
      data: activities.map((activity) => ({
        userId,
        activityType: activity.activityType,
        intensity: activity.intensity,
        startedAt: activity.startedAt,
        endedAt: activity.endedAt,
        duration: activity.duration,
        caloriesBurned: activity.caloriesBurned,
        averageHeartRate: activity.averageHeartRate,
        maxHeartRate: activity.maxHeartRate,
        distance: activity.distance,
        steps: activity.steps,
        source: activity.source,
        sourceId: activity.sourceId,
        notes: activity.notes,
      })),
      skipDuplicates: false, // Activities don't have unique constraints, so allow duplicates
    });

    return createdActivities;
  }

  /**
   * Get activities with flexible filtering
   */
  async getActivities(userId: string, query: GetActivitiesQuery = {}) {
    const { activityType, intensity, startDate, endDate, source, limit = 100 } = query;

    const where: Prisma.ActivityWhereInput = { userId };

    if (activityType) {
      where.activityType = activityType;
    }

    if (intensity) {
      where.intensity = intensity;
    }

    if (source) {
      where.source = source;
    }

    if (startDate || endDate) {
      where.startedAt = {};
      if (startDate) {
        where.startedAt.gte = startDate;
      }
      if (endDate) {
        where.startedAt.lte = endDate;
      }
    }

    const activities = await prisma.activity.findMany({
      where,
      orderBy: {
        startedAt: 'desc',
      },
      take: limit,
    });

    return activities;
  }

  /**
   * Get specific activity by ID
   */
  async getActivityById(userId: string, activityId: string) {
    const activity = await prisma.activity.findFirst({
      where: {
        id: activityId,
        userId,
      },
    });

    if (!activity) {
      throw new Error('Activity not found');
    }

    return activity;
  }

  /**
   * Update an activity
   */
  async updateActivity(userId: string, activityId: string, data: UpdateActivityInput) {
    // Verify activity belongs to user
    await this.getActivityById(userId, activityId);

    const activity = await prisma.activity.update({
      where: { id: activityId },
      data,
    });

    return activity;
  }

  /**
   * Delete an activity
   */
  async deleteActivity(userId: string, activityId: string) {
    // Verify activity belongs to user
    await this.getActivityById(userId, activityId);

    await prisma.activity.delete({
      where: { id: activityId },
    });

    return { message: 'Activity deleted successfully' };
  }

  /**
   * Get daily activity summary
   */
  async getDailySummary(userId: string, date?: Date) {
    const targetDate = date || new Date();

    const startOfDay = new Date(targetDate);
    startOfDay.setHours(0, 0, 0, 0);

    const endOfDay = new Date(targetDate);
    endOfDay.setHours(23, 59, 59, 999);

    const activities = await prisma.activity.findMany({
      where: {
        userId,
        startedAt: {
          gte: startOfDay,
          lte: endOfDay,
        },
      },
      orderBy: {
        startedAt: 'asc',
      },
    });

    const summary = activities.reduce(
      (acc, activity) => ({
        totalDuration: acc.totalDuration + activity.duration,
        totalCalories: acc.totalCalories + (activity.caloriesBurned || 0),
        totalDistance: acc.totalDistance + (activity.distance || 0),
        totalSteps: acc.totalSteps + (activity.steps || 0),
        activityCount: acc.activityCount + 1,
      }),
      {
        totalDuration: 0,
        totalCalories: 0,
        totalDistance: 0,
        totalSteps: 0,
        activityCount: 0,
      }
    );

    return {
      date: targetDate.toISOString().split('T')[0],
      ...summary,
      activities,
    };
  }

  /**
   * Get weekly activity summary
   */
  async getWeeklySummary(userId: string) {
    const today = new Date();
    const sevenDaysAgo = new Date(today);
    sevenDaysAgo.setDate(today.getDate() - 7);
    sevenDaysAgo.setHours(0, 0, 0, 0);

    const activities = await prisma.activity.findMany({
      where: {
        userId,
        startedAt: {
          gte: sevenDaysAgo,
        },
      },
      orderBy: {
        startedAt: 'asc',
      },
    });

    // Group by day
    const dailyData: Record<string, any> = {};

    activities.forEach((activity) => {
      const dateKey = activity.startedAt.toISOString().split('T')[0];

      if (!dailyData[dateKey]) {
        dailyData[dateKey] = {
          date: dateKey,
          totalDuration: 0,
          totalCalories: 0,
          totalDistance: 0,
          totalSteps: 0,
          activityCount: 0,
        };
      }

      dailyData[dateKey].totalDuration += activity.duration;
      dailyData[dateKey].totalCalories += activity.caloriesBurned || 0;
      dailyData[dateKey].totalDistance += activity.distance || 0;
      dailyData[dateKey].totalSteps += activity.steps || 0;
      dailyData[dateKey].activityCount += 1;
    });

    return Object.values(dailyData);
  }

  /**
   * Get activity statistics by type
   */
  async getActivityStatsByType(userId: string, activityType: ActivityType, days: number = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    const activities = await prisma.activity.findMany({
      where: {
        userId,
        activityType,
        startedAt: {
          gte: startDate,
        },
      },
    });

    if (activities.length === 0) {
      return null;
    }

    const totalDuration = activities.reduce((sum, a) => sum + a.duration, 0);
    const totalCalories = activities.reduce((sum, a) => sum + (a.caloriesBurned || 0), 0);
    const totalDistance = activities.reduce((sum, a) => sum + (a.distance || 0), 0);
    const avgDuration = totalDuration / activities.length;
    const avgCalories = totalCalories / activities.length;

    // Heart rate stats (if available)
    const activitiesWithHR = activities.filter((a) => a.averageHeartRate);
    const avgHeartRate = activitiesWithHR.length
      ? activitiesWithHR.reduce((sum, a) => sum + (a.averageHeartRate || 0), 0) /
        activitiesWithHR.length
      : null;

    return {
      activityType,
      days,
      count: activities.length,
      totalDuration,
      totalCalories,
      totalDistance,
      avgDuration,
      avgCalories,
      avgHeartRate,
    };
  }

  /**
   * Get latest activity
   */
  async getLatestActivity(userId: string) {
    const activity = await prisma.activity.findFirst({
      where: { userId },
      orderBy: {
        startedAt: 'desc',
      },
    });

    return activity;
  }

  /**
   * Get recovery time (hours since last high-intensity activity)
   */
  async getRecoveryTime(userId: string) {
    const lastHighIntensity = await prisma.activity.findFirst({
      where: {
        userId,
        intensity: {
          in: ['HIGH', 'MAXIMUM'],
        },
      },
      orderBy: {
        endedAt: 'desc',
      },
    });

    if (!lastHighIntensity) {
      return null;
    }

    const now = new Date();
    const hoursSinceActivity = (now.getTime() - lastHighIntensity.endedAt.getTime()) / (1000 * 60 * 60);

    return {
      lastActivity: lastHighIntensity,
      hoursSinceActivity: Math.round(hoursSinceActivity * 10) / 10,
      fullyRecovered: hoursSinceActivity >= 48, // Simple heuristic: 48 hours for full recovery
    };
  }

  /**
   * Delete all activities for a user (for privacy/GDPR)
   */
  async deleteAllActivities(userId: string) {
    const result = await prisma.activity.deleteMany({
      where: { userId },
    });

    return { message: `Deleted ${result.count} activities` };
  }
}

export const activityService = new ActivityService();
