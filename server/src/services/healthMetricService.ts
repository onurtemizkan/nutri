import prisma from '../config/database';
import { CreateHealthMetricInput, GetHealthMetricsQuery, HealthMetricType } from '../types';
import { Prisma } from '@prisma/client';
import { DEFAULT_PAGE_LIMIT, DEFAULT_TIME_PERIOD_DAYS } from '../config/constants';

export class HealthMetricService {
  /**
   * Create a new health metric entry
   */
  async createHealthMetric(userId: string, data: CreateHealthMetricInput) {
    const metric = await prisma.healthMetric.create({
      data: {
        userId,
        metricType: data.metricType,
        value: data.value,
        unit: data.unit,
        recordedAt: data.recordedAt,
        source: data.source,
        sourceId: data.sourceId,
        metadata: data.metadata || {},
      },
    });

    return metric;
  }

  /**
   * Bulk create health metrics (for wearable sync)
   */
  async createBulkHealthMetrics(userId: string, metrics: CreateHealthMetricInput[]) {
    const createdMetrics = await prisma.healthMetric.createMany({
      data: metrics.map((metric) => ({
        userId,
        metricType: metric.metricType,
        value: metric.value,
        unit: metric.unit,
        recordedAt: metric.recordedAt,
        source: metric.source,
        sourceId: metric.sourceId,
        metadata: metric.metadata || {},
      })),
      skipDuplicates: true, // Skip if exact same metric already exists (based on unique constraint)
    });

    return createdMetrics;
  }

  /**
   * Get health metrics with flexible filtering
   */
  async getHealthMetrics(userId: string, query: GetHealthMetricsQuery = {}) {
    const { metricType, startDate, endDate, source, limit = DEFAULT_PAGE_LIMIT } = query;

    const where: Prisma.HealthMetricWhereInput = { userId };

    if (metricType) {
      where.metricType = metricType;
    }

    if (source) {
      where.source = source;
    }

    if (startDate || endDate) {
      where.recordedAt = {};
      if (startDate) {
        where.recordedAt.gte = startDate;
      }
      if (endDate) {
        where.recordedAt.lte = endDate;
      }
    }

    const metrics = await prisma.healthMetric.findMany({
      where,
      orderBy: {
        recordedAt: 'desc',
      },
      take: limit,
    });

    return metrics;
  }

  /**
   * Get specific health metric by ID
   */
  async getHealthMetricById(userId: string, metricId: string) {
    const metric = await prisma.healthMetric.findFirst({
      where: {
        id: metricId,
        userId,
      },
    });

    if (!metric) {
      throw new Error('Health metric not found');
    }

    return metric;
  }

  /**
   * Get latest metric value for a specific type
   */
  async getLatestMetric(userId: string, metricType: HealthMetricType) {
    const metric = await prisma.healthMetric.findFirst({
      where: {
        userId,
        metricType,
      },
      orderBy: {
        recordedAt: 'desc',
      },
    });

    return metric;
  }

  /**
   * Get daily average for a specific metric type
   */
  async getDailyAverage(userId: string, metricType: HealthMetricType, date?: Date) {
    const targetDate = date || new Date();

    const startOfDay = new Date(targetDate);
    startOfDay.setHours(0, 0, 0, 0);

    const endOfDay = new Date(targetDate);
    endOfDay.setHours(23, 59, 59, 999);

    const metrics = await prisma.healthMetric.findMany({
      where: {
        userId,
        metricType,
        recordedAt: {
          gte: startOfDay,
          lte: endOfDay,
        },
      },
    });

    if (metrics.length === 0) {
      return null;
    }

    const average = metrics.reduce((sum, m) => sum + m.value, 0) / metrics.length;

    return {
      date: targetDate.toISOString().split('T')[0],
      metricType,
      average,
      count: metrics.length,
      unit: metrics[0].unit,
    };
  }

  /**
   * Get 7-day rolling average for a specific metric
   */
  async getWeeklyAverage(userId: string, metricType: HealthMetricType) {
    const today = new Date();
    const sevenDaysAgo = new Date(today);
    sevenDaysAgo.setDate(today.getDate() - 7);
    sevenDaysAgo.setHours(0, 0, 0, 0);

    const metrics = await prisma.healthMetric.findMany({
      where: {
        userId,
        metricType,
        recordedAt: {
          gte: sevenDaysAgo,
        },
      },
    });

    if (metrics.length === 0) {
      return null;
    }

    const average = metrics.reduce((sum, m) => sum + m.value, 0) / metrics.length;

    return {
      metricType,
      average,
      count: metrics.length,
      unit: metrics[0].unit,
      startDate: sevenDaysAgo,
      endDate: today,
    };
  }

  /**
   * Get time series data for charting
   */
  async getTimeSeries(
    userId: string,
    metricType: HealthMetricType,
    startDate: Date,
    endDate: Date
  ) {
    const metrics = await prisma.healthMetric.findMany({
      where: {
        userId,
        metricType,
        recordedAt: {
          gte: startDate,
          lte: endDate,
        },
      },
      orderBy: {
        recordedAt: 'asc',
      },
    });

    return metrics.map((m) => ({
      timestamp: m.recordedAt,
      value: m.value,
      unit: m.unit,
      source: m.source,
    }));
  }

  /**
   * Delete a health metric
   */
  async deleteHealthMetric(userId: string, metricId: string) {
    // Verify metric belongs to user
    await this.getHealthMetricById(userId, metricId);

    await prisma.healthMetric.delete({
      where: { id: metricId },
    });

    return { message: 'Health metric deleted successfully' };
  }

  /**
   * Delete all health metrics for a user (for privacy/GDPR)
   */
  async deleteAllHealthMetrics(userId: string) {
    const result = await prisma.healthMetric.deleteMany({
      where: { userId },
    });

    return { message: `Deleted ${result.count} health metrics` };
  }

  /**
   * Get summary statistics for a metric type
   */
  async getMetricStats(userId: string, metricType: HealthMetricType, days: number = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    const metrics = await prisma.healthMetric.findMany({
      where: {
        userId,
        metricType,
        recordedAt: {
          gte: startDate,
        },
      },
    });

    if (metrics.length === 0) {
      return null;
    }

    const values = metrics.map((m) => m.value);
    const sum = values.reduce((a, b) => a + b, 0);
    const avg = sum / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);

    // Calculate standard deviation
    const variance = values.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);

    return {
      metricType,
      days,
      count: metrics.length,
      average: avg,
      min,
      max,
      stdDev,
      unit: metrics[0].unit,
    };
  }
}

export const healthMetricService = new HealthMetricService();
