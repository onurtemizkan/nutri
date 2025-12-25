import prisma from '../config/database';
import { CreateHealthMetricInput, GetHealthMetricsQuery, HealthMetricType } from '../types';
import { Prisma } from '@prisma/client';
import { DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT, WEEK_IN_DAYS } from '../config/constants';
import { getDayBoundaries, getDaysAgo } from '../utils/dateHelpers';

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
        metadata: (data.metadata as Prisma.InputJsonValue) || Prisma.JsonNull,
      },
    });

    return metric;
  }

  /**
   * Bulk create health metrics (for wearable sync)
   * Returns { created, updated, errors } format expected by frontend
   */
  async createBulkHealthMetrics(userId: string, metrics: CreateHealthMetricInput[]) {
    const result = await prisma.healthMetric.createMany({
      data: metrics.map((metric) => ({
        userId,
        metricType: metric.metricType,
        value: metric.value,
        unit: metric.unit,
        recordedAt: metric.recordedAt,
        source: metric.source,
        sourceId: metric.sourceId,
        metadata: (metric.metadata as Prisma.InputJsonValue) || Prisma.JsonNull,
      })),
      skipDuplicates: true, // Skip if exact same metric already exists (based on unique constraint)
    });

    // Return format expected by frontend
    // Note: Prisma createMany with skipDuplicates doesn't tell us how many were skipped vs created
    // So we return count as "created" and 0 for "updated" (since createMany doesn't update)
    return {
      created: result.count,
      updated: 0,
      errors: [] as { index: number; error: string }[],
    };
  }

  /**
   * Get health metrics with flexible filtering
   */
  async getHealthMetrics(userId: string, query: GetHealthMetricsQuery = {}) {
    const { metricType, startDate, endDate, source, limit = DEFAULT_PAGE_LIMIT } = query;

    // Cap limit to MAX_PAGE_LIMIT to prevent excessive data retrieval
    const cappedLimit = Math.min(limit, MAX_PAGE_LIMIT);

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
      take: cappedLimit,
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
    const { startOfDay, endOfDay } = getDayBoundaries(date);

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
      date: startOfDay.toISOString().split('T')[0],
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
    const sevenDaysAgo = getDaysAgo(WEEK_IN_DAYS);

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
      date: m.recordedAt,
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
    const startDate = getDaysAgo(days);

    const metrics = await prisma.healthMetric.findMany({
      where: {
        userId,
        metricType,
        recordedAt: {
          gte: startDate,
        },
      },
      orderBy: {
        recordedAt: 'asc',
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

    // Calculate trend by comparing first half vs second half of the period
    const { trend, percentChange } = this.calculateTrend(metrics.map(m => m.value));

    return {
      metricType,
      days,
      count: metrics.length,
      average: avg,
      min,
      max,
      stdDev,
      unit: metrics[0].unit,
      trend,
      percentChange,
    };
  }

  /**
   * Calculate trend direction by comparing first half vs second half of values
   * Returns trend direction ('up', 'down', 'stable') and percent change
   */
  private calculateTrend(values: number[]): { trend: 'up' | 'down' | 'stable'; percentChange: number } {
    if (values.length < 2) {
      return { trend: 'stable', percentChange: 0 };
    }

    // Split values into first half and second half (more recent)
    const midpoint = Math.floor(values.length / 2);
    const firstHalf = values.slice(0, midpoint);
    const secondHalf = values.slice(midpoint);

    // Calculate averages for each half
    const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

    // Avoid division by zero
    if (firstAvg === 0) {
      return { trend: 'stable', percentChange: 0 };
    }

    // Calculate percent change
    const percentChange = ((secondAvg - firstAvg) / firstAvg) * 100;

    // Determine trend direction with a threshold to avoid noise
    // Use 3% threshold to filter out small variations
    const TREND_THRESHOLD = 3;

    let trend: 'up' | 'down' | 'stable';
    if (percentChange > TREND_THRESHOLD) {
      trend = 'up';
    } else if (percentChange < -TREND_THRESHOLD) {
      trend = 'down';
    } else {
      trend = 'stable';
    }

    return { trend, percentChange };
  }
}

export const healthMetricService = new HealthMetricService();
