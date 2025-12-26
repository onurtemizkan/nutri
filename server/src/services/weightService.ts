import prisma from '../config/database';
import { WEEK_IN_DAYS, MONTH_IN_DAYS } from '../config/constants';
import {
  CreateWeightRecordInput,
  UpdateWeightRecordInput,
  WeightTrendsResult,
  WeightProgressResult,
} from '../types';
import { getDayBoundaries, getDaysAgo } from '../utils/dateHelpers';

/**
 * Weight Record Service
 * Handles weight tracking CRUD operations, trend analysis, and progress calculations
 */
export class WeightService {
  /**
   * Create a new weight record
   * Also updates the user's currentWeight if this is the latest record
   */
  async createWeightRecord(userId: string, data: CreateWeightRecordInput) {
    const recordedAt = data.recordedAt ? new Date(data.recordedAt) : new Date();

    // Create the weight record
    const weightRecord = await prisma.weightRecord.create({
      data: {
        userId,
        weight: data.weight,
        recordedAt,
      },
    });

    // Check if this is the latest record and update user's currentWeight if so
    const latestRecord = await prisma.weightRecord.findFirst({
      where: { userId },
      orderBy: { recordedAt: 'desc' },
    });

    if (latestRecord && latestRecord.id === weightRecord.id) {
      await prisma.user.update({
        where: { id: userId },
        data: { currentWeight: data.weight },
      });
    }

    return weightRecord;
  }

  /**
   * Get weight records for a user
   * Optionally filter by date range
   */
  async getWeightRecords(
    userId: string,
    options: {
      startDate?: Date;
      endDate?: Date;
      limit?: number;
    } = {}
  ) {
    const { startDate, endDate, limit = 100 } = options;

    const where: {
      userId: string;
      recordedAt?: { gte?: Date; lte?: Date };
    } = { userId };

    if (startDate || endDate) {
      where.recordedAt = {};
      if (startDate) where.recordedAt.gte = startDate;
      if (endDate) where.recordedAt.lte = endDate;
    }

    const records = await prisma.weightRecord.findMany({
      where,
      orderBy: { recordedAt: 'desc' },
      take: limit,
    });

    return records;
  }

  /**
   * Get a specific weight record by ID
   */
  async getWeightRecordById(userId: string, recordId: string) {
    const record = await prisma.weightRecord.findFirst({
      where: {
        id: recordId,
        userId,
      },
    });

    if (!record) {
      throw new Error('Weight record not found');
    }

    return record;
  }

  /**
   * Get weight records for a specific day
   */
  async getWeightRecordsForDay(userId: string, date?: Date) {
    const { startOfDay, endOfDay } = getDayBoundaries(date);

    const records = await prisma.weightRecord.findMany({
      where: {
        userId,
        recordedAt: {
          gte: startOfDay,
          lte: endOfDay,
        },
      },
      orderBy: { recordedAt: 'desc' },
    });

    return records;
  }

  /**
   * Update a weight record
   */
  async updateWeightRecord(userId: string, recordId: string, data: UpdateWeightRecordInput) {
    // Verify record belongs to user
    await this.getWeightRecordById(userId, recordId);

    const updateData: { weight?: number; recordedAt?: Date } = {};
    if (data.weight !== undefined) updateData.weight = data.weight;
    if (data.recordedAt !== undefined) updateData.recordedAt = new Date(data.recordedAt);

    const record = await prisma.weightRecord.update({
      where: { id: recordId },
      data: updateData,
    });

    // Check if this affects the user's currentWeight
    const latestRecord = await prisma.weightRecord.findFirst({
      where: { userId },
      orderBy: { recordedAt: 'desc' },
    });

    if (latestRecord) {
      await prisma.user.update({
        where: { id: userId },
        data: { currentWeight: latestRecord.weight },
      });
    }

    return record;
  }

  /**
   * Delete a weight record
   */
  async deleteWeightRecord(userId: string, recordId: string) {
    // Verify record belongs to user
    await this.getWeightRecordById(userId, recordId);

    await prisma.weightRecord.delete({
      where: { id: recordId },
    });

    // Update user's currentWeight to the latest remaining record
    const latestRecord = await prisma.weightRecord.findFirst({
      where: { userId },
      orderBy: { recordedAt: 'desc' },
    });

    if (latestRecord) {
      await prisma.user.update({
        where: { id: userId },
        data: { currentWeight: latestRecord.weight },
      });
    }

    return { message: 'Weight record deleted successfully' };
  }

  /**
   * Get weight trends with moving averages
   * Returns 7-day and 30-day moving averages for trend visualization
   */
  async getWeightTrends(userId: string, days: number = MONTH_IN_DAYS): Promise<WeightTrendsResult> {
    const startDate = getDaysAgo(days);

    const records = await prisma.weightRecord.findMany({
      where: {
        userId,
        recordedAt: { gte: startDate },
      },
      orderBy: { recordedAt: 'asc' },
    });

    if (records.length === 0) {
      return {
        records: [],
        movingAverage7Day: [],
        movingAverage30Day: [],
        minWeight: null,
        maxWeight: null,
        averageWeight: null,
        totalChange: null,
        weeklyChange: null,
      };
    }

    // Calculate moving averages
    const movingAverage7Day: Array<{ date: string; value: number }> = [];
    const movingAverage30Day: Array<{ date: string; value: number }> = [];

    // Group records by date
    const recordsByDate = new Map<string, number[]>();
    for (const record of records) {
      const dateKey = record.recordedAt.toISOString().split('T')[0];
      if (!recordsByDate.has(dateKey)) {
        recordsByDate.set(dateKey, []);
      }
      recordsByDate.get(dateKey)!.push(record.weight);
    }

    // Calculate daily averages
    const dailyAverages: Array<{ date: string; weight: number }> = [];
    for (const [date, weights] of recordsByDate) {
      const avg = weights.reduce((sum, w) => sum + w, 0) / weights.length;
      dailyAverages.push({ date, weight: avg });
    }
    dailyAverages.sort((a, b) => a.date.localeCompare(b.date));

    // Calculate moving averages
    for (let i = 0; i < dailyAverages.length; i++) {
      const current = dailyAverages[i];

      // 7-day moving average
      const start7 = Math.max(0, i - 6);
      const slice7 = dailyAverages.slice(start7, i + 1);
      const avg7 = slice7.reduce((sum, d) => sum + d.weight, 0) / slice7.length;
      movingAverage7Day.push({ date: current.date, value: Math.round(avg7 * 100) / 100 });

      // 30-day moving average
      const start30 = Math.max(0, i - 29);
      const slice30 = dailyAverages.slice(start30, i + 1);
      const avg30 = slice30.reduce((sum, d) => sum + d.weight, 0) / slice30.length;
      movingAverage30Day.push({ date: current.date, value: Math.round(avg30 * 100) / 100 });
    }

    // Calculate statistics
    const weights = records.map((r) => r.weight);
    const minWeight = Math.min(...weights);
    const maxWeight = Math.max(...weights);
    const averageWeight =
      Math.round((weights.reduce((sum, w) => sum + w, 0) / weights.length) * 100) / 100;

    // Calculate total change (first to last)
    const totalChange =
      Math.round((records[records.length - 1].weight - records[0].weight) * 100) / 100;

    // Calculate weekly change (last 7 days)
    const sevenDaysAgo = getDaysAgo(WEEK_IN_DAYS);
    const recentRecords = records.filter((r) => r.recordedAt >= sevenDaysAgo);
    let weeklyChange: number | null = null;
    if (recentRecords.length >= 2) {
      weeklyChange =
        Math.round(
          (recentRecords[recentRecords.length - 1].weight - recentRecords[0].weight) * 100
        ) / 100;
    }

    return {
      records: records.map((r) => ({
        id: r.id,
        weight: r.weight,
        recordedAt: r.recordedAt.toISOString(),
      })),
      movingAverage7Day,
      movingAverage30Day,
      minWeight,
      maxWeight,
      averageWeight,
      totalChange,
      weeklyChange,
    };
  }

  /**
   * Get weight progress towards goal
   */
  async getWeightProgress(userId: string): Promise<WeightProgressResult> {
    // Get user's goal weight and current weight
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        currentWeight: true,
        goalWeight: true,
        height: true,
      },
    });

    if (!user) {
      throw new Error('User not found');
    }

    // Get first and latest weight records
    const [firstRecord, latestRecord] = await Promise.all([
      prisma.weightRecord.findFirst({
        where: { userId },
        orderBy: { recordedAt: 'asc' },
      }),
      prisma.weightRecord.findFirst({
        where: { userId },
        orderBy: { recordedAt: 'desc' },
      }),
    ]);

    const startWeight = firstRecord?.weight ?? null;
    const currentWeight = latestRecord?.weight ?? user.currentWeight ?? null;
    const goalWeight = user.goalWeight ?? null;

    let progressPercentage: number | null = null;
    let remainingWeight: number | null = null;
    let isOnTrack: boolean | null = null;

    if (startWeight !== null && currentWeight !== null && goalWeight !== null) {
      const totalToLose = startWeight - goalWeight;
      const lostSoFar = startWeight - currentWeight;

      if (totalToLose !== 0) {
        progressPercentage = Math.round((lostSoFar / totalToLose) * 10000) / 100; // 2 decimal places
        progressPercentage = Math.max(-100, Math.min(200, progressPercentage)); // Clamp between -100% and 200%
      }

      remainingWeight = Math.round((currentWeight - goalWeight) * 100) / 100;

      // On track if moving in the right direction
      if (goalWeight < startWeight) {
        isOnTrack = currentWeight <= startWeight;
      } else if (goalWeight > startWeight) {
        isOnTrack = currentWeight >= startWeight;
      } else {
        isOnTrack = true;
      }
    }

    // Calculate BMI if height is available
    let bmi: number | null = null;
    let bmiCategory: string | null = null;
    if (currentWeight !== null && user.height !== null && user.height > 0) {
      // Height is stored in cm, convert to meters
      const heightInMeters = user.height / 100;
      bmi = Math.round((currentWeight / (heightInMeters * heightInMeters)) * 10) / 10;
      bmiCategory = this.getBmiCategory(bmi);
    }

    return {
      startWeight,
      currentWeight,
      goalWeight,
      progressPercentage,
      remainingWeight,
      isOnTrack,
      bmi,
      bmiCategory,
      startDate: firstRecord?.recordedAt.toISOString() ?? null,
      latestRecordDate: latestRecord?.recordedAt.toISOString() ?? null,
    };
  }

  /**
   * Update user's goal weight
   */
  async updateGoalWeight(userId: string, goalWeight: number) {
    const user = await prisma.user.update({
      where: { id: userId },
      data: { goalWeight },
      select: {
        id: true,
        goalWeight: true,
        currentWeight: true,
      },
    });

    return user;
  }

  /**
   * Get BMI category based on WHO standards
   */
  private getBmiCategory(bmi: number): string {
    if (bmi < 18.5) return 'Underweight';
    if (bmi < 25) return 'Normal';
    if (bmi < 30) return 'Overweight';
    if (bmi < 35) return 'Obese Class I';
    if (bmi < 40) return 'Obese Class II';
    return 'Obese Class III';
  }

  /**
   * Calculate BMI for a given weight and height
   */
  calculateBmi(weightKg: number, heightCm: number): { bmi: number; category: string } | null {
    if (weightKg <= 0 || heightCm <= 0) return null;
    const heightInMeters = heightCm / 100;
    const bmi = Math.round((weightKg / (heightInMeters * heightInMeters)) * 10) / 10;
    return {
      bmi,
      category: this.getBmiCategory(bmi),
    };
  }

  /**
   * Get weight summary for dashboard widget
   */
  async getWeightSummary(userId: string) {
    const [latestRecord, weekAgoRecord, progress] = await Promise.all([
      prisma.weightRecord.findFirst({
        where: { userId },
        orderBy: { recordedAt: 'desc' },
      }),
      prisma.weightRecord.findFirst({
        where: {
          userId,
          recordedAt: { lte: getDaysAgo(WEEK_IN_DAYS) },
        },
        orderBy: { recordedAt: 'desc' },
      }),
      this.getWeightProgress(userId),
    ]);

    const currentWeight = latestRecord?.weight ?? null;
    let weeklyChange: number | null = null;

    if (latestRecord && weekAgoRecord) {
      weeklyChange = Math.round((latestRecord.weight - weekAgoRecord.weight) * 100) / 100;
    }

    return {
      currentWeight,
      weeklyChange,
      goalWeight: progress.goalWeight,
      progressPercentage: progress.progressPercentage,
      bmi: progress.bmi,
      bmiCategory: progress.bmiCategory,
      lastRecordDate: latestRecord?.recordedAt.toISOString() ?? null,
    };
  }
}

export const weightService = new WeightService();
