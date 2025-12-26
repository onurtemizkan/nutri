import prisma from '../config/database';
import { WEEK_IN_DAYS } from '../config/constants';
import { CreateWaterIntakeInput, UpdateWaterIntakeInput } from '../types';
import { getDayBoundaries, getDaysAgo, getStartOfDay } from '../utils/dateHelpers';

// Preset water amounts in ml
const WATER_PRESETS = {
  glass: 250, // Standard glass
  bottle: 500, // Standard bottle
  cup: 200, // Coffee cup
} as const;

export class WaterService {
  /**
   * Create a new water intake record
   */
  async createWaterIntake(userId: string, data: CreateWaterIntakeInput) {
    const waterIntake = await prisma.waterIntake.create({
      data: {
        userId,
        amount: data.amount,
        recordedAt: data.recordedAt || new Date(),
      },
    });

    return waterIntake;
  }

  /**
   * Quick add water using preset amounts
   */
  async quickAddWater(
    userId: string,
    preset: 'glass' | 'bottle' | 'cup' | 'custom',
    customAmount?: number
  ) {
    const amount =
      preset === 'custom' && customAmount !== undefined
        ? customAmount
        : WATER_PRESETS[preset as keyof typeof WATER_PRESETS] || 250;

    return this.createWaterIntake(userId, { amount });
  }

  /**
   * Get water intakes for a specific day
   */
  async getWaterIntakes(userId: string, date?: Date) {
    const { startOfDay, endOfDay } = getDayBoundaries(date);

    const intakes = await prisma.waterIntake.findMany({
      where: {
        userId,
        recordedAt: {
          gte: startOfDay,
          lte: endOfDay,
        },
      },
      orderBy: {
        recordedAt: 'desc',
      },
    });

    return intakes;
  }

  /**
   * Get a single water intake by ID
   */
  async getWaterIntakeById(userId: string, intakeId: string) {
    const intake = await prisma.waterIntake.findFirst({
      where: {
        id: intakeId,
        userId,
      },
    });

    if (!intake) {
      throw new Error('Water intake not found');
    }

    return intake;
  }

  /**
   * Update a water intake record
   */
  async updateWaterIntake(userId: string, intakeId: string, data: UpdateWaterIntakeInput) {
    // Verify intake belongs to user
    await this.getWaterIntakeById(userId, intakeId);

    const intake = await prisma.waterIntake.update({
      where: { id: intakeId },
      data: {
        amount: data.amount,
        recordedAt: data.recordedAt,
      },
    });

    return intake;
  }

  /**
   * Delete a water intake record
   */
  async deleteWaterIntake(userId: string, intakeId: string) {
    // Verify intake belongs to user
    await this.getWaterIntakeById(userId, intakeId);

    await prisma.waterIntake.delete({
      where: { id: intakeId },
    });

    return { message: 'Water intake deleted successfully' };
  }

  /**
   * Get daily water summary
   */
  async getDailySummary(userId: string, date?: Date) {
    const intakes = await this.getWaterIntakes(userId, date);
    const targetDate = date || new Date();

    // Get user's water goal
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { goalWater: true },
    });

    const goalAmount = user?.goalWater || 2000; // Default 2L if not set
    const totalAmount = intakes.reduce((sum, intake) => sum + intake.amount, 0);
    const percentageComplete = Math.min(100, Math.round((totalAmount / goalAmount) * 100));

    return {
      date: getStartOfDay(targetDate).toISOString(),
      totalAmount,
      goalAmount,
      percentageComplete,
      intakeCount: intakes.length,
      intakes: intakes.map((intake) => ({
        id: intake.id,
        amount: intake.amount,
        recordedAt: intake.recordedAt.toISOString(),
      })),
    };
  }

  /**
   * Get weekly water summary
   */
  async getWeeklySummary(userId: string) {
    const sevenDaysAgo = getDaysAgo(WEEK_IN_DAYS);
    const today = new Date();

    // Get user's water goal
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { goalWater: true },
    });

    const goalAmount = user?.goalWater || 2000;

    // Get all intakes for the week
    const intakes = await prisma.waterIntake.findMany({
      where: {
        userId,
        recordedAt: {
          gte: sevenDaysAgo,
        },
      },
      orderBy: {
        recordedAt: 'asc',
      },
    });

    // Group intakes by day
    const dailyTotals = new Map<string, number>();

    // Initialize all days with 0
    for (let i = WEEK_IN_DAYS - 1; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateKey = getStartOfDay(date).toISOString().split('T')[0];
      dailyTotals.set(dateKey, 0);
    }

    // Sum up daily intakes
    intakes.forEach((intake) => {
      const dateKey = getStartOfDay(intake.recordedAt).toISOString().split('T')[0];
      const current = dailyTotals.get(dateKey) || 0;
      dailyTotals.set(dateKey, current + intake.amount);
    });

    // Calculate statistics
    const dailySummaries = Array.from(dailyTotals.entries()).map(([date, total]) => ({
      date,
      totalAmount: total,
      goalAmount,
      percentageComplete: Math.min(100, Math.round((total / goalAmount) * 100)),
    }));

    const totalAmount = dailySummaries.reduce((sum, day) => sum + day.totalAmount, 0);
    const daysMetGoal = dailySummaries.filter((day) => day.totalAmount >= goalAmount).length;

    return {
      startDate: getStartOfDay(sevenDaysAgo).toISOString(),
      endDate: getStartOfDay(today).toISOString(),
      dailyAverage: Math.round(totalAmount / WEEK_IN_DAYS),
      totalAmount,
      goalAmount: goalAmount * WEEK_IN_DAYS,
      daysMetGoal,
      dailySummaries,
    };
  }

  /**
   * Update user's water goal
   */
  async updateWaterGoal(userId: string, goalWater: number) {
    const user = await prisma.user.update({
      where: { id: userId },
      data: { goalWater },
      select: { goalWater: true },
    });

    return user;
  }

  /**
   * Get user's water goal
   */
  async getWaterGoal(userId: string) {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { goalWater: true },
    });

    return { goalWater: user?.goalWater || 2000 };
  }
}

export const waterService = new WaterService();
