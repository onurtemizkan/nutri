import prisma from '../config/database';
import { CreateMealInput, UpdateMealInput } from '../types';

interface DailySummaryData {
  date: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  mealCount: number;
}

export class MealService {
  async createMeal(userId: string, data: CreateMealInput) {
    const meal = await prisma.meal.create({
      data: {
        userId,
        name: data.name,
        mealType: data.mealType,
        calories: data.calories,
        protein: data.protein,
        carbs: data.carbs,
        fat: data.fat,
        fiber: data.fiber,
        sugar: data.sugar,
        servingSize: data.servingSize,
        notes: data.notes,
        consumedAt: data.consumedAt || new Date(),
      },
    });

    return meal;
  }

  /**
   * Get meals for a specific day
   * @param userId - User ID
   * @param date - Optional date (defaults to today)
   * @returns Array of meals for the specified day
   */
  async getMeals(userId: string, date?: Date) {
    // Default to today if no date is provided (common use case for nutrition tracking)
    const targetDate = date || new Date();

    const startOfDay = new Date(targetDate);
    startOfDay.setHours(0, 0, 0, 0);

    const endOfDay = new Date(targetDate);
    endOfDay.setHours(23, 59, 59, 999);

    const meals = await prisma.meal.findMany({
      where: {
        userId,
        consumedAt: {
          gte: startOfDay,
          lte: endOfDay,
        },
      },
      orderBy: {
        consumedAt: 'asc',
      },
    });

    return meals;
  }

  async getMealById(userId: string, mealId: string) {
    const meal = await prisma.meal.findFirst({
      where: {
        id: mealId,
        userId,
      },
    });

    if (!meal) {
      throw new Error('Meal not found');
    }

    return meal;
  }

  async updateMeal(userId: string, mealId: string, data: UpdateMealInput) {
    // Verify meal belongs to user
    await this.getMealById(userId, mealId);

    const meal = await prisma.meal.update({
      where: { id: mealId },
      data,
    });

    return meal;
  }

  async deleteMeal(userId: string, mealId: string) {
    // Verify meal belongs to user
    await this.getMealById(userId, mealId);

    await prisma.meal.delete({
      where: { id: mealId },
    });

    return { message: 'Meal deleted successfully' };
  }

  async getDailySummary(userId: string, date?: Date) {
    const meals = await this.getMeals(userId, date);

    const summary = meals.reduce(
      (acc, meal) => ({
        totalCalories: acc.totalCalories + meal.calories,
        totalProtein: acc.totalProtein + meal.protein,
        totalCarbs: acc.totalCarbs + meal.carbs,
        totalFat: acc.totalFat + meal.fat,
        totalFiber: acc.totalFiber + (meal.fiber || 0),
        totalSugar: acc.totalSugar + (meal.sugar || 0),
        mealCount: acc.mealCount + 1,
      }),
      {
        totalCalories: 0,
        totalProtein: 0,
        totalCarbs: 0,
        totalFat: 0,
        totalFiber: 0,
        totalSugar: 0,
        mealCount: 0,
      }
    );

    // Get user goals
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        goalCalories: true,
        goalProtein: true,
        goalCarbs: true,
        goalFat: true,
      },
    });

    return {
      ...summary,
      goals: user,
      meals,
    };
  }

  async getWeeklySummary(userId: string) {
    const today = new Date();
    const sevenDaysAgo = new Date(today);
    sevenDaysAgo.setDate(today.getDate() - 7);
    sevenDaysAgo.setHours(0, 0, 0, 0);

    const meals = await prisma.meal.findMany({
      where: {
        userId,
        consumedAt: {
          gte: sevenDaysAgo,
        },
      },
      orderBy: {
        consumedAt: 'asc',
      },
    });

    // Group by day
    const dailyData: Record<string, DailySummaryData> = {};

    meals.forEach((meal) => {
      const dateKey = meal.consumedAt.toISOString().split('T')[0];

      if (!dailyData[dateKey]) {
        dailyData[dateKey] = {
          date: dateKey,
          calories: 0,
          protein: 0,
          carbs: 0,
          fat: 0,
          mealCount: 0,
        };
      }

      dailyData[dateKey].calories += meal.calories;
      dailyData[dateKey].protein += meal.protein;
      dailyData[dateKey].carbs += meal.carbs;
      dailyData[dateKey].fat += meal.fat;
      dailyData[dateKey].mealCount += 1;
    });

    return Object.values(dailyData);
  }
}

export const mealService = new MealService();
