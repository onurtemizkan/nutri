import prisma from '../config/database';
import { USER_GOALS_SELECT_FIELDS, WEEK_IN_DAYS } from '../config/constants';
import { CreateMealInput, UpdateMealInput } from '../types';
import { getDayBoundaries, getDaysAgo } from '../utils/dateHelpers';

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

        // Fat breakdown
        saturatedFat: data.saturatedFat,
        transFat: data.transFat,
        cholesterol: data.cholesterol,

        // Minerals
        sodium: data.sodium,
        potassium: data.potassium,
        calcium: data.calcium,
        iron: data.iron,
        magnesium: data.magnesium,
        zinc: data.zinc,
        phosphorus: data.phosphorus,

        // Vitamins
        vitaminA: data.vitaminA,
        vitaminC: data.vitaminC,
        vitaminD: data.vitaminD,
        vitaminE: data.vitaminE,
        vitaminK: data.vitaminK,
        vitaminB6: data.vitaminB6,
        vitaminB12: data.vitaminB12,
        folate: data.folate,
        thiamin: data.thiamin,
        riboflavin: data.riboflavin,
        niacin: data.niacin,
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
    const { startOfDay, endOfDay } = getDayBoundaries(date);

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
      select: USER_GOALS_SELECT_FIELDS,
    });

    return {
      ...summary,
      goals: user,
      meals,
    };
  }

  async getWeeklySummary(userId: string) {
    const sevenDaysAgo = getDaysAgo(WEEK_IN_DAYS);

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

    // Calculate totals
    const totalCalories = meals.reduce((sum, meal) => sum + meal.calories, 0);
    const totalProtein = meals.reduce((sum, meal) => sum + meal.protein, 0);
    const totalCarbs = meals.reduce((sum, meal) => sum + meal.carbs, 0);
    const totalFat = meals.reduce((sum, meal) => sum + meal.fat, 0);
    const mealCount = meals.length;

    // Calculate daily averages (over 7 days)
    const averageDailyCalories = totalCalories / WEEK_IN_DAYS;
    const averageDailyProtein = totalProtein / WEEK_IN_DAYS;
    const averageDailyCarbs = totalCarbs / WEEK_IN_DAYS;
    const averageDailyFat = totalFat / WEEK_IN_DAYS;

    return {
      totalCalories,
      totalProtein,
      totalCarbs,
      totalFat,
      mealCount,
      averageDailyCalories,
      averageDailyProtein,
      averageDailyCarbs,
      averageDailyFat,
    };
  }
}

export const mealService = new MealService();
