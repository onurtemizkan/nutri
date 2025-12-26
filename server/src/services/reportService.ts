import prisma from '../config/database';
import { USER_GOALS_SELECT_FIELDS, WEEK_IN_DAYS } from '../config/constants';
import { getDaysAgo } from '../utils/dateHelpers';
import {
  WeeklyReport,
  MonthlyReport,
  DailyBreakdown,
  NutritionTotals,
  GoalProgress,
  TrendComparison,
  MetricTrend,
  TopFood,
  HealthMetricSummary,
  ActivitySummary,
  ReportInsight,
  ReportAchievement,
  WeeklyBreakdown,
  ReportGenerationOptions,
} from '../types/reports';
import { HealthMetricType } from '../types';

// Default options for report generation
const DEFAULT_OPTIONS: ReportGenerationOptions = {
  includeHealthMetrics: true,
  includeActivities: true,
  includeInsights: true,
  includeAchievements: true,
  topFoodsLimit: 10,
};

// Trend threshold percentage - changes below this are considered stable
const TREND_THRESHOLD = 3;

export class ReportService {
  // ==========================================================================
  // WEEKLY REPORT
  // ==========================================================================

  /**
   * Generate a comprehensive weekly nutrition report
   * @param userId - User ID
   * @param weekDate - Any date within the desired week (defaults to current week)
   * @param options - Generation options
   */
  async generateWeeklyReport(
    userId: string,
    weekDate?: Date,
    options: ReportGenerationOptions = DEFAULT_OPTIONS
  ): Promise<WeeklyReport> {
    const { weekStart, weekEnd } = this.getWeekBoundaries(weekDate || new Date());

    // Fetch all data in parallel for optimal performance
    const [user, meals, previousWeekMeals, _healthMetrics, activities] = await Promise.all([
      prisma.user.findUnique({
        where: { id: userId },
        select: USER_GOALS_SELECT_FIELDS,
      }),
      this.getMealsForPeriod(userId, weekStart, weekEnd),
      this.getMealsForPeriod(
        userId,
        this.subtractDays(weekStart, WEEK_IN_DAYS),
        this.subtractDays(weekEnd, WEEK_IN_DAYS)
      ),
      options.includeHealthMetrics
        ? this.getHealthMetricsForPeriod(userId, weekStart, weekEnd)
        : Promise.resolve([]),
      options.includeActivities
        ? this.getActivitiesForPeriod(userId, weekStart, weekEnd)
        : Promise.resolve([]),
    ]);

    if (!user) {
      throw new Error('User not found');
    }

    // Calculate daily breakdowns
    const dailyBreakdowns = this.calculateDailyBreakdowns(meals, weekStart, weekEnd, user);

    // Calculate totals and averages
    const totals = this.calculateTotals(meals);
    const daysWithData = dailyBreakdowns.filter((d) => d.mealCount > 0).length;
    const averages = this.calculateAverages(totals, daysWithData || 1);

    // Calculate goal progress
    const goalProgress = this.calculateGoalProgress(averages, user);

    // Calculate days with goals met
    const daysGoalsMet = dailyBreakdowns.filter((d) => d.goalCompletion >= 90).length;
    const goalCompletionRate = daysWithData > 0 ? (daysGoalsMet / daysWithData) * 100 : 0;

    // Calculate trends compared to previous week
    const previousTotals = this.calculateTotals(previousWeekMeals);
    const previousDaysWithData = this.countDaysWithData(previousWeekMeals);
    const previousAverages = this.calculateAverages(previousTotals, previousDaysWithData || 1);
    const trends = this.calculateTrends(averages, previousAverages);

    // Get top foods
    const topFoods = await this.getTopFoods(
      userId,
      weekStart,
      weekEnd,
      options.topFoodsLimit || 10
    );

    // Health metrics summary
    const healthMetricsSummary = options.includeHealthMetrics
      ? await this.getHealthMetricsSummary(userId, weekStart, weekEnd)
      : [];

    // Activity summary
    const activitySummary = options.includeActivities
      ? this.calculateActivitySummary(activities)
      : null;

    // Generate insights
    const insights = options.includeInsights
      ? this.generateWeeklyInsights(
          dailyBreakdowns,
          goalProgress,
          trends,
          healthMetricsSummary,
          activitySummary
        )
      : [];

    // Get achievements
    const achievements = options.includeAchievements
      ? await this.getAchievements(userId, weekStart, weekEnd)
      : [];

    // Calculate streak
    const streak = await this.calculateStreak(userId);

    // Meal type breakdown
    const mealTypeBreakdown = this.calculateMealTypeBreakdown(meals);

    // Macro distribution
    const macroDistribution = this.calculateMacroDistribution(totals);

    return {
      userId,
      generatedAt: new Date().toISOString(),
      periodStart: weekStart.toISOString(),
      periodEnd: weekEnd.toISOString(),
      dailyBreakdowns,
      totals,
      averages,
      goalProgress,
      daysGoalsMet,
      goalCompletionRate: Math.round(goalCompletionRate * 10) / 10,
      trends,
      topFoods,
      healthMetricsSummary,
      activitySummary,
      insights,
      achievements,
      streak,
      mealTypeBreakdown,
      macroDistribution,
    };
  }

  // ==========================================================================
  // MONTHLY REPORT
  // ==========================================================================

  /**
   * Generate a comprehensive monthly nutrition report
   * @param userId - User ID
   * @param month - Month in YYYY-MM format (defaults to current month)
   * @param options - Generation options
   */
  async generateMonthlyReport(
    userId: string,
    month?: string,
    options: ReportGenerationOptions = DEFAULT_OPTIONS
  ): Promise<MonthlyReport> {
    const { monthStart, monthEnd, monthString } = this.getMonthBoundaries(month);

    // Get previous month boundaries for trend comparison
    const previousMonth = new Date(monthStart);
    previousMonth.setMonth(previousMonth.getMonth() - 1);
    const { monthStart: prevMonthStart, monthEnd: prevMonthEnd } = this.getMonthBoundaries(
      `${previousMonth.getFullYear()}-${String(previousMonth.getMonth() + 1).padStart(2, '0')}`
    );

    // Fetch all data in parallel
    const [user, meals, previousMonthMeals, _healthMetrics, activities] = await Promise.all([
      prisma.user.findUnique({
        where: { id: userId },
        select: USER_GOALS_SELECT_FIELDS,
      }),
      this.getMealsForPeriod(userId, monthStart, monthEnd),
      this.getMealsForPeriod(userId, prevMonthStart, prevMonthEnd),
      options.includeHealthMetrics
        ? this.getHealthMetricsForPeriod(userId, monthStart, monthEnd)
        : Promise.resolve([]),
      options.includeActivities
        ? this.getActivitiesForPeriod(userId, monthStart, monthEnd)
        : Promise.resolve([]),
    ]);

    if (!user) {
      throw new Error('User not found');
    }

    // Calculate daily breakdowns for the entire month
    const dailyBreakdowns = this.calculateDailyBreakdowns(meals, monthStart, monthEnd, user);

    // Calculate weekly breakdowns
    const weeklyBreakdowns = this.calculateWeeklyBreakdowns(dailyBreakdowns, monthStart, monthEnd);

    // Calculate totals and averages
    const totals = this.calculateTotals(meals);
    const totalDaysTracked = dailyBreakdowns.filter((d) => d.mealCount > 0).length;
    const averages = this.calculateAverages(totals, totalDaysTracked || 1);

    // Calculate goal progress
    const goalProgress = this.calculateGoalProgress(averages, user);

    // Calculate days with goals met
    const daysGoalsMet = dailyBreakdowns.filter((d) => d.goalCompletion >= 90).length;
    const goalCompletionRate = totalDaysTracked > 0 ? (daysGoalsMet / totalDaysTracked) * 100 : 0;

    // Calculate trends compared to previous month
    const previousTotals = this.calculateTotals(previousMonthMeals);
    const previousDaysWithData = this.countDaysWithData(previousMonthMeals);
    const previousAverages = this.calculateAverages(previousTotals, previousDaysWithData || 1);
    const trends = this.calculateTrends(averages, previousAverages);

    // Get best and worst days
    const sortedDays = [...dailyBreakdowns]
      .filter((d) => d.mealCount > 0)
      .sort((a, b) => b.goalCompletion - a.goalCompletion);
    const bestDays = sortedDays.slice(0, 5);
    const worstDays = sortedDays.slice(-5).reverse();

    // Get top foods
    const topFoods = await this.getTopFoods(
      userId,
      monthStart,
      monthEnd,
      options.topFoodsLimit || 10
    );

    // Health metrics summary
    const healthMetricsSummary = options.includeHealthMetrics
      ? await this.getHealthMetricsSummary(userId, monthStart, monthEnd)
      : [];

    // Activity summary
    const activitySummary = options.includeActivities
      ? this.calculateActivitySummary(activities)
      : null;

    // Generate insights
    const insights = options.includeInsights
      ? this.generateMonthlyInsights(weeklyBreakdowns, goalProgress, trends, bestDays, worstDays)
      : [];

    // Get achievements
    const achievements = options.includeAchievements
      ? await this.getAchievements(userId, monthStart, monthEnd)
      : [];

    // Calculate streak
    const streak = await this.calculateMonthlyStreak(userId, monthStart, monthEnd);

    // Calculate weekly trends (averages per week)
    const weeklyTrends = this.calculateWeeklyTrends(weeklyBreakdowns);

    // Meal type breakdown
    const mealTypeBreakdown = this.calculateMealTypeBreakdown(meals);

    // Macro distribution
    const macroDistribution = this.calculateMacroDistribution(totals);

    // Year-over-year comparison
    const yearOverYear = await this.calculateYearOverYear(userId, monthStart, monthEnd, averages);

    return {
      userId,
      generatedAt: new Date().toISOString(),
      month: monthString,
      periodStart: monthStart.toISOString(),
      periodEnd: monthEnd.toISOString(),
      weeklyBreakdowns,
      dailyBreakdowns,
      totals,
      averages,
      goalProgress,
      daysGoalsMet,
      totalDaysTracked,
      goalCompletionRate: Math.round(goalCompletionRate * 10) / 10,
      trends,
      bestDays,
      worstDays,
      topFoods,
      healthMetricsSummary,
      activitySummary,
      insights,
      achievements,
      streak,
      weeklyTrends,
      mealTypeBreakdown,
      macroDistribution,
      yearOverYear,
    };
  }

  // ==========================================================================
  // PRIVATE HELPER METHODS - DATA FETCHING
  // ==========================================================================

  private async getMealsForPeriod(userId: string, startDate: Date, endDate: Date) {
    return prisma.meal.findMany({
      where: {
        userId,
        consumedAt: {
          gte: startDate,
          lte: endDate,
        },
      },
      orderBy: {
        consumedAt: 'asc',
      },
    });
  }

  private async getHealthMetricsForPeriod(userId: string, startDate: Date, endDate: Date) {
    return prisma.healthMetric.findMany({
      where: {
        userId,
        recordedAt: {
          gte: startDate,
          lte: endDate,
        },
      },
      orderBy: {
        recordedAt: 'asc',
      },
    });
  }

  private async getActivitiesForPeriod(userId: string, startDate: Date, endDate: Date) {
    return prisma.activity.findMany({
      where: {
        userId,
        startedAt: {
          gte: startDate,
          lte: endDate,
        },
      },
      orderBy: {
        startedAt: 'asc',
      },
    });
  }

  private async getTopFoods(
    userId: string,
    startDate: Date,
    endDate: Date,
    limit: number
  ): Promise<TopFood[]> {
    // Group meals by name to get top foods
    const meals = await prisma.meal.findMany({
      where: {
        userId,
        consumedAt: {
          gte: startDate,
          lte: endDate,
        },
      },
      select: {
        name: true,
        calories: true,
        protein: true,
      },
    });

    // Aggregate by food name
    const foodMap = new Map<
      string,
      { count: number; totalCalories: number; totalProtein: number }
    >();

    meals.forEach((meal) => {
      const existing = foodMap.get(meal.name) || {
        count: 0,
        totalCalories: 0,
        totalProtein: 0,
      };
      foodMap.set(meal.name, {
        count: existing.count + 1,
        totalCalories: existing.totalCalories + meal.calories,
        totalProtein: existing.totalProtein + meal.protein,
      });
    });

    // Convert to array and sort by count
    const topFoods: TopFood[] = Array.from(foodMap.entries())
      .map(([name, data]) => ({
        name,
        count: data.count,
        totalCalories: Math.round(data.totalCalories),
        totalProtein: Math.round(data.totalProtein),
        avgCaloriesPerServing: Math.round(data.totalCalories / data.count),
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, limit);

    return topFoods;
  }

  // ==========================================================================
  // PRIVATE HELPER METHODS - CALCULATIONS
  // ==========================================================================

  private getWeekBoundaries(date: Date): { weekStart: Date; weekEnd: Date } {
    const dayOfWeek = date.getDay();
    const weekStart = new Date(date);
    // Set to Monday (ISO 8601 week start)
    // Sunday (0) needs to go back 6 days, Monday (1) stays, Tuesday (2) goes back 1, etc.
    const daysToSubtract = dayOfWeek === 0 ? 6 : dayOfWeek - 1;
    weekStart.setDate(date.getDate() - daysToSubtract);
    weekStart.setHours(0, 0, 0, 0);

    const weekEnd = new Date(weekStart);
    weekEnd.setDate(weekStart.getDate() + 6);
    weekEnd.setHours(23, 59, 59, 999);

    return { weekStart, weekEnd };
  }

  private getMonthBoundaries(month?: string): {
    monthStart: Date;
    monthEnd: Date;
    monthString: string;
  } {
    let year: number;
    let monthIndex: number;

    if (month) {
      const [y, m] = month.split('-').map(Number);
      year = y;
      monthIndex = m - 1;
    } else {
      const now = new Date();
      year = now.getFullYear();
      monthIndex = now.getMonth();
    }

    const monthStart = new Date(year, monthIndex, 1, 0, 0, 0, 0);
    const monthEnd = new Date(year, monthIndex + 1, 0, 23, 59, 59, 999);
    const monthString = `${year}-${String(monthIndex + 1).padStart(2, '0')}`;

    return { monthStart, monthEnd, monthString };
  }

  private subtractDays(date: Date, days: number): Date {
    const result = new Date(date);
    result.setDate(result.getDate() - days);
    return result;
  }

  private calculateDailyBreakdowns(
    meals: Array<{
      consumedAt: Date;
      calories: number;
      protein: number;
      carbs: number;
      fat: number;
      fiber: number | null;
      sugar: number | null;
    }>,
    startDate: Date,
    endDate: Date,
    userGoals: {
      goalCalories: number;
      goalProtein: number;
      goalCarbs: number;
      goalFat: number;
    }
  ): DailyBreakdown[] {
    const breakdowns: DailyBreakdown[] = [];

    // Group meals by date
    const mealsByDate = new Map<string, typeof meals>();
    meals.forEach((meal) => {
      const dateKey = meal.consumedAt.toISOString().split('T')[0];
      const existing = mealsByDate.get(dateKey) || [];
      existing.push(meal);
      mealsByDate.set(dateKey, existing);
    });

    // Create breakdown for each day in range
    const currentDate = new Date(startDate);
    while (currentDate <= endDate) {
      const dateKey = currentDate.toISOString().split('T')[0];
      const dayMeals = mealsByDate.get(dateKey) || [];

      const totals = dayMeals.reduce(
        (acc, meal) => ({
          calories: acc.calories + meal.calories,
          protein: acc.protein + meal.protein,
          carbs: acc.carbs + meal.carbs,
          fat: acc.fat + meal.fat,
          fiber: acc.fiber + (meal.fiber || 0),
          sugar: acc.sugar + (meal.sugar || 0),
        }),
        { calories: 0, protein: 0, carbs: 0, fat: 0, fiber: 0, sugar: 0 }
      );

      // Calculate goal completion as average of all macros
      const caloriePercent =
        userGoals.goalCalories > 0
          ? Math.min((totals.calories / userGoals.goalCalories) * 100, 100)
          : 0;
      const proteinPercent =
        userGoals.goalProtein > 0
          ? Math.min((totals.protein / userGoals.goalProtein) * 100, 100)
          : 0;
      const carbsPercent =
        userGoals.goalCarbs > 0 ? Math.min((totals.carbs / userGoals.goalCarbs) * 100, 100) : 0;
      const fatPercent =
        userGoals.goalFat > 0 ? Math.min((totals.fat / userGoals.goalFat) * 100, 100) : 0;

      const goalCompletion = (caloriePercent + proteinPercent + carbsPercent + fatPercent) / 4;

      breakdowns.push({
        date: dateKey,
        calories: Math.round(totals.calories),
        protein: Math.round(totals.protein),
        carbs: Math.round(totals.carbs),
        fat: Math.round(totals.fat),
        fiber: Math.round(totals.fiber),
        sugar: Math.round(totals.sugar),
        goalCompletion: Math.round(goalCompletion * 10) / 10,
        mealCount: dayMeals.length,
      });

      currentDate.setDate(currentDate.getDate() + 1);
    }

    return breakdowns;
  }

  private calculateWeeklyBreakdowns(
    dailyBreakdowns: DailyBreakdown[],
    monthStart: Date,
    monthEnd: Date
  ): WeeklyBreakdown[] {
    const weeks: WeeklyBreakdown[] = [];
    let weekNumber = 1;
    let currentWeekStart = new Date(monthStart);

    while (currentWeekStart <= monthEnd) {
      const currentWeekEnd = new Date(currentWeekStart);
      currentWeekEnd.setDate(currentWeekStart.getDate() + 6);

      // Cap week end at month end
      const effectiveWeekEnd = currentWeekEnd > monthEnd ? monthEnd : currentWeekEnd;

      // Filter daily breakdowns for this week
      const weekDays = dailyBreakdowns.filter((d) => {
        const date = new Date(d.date);
        return date >= currentWeekStart && date <= effectiveWeekEnd;
      });

      const daysTracked = weekDays.filter((d) => d.mealCount > 0).length;

      if (weekDays.length > 0) {
        const totals = weekDays.reduce(
          (acc, day) => ({
            calories: acc.calories + day.calories,
            protein: acc.protein + day.protein,
            carbs: acc.carbs + day.carbs,
            fat: acc.fat + day.fat,
            fiber: acc.fiber + day.fiber,
            sugar: acc.sugar + day.sugar,
          }),
          { calories: 0, protein: 0, carbs: 0, fat: 0, fiber: 0, sugar: 0 }
        );

        const averages = this.calculateAverages(totals, daysTracked || 1);
        const daysGoalsMet = weekDays.filter((d) => d.goalCompletion >= 90).length;

        weeks.push({
          weekNumber,
          weekStart: currentWeekStart.toISOString(),
          weekEnd: effectiveWeekEnd.toISOString(),
          totals,
          averages,
          daysTracked,
          daysGoalsMet,
          goalCompletionRate: daysTracked > 0 ? (daysGoalsMet / daysTracked) * 100 : 0,
        });
      }

      currentWeekStart.setDate(currentWeekStart.getDate() + 7);
      weekNumber++;
    }

    return weeks;
  }

  private calculateTotals(
    meals: Array<{
      calories: number;
      protein: number;
      carbs: number;
      fat: number;
      fiber: number | null;
      sugar: number | null;
    }>
  ): NutritionTotals {
    const initial: NutritionTotals = {
      calories: 0,
      protein: 0,
      carbs: 0,
      fat: 0,
      fiber: 0,
      sugar: 0,
    };
    return meals.reduce<NutritionTotals>(
      (acc, meal) => ({
        calories: acc.calories + meal.calories,
        protein: acc.protein + meal.protein,
        carbs: acc.carbs + meal.carbs,
        fat: acc.fat + meal.fat,
        fiber: acc.fiber + (meal.fiber || 0),
        sugar: acc.sugar + (meal.sugar || 0),
      }),
      initial
    );
  }

  private calculateAverages(totals: NutritionTotals, days: number): NutritionTotals {
    return {
      calories: Math.round((totals.calories / days) * 10) / 10,
      protein: Math.round((totals.protein / days) * 10) / 10,
      carbs: Math.round((totals.carbs / days) * 10) / 10,
      fat: Math.round((totals.fat / days) * 10) / 10,
      fiber: Math.round((totals.fiber / days) * 10) / 10,
      sugar: Math.round((totals.sugar / days) * 10) / 10,
    };
  }

  private countDaysWithData(meals: Array<{ consumedAt: Date }>): number {
    const uniqueDates = new Set(meals.map((m) => m.consumedAt.toISOString().split('T')[0]));
    return uniqueDates.size;
  }

  private calculateGoalProgress(
    averages: NutritionTotals,
    userGoals: {
      goalCalories: number;
      goalProtein: number;
      goalCarbs: number;
      goalFat: number;
    }
  ): GoalProgress {
    return {
      caloriesGoal: userGoals.goalCalories,
      caloriesActual: averages.calories,
      caloriesPercent:
        userGoals.goalCalories > 0
          ? Math.round((averages.calories / userGoals.goalCalories) * 1000) / 10
          : 0,
      proteinGoal: userGoals.goalProtein,
      proteinActual: averages.protein,
      proteinPercent:
        userGoals.goalProtein > 0
          ? Math.round((averages.protein / userGoals.goalProtein) * 1000) / 10
          : 0,
      carbsGoal: userGoals.goalCarbs,
      carbsActual: averages.carbs,
      carbsPercent:
        userGoals.goalCarbs > 0
          ? Math.round((averages.carbs / userGoals.goalCarbs) * 1000) / 10
          : 0,
      fatGoal: userGoals.goalFat,
      fatActual: averages.fat,
      fatPercent:
        userGoals.goalFat > 0 ? Math.round((averages.fat / userGoals.goalFat) * 1000) / 10 : 0,
    };
  }

  private calculateTrends(current: NutritionTotals, previous: NutritionTotals): TrendComparison {
    return {
      calories: this.calculateMetricTrend(current.calories, previous.calories),
      protein: this.calculateMetricTrend(current.protein, previous.protein),
      carbs: this.calculateMetricTrend(current.carbs, previous.carbs),
      fat: this.calculateMetricTrend(current.fat, previous.fat),
      fiber: this.calculateMetricTrend(current.fiber, previous.fiber),
      sugar: this.calculateMetricTrend(current.sugar, previous.sugar),
    };
  }

  private calculateMetricTrend(current: number, previous: number): MetricTrend {
    const percentChange = previous > 0 ? ((current - previous) / previous) * 100 : 0;

    let trend: 'up' | 'down' | 'stable';
    if (percentChange > TREND_THRESHOLD) {
      trend = 'up';
    } else if (percentChange < -TREND_THRESHOLD) {
      trend = 'down';
    } else {
      trend = 'stable';
    }

    return {
      current: Math.round(current * 10) / 10,
      previous: Math.round(previous * 10) / 10,
      percentChange: Math.round(percentChange * 10) / 10,
      trend,
    };
  }

  private async getHealthMetricsSummary(
    userId: string,
    startDate: Date,
    endDate: Date
  ): Promise<HealthMetricSummary[]> {
    // Get all health metrics for the period
    const metrics = await prisma.healthMetric.findMany({
      where: {
        userId,
        recordedAt: {
          gte: startDate,
          lte: endDate,
        },
      },
      orderBy: {
        recordedAt: 'asc',
      },
    });

    // Group by metric type
    const metricsByType = new Map<string, typeof metrics>();
    metrics.forEach((m) => {
      const existing = metricsByType.get(m.metricType) || [];
      existing.push(m);
      metricsByType.set(m.metricType, existing);
    });

    // Calculate summary for each type
    const summaries: HealthMetricSummary[] = [];
    metricsByType.forEach((typeMetrics, metricType) => {
      if (typeMetrics.length === 0) return;

      const values = typeMetrics.map((m) => m.value);
      const sum = values.reduce((a, b) => a + b, 0);
      const average = sum / values.length;
      const min = Math.min(...values);
      const max = Math.max(...values);

      // Calculate trend
      const midpoint = Math.floor(values.length / 2);
      const firstHalf = values.slice(0, midpoint);
      const secondHalf = values.slice(midpoint);
      const firstAvg =
        firstHalf.length > 0 ? firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length : 0;
      const secondAvg =
        secondHalf.length > 0 ? secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length : 0;
      const percentChange = firstAvg > 0 ? ((secondAvg - firstAvg) / firstAvg) * 100 : 0;

      let trend: 'up' | 'down' | 'stable';
      if (percentChange > TREND_THRESHOLD) {
        trend = 'up';
      } else if (percentChange < -TREND_THRESHOLD) {
        trend = 'down';
      } else {
        trend = 'stable';
      }

      summaries.push({
        metricType: metricType as HealthMetricType,
        average: Math.round(average * 100) / 100,
        min,
        max,
        trend,
        percentChange: Math.round(percentChange * 10) / 10,
        unit: typeMetrics[0].unit,
        dataPoints: typeMetrics.length,
      });
    });

    return summaries;
  }

  private calculateActivitySummary(
    activities: Array<{
      duration: number;
      caloriesBurned: number | null;
      distance: number | null;
      steps: number | null;
      activityType: string;
      intensity: string;
    }>
  ): ActivitySummary | null {
    if (activities.length === 0) {
      return null;
    }

    const totalDuration = activities.reduce((sum, a) => sum + a.duration, 0);
    const totalCaloriesBurned = activities.reduce((sum, a) => sum + (a.caloriesBurned || 0), 0);
    const totalDistance = activities.reduce((sum, a) => sum + (a.distance || 0), 0);
    const totalSteps = activities.reduce((sum, a) => sum + (a.steps || 0), 0);

    // Find most frequent activity
    const activityCounts = new Map<string, number>();
    activities.forEach((a) => {
      activityCounts.set(a.activityType, (activityCounts.get(a.activityType) || 0) + 1);
    });
    let mostFrequentActivity: string | null = null;
    let maxCount = 0;
    activityCounts.forEach((count, type) => {
      if (count > maxCount) {
        maxCount = count;
        mostFrequentActivity = type;
      }
    });

    // Intensity breakdown
    const intensityBreakdown = {
      low: activities.filter((a) => a.intensity === 'LOW').length,
      moderate: activities.filter((a) => a.intensity === 'MODERATE').length,
      high: activities.filter((a) => a.intensity === 'HIGH').length,
      maximum: activities.filter((a) => a.intensity === 'MAXIMUM').length,
    };

    return {
      totalDuration,
      totalCaloriesBurned,
      totalDistance,
      totalSteps,
      activityCount: activities.length,
      averageDurationPerActivity: Math.round(totalDuration / activities.length),
      mostFrequentActivity,
      intensityBreakdown,
    };
  }

  private async calculateStreak(userId: string): Promise<{
    current: number;
    longest: number;
    isOnStreak: boolean;
  }> {
    // Get meals from the past 90 days to calculate streak
    const ninetyDaysAgo = getDaysAgo(90);
    const meals = await prisma.meal.findMany({
      where: {
        userId,
        consumedAt: {
          gte: ninetyDaysAgo,
        },
      },
      select: {
        consumedAt: true,
      },
      orderBy: {
        consumedAt: 'desc',
      },
    });

    // Get unique dates with meals
    const datesWithMeals = new Set(meals.map((m) => m.consumedAt.toISOString().split('T')[0]));

    // Calculate current streak
    let currentStreak = 0;
    let longestStreak = 0;
    let tempStreak = 0;
    const today = new Date();
    const checkDate = new Date(today);

    for (let i = 0; i < 90; i++) {
      const dateKey = checkDate.toISOString().split('T')[0];
      if (datesWithMeals.has(dateKey)) {
        tempStreak++;
        if (i === currentStreak) {
          currentStreak = tempStreak;
        }
      } else {
        longestStreak = Math.max(longestStreak, tempStreak);
        tempStreak = 0;
      }
      checkDate.setDate(checkDate.getDate() - 1);
    }
    longestStreak = Math.max(longestStreak, tempStreak);

    const todayKey = today.toISOString().split('T')[0];
    const isOnStreak = datesWithMeals.has(todayKey) && currentStreak > 0;

    return {
      current: currentStreak,
      longest: longestStreak,
      isOnStreak,
    };
  }

  private async calculateMonthlyStreak(
    userId: string,
    monthStart: Date,
    monthEnd: Date
  ): Promise<{
    current: number;
    longest: number;
    longestThisMonth: number;
  }> {
    const baseStreak = await this.calculateStreak(userId);

    // Calculate longest streak within the month
    const meals = await prisma.meal.findMany({
      where: {
        userId,
        consumedAt: {
          gte: monthStart,
          lte: monthEnd,
        },
      },
      select: {
        consumedAt: true,
      },
    });

    const datesWithMeals = new Set(meals.map((m) => m.consumedAt.toISOString().split('T')[0]));

    let longestThisMonth = 0;
    let tempStreak = 0;
    const checkDate = new Date(monthStart);

    while (checkDate <= monthEnd) {
      const dateKey = checkDate.toISOString().split('T')[0];
      if (datesWithMeals.has(dateKey)) {
        tempStreak++;
        longestThisMonth = Math.max(longestThisMonth, tempStreak);
      } else {
        tempStreak = 0;
      }
      checkDate.setDate(checkDate.getDate() + 1);
    }

    return {
      current: baseStreak.current,
      longest: baseStreak.longest,
      longestThisMonth,
    };
  }

  private calculateMealTypeBreakdown(meals: Array<{ mealType: string; calories: number }>): {
    breakfast: { count: number; avgCalories: number };
    lunch: { count: number; avgCalories: number };
    dinner: { count: number; avgCalories: number };
    snack: { count: number; avgCalories: number };
  } {
    const breakdown = {
      breakfast: { count: 0, totalCalories: 0 },
      lunch: { count: 0, totalCalories: 0 },
      dinner: { count: 0, totalCalories: 0 },
      snack: { count: 0, totalCalories: 0 },
    };

    meals.forEach((meal) => {
      const type = meal.mealType.toLowerCase() as keyof typeof breakdown;
      if (breakdown[type]) {
        breakdown[type].count++;
        breakdown[type].totalCalories += meal.calories;
      }
    });

    return {
      breakfast: {
        count: breakdown.breakfast.count,
        avgCalories:
          breakdown.breakfast.count > 0
            ? Math.round(breakdown.breakfast.totalCalories / breakdown.breakfast.count)
            : 0,
      },
      lunch: {
        count: breakdown.lunch.count,
        avgCalories:
          breakdown.lunch.count > 0
            ? Math.round(breakdown.lunch.totalCalories / breakdown.lunch.count)
            : 0,
      },
      dinner: {
        count: breakdown.dinner.count,
        avgCalories:
          breakdown.dinner.count > 0
            ? Math.round(breakdown.dinner.totalCalories / breakdown.dinner.count)
            : 0,
      },
      snack: {
        count: breakdown.snack.count,
        avgCalories:
          breakdown.snack.count > 0
            ? Math.round(breakdown.snack.totalCalories / breakdown.snack.count)
            : 0,
      },
    };
  }

  private calculateMacroDistribution(totals: NutritionTotals): {
    proteinPercent: number;
    carbsPercent: number;
    fatPercent: number;
  } {
    // Calculate calories from each macro
    // Protein: 4 cal/g, Carbs: 4 cal/g, Fat: 9 cal/g
    const proteinCals = totals.protein * 4;
    const carbsCals = totals.carbs * 4;
    const fatCals = totals.fat * 9;
    const totalMacroCals = proteinCals + carbsCals + fatCals;

    if (totalMacroCals === 0) {
      return { proteinPercent: 0, carbsPercent: 0, fatPercent: 0 };
    }

    return {
      proteinPercent: Math.round((proteinCals / totalMacroCals) * 1000) / 10,
      carbsPercent: Math.round((carbsCals / totalMacroCals) * 1000) / 10,
      fatPercent: Math.round((fatCals / totalMacroCals) * 1000) / 10,
    };
  }

  private calculateWeeklyTrends(weeklyBreakdowns: WeeklyBreakdown[]): {
    week1Avg: NutritionTotals | null;
    week2Avg: NutritionTotals | null;
    week3Avg: NutritionTotals | null;
    week4Avg: NutritionTotals | null;
    week5Avg: NutritionTotals | null;
  } {
    return {
      week1Avg: weeklyBreakdowns[0]?.averages || null,
      week2Avg: weeklyBreakdowns[1]?.averages || null,
      week3Avg: weeklyBreakdowns[2]?.averages || null,
      week4Avg: weeklyBreakdowns[3]?.averages || null,
      week5Avg: weeklyBreakdowns[4]?.averages || null,
    };
  }

  private async calculateYearOverYear(
    userId: string,
    monthStart: Date,
    monthEnd: Date,
    currentAverages: NutritionTotals
  ): Promise<{
    available: boolean;
    previousYearAvg: NutritionTotals | null;
    percentChange: number | null;
  }> {
    // Get same month from previous year
    const previousYearStart = new Date(monthStart);
    previousYearStart.setFullYear(previousYearStart.getFullYear() - 1);
    const previousYearEnd = new Date(monthEnd);
    previousYearEnd.setFullYear(previousYearEnd.getFullYear() - 1);

    const previousYearMeals = await prisma.meal.findMany({
      where: {
        userId,
        consumedAt: {
          gte: previousYearStart,
          lte: previousYearEnd,
        },
      },
    });

    if (previousYearMeals.length === 0) {
      return {
        available: false,
        previousYearAvg: null,
        percentChange: null,
      };
    }

    const previousTotals = this.calculateTotals(previousYearMeals);
    const previousDaysWithData = this.countDaysWithData(previousYearMeals);
    const previousYearAvg = this.calculateAverages(previousTotals, previousDaysWithData || 1);

    // Calculate overall percent change based on calories
    const percentChange =
      previousYearAvg.calories > 0
        ? ((currentAverages.calories - previousYearAvg.calories) / previousYearAvg.calories) * 100
        : 0;

    return {
      available: true,
      previousYearAvg,
      percentChange: Math.round(percentChange * 10) / 10,
    };
  }

  // ==========================================================================
  // PRIVATE HELPER METHODS - INSIGHTS
  // ==========================================================================

  private generateWeeklyInsights(
    dailyBreakdowns: DailyBreakdown[],
    goalProgress: GoalProgress,
    trends: TrendComparison,
    _healthMetricsSummary: HealthMetricSummary[],
    activitySummary: ActivitySummary | null
  ): ReportInsight[] {
    const insights: ReportInsight[] = [];

    // Goal achievement insights
    const daysGoalsMet = dailyBreakdowns.filter((d) => d.goalCompletion >= 90).length;
    if (daysGoalsMet >= 5) {
      insights.push({
        id: 'high-goal-achievement',
        title: 'Excellent Goal Achievement',
        description: `You met your nutrition goals on ${daysGoalsMet} out of 7 days this week. Keep up the great work!`,
        category: 'achievement',
        priority: 'high',
      });
    } else if (daysGoalsMet >= 3) {
      insights.push({
        id: 'moderate-goal-achievement',
        title: 'Good Progress on Goals',
        description: `You met your nutrition goals on ${daysGoalsMet} days. Try to be more consistent on weekends.`,
        category: 'nutrition',
        priority: 'medium',
      });
    }

    // Protein insights
    if (goalProgress.proteinPercent < 80) {
      insights.push({
        id: 'low-protein',
        title: 'Protein Intake Below Target',
        description: `You're averaging ${goalProgress.proteinActual}g of protein (${goalProgress.proteinPercent}% of goal). Consider adding more protein-rich foods.`,
        category: 'recommendation',
        priority: 'medium',
      });
    } else if (goalProgress.proteinPercent >= 100) {
      insights.push({
        id: 'protein-goal-met',
        title: 'Protein Goal Achieved',
        description: `Great job hitting your protein target! You averaged ${goalProgress.proteinActual}g daily.`,
        category: 'achievement',
        priority: 'low',
      });
    }

    // Calorie trend insights
    if (trends.calories.trend === 'up' && trends.calories.percentChange > 10) {
      insights.push({
        id: 'calorie-increase',
        title: 'Calorie Intake Increased',
        description: `Your calorie intake is up ${trends.calories.percentChange}% compared to last week. This might be intentional, but worth monitoring.`,
        category: 'nutrition',
        priority: 'medium',
      });
    } else if (trends.calories.trend === 'down' && trends.calories.percentChange < -10) {
      insights.push({
        id: 'calorie-decrease',
        title: 'Calorie Intake Decreased',
        description: `Your calorie intake is down ${Math.abs(trends.calories.percentChange)}% from last week.`,
        category: 'nutrition',
        priority: 'medium',
      });
    }

    // Activity insights
    if (activitySummary && activitySummary.activityCount >= 5) {
      insights.push({
        id: 'active-week',
        title: 'Active Week',
        description: `You completed ${activitySummary.activityCount} workouts and burned ${activitySummary.totalCaloriesBurned} calories through exercise.`,
        category: 'activity',
        priority: 'low',
      });
    }

    return insights;
  }

  private generateMonthlyInsights(
    weeklyBreakdowns: WeeklyBreakdown[],
    _goalProgress: GoalProgress,
    trends: TrendComparison,
    bestDays: DailyBreakdown[],
    _worstDays: DailyBreakdown[]
  ): ReportInsight[] {
    const insights: ReportInsight[] = [];

    // Monthly consistency insight
    const avgGoalCompletion =
      weeklyBreakdowns.length > 0
        ? weeklyBreakdowns.reduce((sum, w) => sum + w.goalCompletionRate, 0) /
          weeklyBreakdowns.length
        : 0;

    if (avgGoalCompletion >= 80) {
      insights.push({
        id: 'monthly-consistency',
        title: 'Outstanding Monthly Consistency',
        description: `You maintained ${Math.round(avgGoalCompletion)}% goal completion rate throughout the month. Exceptional discipline!`,
        category: 'achievement',
        priority: 'high',
      });
    }

    // Weekly improvement trend
    if (weeklyBreakdowns.length >= 3) {
      const firstWeekRate = weeklyBreakdowns[0]?.goalCompletionRate || 0;
      const lastWeekRate = weeklyBreakdowns[weeklyBreakdowns.length - 1]?.goalCompletionRate || 0;
      if (lastWeekRate > firstWeekRate + 10) {
        insights.push({
          id: 'improving-trend',
          title: 'Improving Throughout the Month',
          description: `Your goal completion improved from ${Math.round(firstWeekRate)}% in week 1 to ${Math.round(lastWeekRate)}% by the end of the month.`,
          category: 'achievement',
          priority: 'medium',
        });
      }
    }

    // Best day highlight
    if (bestDays.length > 0 && bestDays[0].goalCompletion >= 95) {
      insights.push({
        id: 'best-day',
        title: 'Your Best Day',
        description: `${new Date(bestDays[0].date).toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })} was your best day with ${bestDays[0].goalCompletion}% goal completion.`,
        category: 'achievement',
        priority: 'low',
      });
    }

    // Month-over-month trend
    if (trends.calories.trend !== 'stable') {
      const direction = trends.calories.trend === 'up' ? 'increased' : 'decreased';
      insights.push({
        id: 'monthly-trend',
        title: 'Monthly Calorie Trend',
        description: `Your average daily calories ${direction} by ${Math.abs(trends.calories.percentChange)}% compared to last month.`,
        category: 'nutrition',
        priority: 'medium',
      });
    }

    return insights;
  }

  private async getAchievements(
    userId: string,
    _startDate: Date,
    _endDate: Date
  ): Promise<ReportAchievement[]> {
    // This would typically query an achievements table
    // For now, we'll generate achievements based on data
    const achievements: ReportAchievement[] = [];

    // Check for streak achievements
    const streak = await this.calculateStreak(userId);
    if (streak.current >= 7) {
      achievements.push({
        id: 'week-streak',
        title: '7-Day Streak',
        description: 'Logged meals for 7 consecutive days',
        earnedAt: new Date().toISOString(),
        type: 'streak',
      });
    }
    if (streak.current >= 30) {
      achievements.push({
        id: 'month-streak',
        title: '30-Day Streak',
        description: 'Logged meals for 30 consecutive days',
        earnedAt: new Date().toISOString(),
        type: 'streak',
      });
    }

    return achievements;
  }
}

export const reportService = new ReportService();
