import prisma from '../config/database';
import { USER_GOALS_SELECT_FIELDS, WEEK_IN_DAYS } from '../config/constants';
import { getDayBoundaries, getDaysAgo, getStartOfDay, getEndOfDay } from '../utils/dateHelpers';

// ============================================================================
// TYPES
// ============================================================================

export interface DailyGoalProgress {
  date: string; // ISO date string
  calories: MacroProgress;
  protein: MacroProgress;
  carbs: MacroProgress;
  fat: MacroProgress;
  goalsMetCount: number; // How many goals were met (0-4)
  allGoalsMet: boolean;
}

export interface MacroProgress {
  consumed: number;
  goal: number;
  percentage: number; // (consumed / goal) * 100, capped at 100 for display
  rawPercentage: number; // uncapped percentage
  isOnTrack: boolean; // Within 90-110% of goal
  isMet: boolean; // >= 100% of goal
}

export interface GoalStreak {
  currentStreak: number; // Consecutive days meeting all goals
  longestStreak: number;
  lastStreakDate: string | null; // Last date a goal was met
}

export interface WeeklyGoalSummary {
  weekStart: string;
  weekEnd: string;
  daysTracked: number;
  daysGoalsMet: number;
  averageProgress: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
  trend: 'improving' | 'declining' | 'stable'; // Compared to previous week
}

export interface MonthlyGoalSummary {
  month: string; // YYYY-MM
  daysTracked: number;
  daysGoalsMet: number;
  successRate: number; // Percentage of days goals were met
  averageProgress: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
}

export interface GoalProgressDashboard {
  today: DailyGoalProgress;
  streak: GoalStreak;
  weeklyAverage: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
  weeklyTrend: 'improving' | 'declining' | 'stable';
}

export interface HistoricalProgress {
  daily: DailyGoalProgress[];
  weeklySummaries: WeeklyGoalSummary[];
  monthlySummaries: MonthlyGoalSummary[];
  streak: GoalStreak;
}

// ============================================================================
// SERVICE
// ============================================================================

export class GoalProgressService {
  /**
   * Get goal progress for a specific day
   */
  async getDailyProgress(userId: string, date: Date = new Date()): Promise<DailyGoalProgress> {
    const { startOfDay, endOfDay } = getDayBoundaries(date);

    // Get meals for the day
    const meals = await prisma.meal.findMany({
      where: {
        userId,
        consumedAt: {
          gte: startOfDay,
          lte: endOfDay,
        },
      },
      select: {
        calories: true,
        protein: true,
        carbs: true,
        fat: true,
      },
    });

    // Get user goals
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: USER_GOALS_SELECT_FIELDS,
    });

    if (!user) {
      throw new Error('User not found');
    }

    // Calculate totals
    const totals = meals.reduce(
      (acc, meal) => ({
        calories: acc.calories + meal.calories,
        protein: acc.protein + meal.protein,
        carbs: acc.carbs + meal.carbs,
        fat: acc.fat + meal.fat,
      }),
      { calories: 0, protein: 0, carbs: 0, fat: 0 }
    );

    // Calculate progress for each macro
    const caloriesProgress = this.calculateMacroProgress(totals.calories, user.goalCalories);
    const proteinProgress = this.calculateMacroProgress(totals.protein, user.goalProtein);
    const carbsProgress = this.calculateMacroProgress(totals.carbs, user.goalCarbs);
    const fatProgress = this.calculateMacroProgress(totals.fat, user.goalFat);

    // Count goals met
    const goalsMetCount = [caloriesProgress, proteinProgress, carbsProgress, fatProgress].filter(
      (p) => p.isMet
    ).length;

    return {
      date: startOfDay.toISOString(),
      calories: caloriesProgress,
      protein: proteinProgress,
      carbs: carbsProgress,
      fat: fatProgress,
      goalsMetCount,
      allGoalsMet: goalsMetCount === 4,
    };
  }

  /**
   * Calculate progress for a single macro
   */
  private calculateMacroProgress(consumed: number, goal: number): MacroProgress {
    const rawPercentage = goal > 0 ? (consumed / goal) * 100 : 0;
    const percentage = Math.min(rawPercentage, 100);
    const isMet = rawPercentage >= 100;
    const isOnTrack = rawPercentage >= 90 && rawPercentage <= 110;

    return {
      consumed: Math.round(consumed * 10) / 10,
      goal,
      percentage: Math.round(percentage * 10) / 10,
      rawPercentage: Math.round(rawPercentage * 10) / 10,
      isOnTrack,
      isMet,
    };
  }

  /**
   * Get goal progress for a date range
   */
  async getProgressHistory(
    userId: string,
    startDate: Date,
    endDate: Date
  ): Promise<DailyGoalProgress[]> {
    const results: DailyGoalProgress[] = [];
    const currentDate = new Date(startDate);

    while (currentDate <= endDate) {
      const progress = await this.getDailyProgress(userId, new Date(currentDate));
      results.push(progress);
      currentDate.setDate(currentDate.getDate() + 1);
    }

    return results;
  }

  /**
   * Calculate streak information (consecutive days meeting all goals)
   */
  async getStreak(userId: string): Promise<GoalStreak> {
    // Get all meals grouped by day for the last 90 days
    const ninetyDaysAgo = getDaysAgo(90);
    const today = getEndOfDay();

    const meals = await prisma.meal.findMany({
      where: {
        userId,
        consumedAt: {
          gte: ninetyDaysAgo,
          lte: today,
        },
      },
      select: {
        calories: true,
        protein: true,
        carbs: true,
        fat: true,
        consumedAt: true,
      },
      orderBy: {
        consumedAt: 'desc',
      },
    });

    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: USER_GOALS_SELECT_FIELDS,
    });

    if (!user) {
      throw new Error('User not found');
    }

    // Group meals by day
    const mealsByDay = new Map<string, typeof meals>();
    meals.forEach((meal) => {
      const dateKey = getStartOfDay(meal.consumedAt).toISOString();
      const existing = mealsByDay.get(dateKey) || [];
      existing.push(meal);
      mealsByDay.set(dateKey, existing);
    });

    // Check each day to see if goals were met
    let currentStreak = 0;
    let longestStreak = 0;
    let tempStreak = 0;
    let lastStreakDate: string | null = null;
    let streakBroken = false;

    // Start from today and go backwards
    const checkDate = new Date();
    for (let i = 0; i < 90; i++) {
      const dateKey = getStartOfDay(checkDate).toISOString();
      const dayMeals = mealsByDay.get(dateKey) || [];

      // Calculate totals for the day
      const totals = dayMeals.reduce(
        (acc, meal) => ({
          calories: acc.calories + meal.calories,
          protein: acc.protein + meal.protein,
          carbs: acc.carbs + meal.carbs,
          fat: acc.fat + meal.fat,
        }),
        { calories: 0, protein: 0, carbs: 0, fat: 0 }
      );

      // Check if all goals met (within 90-110% for a reasonable streak definition)
      const goalsMet =
        totals.calories >= user.goalCalories * 0.9 &&
        totals.protein >= user.goalProtein * 0.9 &&
        totals.carbs >= user.goalCarbs * 0.9 &&
        totals.fat >= user.goalFat * 0.9;

      if (goalsMet) {
        tempStreak++;
        if (!streakBroken) {
          currentStreak = tempStreak;
        }
        if (lastStreakDate === null) {
          lastStreakDate = dateKey;
        }
      } else {
        longestStreak = Math.max(longestStreak, tempStreak);
        tempStreak = 0;
        if (!streakBroken) {
          streakBroken = true;
        }
      }

      checkDate.setDate(checkDate.getDate() - 1);
    }

    longestStreak = Math.max(longestStreak, tempStreak);

    return {
      currentStreak,
      longestStreak,
      lastStreakDate,
    };
  }

  /**
   * Get weekly summary with trend comparison
   */
  async getWeeklySummary(userId: string, weekOffset: number = 0): Promise<WeeklyGoalSummary> {
    const weekEnd = new Date();
    weekEnd.setDate(weekEnd.getDate() - weekOffset * WEEK_IN_DAYS);
    const weekStart = new Date(weekEnd);
    weekStart.setDate(weekStart.getDate() - (WEEK_IN_DAYS - 1));

    const dailyProgress = await this.getProgressHistory(
      userId,
      getStartOfDay(weekStart),
      getEndOfDay(weekEnd)
    );

    const daysTracked = dailyProgress.filter(
      (d) => d.calories.consumed > 0 || d.protein.consumed > 0
    ).length;

    const daysGoalsMet = dailyProgress.filter((d) => d.allGoalsMet).length;

    // Calculate averages
    const averageProgress = {
      calories:
        daysTracked > 0
          ? dailyProgress.reduce((sum, d) => sum + d.calories.percentage, 0) / daysTracked
          : 0,
      protein:
        daysTracked > 0
          ? dailyProgress.reduce((sum, d) => sum + d.protein.percentage, 0) / daysTracked
          : 0,
      carbs:
        daysTracked > 0
          ? dailyProgress.reduce((sum, d) => sum + d.carbs.percentage, 0) / daysTracked
          : 0,
      fat:
        daysTracked > 0
          ? dailyProgress.reduce((sum, d) => sum + d.fat.percentage, 0) / daysTracked
          : 0,
    };

    // Calculate trend by comparing to previous week
    let trend: 'improving' | 'declining' | 'stable' = 'stable';
    if (weekOffset === 0) {
      const previousWeek = await this.getWeeklySummary(userId, 1);
      const currentAvg =
        (averageProgress.calories +
          averageProgress.protein +
          averageProgress.carbs +
          averageProgress.fat) /
        4;
      const previousAvg =
        (previousWeek.averageProgress.calories +
          previousWeek.averageProgress.protein +
          previousWeek.averageProgress.carbs +
          previousWeek.averageProgress.fat) /
        4;

      if (currentAvg > previousAvg + 5) {
        trend = 'improving';
      } else if (currentAvg < previousAvg - 5) {
        trend = 'declining';
      }
    }

    return {
      weekStart: getStartOfDay(weekStart).toISOString(),
      weekEnd: getEndOfDay(weekEnd).toISOString(),
      daysTracked,
      daysGoalsMet,
      averageProgress: {
        calories: Math.round(averageProgress.calories * 10) / 10,
        protein: Math.round(averageProgress.protein * 10) / 10,
        carbs: Math.round(averageProgress.carbs * 10) / 10,
        fat: Math.round(averageProgress.fat * 10) / 10,
      },
      trend,
    };
  }

  /**
   * Get monthly summary
   */
  async getMonthlySummary(userId: string, monthOffset: number = 0): Promise<MonthlyGoalSummary> {
    const now = new Date();
    const targetMonth = new Date(now.getFullYear(), now.getMonth() - monthOffset, 1);
    const monthEnd = new Date(targetMonth.getFullYear(), targetMonth.getMonth() + 1, 0);

    const dailyProgress = await this.getProgressHistory(
      userId,
      getStartOfDay(targetMonth),
      getEndOfDay(monthEnd)
    );

    const daysTracked = dailyProgress.filter(
      (d) => d.calories.consumed > 0 || d.protein.consumed > 0
    ).length;

    const daysGoalsMet = dailyProgress.filter((d) => d.allGoalsMet).length;
    const successRate = daysTracked > 0 ? (daysGoalsMet / daysTracked) * 100 : 0;

    // Calculate averages
    const averageProgress = {
      calories:
        daysTracked > 0
          ? dailyProgress.reduce((sum, d) => sum + d.calories.percentage, 0) / daysTracked
          : 0,
      protein:
        daysTracked > 0
          ? dailyProgress.reduce((sum, d) => sum + d.protein.percentage, 0) / daysTracked
          : 0,
      carbs:
        daysTracked > 0
          ? dailyProgress.reduce((sum, d) => sum + d.carbs.percentage, 0) / daysTracked
          : 0,
      fat:
        daysTracked > 0
          ? dailyProgress.reduce((sum, d) => sum + d.fat.percentage, 0) / daysTracked
          : 0,
    };

    const monthString = `${targetMonth.getFullYear()}-${String(targetMonth.getMonth() + 1).padStart(2, '0')}`;

    return {
      month: monthString,
      daysTracked,
      daysGoalsMet,
      successRate: Math.round(successRate * 10) / 10,
      averageProgress: {
        calories: Math.round(averageProgress.calories * 10) / 10,
        protein: Math.round(averageProgress.protein * 10) / 10,
        carbs: Math.round(averageProgress.carbs * 10) / 10,
        fat: Math.round(averageProgress.fat * 10) / 10,
      },
    };
  }

  /**
   * Get dashboard data (today's progress, streak, weekly trends)
   */
  async getDashboardProgress(userId: string): Promise<GoalProgressDashboard> {
    const [today, streak, weeklySummary] = await Promise.all([
      this.getDailyProgress(userId),
      this.getStreak(userId),
      this.getWeeklySummary(userId),
    ]);

    return {
      today,
      streak,
      weeklyAverage: weeklySummary.averageProgress,
      weeklyTrend: weeklySummary.trend,
    };
  }

  /**
   * Get full historical progress with all summaries
   */
  async getHistoricalProgress(userId: string, days: number = 30): Promise<HistoricalProgress> {
    const startDate = getDaysAgo(days);
    const endDate = getEndOfDay();

    const [daily, streak] = await Promise.all([
      this.getProgressHistory(userId, startDate, endDate),
      this.getStreak(userId),
    ]);

    // Calculate weekly summaries for the past 4 weeks
    const weeklySummaries: WeeklyGoalSummary[] = [];
    for (let i = 0; i < 4; i++) {
      const summary = await this.getWeeklySummary(userId, i);
      weeklySummaries.push(summary);
    }

    // Calculate monthly summaries for the past 3 months
    const monthlySummaries: MonthlyGoalSummary[] = [];
    for (let i = 0; i < 3; i++) {
      const summary = await this.getMonthlySummary(userId, i);
      monthlySummaries.push(summary);
    }

    return {
      daily,
      weeklySummaries,
      monthlySummaries,
      streak,
    };
  }
}

export const goalProgressService = new GoalProgressService();
