/**
 * Report Types for Mobile App
 * Mirrors backend types from server/src/types/reports.ts
 */

// ============================================================================
// CORE BUILDING BLOCKS
// ============================================================================

/**
 * Nutrition totals for a given period
 */
export interface NutritionTotals {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  sugar: number;
}

/**
 * Daily breakdown of nutrition data
 */
export interface DailyBreakdown {
  date: string; // ISO date string (YYYY-MM-DD)
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  sugar: number;
  goalCompletion: number; // 0-100% - overall goal completion for the day
  mealCount: number;
}

/**
 * Progress toward individual nutrition goals
 */
export interface GoalProgress {
  caloriesGoal: number;
  caloriesActual: number;
  caloriesPercent: number;
  proteinGoal: number;
  proteinActual: number;
  proteinPercent: number;
  carbsGoal: number;
  carbsActual: number;
  carbsPercent: number;
  fatGoal: number;
  fatActual: number;
  fatPercent: number;
}

/**
 * Trend information for a single metric
 */
export interface MetricTrend {
  current: number;
  previous: number;
  percentChange: number;
  trend: 'up' | 'down' | 'stable';
}

/**
 * Trend comparison for all macronutrients
 */
export interface TrendComparison {
  calories: MetricTrend;
  protein: MetricTrend;
  carbs: MetricTrend;
  fat: MetricTrend;
  fiber: MetricTrend;
  sugar: MetricTrend;
}

/**
 * Top consumed foods in a period
 */
export interface TopFood {
  name: string;
  count: number;
  totalCalories: number;
  totalProtein: number;
  avgCaloriesPerServing: number;
}

/**
 * Health metric summary for reports
 */
export interface HealthMetricSummary {
  metricType: string;
  average: number;
  min: number;
  max: number;
  trend: 'up' | 'down' | 'stable';
  percentChange: number;
  unit: string;
  dataPoints: number;
}

/**
 * Activity summary for reports
 */
export interface ActivitySummary {
  totalDuration: number; // in minutes
  totalCaloriesBurned: number;
  totalDistance: number; // in meters
  totalSteps: number;
  activityCount: number;
  averageDurationPerActivity: number;
  mostFrequentActivity: string | null;
  intensityBreakdown: {
    low: number;
    moderate: number;
    high: number;
    maximum: number;
  };
}

/**
 * AI-generated insight for reports
 */
export interface ReportInsight {
  id: string;
  title: string;
  description: string;
  category: 'nutrition' | 'health' | 'activity' | 'achievement' | 'recommendation';
  priority: 'low' | 'medium' | 'high';
  icon?: string;
}

/**
 * Achievement earned during the report period
 */
export interface ReportAchievement {
  id: string;
  title: string;
  description: string;
  earnedAt: string;
  type: 'streak' | 'milestone' | 'goal' | 'improvement';
}

/**
 * Meal type breakdown
 */
export interface MealTypeBreakdown {
  breakfast: { count: number; avgCalories: number };
  lunch: { count: number; avgCalories: number };
  dinner: { count: number; avgCalories: number };
  snack: { count: number; avgCalories: number };
}

/**
 * Macro distribution percentages
 */
export interface MacroDistribution {
  proteinPercent: number;
  carbsPercent: number;
  fatPercent: number;
}

/**
 * Streak information
 */
export interface StreakInfo {
  current: number;
  longest: number;
  isOnStreak?: boolean;
  longestThisMonth?: number;
}

// ============================================================================
// WEEKLY REPORT
// ============================================================================

/**
 * Comprehensive weekly nutrition report
 */
export interface WeeklyReport {
  // Metadata
  userId: string;
  generatedAt: string;
  periodStart: string;
  periodEnd: string;

  // Daily data
  dailyBreakdowns: DailyBreakdown[];

  // Aggregated nutrition data
  totals: NutritionTotals;
  averages: NutritionTotals;

  // Goal progress
  goalProgress: GoalProgress;
  daysGoalsMet: number;
  goalCompletionRate: number;

  // Trends (compared to previous week)
  trends: TrendComparison;

  // Top foods
  topFoods: TopFood[];

  // Health metrics summary
  healthMetricsSummary: HealthMetricSummary[];

  // Activity summary
  activitySummary: ActivitySummary | null;

  // AI-generated insights
  insights: ReportInsight[];

  // Achievements earned this week
  achievements: ReportAchievement[];

  // Streak information
  streak: StreakInfo;

  // Meal type distribution
  mealTypeBreakdown: MealTypeBreakdown;

  // Macro distribution percentages
  macroDistribution: MacroDistribution;
}

// ============================================================================
// MONTHLY REPORT
// ============================================================================

/**
 * Weekly breakdown for monthly report
 */
export interface WeeklyBreakdown {
  weekNumber: number;
  weekStart: string;
  weekEnd: string;
  totals: NutritionTotals;
  averages: NutritionTotals;
  daysTracked: number;
  daysGoalsMet: number;
  goalCompletionRate: number;
}

/**
 * Year-over-year comparison
 */
export interface YearOverYear {
  available: boolean;
  previousYearAvg: NutritionTotals | null;
  percentChange: number | null;
}

/**
 * Comprehensive monthly nutrition report
 */
export interface MonthlyReport {
  // Metadata
  userId: string;
  generatedAt: string;
  month: string; // YYYY-MM format
  periodStart: string;
  periodEnd: string;

  // Weekly breakdowns
  weeklyBreakdowns: WeeklyBreakdown[];

  // Daily data
  dailyBreakdowns: DailyBreakdown[];

  // Aggregated nutrition data
  totals: NutritionTotals;
  averages: NutritionTotals;

  // Goal progress
  goalProgress: GoalProgress;
  daysGoalsMet: number;
  totalDaysTracked: number;
  goalCompletionRate: number;

  // Trends (compared to previous month)
  trends: TrendComparison;

  // Best and worst days
  bestDays: DailyBreakdown[];
  worstDays: DailyBreakdown[];

  // Top foods
  topFoods: TopFood[];

  // Health metrics summary
  healthMetricsSummary: HealthMetricSummary[];

  // Activity summary
  activitySummary: ActivitySummary | null;

  // AI-generated insights
  insights: ReportInsight[];

  // Achievements earned this month
  achievements: ReportAchievement[];

  // Streak information
  streak: StreakInfo;

  // Weekly trends
  weeklyTrends: {
    week1Avg: NutritionTotals | null;
    week2Avg: NutritionTotals | null;
    week3Avg: NutritionTotals | null;
    week4Avg: NutritionTotals | null;
    week5Avg: NutritionTotals | null;
  };

  // Meal type distribution
  mealTypeBreakdown: MealTypeBreakdown;

  // Macro distribution percentages
  macroDistribution: MacroDistribution;

  // Year-over-year comparison
  yearOverYear: YearOverYear;
}

// ============================================================================
// EXPORT TYPES
// ============================================================================

/**
 * Export format for reports
 */
export type ReportExportFormat = 'pdf' | 'image' | 'json';

/**
 * Export result
 */
export interface ReportExportResult {
  success: boolean;
  format: ReportExportFormat;
  data: WeeklyReport | MonthlyReport;
  message?: string;
  filename: string;
  mimeType: string;
}
