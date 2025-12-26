// ============================================================================
// NUTRITION REPORTS TYPES
// ============================================================================

import { HealthMetricType } from './index';

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
  metricType: HealthMetricType;
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
  periodStart: string; // ISO date string
  periodEnd: string; // ISO date string

  // Daily data
  dailyBreakdowns: DailyBreakdown[];

  // Aggregated nutrition data
  totals: NutritionTotals;
  averages: NutritionTotals;

  // Goal progress
  goalProgress: GoalProgress;
  daysGoalsMet: number; // Number of days where goals were met
  goalCompletionRate: number; // Percentage of days with goals met

  // Trends (compared to previous week)
  trends: TrendComparison;

  // Top foods
  topFoods: TopFood[];

  // Health metrics summary (if available)
  healthMetricsSummary: HealthMetricSummary[];

  // Activity summary (if available)
  activitySummary: ActivitySummary | null;

  // AI-generated insights
  insights: ReportInsight[];

  // Achievements earned this week
  achievements: ReportAchievement[];

  // Streak information
  streak: {
    current: number;
    longest: number;
    isOnStreak: boolean;
  };

  // Meal type distribution
  mealTypeBreakdown: {
    breakfast: { count: number; avgCalories: number };
    lunch: { count: number; avgCalories: number };
    dinner: { count: number; avgCalories: number };
    snack: { count: number; avgCalories: number };
  };

  // Macro distribution percentages
  macroDistribution: {
    proteinPercent: number;
    carbsPercent: number;
    fatPercent: number;
  };
}

// ============================================================================
// MONTHLY REPORT
// ============================================================================

/**
 * Weekly breakdown for monthly report
 */
export interface WeeklyBreakdown {
  weekNumber: number; // 1-5 (week of the month)
  weekStart: string;
  weekEnd: string;
  totals: NutritionTotals;
  averages: NutritionTotals;
  daysTracked: number;
  daysGoalsMet: number;
  goalCompletionRate: number;
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

  // Daily data for the entire month
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
  bestDays: DailyBreakdown[]; // Top 5 days by goal completion
  worstDays: DailyBreakdown[]; // Bottom 5 days by goal completion

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
  streak: {
    current: number;
    longest: number;
    longestThisMonth: number;
  };

  // Weekly trends
  weeklyTrends: {
    week1Avg: NutritionTotals | null;
    week2Avg: NutritionTotals | null;
    week3Avg: NutritionTotals | null;
    week4Avg: NutritionTotals | null;
    week5Avg: NutritionTotals | null; // May not exist in all months
  };

  // Meal type distribution
  mealTypeBreakdown: {
    breakfast: { count: number; avgCalories: number };
    lunch: { count: number; avgCalories: number };
    dinner: { count: number; avgCalories: number };
    snack: { count: number; avgCalories: number };
  };

  // Macro distribution percentages
  macroDistribution: {
    proteinPercent: number;
    carbsPercent: number;
    fatPercent: number;
  };

  // Year-over-year comparison (if data available)
  yearOverYear: {
    available: boolean;
    previousYearAvg: NutritionTotals | null;
    percentChange: number | null;
  };
}

// ============================================================================
// REPORT GENERATION OPTIONS
// ============================================================================

/**
 * Options for generating reports
 */
export interface ReportGenerationOptions {
  includeHealthMetrics?: boolean;
  includeActivities?: boolean;
  includeInsights?: boolean;
  includeAchievements?: boolean;
  topFoodsLimit?: number;
}

/**
 * Export format for reports
 */
export type ReportExportFormat = 'pdf' | 'image' | 'json';

/**
 * Export options
 */
export interface ReportExportOptions {
  format: ReportExportFormat;
  includeCharts?: boolean;
  includeBranding?: boolean;
  paperSize?: 'letter' | 'a4';
}

/**
 * Export result
 */
export interface ReportExportResult {
  success: boolean;
  format: ReportExportFormat;
  data: string; // Base64 encoded data for pdf/image, or JSON string
  filename: string;
  mimeType: string;
  sizeBytes: number;
}

// ============================================================================
// API QUERY TYPES
// ============================================================================

/**
 * Query parameters for weekly report
 */
export interface WeeklyReportQuery {
  date?: string; // ISO date string - any date within the desired week
}

/**
 * Query parameters for monthly report
 */
export interface MonthlyReportQuery {
  month?: string; // YYYY-MM format
}

/**
 * Query parameters for report export
 */
export interface ReportExportQuery {
  date?: string; // For weekly reports
  month?: string; // For monthly reports
  format: ReportExportFormat;
}
