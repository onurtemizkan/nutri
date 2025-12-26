/**
 * Weight Tracking Types
 * TypeScript types for weight tracking functionality
 */

// ============================================================================
// WEIGHT RECORD TYPES
// ============================================================================

/**
 * Weight record entity
 */
export interface WeightRecord {
  id: string;
  userId: string;
  weight: number; // in kg
  recordedAt: string; // ISO datetime string
  createdAt: string;
}

/**
 * Input for creating a new weight record
 */
export interface CreateWeightRecordInput {
  weight: number; // in kg
  recordedAt?: string; // ISO datetime string (optional, defaults to now)
}

/**
 * Input for updating a weight record
 */
export interface UpdateWeightRecordInput {
  weight?: number; // in kg
  recordedAt?: string; // ISO datetime string
}

// ============================================================================
// WEIGHT TRENDS TYPES
// ============================================================================

/**
 * Moving average data point
 */
export interface MovingAveragePoint {
  date: string; // YYYY-MM-DD format
  value: number;
}

/**
 * Weight trends result with moving averages and statistics
 */
export interface WeightTrendsResult {
  records: {
    id: string;
    weight: number;
    recordedAt: string;
  }[];
  movingAverage7Day: MovingAveragePoint[];
  movingAverage30Day: MovingAveragePoint[];
  minWeight: number | null;
  maxWeight: number | null;
  averageWeight: number | null;
  totalChange: number | null; // Weight change from first to last record
  weeklyChange: number | null; // Weight change in the last 7 days
}

// ============================================================================
// WEIGHT PROGRESS TYPES
// ============================================================================

/**
 * Weight progress towards goal
 */
export interface WeightProgressResult {
  startWeight: number | null;
  currentWeight: number | null;
  goalWeight: number | null;
  progressPercentage: number | null; // 0-100+ (can exceed 100 if goal reached)
  remainingWeight: number | null; // kg remaining to goal
  isOnTrack: boolean | null; // Moving in the right direction
  bmi: number | null;
  bmiCategory: string | null; // 'Underweight' | 'Normal' | 'Overweight' | 'Obese Class I/II/III'
  startDate: string | null; // When tracking started
  latestRecordDate: string | null; // Most recent entry date
}

// ============================================================================
// WEIGHT SUMMARY TYPES
// ============================================================================

/**
 * Weight summary for dashboard widget
 */
export interface WeightSummary {
  currentWeight: number | null;
  weeklyChange: number | null; // kg change in last 7 days
  goalWeight: number | null;
  progressPercentage: number | null;
  bmi: number | null;
  bmiCategory: string | null;
  lastRecordDate: string | null;
}

// ============================================================================
// WEIGHT QUERY TYPES
// ============================================================================

/**
 * Query parameters for fetching weight records
 */
export interface GetWeightRecordsParams {
  startDate?: string;
  endDate?: string;
  limit?: number;
}

/**
 * Query parameters for weight trends
 */
export interface GetWeightTrendsParams {
  days?: number; // Default 30, min 7, max 365
}

// ============================================================================
// WEIGHT UNIT CONVERSION
// ============================================================================

/**
 * Weight unit preference
 */
export type WeightUnit = 'kg' | 'lb';

/**
 * Convert kg to pounds
 */
export function kgToLb(kg: number): number {
  return Math.round(kg * 2.20462 * 10) / 10;
}

/**
 * Convert pounds to kg
 */
export function lbToKg(lb: number): number {
  return Math.round((lb / 2.20462) * 10) / 10;
}

/**
 * Format weight with unit
 */
export function formatWeight(kg: number, unit: WeightUnit = 'kg', decimals: number = 1): string {
  if (unit === 'lb') {
    return `${kgToLb(kg).toFixed(decimals)} lb`;
  }
  return `${kg.toFixed(decimals)} kg`;
}

/**
 * Format weight change with sign
 */
export function formatWeightChange(changeKg: number, unit: WeightUnit = 'kg'): string {
  const value = unit === 'lb' ? kgToLb(changeKg) : changeKg;
  const sign = value > 0 ? '+' : '';
  const unitSuffix = unit === 'lb' ? ' lb' : ' kg';
  return `${sign}${value.toFixed(1)}${unitSuffix}`;
}

// ============================================================================
// BMI HELPERS
// ============================================================================

/**
 * BMI category colors for UI
 */
export const BMI_CATEGORY_COLORS: Record<string, string> = {
  Underweight: '#3B82F6', // Blue
  Normal: '#10B981', // Green
  Overweight: '#F59E0B', // Yellow/Orange
  'Obese Class I': '#EF4444', // Red
  'Obese Class II': '#DC2626', // Darker red
  'Obese Class III': '#991B1B', // Even darker red
};

/**
 * Get color for BMI category
 */
export function getBmiColor(category: string | null): string {
  if (!category) return '#6B7280'; // Gray for null
  return BMI_CATEGORY_COLORS[category] || '#6B7280';
}
