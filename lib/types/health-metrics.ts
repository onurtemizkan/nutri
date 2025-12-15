/**
 * Health Metrics Types
 * TypeScript types for health metrics matching the backend API contracts
 */

/**
 * Health Metric Type enum - matches server/src/validation/schemas.ts healthMetricTypeSchema
 */
export type HealthMetricType =
  | 'RESTING_HEART_RATE'
  | 'HEART_RATE_VARIABILITY_SDNN'
  | 'HEART_RATE_VARIABILITY_RMSSD'
  | 'BLOOD_PRESSURE_SYSTOLIC'
  | 'BLOOD_PRESSURE_DIASTOLIC'
  | 'RESPIRATORY_RATE'
  | 'OXYGEN_SATURATION'
  | 'VO2_MAX'
  | 'SLEEP_DURATION'
  | 'DEEP_SLEEP_DURATION'
  | 'REM_SLEEP_DURATION'
  | 'SLEEP_EFFICIENCY'
  | 'SLEEP_SCORE'
  | 'STEPS'
  | 'ACTIVE_CALORIES'
  | 'TOTAL_CALORIES'
  | 'EXERCISE_MINUTES'
  | 'STANDING_HOURS'
  | 'RECOVERY_SCORE'
  | 'STRAIN_SCORE'
  | 'READINESS_SCORE'
  | 'BODY_FAT_PERCENTAGE'
  | 'MUSCLE_MASS'
  | 'BONE_MASS'
  | 'WATER_PERCENTAGE'
  | 'SKIN_TEMPERATURE'
  | 'BLOOD_GLUCOSE'
  | 'STRESS_LEVEL';

/**
 * Health Metric Source - matches server/src/validation/schemas.ts healthMetricSourceSchema
 */
export type HealthMetricSource =
  | 'apple_health'
  | 'fitbit'
  | 'garmin'
  | 'oura'
  | 'whoop'
  | 'manual';

/**
 * Trend direction for metric statistics
 */
export type TrendDirection = 'up' | 'down' | 'stable';

/**
 * Health Metric entity returned from API
 */
export interface HealthMetric {
  id: string;
  userId: string;
  metricType: HealthMetricType;
  value: number;
  unit: string;
  recordedAt: string;
  source: HealthMetricSource;
  sourceId?: string;
  metadata?: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

/**
 * Input for creating a new health metric
 */
export interface CreateHealthMetricInput {
  metricType: HealthMetricType;
  value: number;
  unit: string;
  recordedAt: string; // ISO 8601 datetime string
  source: HealthMetricSource;
  sourceId?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Input for updating an existing health metric
 */
export interface UpdateHealthMetricInput {
  value?: number;
  unit?: string;
  recordedAt?: string; // ISO 8601 datetime string
  metadata?: Record<string, unknown>;
}

/**
 * Statistics for a health metric type
 */
export interface HealthMetricStats {
  average: number;
  min: number;
  max: number;
  count: number;
  trend: TrendDirection;
  percentChange: number;
}

/**
 * Time series data point for charts
 */
export interface TimeSeriesDataPoint {
  date: string;
  value: number;
  source?: HealthMetricSource;
}

/**
 * Average response from daily/weekly average endpoints
 */
export interface AverageResponse {
  average: number;
  count: number;
}

/**
 * Metric configuration for display and validation
 */
export interface MetricConfig {
  unit: string;
  displayName: string;
  shortName: string;
  minValue?: number;
  maxValue?: number;
  icon: string; // Ionicons name
  category: MetricCategory;
  description?: string;
}

/**
 * Categories for grouping metrics
 */
export type MetricCategory =
  | 'cardiovascular'
  | 'sleep'
  | 'activity'
  | 'recovery'
  | 'body_composition'
  | 'other';

/**
 * Metric configuration constant - display info and validation rules per metric type
 */
export const METRIC_CONFIG: Record<HealthMetricType, MetricConfig> = {
  // Cardiovascular
  RESTING_HEART_RATE: {
    unit: 'bpm',
    displayName: 'Resting Heart Rate',
    shortName: 'RHR',
    minValue: 30,
    maxValue: 220,
    icon: 'heart-outline',
    category: 'cardiovascular',
    description: 'Your heart rate while at rest',
  },
  HEART_RATE_VARIABILITY_SDNN: {
    unit: 'ms',
    displayName: 'Heart Rate Variability (SDNN)',
    shortName: 'HRV',
    minValue: 0,
    maxValue: 300,
    icon: 'pulse-outline',
    category: 'cardiovascular',
    description: 'Standard deviation of heartbeat intervals',
  },
  HEART_RATE_VARIABILITY_RMSSD: {
    unit: 'ms',
    displayName: 'Heart Rate Variability (RMSSD)',
    shortName: 'HRV RMSSD',
    minValue: 0,
    maxValue: 300,
    icon: 'pulse-outline',
    category: 'cardiovascular',
    description: 'Root mean square of successive differences',
  },
  BLOOD_PRESSURE_SYSTOLIC: {
    unit: 'mmHg',
    displayName: 'Blood Pressure (Systolic)',
    shortName: 'BP Sys',
    minValue: 70,
    maxValue: 250,
    icon: 'water-outline',
    category: 'cardiovascular',
    description: 'Pressure when heart beats',
  },
  BLOOD_PRESSURE_DIASTOLIC: {
    unit: 'mmHg',
    displayName: 'Blood Pressure (Diastolic)',
    shortName: 'BP Dia',
    minValue: 40,
    maxValue: 150,
    icon: 'water-outline',
    category: 'cardiovascular',
    description: 'Pressure between heartbeats',
  },
  RESPIRATORY_RATE: {
    unit: 'brpm',
    displayName: 'Respiratory Rate',
    shortName: 'Resp',
    minValue: 4,
    maxValue: 60,
    icon: 'cloud-outline',
    category: 'cardiovascular',
    description: 'Breaths per minute',
  },
  OXYGEN_SATURATION: {
    unit: '%',
    displayName: 'Oxygen Saturation',
    shortName: 'SpO2',
    minValue: 70,
    maxValue: 100,
    icon: 'water-outline',
    category: 'cardiovascular',
    description: 'Blood oxygen level',
  },
  VO2_MAX: {
    unit: 'ml/kg/min',
    displayName: 'VO2 Max',
    shortName: 'VO2',
    minValue: 10,
    maxValue: 100,
    icon: 'fitness-outline',
    category: 'cardiovascular',
    description: 'Maximum oxygen uptake',
  },

  // Sleep
  SLEEP_DURATION: {
    unit: 'hours',
    displayName: 'Sleep Duration',
    shortName: 'Sleep',
    minValue: 0,
    maxValue: 24,
    icon: 'moon-outline',
    category: 'sleep',
    description: 'Total time asleep',
  },
  DEEP_SLEEP_DURATION: {
    unit: 'hours',
    displayName: 'Deep Sleep Duration',
    shortName: 'Deep',
    minValue: 0,
    maxValue: 12,
    icon: 'moon-outline',
    category: 'sleep',
    description: 'Time in deep sleep stage',
  },
  REM_SLEEP_DURATION: {
    unit: 'hours',
    displayName: 'REM Sleep Duration',
    shortName: 'REM',
    minValue: 0,
    maxValue: 12,
    icon: 'moon-outline',
    category: 'sleep',
    description: 'Time in REM sleep stage',
  },
  SLEEP_EFFICIENCY: {
    unit: '%',
    displayName: 'Sleep Efficiency',
    shortName: 'Efficiency',
    minValue: 0,
    maxValue: 100,
    icon: 'moon-outline',
    category: 'sleep',
    description: 'Percentage of time in bed spent sleeping',
  },
  SLEEP_SCORE: {
    unit: 'pts',
    displayName: 'Sleep Score',
    shortName: 'Score',
    minValue: 0,
    maxValue: 100,
    icon: 'moon-outline',
    category: 'sleep',
    description: 'Overall sleep quality score',
  },

  // Activity
  STEPS: {
    unit: 'steps',
    displayName: 'Steps',
    shortName: 'Steps',
    minValue: 0,
    maxValue: 100000,
    icon: 'footsteps-outline',
    category: 'activity',
    description: 'Total steps taken',
  },
  ACTIVE_CALORIES: {
    unit: 'kcal',
    displayName: 'Active Calories',
    shortName: 'Active',
    minValue: 0,
    maxValue: 10000,
    icon: 'flame-outline',
    category: 'activity',
    description: 'Calories burned through activity',
  },
  TOTAL_CALORIES: {
    unit: 'kcal',
    displayName: 'Total Calories',
    shortName: 'Total',
    minValue: 0,
    maxValue: 15000,
    icon: 'flame-outline',
    category: 'activity',
    description: 'Total calories burned',
  },
  EXERCISE_MINUTES: {
    unit: 'min',
    displayName: 'Exercise Minutes',
    shortName: 'Exercise',
    minValue: 0,
    maxValue: 1440,
    icon: 'stopwatch-outline',
    category: 'activity',
    description: 'Minutes of exercise',
  },
  STANDING_HOURS: {
    unit: 'hours',
    displayName: 'Standing Hours',
    shortName: 'Stand',
    minValue: 0,
    maxValue: 24,
    icon: 'person-outline',
    category: 'activity',
    description: 'Hours spent standing',
  },

  // Recovery
  RECOVERY_SCORE: {
    unit: '%',
    displayName: 'Recovery Score',
    shortName: 'Recovery',
    minValue: 0,
    maxValue: 100,
    icon: 'battery-charging-outline',
    category: 'recovery',
    description: 'Overall recovery level',
  },
  STRAIN_SCORE: {
    unit: 'pts',
    displayName: 'Strain Score',
    shortName: 'Strain',
    minValue: 0,
    maxValue: 21,
    icon: 'barbell-outline',
    category: 'recovery',
    description: 'Daily strain from activity',
  },
  READINESS_SCORE: {
    unit: 'pts',
    displayName: 'Readiness Score',
    shortName: 'Ready',
    minValue: 0,
    maxValue: 100,
    icon: 'checkmark-circle-outline',
    category: 'recovery',
    description: 'Body readiness for activity',
  },
  STRESS_LEVEL: {
    unit: 'pts',
    displayName: 'Stress Level',
    shortName: 'Stress',
    minValue: 0,
    maxValue: 100,
    icon: 'alert-circle-outline',
    category: 'recovery',
    description: 'Current stress level',
  },

  // Body Composition
  BODY_FAT_PERCENTAGE: {
    unit: '%',
    displayName: 'Body Fat Percentage',
    shortName: 'Body Fat',
    minValue: 1,
    maxValue: 60,
    icon: 'body-outline',
    category: 'body_composition',
    description: 'Percentage of body fat',
  },
  MUSCLE_MASS: {
    unit: 'kg',
    displayName: 'Muscle Mass',
    shortName: 'Muscle',
    minValue: 10,
    maxValue: 100,
    icon: 'body-outline',
    category: 'body_composition',
    description: 'Total muscle mass',
  },
  BONE_MASS: {
    unit: 'kg',
    displayName: 'Bone Mass',
    shortName: 'Bone',
    minValue: 1,
    maxValue: 10,
    icon: 'body-outline',
    category: 'body_composition',
    description: 'Total bone mass',
  },
  WATER_PERCENTAGE: {
    unit: '%',
    displayName: 'Water Percentage',
    shortName: 'Water',
    minValue: 30,
    maxValue: 80,
    icon: 'water-outline',
    category: 'body_composition',
    description: 'Body water percentage',
  },
  SKIN_TEMPERATURE: {
    unit: '°C',
    displayName: 'Skin Temperature',
    shortName: 'Temp',
    minValue: 30,
    maxValue: 42,
    icon: 'thermometer-outline',
    category: 'body_composition',
    description: 'Skin surface temperature',
  },
  BLOOD_GLUCOSE: {
    unit: 'mg/dL',
    displayName: 'Blood Glucose',
    shortName: 'Glucose',
    minValue: 20,
    maxValue: 600,
    icon: 'water-outline',
    category: 'other',
    description: 'Blood sugar level',
  },
};

/**
 * All health metric types as an array for iteration
 */
export const HEALTH_METRIC_TYPES: HealthMetricType[] = [
  'RESTING_HEART_RATE',
  'HEART_RATE_VARIABILITY_SDNN',
  'HEART_RATE_VARIABILITY_RMSSD',
  'BLOOD_PRESSURE_SYSTOLIC',
  'BLOOD_PRESSURE_DIASTOLIC',
  'RESPIRATORY_RATE',
  'OXYGEN_SATURATION',
  'VO2_MAX',
  'SLEEP_DURATION',
  'DEEP_SLEEP_DURATION',
  'REM_SLEEP_DURATION',
  'SLEEP_EFFICIENCY',
  'SLEEP_SCORE',
  'STEPS',
  'ACTIVE_CALORIES',
  'TOTAL_CALORIES',
  'EXERCISE_MINUTES',
  'STANDING_HOURS',
  'RECOVERY_SCORE',
  'STRAIN_SCORE',
  'READINESS_SCORE',
  'BODY_FAT_PERCENTAGE',
  'MUSCLE_MASS',
  'BONE_MASS',
  'WATER_PERCENTAGE',
  'SKIN_TEMPERATURE',
  'BLOOD_GLUCOSE',
  'STRESS_LEVEL',
];

/**
 * Health metric sources as an array for iteration
 */
export const HEALTH_METRIC_SOURCES: HealthMetricSource[] = [
  'apple_health',
  'fitbit',
  'garmin',
  'oura',
  'whoop',
  'manual',
];

/**
 * Source display configuration
 */
export const SOURCE_CONFIG: Record<HealthMetricSource, { displayName: string; icon: string }> = {
  apple_health: { displayName: 'Apple Health', icon: 'logo-apple' },
  fitbit: { displayName: 'Fitbit', icon: 'watch-outline' },
  garmin: { displayName: 'Garmin', icon: 'watch-outline' },
  oura: { displayName: 'Oura', icon: 'ellipse-outline' },
  whoop: { displayName: 'WHOOP', icon: 'fitness-outline' },
  manual: { displayName: 'Manual Entry', icon: 'create-outline' },
};

/**
 * Key metrics to display on the dashboard
 */
export const DASHBOARD_METRICS: HealthMetricType[] = [
  'RESTING_HEART_RATE',
  'HEART_RATE_VARIABILITY_SDNN',
  'SLEEP_DURATION',
  'RECOVERY_SCORE',
];

/**
 * Get metrics grouped by category
 */
export function getMetricsByCategory(): Record<MetricCategory, HealthMetricType[]> {
  const grouped: Record<MetricCategory, HealthMetricType[]> = {
    cardiovascular: [],
    sleep: [],
    activity: [],
    recovery: [],
    body_composition: [],
    other: [],
  };

  for (const metricType of HEALTH_METRIC_TYPES) {
    const config = METRIC_CONFIG[metricType];
    grouped[config.category].push(metricType);
  }

  return grouped;
}

/**
 * Category display names
 */
export const CATEGORY_DISPLAY_NAMES: Record<MetricCategory, string> = {
  cardiovascular: 'Cardiovascular',
  sleep: 'Sleep',
  activity: 'Activity',
  recovery: 'Recovery',
  body_composition: 'Body Composition',
  other: 'Other',
};

/**
 * Validate a metric value against its configured min/max range
 */
export function validateMetricValue(
  metricType: HealthMetricType,
  value: number
): { valid: boolean; error?: string } {
  const config = METRIC_CONFIG[metricType];

  if (config.minValue !== undefined && value < config.minValue) {
    return {
      valid: false,
      error: `Value must be at least ${config.minValue} ${config.unit}`,
    };
  }

  if (config.maxValue !== undefined && value > config.maxValue) {
    return {
      valid: false,
      error: `Value must be at most ${config.maxValue} ${config.unit}`,
    };
  }

  return { valid: true };
}
