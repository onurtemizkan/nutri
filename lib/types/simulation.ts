/**
 * Simulation Types
 *
 * TypeScript interfaces for the What-If Simulation Engine.
 * Matches the Pydantic schemas from ml-service/app/schemas/interpretability.py
 */

/**
 * Health metrics available for prediction
 */
export type PredictionMetric =
  | 'RESTING_HEART_RATE'
  | 'HEART_RATE_VARIABILITY_SDNN'
  | 'SLEEP_QUALITY_SCORE'
  | 'RECOVERY_SCORE';

/**
 * Supported simulation durations
 */
export type SimulationDuration = 7 | 14 | 30;

/**
 * A nutrition change to apply during simulation
 */
export interface NutritionChange {
  /** Feature to modify (e.g., 'nutrition_protein_daily') */
  feature_name: string;
  /** Change amount (absolute, e.g., +30 for 30g more protein) */
  delta: number;
  /** Human-readable change (e.g., '+30g protein') */
  change_description: string;
}

/**
 * A single point in the trajectory
 */
export interface TrajectoryPoint {
  /** Day number in the simulation (0 = baseline) */
  day: number;
  /** Actual date of this prediction */
  timestamp: string;
  /** Predicted metric value */
  predicted_value: number;
  /** Lower bound of confidence interval */
  confidence_lower: number;
  /** Upper bound of confidence interval */
  confidence_upper: number;
}

/**
 * Multi-day trajectory projection for a single metric
 */
export interface SimulationTrajectory {
  /** Health metric being projected */
  metric: PredictionMetric;
  /** Starting value (day 0 prediction) */
  baseline_value: number;
  /** Final value at end of simulation */
  projected_final_value: number;
  /** Difference from baseline to final */
  change_from_baseline: number;
  /** Percentage change from baseline */
  percent_change: number;
  /** Day-by-day trajectory points */
  trajectory: TrajectoryPoint[];
  /** Minimum value in trajectory */
  min_value: number;
  /** Maximum value in trajectory */
  max_value: number;
  /** Average value across trajectory */
  average_value: number;
  /** Detected optimal lag from nutrition changes to metric effect */
  optimal_lag_hours?: number | null;
  /** Human-readable lag description */
  lag_description?: string | null;
}

/**
 * Request to generate a multi-day trajectory simulation
 */
export interface TrajectoryRequest {
  /** User ID */
  user_id: string;
  /** Nutrition changes to simulate */
  nutrition_changes: NutritionChange[];
  /** Simulation duration (7, 14, or 30 days) */
  duration_days: SimulationDuration;
  /** Metrics to project (1-3) */
  metrics_to_predict: PredictionMetric[];
  /** Start date (defaults to today) */
  start_date?: string | null;
  /** Include trajectory without changes for comparison */
  include_no_change_baseline?: boolean;
}

/**
 * Response with multi-day trajectory simulations
 */
export interface TrajectoryResponse {
  /** User ID */
  user_id: string;
  /** Simulation start date */
  start_date: string;
  /** Simulation end date */
  end_date: string;
  /** Duration in days */
  duration_days: number;
  /** Applied nutrition changes */
  nutrition_changes: NutritionChange[];
  /** Trajectory projections for each metric */
  trajectories: SimulationTrajectory[];
  /** Trajectories without any changes for comparison */
  baseline_trajectories?: SimulationTrajectory[] | null;
  /** Overall model confidence (0-1) */
  model_confidence: number;
  /** Warning about data quality if applicable */
  data_quality_warning?: string | null;
  /** Natural language summary of projections */
  summary: string;
  /** Recommendation based on trajectory outcomes */
  recommendation: string;
}

/**
 * Nutrition slider configuration
 */
export interface NutritionSliderConfig {
  /** Feature name for API */
  featureName: string;
  /** Display label */
  label: string;
  /** Minimum delta value */
  min: number;
  /** Maximum delta value */
  max: number;
  /** Step size */
  step: number;
  /** Unit (e.g., 'g', 'kcal') */
  unit: string;
  /** Format function for display */
  formatValue: (value: number) => string;
}

/**
 * Metric display configuration
 */
export interface MetricDisplayConfig {
  /** Metric type */
  metric: PredictionMetric;
  /** Display label */
  label: string;
  /** Short label for badges */
  shortLabel: string;
  /** Unit (e.g., 'BPM', 'ms') */
  unit: string;
  /** Whether lower values are better */
  lowerIsBetter: boolean;
  /** Color for positive changes */
  positiveColor: string;
  /** Color for negative changes */
  negativeColor: string;
}

/**
 * Default nutrition slider configurations
 */
export const NUTRITION_SLIDERS: NutritionSliderConfig[] = [
  {
    featureName: 'nutrition_protein_daily',
    label: 'Protein',
    min: -50,
    max: 50,
    step: 5,
    unit: 'g',
    formatValue: (v) => `${v >= 0 ? '+' : ''}${v}g`,
  },
  {
    featureName: 'nutrition_carbs_daily',
    label: 'Carbohydrates',
    min: -100,
    max: 100,
    step: 10,
    unit: 'g',
    formatValue: (v) => `${v >= 0 ? '+' : ''}${v}g`,
  },
  {
    featureName: 'nutrition_fat_daily',
    label: 'Fat',
    min: -30,
    max: 30,
    step: 5,
    unit: 'g',
    formatValue: (v) => `${v >= 0 ? '+' : ''}${v}g`,
  },
  {
    featureName: 'nutrition_calories_daily',
    label: 'Calories',
    min: -500,
    max: 500,
    step: 50,
    unit: 'kcal',
    formatValue: (v) => `${v >= 0 ? '+' : ''}${v}`,
  },
  {
    featureName: 'nutrition_sugar_daily',
    label: 'Sugar',
    min: -30,
    max: 30,
    step: 5,
    unit: 'g',
    formatValue: (v) => `${v >= 0 ? '+' : ''}${v}g`,
  },
];

/**
 * Default metric display configurations
 */
export const METRIC_CONFIGS: MetricDisplayConfig[] = [
  {
    metric: 'RESTING_HEART_RATE',
    label: 'Resting Heart Rate',
    shortLabel: 'RHR',
    unit: 'BPM',
    lowerIsBetter: true,
    positiveColor: '#EF4444', // Red (higher RHR is worse)
    negativeColor: '#10B981', // Green (lower RHR is better)
  },
  {
    metric: 'HEART_RATE_VARIABILITY_SDNN',
    label: 'Heart Rate Variability',
    shortLabel: 'HRV',
    unit: 'ms',
    lowerIsBetter: false,
    positiveColor: '#10B981', // Green (higher HRV is better)
    negativeColor: '#EF4444', // Red (lower HRV is worse)
  },
  {
    metric: 'RECOVERY_SCORE',
    label: 'Recovery Score',
    shortLabel: 'Recovery',
    unit: '%',
    lowerIsBetter: false,
    positiveColor: '#10B981', // Green
    negativeColor: '#EF4444', // Red
  },
];

/**
 * Get display config for a metric
 */
export function getMetricConfig(metric: PredictionMetric): MetricDisplayConfig | undefined {
  return METRIC_CONFIGS.find((c) => c.metric === metric);
}

/**
 * Format change value with color indication
 */
export function formatChange(
  value: number,
  config: MetricDisplayConfig
): { text: string; color: string } {
  const isPositiveChange = value > 0;
  const isImprovement = config.lowerIsBetter ? !isPositiveChange : isPositiveChange;

  return {
    text: `${value >= 0 ? '+' : ''}${value.toFixed(1)}${config.unit}`,
    color: isImprovement ? config.positiveColor : config.negativeColor,
  };
}
