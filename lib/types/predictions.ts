/**
 * TypeScript types for ML Predictions
 *
 * These types mirror the ML service schemas from ml-service/app/schemas/predictions.py
 * Used for health metric predictions (RHR, HRV forecasting)
 */

// ============================================================================
// ENUMS
// ============================================================================

/**
 * Health metrics that can be predicted.
 * Matches PredictionMetric enum (ml-service/app/schemas/predictions.py lines 12-19)
 */
export type PredictionMetric =
  | 'RESTING_HEART_RATE'
  | 'HEART_RATE_VARIABILITY_SDNN'
  | 'HEART_RATE_VARIABILITY_RMSSD'
  | 'SLEEP_DURATION'
  | 'RECOVERY_SCORE';

/**
 * ML model architectures.
 * Matches ModelArchitecture enum (ml-service/app/schemas/predictions.py lines 22-27)
 */
export type ModelArchitecture = 'lstm' | 'xgboost' | 'linear';

// ============================================================================
// TRAINING
// ============================================================================

/**
 * Training metrics for model evaluation.
 * Matches TrainingMetrics (ml-service/app/schemas/predictions.py lines 78-95)
 */
export interface TrainingMetrics {
  /** Final training loss (MSE) */
  train_loss: number;
  /** Final validation loss (MSE) */
  val_loss: number;
  /** Best validation loss achieved */
  best_val_loss: number;
  /** Mean Absolute Error */
  mae: number;
  /** Root Mean Squared Error */
  rmse: number;
  /** R² coefficient of determination */
  r2_score: number;
  /** Mean Absolute Percentage Error */
  mape: number;
  /** Number of epochs trained */
  epochs_trained: number;
  /** True if early stopping triggered */
  early_stopped: boolean;
  /** Total training time in seconds */
  training_time_seconds: number;
}

/**
 * Response from model training.
 * Matches TrainModelResponse (ml-service/app/schemas/predictions.py lines 98-126)
 */
export interface TrainModelResponse {
  user_id: string;
  metric: PredictionMetric;
  architecture: ModelArchitecture;
  /** Unique model identifier */
  model_id: string;
  /** Model version (e.g., 'v1.0.0') */
  model_version: string;
  trained_at: string;
  training_metrics: TrainingMetrics;
  /** Total training samples */
  total_samples: number;
  /** Input sequence length */
  sequence_length: number;
  /** Number of input features */
  num_features: number;
  /** Path to saved model file */
  model_path: string;
  /** Model file size in MB */
  model_size_mb: number;
  /** True if model meets quality thresholds */
  is_production_ready: boolean;
  /** Quality issues if not production-ready */
  quality_issues: string[];
}

// ============================================================================
// PREDICTION
// ============================================================================

/**
 * Result of a single prediction.
 * Matches PredictionResult (ml-service/app/schemas/predictions.py lines 155-182)
 */
export interface PredictionResult {
  metric: PredictionMetric;
  target_date: string;
  predicted_at: string;
  /** Predicted metric value */
  predicted_value: number;
  /** 95% CI lower bound */
  confidence_interval_lower: number;
  /** 95% CI upper bound */
  confidence_interval_upper: number;
  /** Prediction confidence (0-1) */
  confidence_score: number;
  /** User's 30-day average */
  historical_average: number;
  /** Predicted value - historical average */
  deviation_from_average: number;
  /** Percentile compared to user's history (0-100) */
  percentile: number;
  /** Model used for prediction */
  model_id: string;
  /** Model version */
  model_version: string;
  /** Model architecture */
  architecture: ModelArchitecture;
}

/**
 * Response containing prediction.
 * Matches PredictResponse (ml-service/app/schemas/predictions.py lines 185-201)
 */
export interface PredictResponse {
  user_id: string;
  prediction: PredictionResult;
  /** Number of features used */
  features_used: number;
  /** Days of input sequence */
  sequence_length: number;
  /** Input data quality (0-1) */
  data_quality_score: number;
  /** Natural language interpretation */
  interpretation: string;
  /** Actionable recommendation (may be null) */
  recommendation: string | null;
  /** True if prediction was cached */
  cached: boolean;
}

// ============================================================================
// BATCH PREDICTION
// ============================================================================

/**
 * Response containing multiple predictions.
 * Matches BatchPredictResponse (ml-service/app/schemas/predictions.py lines 224-242)
 */
export interface BatchPredictResponse {
  user_id: string;
  target_date: string;
  predicted_at: string;
  /** Predictions keyed by metric name */
  predictions: Record<string, PredictionResult>;
  /** Overall data quality (0-1) */
  overall_data_quality: number;
  /** True if all metrics predicted successfully */
  all_predictions_successful: boolean;
  /** Metrics that failed to predict */
  failed_metrics: string[];
}

// ============================================================================
// MODEL MANAGEMENT
// ============================================================================

/**
 * Information about a trained model.
 * Matches ModelInfo (ml-service/app/schemas/predictions.py lines 250-267)
 */
export interface ModelInfo {
  model_id: string;
  user_id: string;
  metric: PredictionMetric;
  architecture: ModelArchitecture;
  version: string;
  trained_at: string;
  training_metrics: TrainingMetrics;
  sequence_length: number;
  num_features: number;
  model_size_mb: number;
  /** True if currently in use */
  is_active: boolean;
  is_production_ready: boolean;
}

/**
 * Response listing user's models.
 * Matches ListModelsResponse (ml-service/app/schemas/predictions.py lines 270-275)
 */
export interface ListModelsResponse {
  user_id: string;
  models: ModelInfo[];
  total_models: number;
}

/**
 * Historical performance of a model.
 * Matches ModelPerformanceHistory (ml-service/app/schemas/predictions.py lines 278-296)
 */
export interface ModelPerformanceHistory {
  model_id: string;
  metric: PredictionMetric;
  /** Total predictions made */
  predictions_made: number;
  /** Average MAE on actuals */
  avg_mae: number;
  /** Average confidence score */
  avg_confidence: number;
  /** List of {date, predicted, actual, error} */
  actual_vs_predicted: Array<{
    date: string;
    predicted: number;
    actual: number;
    error: number;
  }>;
  /** True if performance is degrading */
  is_drifting: boolean;
  /** True if model should be retrained */
  should_retrain: boolean;
}

// ============================================================================
// WHAT-IF SCENARIOS
// ============================================================================

/**
 * A hypothetical nutrition/activity scenario.
 * Matches WhatIfScenario (ml-service/app/schemas/predictions.py lines 304-316)
 */
export interface WhatIfScenario {
  /** Name of scenario */
  scenario_name: string;
  /** Changes to nutrition features (e.g., {'protein_daily': 200}) */
  nutrition_changes?: Record<string, number>;
  /** Changes to activity features (e.g., {'workout_intensity': 0.8}) */
  activity_changes?: Record<string, number>;
}

/**
 * Request to test what-if scenarios.
 * Matches WhatIfRequest (ml-service/app/schemas/predictions.py lines 319-328)
 */
export interface WhatIfRequest {
  user_id: string;
  metric: PredictionMetric;
  target_date: string;
  /** Scenarios to test (1-5) */
  scenarios: WhatIfScenario[];
}

/**
 * Result of a what-if scenario.
 * Matches WhatIfResult (ml-service/app/schemas/predictions.py lines 331-341)
 */
export interface WhatIfResult {
  scenario_name: string;
  predicted_value: number;
  confidence_score: number;
  /** Difference from baseline prediction */
  change_from_baseline: number;
  /** Percentage change from baseline */
  percent_change: number;
}

/**
 * Response containing what-if scenario results.
 * Matches WhatIfResponse (ml-service/app/schemas/predictions.py lines 344-362)
 */
export interface WhatIfResponse {
  user_id: string;
  metric: PredictionMetric;
  target_date: string;
  /** Prediction with current data */
  baseline_prediction: number;
  /** Scenario results */
  scenarios: WhatIfResult[];
  /** Scenario with best predicted outcome */
  best_scenario: string;
  /** Scenario with worst predicted outcome */
  worst_scenario: string;
  /** Natural language interpretation of scenarios */
  interpretation: string;
}

// ============================================================================
// DISPLAY HELPERS
// ============================================================================

/**
 * Configuration for displaying prediction metrics
 */
export interface PredictionMetricConfig {
  displayName: string;
  unit: string;
  icon: string;
  description: string;
  decimalPlaces: number;
}

/**
 * Display configuration for each prediction metric
 */
export const PREDICTION_METRIC_CONFIG: Record<PredictionMetric, PredictionMetricConfig> = {
  RESTING_HEART_RATE: {
    displayName: 'Resting Heart Rate',
    unit: 'bpm',
    icon: 'heart-outline',
    description: 'Predicted resting heart rate for tomorrow',
    decimalPlaces: 0,
  },
  HEART_RATE_VARIABILITY_SDNN: {
    displayName: 'HRV (SDNN)',
    unit: 'ms',
    icon: 'pulse-outline',
    description: 'Predicted heart rate variability for tomorrow',
    decimalPlaces: 0,
  },
  HEART_RATE_VARIABILITY_RMSSD: {
    displayName: 'HRV (RMSSD)',
    unit: 'ms',
    icon: 'pulse-outline',
    description: 'Predicted HRV RMSSD for tomorrow',
    decimalPlaces: 0,
  },
  SLEEP_DURATION: {
    displayName: 'Sleep Duration',
    unit: 'hrs',
    icon: 'moon-outline',
    description: 'Predicted sleep duration for tonight',
    decimalPlaces: 1,
  },
  RECOVERY_SCORE: {
    displayName: 'Recovery Score',
    unit: '%',
    icon: 'fitness-outline',
    description: 'Predicted recovery score for tomorrow',
    decimalPlaces: 0,
  },
};

/**
 * Primary metrics shown on predictions dashboard
 */
export const PRIMARY_PREDICTION_METRICS: PredictionMetric[] = [
  'RESTING_HEART_RATE',
  'HEART_RATE_VARIABILITY_SDNN',
];

/**
 * Helper to get tomorrow's date as ISO string (YYYY-MM-DD)
 */
export function getTomorrowDate(): string {
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  return tomorrow.toISOString().split('T')[0];
}

/**
 * Format confidence score as percentage
 */
export function formatConfidenceScore(score: number): string {
  return `${Math.round(score * 100)}%`;
}

/**
 * Determine trend direction based on deviation from average
 */
export function getTrendDirection(deviation: number): 'up' | 'down' | 'neutral' {
  if (deviation > 0.5) return 'up';
  if (deviation < -0.5) return 'down';
  return 'neutral';
}
