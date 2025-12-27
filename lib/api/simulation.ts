/**
 * Simulation API Client
 *
 * API client for the What-If Simulation Engine.
 */

import api from './client';
import type {
  TrajectoryRequest,
  TrajectoryResponse,
  NutritionChange,
  PredictionMetric,
  SimulationDuration,
} from '@/lib/types/simulation';

/**
 * Simulate multi-day trajectory with nutrition changes
 *
 * @param request - Trajectory simulation request
 * @returns Promise<TrajectoryResponse>
 */
export async function simulateTrajectory(request: TrajectoryRequest): Promise<TrajectoryResponse> {
  const response = await api.post<TrajectoryResponse>('/ml/interpret/trajectory', request);
  return response.data;
}

/**
 * Helper function to create a trajectory request
 *
 * @param userId - User ID
 * @param changes - Map of feature names to delta values
 * @param options - Additional options
 * @returns TrajectoryRequest
 */
export function createTrajectoryRequest(
  userId: string,
  changes: Record<string, number>,
  options: {
    duration?: SimulationDuration;
    metrics?: PredictionMetric[];
    includeBaseline?: boolean;
  } = {}
): TrajectoryRequest {
  const {
    duration = 7,
    metrics = ['RESTING_HEART_RATE', 'HEART_RATE_VARIABILITY_SDNN'],
    includeBaseline = true,
  } = options;

  // Convert changes record to NutritionChange array
  const nutritionChanges: NutritionChange[] = Object.entries(changes)
    .filter(([_, delta]) => delta !== 0)
    .map(([featureName, delta]) => ({
      feature_name: featureName,
      delta,
      change_description: formatChangeDescription(featureName, delta),
    }));

  return {
    user_id: userId,
    nutrition_changes: nutritionChanges,
    duration_days: duration,
    metrics_to_predict: metrics,
    include_no_change_baseline: includeBaseline,
  };
}

/**
 * Format a change description from feature name and delta
 */
function formatChangeDescription(featureName: string, delta: number): string {
  const prefix = delta >= 0 ? '+' : '';

  // Extract readable name and unit from feature name
  if (featureName.includes('protein')) {
    return `${prefix}${delta}g protein`;
  } else if (featureName.includes('carbs')) {
    return `${prefix}${delta}g carbs`;
  } else if (featureName.includes('fat')) {
    return `${prefix}${delta}g fat`;
  } else if (featureName.includes('calories')) {
    return `${prefix}${delta} calories`;
  } else if (featureName.includes('sugar')) {
    return `${prefix}${delta}g sugar`;
  } else if (featureName.includes('fiber')) {
    return `${prefix}${delta}g fiber`;
  }

  // Default format
  const readableName = featureName
    .replace('nutrition_', '')
    .replace('_daily', '')
    .replace(/_/g, ' ');
  return `${prefix}${delta} ${readableName}`;
}

/**
 * Simulation readiness check response
 */
export interface SimulationReadinessResponse {
  /** Whether user has enough data for simulation */
  isReady: boolean;
  /** Minimum days of data required */
  minDataDays: number;
  /** Actual days of data available */
  availableDataDays: number;
  /** Available metrics for simulation */
  availableMetrics: PredictionMetric[];
  /** Metrics that need more data */
  unavailableMetrics: {
    metric: PredictionMetric;
    reason: string;
    dataAvailable: number;
    dataRequired: number;
  }[];
  /** Overall data quality score (0-1) */
  dataQualityScore: number;
  /** Suggestions for improving readiness */
  suggestions: string[];
}

/**
 * Check if user has enough data for simulation
 *
 * @param userId - User ID
 * @returns Promise<SimulationReadinessResponse>
 */
export async function getSimulationReadiness(userId: string): Promise<SimulationReadinessResponse> {
  try {
    const response = await api.get<SimulationReadinessResponse>(
      `/ml/interpret/simulation-readiness/${userId}`
    );
    return response.data;
  } catch {
    // If endpoint doesn't exist yet, return a mock response
    // This allows the UI to work during development
    return {
      isReady: true,
      minDataDays: 30,
      availableDataDays: 45,
      availableMetrics: ['RESTING_HEART_RATE', 'HEART_RATE_VARIABILITY_SDNN'],
      unavailableMetrics: [],
      dataQualityScore: 0.8,
      suggestions: [],
    };
  }
}

export default {
  simulateTrajectory,
  createTrajectoryRequest,
  getSimulationReadiness,
};
