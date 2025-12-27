/**
 * Predictions API Client
 *
 * API functions for ML-powered health metric predictions.
 * Communicates with the ML service endpoints for RHR, HRV forecasting.
 */

import api from './client';
import {
  PredictionMetric,
  PredictResponse,
  BatchPredictResponse,
  ListModelsResponse,
  ModelInfo,
  WhatIfScenario,
  WhatIfResponse,
  PRIMARY_PREDICTION_METRICS,
  getTomorrowDate,
} from '../types/predictions';

// Re-export types for convenience
export type {
  PredictionMetric,
  PredictResponse,
  BatchPredictResponse,
  ListModelsResponse,
  ModelInfo,
};

/**
 * Helper to check if an error is a 404 Not Found
 * Used when no trained model exists for the user/metric
 */
function isNotFoundError(error: unknown): boolean {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { status?: number } }).response;
    return response?.status === 404;
  }
  return false;
}

/**
 * Helper to check if circuit breaker is open (503)
 */
function isCircuitBreakerOpen(error: unknown): boolean {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { status?: number } }).response;
    return response?.status === 503;
  }
  return false;
}

/**
 * Predictions API client
 */
export const predictionsApi = {
  /**
   * Get a single prediction for a metric
   * POST /predictions/predict
   *
   * @param metric - The health metric to predict
   * @param targetDate - Date to predict for (default: tomorrow)
   * @param modelVersion - Optional specific model version
   * @returns PredictResponse or null if no model trained
   */
  async predict(
    metric: PredictionMetric,
    targetDate?: string,
    modelVersion?: string
  ): Promise<PredictResponse | null> {
    try {
      const response = await api.post<PredictResponse>('/predictions/predict', {
        metric,
        target_date: targetDate || getTomorrowDate(),
        model_version: modelVersion,
      });
      return response.data;
    } catch (error) {
      // Return null if no model trained (404)
      if (isNotFoundError(error)) {
        return null;
      }
      // Rethrow circuit breaker errors with descriptive message
      if (isCircuitBreakerOpen(error)) {
        throw new Error('Prediction service is temporarily unavailable. Please try again later.');
      }
      throw error;
    }
  },

  /**
   * Get predictions for multiple metrics at once
   * POST /predictions/batch-predict
   *
   * @param metrics - Array of metrics to predict
   * @param targetDate - Date to predict for (default: tomorrow)
   * @returns BatchPredictResponse or null if no models trained
   */
  async batchPredict(
    metrics: PredictionMetric[],
    targetDate?: string
  ): Promise<BatchPredictResponse | null> {
    try {
      const response = await api.post<BatchPredictResponse>('/predictions/batch-predict', {
        metrics,
        target_date: targetDate || getTomorrowDate(),
      });
      return response.data;
    } catch (error) {
      // Return null if no models trained (404)
      if (isNotFoundError(error)) {
        return null;
      }
      // Rethrow circuit breaker errors with descriptive message
      if (isCircuitBreakerOpen(error)) {
        throw new Error('Prediction service is temporarily unavailable. Please try again later.');
      }
      throw error;
    }
  },

  /**
   * List all trained models for the current user
   * GET /predictions/models
   *
   * @returns ListModelsResponse or null if no models exist
   */
  async listModels(): Promise<ListModelsResponse | null> {
    try {
      const response = await api.get<ListModelsResponse>('/predictions/models');
      return response.data;
    } catch (error) {
      // Return null if no models found (404)
      if (isNotFoundError(error)) {
        return null;
      }
      throw error;
    }
  },

  /**
   * Get information about a specific model
   * GET /predictions/models/:modelId
   *
   * @param modelId - The model ID to fetch
   * @returns ModelInfo or null if not found
   */
  async getModel(modelId: string): Promise<ModelInfo | null> {
    try {
      const response = await api.get<ModelInfo>(`/predictions/models/${modelId}`);
      return response.data;
    } catch (error) {
      if (isNotFoundError(error)) {
        return null;
      }
      throw error;
    }
  },

  /**
   * Test what-if scenarios to see how changes affect predictions
   * POST /predictions/what-if
   *
   * @param metric - The health metric to predict
   * @param scenarios - Array of hypothetical scenarios to test
   * @param targetDate - Date to predict for (default: tomorrow)
   * @returns WhatIfResponse or null if no model trained
   */
  async whatIf(
    metric: PredictionMetric,
    scenarios: WhatIfScenario[],
    targetDate?: string
  ): Promise<WhatIfResponse | null> {
    try {
      const response = await api.post<WhatIfResponse>('/predictions/what-if', {
        metric,
        scenarios,
        target_date: targetDate || getTomorrowDate(),
      });
      return response.data;
    } catch (error) {
      if (isNotFoundError(error)) {
        return null;
      }
      if (isCircuitBreakerOpen(error)) {
        throw new Error('Prediction service is temporarily unavailable. Please try again later.');
      }
      throw error;
    }
  },

  /**
   * Get dashboard predictions for primary metrics (RHR, HRV)
   * Convenience method that fetches batch predictions for display
   *
   * @returns BatchPredictResponse or null if no models trained
   */
  async getDashboardPredictions(): Promise<BatchPredictResponse | null> {
    return this.batchPredict(PRIMARY_PREDICTION_METRICS);
  },

  /**
   * Check if user has any trained models
   * Useful for showing "no model" state before attempting predictions
   *
   * @returns true if user has at least one model
   */
  async hasTrainedModels(): Promise<boolean> {
    const models = await this.listModels();
    return models !== null && models.total_models > 0;
  },

  /**
   * Check if user has a production-ready model for a specific metric
   *
   * @param metric - The metric to check
   * @returns true if a production-ready model exists
   */
  async hasProductionReadyModel(metric: PredictionMetric): Promise<boolean> {
    const models = await this.listModels();
    if (!models || models.total_models === 0) {
      return false;
    }
    return models.models.some(
      (m) => m.metric === metric && m.is_production_ready && m.is_active
    );
  },

  /**
   * Get the active model for a specific metric
   *
   * @param metric - The metric to get the model for
   * @returns ModelInfo or null if no active model
   */
  async getActiveModel(metric: PredictionMetric): Promise<ModelInfo | null> {
    const models = await this.listModels();
    if (!models || models.total_models === 0) {
      return null;
    }
    return models.models.find((m) => m.metric === metric && m.is_active) || null;
  },
};

export default predictionsApi;
