import axios, { AxiosError } from 'axios';
import api from './client';
import {
  FoodAnalysisRequest,
  FoodAnalysisResponse,
  FoodAnalysisErrorResponse,
  MLServiceConfig,
  FoodItem,
  NutritionInfo,
  USDAClassificationResult,
  ARMeasurement,
} from '@/lib/types/food-analysis';

/**
 * Transform ML service response (snake_case) to TypeScript types (camelCase)
 */
function transformMLResponse(data: Record<string, unknown>): FoodAnalysisResponse {
  const foodItems = (data.food_items as Record<string, unknown>[]) || [];

  return {
    foodItems: foodItems.map((item): FoodItem => {
      const nutrition = item.nutrition as Record<string, unknown>;
      return {
        name: item.name as string,
        confidence: item.confidence as number,
        portionSize: item.portion_size as string,
        portionWeight: item.portion_weight as number,
        nutrition: {
          calories: nutrition.calories as number,
          protein: nutrition.protein as number,
          carbs: nutrition.carbs as number,
          fat: nutrition.fat as number,
          fiber: nutrition.fiber as number | undefined,
          sugar: nutrition.sugar as number | undefined,
          sodium: nutrition.sodium as number | undefined,
          saturatedFat: nutrition.saturated_fat as number | undefined,
          lysine: nutrition.lysine as number | undefined,
          arginine: nutrition.arginine as number | undefined,
        } as NutritionInfo,
        category: item.category as string | undefined,
        alternatives: item.alternatives ? (item.alternatives as Record<string, unknown>[]).map(alt => ({
          name: alt.name as string,
          confidence: alt.confidence as number,
        })) : undefined,
      };
    }),
    measurementQuality: data.measurement_quality as 'high' | 'medium' | 'low',
    processingTime: data.processing_time as number,
    suggestions: data.suggestions as string[] | undefined,
    error: data.error as string | undefined,
    imageHash: data.image_hash as string | undefined,
  };
}

/**
 * Custom error class for Food Analysis errors
 */
export class FoodAnalysisError extends Error {
  constructor(
    message: string,
    public readonly error: FoodAnalysisErrorResponse['error'],
    public readonly retryable: boolean
  ) {
    super(message);
    this.name = 'FoodAnalysisError';
  }
}

// Food Analysis configuration
// Uses the backend API which proxies to the ML service
const config: MLServiceConfig = {
  baseUrl: '', // Will use the api client's baseURL
  timeout: 30000, // 30 seconds
  maxRetries: 2,
};

/**
 * Food Analysis API client
 * Routes requests through the backend API which proxies to the ML service
 */
class FoodAnalysisAPI {
  private timeout: number;
  private maxRetries: number;

  constructor(cfg: MLServiceConfig) {
    this.timeout = cfg.timeout;
    this.maxRetries = cfg.maxRetries;
  }

  /**
   * Analyze food image and estimate nutrition
   */
  async analyzeFood(
    request: FoodAnalysisRequest
  ): Promise<FoodAnalysisResponse> {
    try {
      // Determine file extension and MIME type from URI
      const extension = request.imageUri.split('.').pop()?.toLowerCase() || 'jpg';
      const mimeType = extension === 'png' ? 'image/png' : 'image/jpeg';

      // Create form data for the backend API
      const formData = new FormData();

      // Create a blob-like object for React Native
      // React Native's FormData accepts objects with uri, type, name properties
      const imageFile = {
        uri: request.imageUri,
        type: mimeType,
        name: `food.${extension}`,
      } as unknown as Blob;

      formData.append('image', imageFile);

      // Add measurements if available
      if (request.measurements) {
        formData.append('dimensions', JSON.stringify({
          width: request.measurements.width,
          height: request.measurements.height,
          depth: request.measurements.depth,
        }));
      }

      // Add cooking method if specified
      if (request.cookingMethod) {
        formData.append('cooking_method', request.cookingMethod);
      }

      // Make request through backend API with retry logic
      const response = await this.makeRequestWithRetry<Record<string, unknown>>(
        '/food/analyze',
        {
          method: 'POST',
          data: formData,
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: this.timeout,
        }
      );

      // Transform snake_case ML service response to camelCase TypeScript types
      return transformMLResponse(response.data);
    } catch (error) {
      // Log the actual error for debugging
      console.error('Food analysis request failed:', error);
      if (axios.isAxiosError(error)) {
        console.error('Axios error details:', {
          status: error.response?.status,
          data: error.response?.data,
          message: error.message,
          code: error.code,
        });
      }
      throw this.handleError(error);
    }
  }

  /**
   * Search nutrition database by food name
   */
  async searchNutritionDB(query: string): Promise<unknown[]> {
    try {
      const response = await this.makeRequestWithRetry<{ results: unknown[] }>(
        '/food/nutrition-db/search',
        {
          method: 'GET',
          params: { q: query },
          timeout: 10000,
        }
      );

      return response.data.results;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get ML models info
   */
  async getModelsInfo(): Promise<{
    available_models: string[];
    active_model: string;
  }> {
    try {
      const response = await this.makeRequestWithRetry<{
        available_models: string[];
        active_model: string;
      }>('/food/models/info', {
        method: 'GET',
        timeout: 5000,
      });

      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get supported cooking methods
   */
  async getCookingMethods(): Promise<string[]> {
    try {
      const response = await this.makeRequestWithRetry<{
        cooking_methods: string[];
        total: number;
      }>('/food/cooking-methods', {
        method: 'GET',
        timeout: 5000,
      });

      return response.data.cooking_methods;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Classify food image and search USDA database
   * Uses the classify-and-search endpoint for combined ML + USDA lookup
   */
  async classifyAndSearch(
    imageUri: string,
    measurements?: ARMeasurement
  ): Promise<USDAClassificationResult> {
    try {
      // Determine file extension and MIME type from URI
      const extension = imageUri.split('.').pop()?.toLowerCase() || 'jpg';
      const mimeType = extension === 'png' ? 'image/png' : 'image/jpeg';

      // Create form data for the backend API
      const formData = new FormData();

      // Create a blob-like object for React Native
      const imageFile = {
        uri: imageUri,
        type: mimeType,
        name: `food.${extension}`,
      } as unknown as Blob;

      formData.append('image', imageFile);

      // Add dimensions if available from AR measurement
      if (measurements) {
        formData.append('dimensions', JSON.stringify({
          width: measurements.width,
          height: measurements.height,
          depth: measurements.depth,
        }));
      }

      // Make request through backend API
      const response = await this.makeRequestWithRetry<USDAClassificationResult>(
        '/food/classify-and-search',
        {
          method: 'POST',
          data: formData,
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: this.timeout,
        }
      );

      return response.data;
    } catch (error) {
      console.error('Classify and search request failed:', error);
      if (axios.isAxiosError(error)) {
        console.error('Axios error details:', {
          status: error.response?.status,
          data: error.response?.data,
          message: error.message,
          code: error.code,
        });
      }
      throw this.handleError(error);
    }
  }

  /**
   * Check food analysis service health (through backend)
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await api.get('/food/health', {
        timeout: 5000,
      });
      return response.status === 200 && response.data?.status === 'healthy';
    } catch {
      return false;
    }
  }

  /**
   * Make request with retry logic using the backend API client
   */
  private async makeRequestWithRetry<T>(
    endpoint: string,
    config: {
      method: string;
      data?: unknown;
      params?: unknown;
      headers?: Record<string, string>;
      timeout: number;
    }
  ): Promise<{ data: T }> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        let response;
        const axiosConfig = {
          headers: config.headers,
          timeout: config.timeout,
          params: config.params,
        };

        // Use the api client (already configured with auth and base URL)
        if (config.method === 'POST') {
          response = await api.post(endpoint, config.data, axiosConfig);
        } else if (config.method === 'GET') {
          response = await api.get(endpoint, axiosConfig);
        } else if (config.method === 'PUT') {
          response = await api.put(endpoint, config.data, axiosConfig);
        } else if (config.method === 'DELETE') {
          response = await api.delete(endpoint, axiosConfig);
        } else {
          response = await api.request({
            ...config,
            url: endpoint,
          });
        }

        return { data: response.data as T };
      } catch (error) {
        lastError = error as Error;

        // Don't retry on client errors (4xx)
        if (axios.isAxiosError(error) && error.response?.status) {
          const status = error.response.status;
          if (status >= 400 && status < 500) {
            throw error;
          }
        }

        // Wait before retry (exponential backoff)
        if (attempt < this.maxRetries - 1) {
          await new Promise((resolve) =>
            setTimeout(resolve, Math.pow(2, attempt) * 1000)
          );
        }
      }
    }

    throw lastError;
  }

  /**
   * Handle API errors
   */
  private handleError(error: unknown): FoodAnalysisError {
    // Check if it's an axios error (works with both real axios and mocked errors)
    const isAxios = axios.isAxiosError(error) || (error && typeof error === 'object' && 'isAxiosError' in error);

    if (isAxios) {
      const axiosError = error as AxiosError;

      // Timeout errors
      if (axiosError.code === 'ECONNABORTED') {
        return new FoodAnalysisError(
          'Food analysis request timed out. Please try again.',
          'timeout',
          true
        );
      }

      // Network errors
      if (axiosError.code === 'ERR_NETWORK') {
        return new FoodAnalysisError(
          'Network error. Please check your connection and try again.',
          'network-error',
          true
        );
      }

      // Permission errors (401/403)
      if (axiosError.response?.status === 403 || axiosError.response?.status === 401) {
        return new FoodAnalysisError(
          'Permission denied. Please check your credentials.',
          'permission-denied',
          false
        );
      }

      // Invalid image (400)
      if (axiosError.response?.status === 400) {
        return new FoodAnalysisError(
          'Invalid image. Please try taking a clearer photo.',
          'invalid-image',
          false
        );
      }

      // Server errors (5xx)
      if (axiosError.response?.status && axiosError.response.status >= 500) {
        return new FoodAnalysisError(
          'Food analysis failed. Please try again later.',
          'analysis-failed',
          true
        );
      }
    }

    return new FoodAnalysisError(
      'An unexpected error occurred. Please try again.',
      'unknown',
      true
    );
  }
}

// Export singleton instance
export const foodAnalysisApi = new FoodAnalysisAPI(config);

// Export class for testing
export { FoodAnalysisAPI };
