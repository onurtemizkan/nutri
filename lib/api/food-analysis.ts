import axios, { AxiosError } from 'axios';
import * as FileSystem from 'expo-file-system';
import {
  FoodAnalysisRequest,
  FoodAnalysisResponse,
  FoodAnalysisErrorResponse,
  MLServiceConfig,
} from '@/lib/types/food-analysis';

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

// ML Service configuration
const ML_SERVICE_URL = __DEV__
  ? 'http://localhost:8000' // Development - ML service running locally
  : 'https://your-ml-service.com'; // Production

const config: MLServiceConfig = {
  baseUrl: ML_SERVICE_URL,
  timeout: 30000, // 30 seconds
  maxRetries: 2,
};

/**
 * Food Analysis API client
 */
class FoodAnalysisAPI {
  private baseUrl: string;
  private timeout: number;
  private maxRetries: number;

  constructor(cfg: MLServiceConfig) {
    this.baseUrl = cfg.baseUrl;
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
      // Read image file as base64
      const imageBase64 = await FileSystem.readAsStringAsync(request.imageUri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // Create form data
      const formData = new FormData();
      formData.append('image', imageBase64);

      // Add measurements if available
      if (request.measurements) {
        formData.append('dimensions', JSON.stringify({
          width: request.measurements.width,
          height: request.measurements.height,
          depth: request.measurements.depth,
        }));
        formData.append('confidence', request.measurements.confidence);
      }

      // Make request with retry logic
      const response = await this.makeRequestWithRetry<FoodAnalysisResponse>(
        '/api/food/analyze',
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
      throw this.handleError(error);
    }
  }

  /**
   * Search nutrition database by food name
   */
  async searchNutritionDB(query: string): Promise<unknown[]> {
    try {
      const response = await this.makeRequestWithRetry<{ results: unknown[] }>(
        '/api/food/nutrition-db/search',
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
      }>('/api/food/models/info', {
        method: 'GET',
        timeout: 5000,
      });

      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Check ML service health
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await axios.get(`${this.baseUrl}/health`, {
        timeout: 5000,
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  /**
   * Make request with retry logic
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
    const url = `${this.baseUrl}${endpoint}`;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        let response;
        const axiosConfig = {
          headers: config.headers,
          timeout: config.timeout,
          params: config.params,
        };

        // Use specific axios methods based on HTTP method
        if (config.method === 'POST') {
          response = await axios.post(url, config.data, axiosConfig);
        } else if (config.method === 'GET') {
          response = await axios.get(url, axiosConfig);
        } else if (config.method === 'PUT') {
          response = await axios.put(url, config.data, axiosConfig);
        } else if (config.method === 'DELETE') {
          response = await axios.delete(url, axiosConfig);
        } else {
          // Fallback to generic axios call
          response = await axios({
            ...config,
            url,
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
