import axios, { AxiosError } from 'axios';
import {
  FoodAnalysisRequest,
  FoodAnalysisResponse,
  FoodAnalysisErrorResponse,
  MLServiceConfig,
} from '@/lib/types/food-analysis';

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
      // Create form data
      const formData = new FormData();
      formData.append('image', {
        uri: request.imageUri,
        type: 'image/jpeg',
        name: 'food.jpg',
      } as unknown as Blob);

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

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const response = await axios({
          ...config,
          url: `${this.baseUrl}${endpoint}`,
        });

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
  private handleError(error: unknown): FoodAnalysisErrorResponse {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;

      if (axiosError.code === 'ECONNABORTED') {
        return {
          error: 'timeout',
          message: 'Food analysis request timed out. Please try again.',
          retryable: true,
        };
      }

      if (axiosError.code === 'ERR_NETWORK') {
        return {
          error: 'network-error',
          message:
            'Network error. Please check your connection and try again.',
          retryable: true,
        };
      }

      if (axiosError.response?.status === 403 || axiosError.response?.status === 401) {
        return {
          error: 'permission-denied',
          message: 'Permission denied. Please check your credentials.',
          retryable: false,
        };
      }

      if (axiosError.response?.status === 400) {
        return {
          error: 'invalid-image',
          message: 'Invalid image. Please try taking a clearer photo.',
          retryable: false,
        };
      }

      if (axiosError.response?.status && axiosError.response.status >= 500) {
        return {
          error: 'analysis-failed',
          message: 'Food analysis failed. Please try again later.',
          retryable: true,
        };
      }
    }

    return {
      error: 'unknown',
      message: 'An unexpected error occurred. Please try again.',
      retryable: true,
    };
  }
}

// Export singleton instance
export const foodAnalysisApi = new FoodAnalysisAPI(config);

// Export class for testing
export { FoodAnalysisAPI };
