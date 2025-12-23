/**
 * USDA FoodData Central API Service
 *
 * Provides access to USDA's food database with:
 * - Rate limiting (1000 requests/hour)
 * - Exponential backoff retry logic
 * - Response caching hooks
 * - Type-safe responses
 *
 * API Documentation: https://fdc.nal.usda.gov/api-spec/fdc_api.html
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import { createChildLogger, Logger } from '../config/logger';
import {
  USDASearchOptions,
  USDASearchResponse,
  USDAFoodItem,
  USDAGetFoodOptions,
  USDAApiError,
  USDAApiRateLimitError,
  USDAApiTimeoutError,
} from '../types/usda';

// ============================================================================
// CONSTANTS
// ============================================================================

const USDA_API_BASE_URL =
  process.env.USDA_API_BASE_URL || 'https://api.nal.usda.gov/fdc/v1';
const USDA_RATE_LIMIT_PER_HOUR = parseInt(
  process.env.USDA_RATE_LIMIT_PER_HOUR || '1000',
  10
);
const USDA_REQUEST_TIMEOUT_MS = parseInt(
  process.env.USDA_REQUEST_TIMEOUT_MS || '10000',
  10
);

// Retry configuration
const MAX_RETRIES = 3;
const INITIAL_RETRY_DELAY_MS = 1000;
const MAX_RETRY_DELAY_MS = 10000;

// Default search options
const DEFAULT_PAGE_SIZE = 25;
const MAX_PAGE_SIZE = 200;

// ============================================================================
// RATE LIMITER (Token Bucket Algorithm)
// ============================================================================

class RateLimiter {
  private tokens: number;
  private maxTokens: number;
  private refillRate: number; // tokens per ms
  private lastRefill: number;

  constructor(requestsPerHour: number) {
    this.maxTokens = requestsPerHour;
    this.tokens = requestsPerHour;
    this.refillRate = requestsPerHour / (60 * 60 * 1000); // per ms
    this.lastRefill = Date.now();
  }

  private refillTokens(): void {
    const now = Date.now();
    const elapsed = now - this.lastRefill;
    const tokensToAdd = elapsed * this.refillRate;
    this.tokens = Math.min(this.maxTokens, this.tokens + tokensToAdd);
    this.lastRefill = now;
  }

  async acquire(): Promise<void> {
    this.refillTokens();

    if (this.tokens < 1) {
      const waitTime = Math.ceil((1 - this.tokens) / this.refillRate);
      await this.delay(waitTime);
      this.refillTokens();
    }

    this.tokens -= 1;
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  getAvailableTokens(): number {
    this.refillTokens();
    return Math.floor(this.tokens);
  }
}

// ============================================================================
// USDA API SERVICE
// ============================================================================

export class USDAApiService {
  private readonly client: AxiosInstance;
  private readonly rateLimiter: RateLimiter;
  private readonly log: Logger;
  private readonly apiKey: string;

  constructor() {
    this.log = createChildLogger({ service: 'USDAApiService' });
    // Read API key at construction time for testability
    this.apiKey = process.env.USDA_API_KEY || '';

    if (!this.apiKey) {
      this.log.warn(
        'USDA_API_KEY not configured. USDA API requests will fail.'
      );
    }

    this.client = axios.create({
      baseURL: USDA_API_BASE_URL,
      timeout: USDA_REQUEST_TIMEOUT_MS,
      headers: {
        'Content-Type': 'application/json',
        'X-Api-Key': this.apiKey,
      },
    });

    this.rateLimiter = new RateLimiter(USDA_RATE_LIMIT_PER_HOUR);

    // Add request interceptor for logging
    this.client.interceptors.request.use((config) => {
      this.log.debug({ url: config.url, method: config.method }, 'USDA API request');
      return config;
    });

    // Add response interceptor for logging
    this.client.interceptors.response.use(
      (response) => {
        this.log.debug(
          { url: response.config.url, status: response.status },
          'USDA API response'
        );
        return response;
      },
      (error) => {
        this.log.error(
          {
            url: error.config?.url,
            status: error.response?.status,
            message: error.message,
          },
          'USDA API error'
        );
        return Promise.reject(error);
      }
    );
  }

  // ==========================================================================
  // PUBLIC METHODS
  // ==========================================================================

  /**
   * Search for foods in USDA FoodData Central
   */
  async searchFoods(options: USDASearchOptions): Promise<USDASearchResponse> {
    const {
      query,
      dataType,
      pageNumber = 1,
      pageSize = DEFAULT_PAGE_SIZE,
      sortBy = 'dataType.keyword',
      sortOrder = 'asc',
      brandOwner,
      requireAllWords = false,
    } = options;

    if (!query || query.trim().length === 0) {
      throw new USDAApiError('Search query is required', 400, 'INVALID_QUERY');
    }

    const requestBody = {
      query: query.trim(),
      dataType: dataType,
      pageNumber,
      pageSize: Math.min(pageSize, MAX_PAGE_SIZE),
      sortBy,
      sortOrder,
      brandOwner,
      requireAllWords,
    };

    return this.executeWithRetry<USDASearchResponse>(
      () =>
        this.client.post<USDASearchResponse>('/foods/search', requestBody),
      'searchFoods'
    );
  }

  /**
   * Get a single food by FDC ID
   */
  async getFoodById(
    fdcId: number,
    options: USDAGetFoodOptions = {}
  ): Promise<USDAFoodItem> {
    if (!fdcId || fdcId <= 0) {
      throw new USDAApiError('Valid FDC ID is required', 400, 'INVALID_FDC_ID');
    }

    const { format = 'full', nutrients } = options;

    const params: Record<string, string | string[]> = {
      format,
    };

    if (nutrients && nutrients.length > 0) {
      params.nutrients = nutrients.map(String);
    }

    return this.executeWithRetry<USDAFoodItem>(
      () =>
        this.client.get<USDAFoodItem>(`/food/${fdcId}`, { params }),
      'getFoodById'
    );
  }

  /**
   * Get multiple foods by FDC IDs (bulk fetch)
   */
  async getFoodsByIds(
    fdcIds: number[],
    options: USDAGetFoodOptions = {}
  ): Promise<USDAFoodItem[]> {
    if (!fdcIds || fdcIds.length === 0) {
      throw new USDAApiError(
        'At least one FDC ID is required',
        400,
        'INVALID_FDC_IDS'
      );
    }

    // USDA API limits bulk requests to 20 items
    const MAX_BULK_SIZE = 20;
    const validIds = fdcIds.filter((id) => id > 0);

    if (validIds.length > MAX_BULK_SIZE) {
      // Batch requests if more than 20 items
      const results: USDAFoodItem[] = [];
      for (let i = 0; i < validIds.length; i += MAX_BULK_SIZE) {
        const batch = validIds.slice(i, i + MAX_BULK_SIZE);
        const batchResults = await this.fetchFoodsBatch(batch, options);
        results.push(...batchResults);
      }
      return results;
    }

    return this.fetchFoodsBatch(validIds, options);
  }

  /**
   * Get nutrients for a food (convenience method)
   */
  async getFoodNutrients(fdcId: number): Promise<USDAFoodItem> {
    return this.getFoodById(fdcId, { format: 'full' });
  }

  /**
   * Check if USDA API is configured and accessible
   */
  async healthCheck(): Promise<{ healthy: boolean; message: string }> {
    if (!this.apiKey) {
      return { healthy: false, message: 'USDA_API_KEY not configured' };
    }

    try {
      // Try a minimal search to verify API access
      await this.searchFoods({ query: 'apple', pageSize: 1 });
      return { healthy: true, message: 'USDA API is accessible' };
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Unknown error';
      return { healthy: false, message };
    }
  }

  /**
   * Get rate limiter status
   */
  getRateLimitStatus(): { available: number; max: number } {
    return {
      available: this.rateLimiter.getAvailableTokens(),
      max: USDA_RATE_LIMIT_PER_HOUR,
    };
  }

  // ==========================================================================
  // PRIVATE METHODS
  // ==========================================================================

  /**
   * Fetch a batch of foods by IDs
   */
  private async fetchFoodsBatch(
    fdcIds: number[],
    options: USDAGetFoodOptions
  ): Promise<USDAFoodItem[]> {
    const { format = 'full', nutrients } = options;

    const requestBody: Record<string, unknown> = {
      fdcIds,
      format,
    };

    if (nutrients && nutrients.length > 0) {
      requestBody.nutrients = nutrients;
    }

    return this.executeWithRetry<USDAFoodItem[]>(
      () =>
        this.client.post<USDAFoodItem[]>('/foods', requestBody),
      'getFoodsByIds'
    );
  }

  /**
   * Execute request with rate limiting and retry logic
   */
  private async executeWithRetry<T>(
    requestFn: () => Promise<{ data: T }>,
    operationName: string
  ): Promise<T> {
    let lastError: Error | null = null;
    let delay = INITIAL_RETRY_DELAY_MS;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        // Wait for rate limiter
        await this.rateLimiter.acquire();

        const response = await requestFn();
        return response.data;
      } catch (error) {
        lastError = this.handleError(error, operationName);

        // Only retry on retryable errors
        if (
          lastError instanceof USDAApiError &&
          lastError.retryable &&
          attempt < MAX_RETRIES
        ) {
          this.log.warn(
            {
              operation: operationName,
              attempt,
              delay,
              error: lastError.message,
            },
            'Retrying USDA API request'
          );

          await this.delay(delay);
          delay = Math.min(delay * 2, MAX_RETRY_DELAY_MS);
          continue;
        }

        throw lastError;
      }
    }

    throw lastError || new Error('Unknown error during USDA API request');
  }

  /**
   * Handle and transform errors
   */
  private handleError(error: unknown, operationName: string): USDAApiError {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<{ error?: string; message?: string }>;

      // Timeout error
      if (axiosError.code === 'ECONNABORTED' || axiosError.code === 'ETIMEDOUT') {
        return new USDAApiTimeoutError();
      }

      const status = axiosError.response?.status || 500;
      const responseMessage =
        axiosError.response?.data?.error ||
        axiosError.response?.data?.message ||
        axiosError.message;

      // Rate limit error
      if (status === 429) {
        const retryAfter = parseInt(
          axiosError.response?.headers?.['retry-after'] || '60',
          10
        );
        return new USDAApiRateLimitError(
          `Rate limit exceeded for ${operationName}`,
          retryAfter * 1000
        );
      }

      // Server errors (5xx) are retryable
      if (status >= 500) {
        return new USDAApiError(
          `USDA API server error: ${responseMessage}`,
          status,
          'USDA_SERVER_ERROR',
          true
        );
      }

      // Client errors (4xx) are not retryable
      return new USDAApiError(
        `USDA API error: ${responseMessage}`,
        status,
        'USDA_CLIENT_ERROR',
        false
      );
    }

    // Unknown error type
    if (error instanceof Error) {
      return new USDAApiError(
        `USDA API error: ${error.message}`,
        500,
        'USDA_UNKNOWN_ERROR',
        false
      );
    }

    return new USDAApiError(
      'Unknown USDA API error',
      500,
      'USDA_UNKNOWN_ERROR',
      false
    );
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// Export singleton instance
export const usdaApiService = new USDAApiService();
