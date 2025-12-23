/**
 * Foods API Client
 *
 * API client for USDA FoodData Central integration
 */

import api from './client';
import {
  USDAFood,
  FoodSearchResult,
  FoodSearchOptions,
  FoodNutrients,
  ClassifyAndSearchResult,
  FoodFeedbackInput,
  FoodFeedbackStats,
} from '../types/foods';

export const foodsApi = {
  /**
   * Search for foods in USDA database
   */
  async searchFoods(options: FoodSearchOptions): Promise<FoodSearchResult> {
    const params: Record<string, string | number | undefined> = {
      q: options.query,
      page: options.page,
      limit: options.limit,
      sortBy: options.sortBy,
      sortOrder: options.sortOrder,
      brandOwner: options.brandOwner,
    };

    // Handle dataType array
    if (options.dataType && options.dataType.length > 0) {
      params.dataType = options.dataType.join(',');
    }

    const response = await api.get<FoodSearchResult>('/foods/search', { params });
    return response.data;
  },

  /**
   * Get a single food by FDC ID
   */
  async getFoodById(fdcId: number): Promise<USDAFood> {
    const response = await api.get<USDAFood>(`/foods/${fdcId}`);
    return response.data;
  },

  /**
   * Get nutrients for a food, optionally scaled
   */
  async getFoodNutrients(fdcId: number, grams?: number): Promise<FoodNutrients> {
    const params = grams ? { grams } : undefined;
    const response = await api.get<FoodNutrients>(`/foods/${fdcId}/nutrients`, { params });
    return response.data;
  },

  /**
   * Get multiple foods by IDs
   */
  async getBulkFoods(
    fdcIds: number[],
    options?: { format?: 'abridged' | 'full'; nutrients?: number[] }
  ): Promise<USDAFood[]> {
    const response = await api.post<USDAFood[]>('/foods/bulk', {
      fdcIds,
      ...options,
    });
    return response.data;
  },

  /**
   * Get popular foods
   */
  async getPopularFoods(): Promise<USDAFood[]> {
    const response = await api.get<USDAFood[]>('/foods/popular');
    return response.data;
  },

  /**
   * Get user's recent food selections
   */
  async getRecentFoods(): Promise<USDAFood[]> {
    const response = await api.get<USDAFood[]>('/foods/recent');
    return response.data;
  },

  /**
   * Record a food selection (adds to recent foods)
   */
  async recordFoodSelection(fdcId: number): Promise<void> {
    await api.post(`/foods/${fdcId}/select`);
  },

  /**
   * Classify an image and search for matching foods
   */
  async classifyAndSearch(
    imageBase64: string,
    options?: {
      query?: string;
      dimensions?: { width: number; height: number; depth?: number };
    }
  ): Promise<ClassifyAndSearchResult> {
    const formData = new FormData();

    // Convert base64 to blob for file upload (using Uint8Array.from for efficiency)
    const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, '');
    const byteCharacters = atob(base64Data);
    const byteArray = Uint8Array.from(byteCharacters, (char) => char.charCodeAt(0));
    const blob = new Blob([byteArray], { type: 'image/jpeg' });

    formData.append('image', blob, 'food.jpg');

    if (options?.query) {
      formData.append('query', options.query);
    }
    if (options?.dimensions) {
      formData.append('dimensions', JSON.stringify(options.dimensions));
    }

    const response = await api.post<ClassifyAndSearchResult>(
      '/food/classify-and-search',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout for classification
      }
    );
    return response.data;
  },

  /**
   * Submit feedback for food classification
   */
  async submitFeedback(feedback: FoodFeedbackInput): Promise<{
    success: boolean;
    feedbackId?: string;
    isDuplicate?: boolean;
    patternFlagged?: boolean;
    message: string;
  }> {
    const response = await api.post('/foods/feedback', feedback);
    return response.data;
  },

  /**
   * Get feedback statistics
   */
  async getFeedbackStats(): Promise<FoodFeedbackStats> {
    const response = await api.get<FoodFeedbackStats>('/foods/feedback/stats');
    return response.data;
  },

  /**
   * Check USDA API health
   */
  async healthCheck(): Promise<{
    status: string;
    usdaApi: boolean;
    redis?: boolean;
    cacheStats?: { hits: number; misses: number; hitRate: number };
  }> {
    const response = await api.get('/foods/health');
    return response.data;
  },
};
