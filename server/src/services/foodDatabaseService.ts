/**
 * Food Database Service
 *
 * Provides unified access to USDA FoodData Central with:
 * - Transformed food data matching app schema
 * - Search with filtering and pagination
 * - Popular and recent foods
 */

import { usdaApiService } from './usdaApiService';
import { nutrientMappingService } from './nutrientMappingService';
import { foodCacheService } from './foodCacheService';
import {
  USDADataType,
  USDASearchOptions,
  TransformedUSDAFood,
} from '../types/usda';
import { createChildLogger, Logger } from '../config/logger';
import {
  rankSearchResults,
  isWholeFoodQuery,
  isBrandedQuery,
  type RankingHints,
  type RankedFood,
} from '../utils/foodRanking';

// ============================================================================
// TYPES
// ============================================================================

export interface FoodSearchOptions {
  query: string;
  page?: number;
  limit?: number;
  dataType?: string[];
  sortBy?: 'dataType.keyword' | 'description' | 'fdcId' | 'publishedDate';
  sortOrder?: 'asc' | 'desc';
  brandOwner?: string;
  /** Enable intelligent ranking based on data quality and relevance */
  enableRanking?: boolean;
  /** Classification hints for improved ranking */
  rankingHints?: RankingHints;
}

export interface FoodSearchResult {
  foods: TransformedUSDAFood[] | RankedFood[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNextPage: boolean;
    hasPrevPage: boolean;
  };
  /** Indicates if results are ranked */
  isRanked?: boolean;
}

export interface PopularFood {
  fdcId: number;
  name: string;
  description: string;
  dataType: USDADataType;
  category?: string;
  calories: number;
}

// ============================================================================
// POPULAR FOODS (STATIC LIST)
// ============================================================================

/**
 * Pre-defined list of popular foods for quick access
 * These are Foundation/SR Legacy foods with reliable nutrition data
 */
const POPULAR_FOOD_IDS = [
  171688, // Apples, raw, with skin
  170567, // Bananas, raw
  170182, // Chicken breast, boneless, skinless
  171705, // Eggs, whole, raw
  170903, // Rice, white, long-grain, regular, cooked
  174270, // Bread, whole-wheat
  170379, // Broccoli, raw
  170393, // Carrots, raw
  170556, // Avocados, raw
  171287, // Salmon, Atlantic, wild, raw
  173757, // Milk, whole
  170457, // Spinach, raw
  169655, // Oats, regular and quick
  175167, // Almonds, raw
  175215, // Peanut butter, smooth
  167770, // Pasta, cooked
  170042, // Beef, ground, 85% lean
  168917, // Yogurt, Greek, plain
  173420, // Sweet potato, raw
  171693, // Orange juice, raw
];

// ============================================================================
// FOOD DATABASE SERVICE
// ============================================================================

export class FoodDatabaseService {
  private readonly log: Logger;

  constructor() {
    this.log = createChildLogger({ service: 'FoodDatabaseService' });
  }

  // ==========================================================================
  // PUBLIC METHODS
  // ==========================================================================

  /**
   * Search for foods in USDA database
   */
  async searchFoods(options: FoodSearchOptions): Promise<FoodSearchResult> {
    const {
      query,
      page = 1,
      limit = 25,
      dataType,
      sortBy,
      sortOrder,
      brandOwner,
      enableRanking = true, // Enable by default
      rankingHints,
    } = options;

    // Check cache first
    const cached = await foodCacheService.getCachedSearch(query, page, limit, dataType);
    if (cached) {
      this.log.debug({ query, page }, 'Returning cached search results');
      return cached;
    }

    // Validate and transform data types
    const validDataTypes = this.validateDataTypes(dataType);

    const searchOptions: USDASearchOptions = {
      query,
      pageNumber: page,
      pageSize: limit,
      dataType: validDataTypes,
      sortBy,
      sortOrder,
      brandOwner,
    };

    const response = await usdaApiService.searchFoods(searchOptions);

    // Transform foods to app format
    const transformedFoods = response.foods.map((food) =>
      nutrientMappingService.transformSearchResultFood(food)
    );

    // Apply ranking if enabled
    let finalFoods: TransformedUSDAFood[] | RankedFood[] = transformedFoods;
    let isRanked = false;

    if (enableRanking && transformedFoods.length > 0) {
      // Build ranking hints from query if not provided
      const hints: RankingHints = rankingHints || {
        query,
        isWholeFoodQuery: isWholeFoodQuery(query),
        isBrandedQuery: isBrandedQuery(query),
      };

      finalFoods = rankSearchResults(transformedFoods, hints);
      isRanked = true;

      this.log.debug(
        { query, resultsCount: finalFoods.length },
        'Applied ranking to search results'
      );
    }

    const result: FoodSearchResult = {
      foods: finalFoods,
      pagination: {
        page: response.currentPage,
        limit,
        total: response.totalHits,
        totalPages: response.totalPages,
        hasNextPage: response.currentPage < response.totalPages,
        hasPrevPage: response.currentPage > 1,
      },
      isRanked,
    };

    // Cache the results (cache transformed foods, ranking can be re-applied)
    await foodCacheService.cacheSearchResults(
      query,
      page,
      limit,
      dataType,
      transformedFoods,
      result.pagination
    );

    return result;
  }

  /**
   * Get a single food by FDC ID
   */
  async getFoodById(fdcId: number): Promise<TransformedUSDAFood> {
    // Check cache first
    const cached = await foodCacheService.getCachedFood(fdcId);
    if (cached) {
      this.log.debug({ fdcId }, 'Returning cached food');
      return cached;
    }

    const food = await usdaApiService.getFoodById(fdcId);
    const transformed = nutrientMappingService.transformUSDAFood(food);

    // Cache the result
    await foodCacheService.cacheFood(transformed);

    return transformed;
  }

  /**
   * Get multiple foods by FDC IDs
   */
  async getFoodsByIds(fdcIds: number[]): Promise<TransformedUSDAFood[]> {
    const foods = await usdaApiService.getFoodsByIds(fdcIds);
    return foods.map((food) => nutrientMappingService.transformUSDAFood(food));
  }

  /**
   * Get nutrients for a food by FDC ID
   */
  async getFoodNutrients(fdcId: number): Promise<TransformedUSDAFood['nutrients']> {
    // Check cache first
    const cached = await foodCacheService.getCachedNutrients(fdcId);
    if (cached) {
      this.log.debug({ fdcId }, 'Returning cached nutrients');
      return cached;
    }

    const food = await usdaApiService.getFoodNutrients(fdcId);
    const nutrients = nutrientMappingService.mapUSDANutrients(food.foodNutrients);

    // Cache the result
    await foodCacheService.cacheNutrients(fdcId, nutrients);

    return nutrients;
  }

  /**
   * Get nutrients scaled to a specific serving size
   */
  async getScaledNutrients(
    fdcId: number,
    servingGrams: number
  ): Promise<TransformedUSDAFood['nutrients']> {
    const nutrients = await this.getFoodNutrients(fdcId);
    return nutrientMappingService.scaleNutrients(nutrients, servingGrams);
  }

  /**
   * Get list of popular foods
   * Returns a pre-defined list of common, nutritious foods
   */
  async getPopularFoods(): Promise<PopularFood[]> {
    try {
      // Check cache first
      const cached = await foodCacheService.getCachedPopularFoods();
      if (cached) {
        this.log.debug('Returning cached popular foods');
        return cached.map((food) => ({
          fdcId: food.fdcId,
          name: food.name,
          description: food.description,
          dataType: food.dataType,
          category: food.category,
          calories: food.nutrients.calories,
        }));
      }

      const foods = await this.getFoodsByIds(POPULAR_FOOD_IDS);

      // Cache the full food data
      await foodCacheService.cachePopularFoods(foods);

      return foods.map((food) => ({
        fdcId: food.fdcId,
        name: food.name,
        description: food.description,
        dataType: food.dataType,
        category: food.category,
        calories: food.nutrients.calories,
      }));
    } catch (error) {
      this.log.error({ error }, 'Failed to fetch popular foods');
      // Return empty array if USDA API fails
      return [];
    }
  }

  /**
   * Search with classification hints (for hybrid ML + search)
   * Enhances search based on visual classification results
   */
  async searchWithHints(
    query: string,
    hints: {
      coarseCategory?: string;
      finegrainedSuggestions?: string[];
      brandDetected?: string;
    }
  ): Promise<FoodSearchResult> {
    // Enhance query with classification hints
    let enhancedQuery = query;

    // Add fine-grained suggestions to query if available
    if (hints.finegrainedSuggestions?.length) {
      const topSuggestion = hints.finegrainedSuggestions[0];
      enhancedQuery = `${topSuggestion} ${query}`.trim();
    }

    // Determine appropriate data types based on category
    const dataTypes = this.getDataTypesForCategory(hints.coarseCategory);

    // If brand detected, include branded foods
    if (hints.brandDetected) {
      if (!dataTypes.includes('Branded')) {
        dataTypes.push('Branded');
      }
    }

    // Build ranking hints from classification
    const rankingHints: RankingHints = {
      query: enhancedQuery,
      category: hints.coarseCategory,
      subcategoryHints: hints.finegrainedSuggestions,
      isWholeFoodQuery: !hints.brandDetected && isWholeFoodQuery(query),
      isBrandedQuery: !!hints.brandDetected || isBrandedQuery(query),
    };

    return this.searchFoods({
      query: enhancedQuery,
      dataType: dataTypes,
      limit: 10,
      enableRanking: true,
      rankingHints,
    });
  }

  /**
   * Check USDA API health
   */
  async checkHealth(): Promise<{ healthy: boolean; message: string }> {
    return usdaApiService.healthCheck();
  }

  /**
   * Get rate limit status
   */
  getRateLimitStatus(): { available: number; max: number } {
    return usdaApiService.getRateLimitStatus();
  }

  // ==========================================================================
  // PRIVATE METHODS
  // ==========================================================================

  /**
   * Validate and filter data types
   */
  private validateDataTypes(dataTypes?: string[]): USDADataType[] | undefined {
    if (!dataTypes || dataTypes.length === 0) {
      return undefined;
    }

    const validTypes: USDADataType[] = [
      'Foundation',
      'SR Legacy',
      'Survey (FNDDS)',
      'Branded',
      'Experimental',
    ];

    const filtered = dataTypes.filter((type) =>
      validTypes.includes(type as USDADataType)
    ) as USDADataType[];

    return filtered.length > 0 ? filtered : undefined;
  }

  /**
   * Get appropriate data types for a food category
   */
  private getDataTypesForCategory(category?: string): string[] {
    if (!category) {
      return ['Foundation', 'SR Legacy'];
    }

    const categoryLower = category.toLowerCase();

    // Whole/unprocessed foods - prefer Foundation and SR Legacy
    if (
      categoryLower.includes('fresh') ||
      categoryLower.includes('raw') ||
      categoryLower.includes('whole')
    ) {
      return ['Foundation', 'SR Legacy'];
    }

    // Mixed dishes and prepared foods
    if (
      categoryLower.includes('mixed') ||
      categoryLower.includes('prepared') ||
      categoryLower.includes('dish')
    ) {
      return ['Survey (FNDDS)', 'Branded'];
    }

    // Fast food and restaurant items
    if (
      categoryLower.includes('fast_food') ||
      categoryLower.includes('restaurant')
    ) {
      return ['Branded', 'Survey (FNDDS)'];
    }

    // Snacks and packaged foods
    if (
      categoryLower.includes('snack') ||
      categoryLower.includes('packaged')
    ) {
      return ['Branded'];
    }

    // Beverages
    if (
      categoryLower.includes('beverage') ||
      categoryLower.includes('drink')
    ) {
      return ['Foundation', 'SR Legacy', 'Branded'];
    }

    // Default to whole foods
    return ['Foundation', 'SR Legacy'];
  }
}

// Export singleton instance
export const foodDatabaseService = new FoodDatabaseService();
