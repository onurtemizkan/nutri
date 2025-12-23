/**
 * Food Cache Service
 *
 * Multi-tier caching for USDA food data:
 * - Search results: 1-hour TTL
 * - Individual foods: 24-hour TTL
 * - User recent foods: 30-day TTL (sorted set)
 * - Classification results: 1-hour TTL
 */

import { createHash } from 'crypto';
import { getRedisClient, isRedisAvailable } from '../config/redis';
import { TransformedUSDAFood, TransformedNutrients } from '../types/usda';
import { createChildLogger, Logger } from '../config/logger';

// ============================================================================
// CONSTANTS
// ============================================================================

// TTL values in seconds
const TTL_SEARCH_RESULTS = 3600; // 1 hour
const TTL_FOOD_DATA = 86400; // 24 hours
const TTL_USER_RECENT = 2592000; // 30 days
const TTL_CLASSIFICATION = 3600; // 1 hour

// Cache key prefixes
const KEY_PREFIX = 'nutri:food:';
const SEARCH_PREFIX = `${KEY_PREFIX}search:`;
const FOOD_PREFIX = `${KEY_PREFIX}item:`;
const NUTRIENTS_PREFIX = `${KEY_PREFIX}nutrients:`;
const USER_RECENT_PREFIX = `${KEY_PREFIX}user:`;
const CLASSIFY_PREFIX = `${KEY_PREFIX}classify:`;
const POPULAR_KEY = `${KEY_PREFIX}popular`;

// ============================================================================
// TYPES
// ============================================================================

interface CachedSearchResult {
  foods: TransformedUSDAFood[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNextPage: boolean;
    hasPrevPage: boolean;
  };
  cachedAt: number;
}

interface CacheStats {
  hits: number;
  misses: number;
  hitRate: number;
}

// ============================================================================
// FOOD CACHE SERVICE
// ============================================================================

export class FoodCacheService {
  private readonly log: Logger;
  private stats = { hits: 0, misses: 0 };

  constructor() {
    this.log = createChildLogger({ service: 'FoodCacheService' });
  }

  // ==========================================================================
  // SEARCH RESULT CACHING
  // ==========================================================================

  /**
   * Generate cache key for search query
   */
  private generateSearchKey(
    query: string,
    page: number,
    limit: number,
    dataTypes?: string[]
  ): string {
    const normalizedQuery = query.toLowerCase().trim();
    const dataTypeStr = dataTypes ? dataTypes.sort().join(',') : 'all';
    const keyData = `${normalizedQuery}:${page}:${limit}:${dataTypeStr}`;
    const hash = createHash('sha256').update(keyData).digest('hex').slice(0, 16);
    return `${SEARCH_PREFIX}${hash}`;
  }

  /**
   * Get cached search results
   */
  async getCachedSearch(
    query: string,
    page: number,
    limit: number,
    dataTypes?: string[]
  ): Promise<CachedSearchResult | null> {
    if (!isRedisAvailable()) {
      return null;
    }

    const key = this.generateSearchKey(query, page, limit, dataTypes);
    const redis = getRedisClient();

    try {
      const cached = await redis?.get(key);
      if (cached) {
        this.stats.hits++;
        this.log.debug({ key }, 'Search cache hit');
        return JSON.parse(cached);
      }
      this.stats.misses++;
      return null;
    } catch (error) {
      this.log.warn({ error, key }, 'Error reading search cache');
      return null;
    }
  }

  /**
   * Cache search results
   */
  async cacheSearchResults(
    query: string,
    page: number,
    limit: number,
    dataTypes: string[] | undefined,
    result: CachedSearchResult['foods'],
    pagination: CachedSearchResult['pagination']
  ): Promise<void> {
    if (!isRedisAvailable()) {
      return;
    }

    const key = this.generateSearchKey(query, page, limit, dataTypes);
    const redis = getRedisClient();

    const cacheData: CachedSearchResult = {
      foods: result,
      pagination,
      cachedAt: Date.now(),
    };

    try {
      await redis?.setex(key, TTL_SEARCH_RESULTS, JSON.stringify(cacheData));
      this.log.debug({ key, ttl: TTL_SEARCH_RESULTS }, 'Search results cached');
    } catch (error) {
      this.log.warn({ error, key }, 'Error caching search results');
    }
  }

  // ==========================================================================
  // INDIVIDUAL FOOD CACHING
  // ==========================================================================

  /**
   * Get cached food by FDC ID
   */
  async getCachedFood(fdcId: number): Promise<TransformedUSDAFood | null> {
    if (!isRedisAvailable()) {
      return null;
    }

    const key = `${FOOD_PREFIX}${fdcId}`;
    const redis = getRedisClient();

    try {
      const cached = await redis?.get(key);
      if (cached) {
        this.stats.hits++;
        this.log.debug({ fdcId }, 'Food cache hit');
        return JSON.parse(cached);
      }
      this.stats.misses++;
      return null;
    } catch (error) {
      this.log.warn({ error, fdcId }, 'Error reading food cache');
      return null;
    }
  }

  /**
   * Cache individual food
   */
  async cacheFood(food: TransformedUSDAFood): Promise<void> {
    if (!isRedisAvailable()) {
      return;
    }

    const key = `${FOOD_PREFIX}${food.fdcId}`;
    const redis = getRedisClient();

    try {
      await redis?.setex(key, TTL_FOOD_DATA, JSON.stringify(food));
      this.log.debug({ fdcId: food.fdcId, ttl: TTL_FOOD_DATA }, 'Food cached');
    } catch (error) {
      this.log.warn({ error, fdcId: food.fdcId }, 'Error caching food');
    }
  }

  /**
   * Get cached nutrients for a food
   */
  async getCachedNutrients(fdcId: number): Promise<TransformedNutrients | null> {
    if (!isRedisAvailable()) {
      return null;
    }

    const key = `${NUTRIENTS_PREFIX}${fdcId}`;
    const redis = getRedisClient();

    try {
      const cached = await redis?.get(key);
      if (cached) {
        this.stats.hits++;
        return JSON.parse(cached);
      }
      this.stats.misses++;
      return null;
    } catch (error) {
      this.log.warn({ error, fdcId }, 'Error reading nutrients cache');
      return null;
    }
  }

  /**
   * Cache nutrients for a food
   */
  async cacheNutrients(fdcId: number, nutrients: TransformedNutrients): Promise<void> {
    if (!isRedisAvailable()) {
      return;
    }

    const key = `${NUTRIENTS_PREFIX}${fdcId}`;
    const redis = getRedisClient();

    try {
      await redis?.setex(key, TTL_FOOD_DATA, JSON.stringify(nutrients));
    } catch (error) {
      this.log.warn({ error, fdcId }, 'Error caching nutrients');
    }
  }

  // ==========================================================================
  // USER RECENT FOODS
  // ==========================================================================

  /**
   * Add a food to user's recent selections
   */
  async addUserRecentFood(userId: string, fdcId: number): Promise<void> {
    if (!isRedisAvailable()) {
      return;
    }

    const key = `${USER_RECENT_PREFIX}${userId}:recent`;
    const redis = getRedisClient();
    const now = Date.now();

    try {
      // Add to sorted set with timestamp as score
      await redis?.zadd(key, now, String(fdcId));
      // Set expiry on the key
      await redis?.expire(key, TTL_USER_RECENT);
      // Trim to keep only last 100 items
      await redis?.zremrangebyrank(key, 0, -101);

      this.log.debug({ userId, fdcId }, 'Added recent food');
    } catch (error) {
      this.log.warn({ error, userId, fdcId }, 'Error adding recent food');
    }
  }

  /**
   * Get user's recent food selections
   */
  async getUserRecentFoods(userId: string, limit: number = 20): Promise<number[]> {
    if (!isRedisAvailable()) {
      return [];
    }

    const key = `${USER_RECENT_PREFIX}${userId}:recent`;
    const redis = getRedisClient();

    try {
      // Get most recent items (highest scores)
      const items = await redis?.zrevrange(key, 0, limit - 1);
      if (!items || items.length === 0) {
        return [];
      }
      return items.map((id: string) => parseInt(id, 10));
    } catch (error) {
      this.log.warn({ error, userId }, 'Error getting recent foods');
      return [];
    }
  }

  // ==========================================================================
  // CLASSIFICATION CACHING
  // ==========================================================================

  /**
   * Generate cache key for image classification
   */
  private generateClassifyKey(imageHash: string): string {
    return `${CLASSIFY_PREFIX}${imageHash}`;
  }

  /**
   * Get cached classification result
   */
  async getCachedClassification(
    imageHash: string
  ): Promise<{ category: string; confidence: number; suggestions: string[] } | null> {
    if (!isRedisAvailable()) {
      return null;
    }

    const key = this.generateClassifyKey(imageHash);
    const redis = getRedisClient();

    try {
      const cached = await redis?.get(key);
      if (cached) {
        this.stats.hits++;
        return JSON.parse(cached);
      }
      this.stats.misses++;
      return null;
    } catch (error) {
      this.log.warn({ error, imageHash }, 'Error reading classification cache');
      return null;
    }
  }

  /**
   * Cache classification result
   */
  async cacheClassification(
    imageHash: string,
    result: { category: string; confidence: number; suggestions: string[] }
  ): Promise<void> {
    if (!isRedisAvailable()) {
      return;
    }

    const key = this.generateClassifyKey(imageHash);
    const redis = getRedisClient();

    try {
      await redis?.setex(key, TTL_CLASSIFICATION, JSON.stringify(result));
      this.log.debug({ imageHash, ttl: TTL_CLASSIFICATION }, 'Classification cached');
    } catch (error) {
      this.log.warn({ error, imageHash }, 'Error caching classification');
    }
  }

  // ==========================================================================
  // POPULAR FOODS CACHING
  // ==========================================================================

  /**
   * Get cached popular foods
   */
  async getCachedPopularFoods(): Promise<TransformedUSDAFood[] | null> {
    if (!isRedisAvailable()) {
      return null;
    }

    const redis = getRedisClient();

    try {
      const cached = await redis?.get(POPULAR_KEY);
      if (cached) {
        this.stats.hits++;
        return JSON.parse(cached);
      }
      this.stats.misses++;
      return null;
    } catch (error) {
      this.log.warn({ error }, 'Error reading popular foods cache');
      return null;
    }
  }

  /**
   * Cache popular foods
   */
  async cachePopularFoods(foods: TransformedUSDAFood[]): Promise<void> {
    if (!isRedisAvailable()) {
      return;
    }

    const redis = getRedisClient();

    try {
      // Cache for 24 hours
      await redis?.setex(POPULAR_KEY, TTL_FOOD_DATA, JSON.stringify(foods));
      this.log.debug({ count: foods.length }, 'Popular foods cached');
    } catch (error) {
      this.log.warn({ error }, 'Error caching popular foods');
    }
  }

  // ==========================================================================
  // CACHE MANAGEMENT
  // ==========================================================================

  /**
   * Invalidate a specific food cache
   */
  async invalidateFood(fdcId: number): Promise<void> {
    if (!isRedisAvailable()) {
      return;
    }

    const redis = getRedisClient();
    const keys = [
      `${FOOD_PREFIX}${fdcId}`,
      `${NUTRIENTS_PREFIX}${fdcId}`,
    ];

    try {
      await redis?.del(...keys);
      this.log.debug({ fdcId }, 'Food cache invalidated');
    } catch (error) {
      this.log.warn({ error, fdcId }, 'Error invalidating food cache');
    }
  }

  /**
   * Invalidate all search caches (use sparingly)
   */
  async invalidateSearchCaches(): Promise<void> {
    if (!isRedisAvailable()) {
      return;
    }

    const redis = getRedisClient();

    try {
      // Scan and delete all search keys
      let cursor = '0';
      do {
        const result = await redis?.scan(cursor, 'MATCH', `${SEARCH_PREFIX}*`, 'COUNT', 100);
        if (result) {
          cursor = result[0];
          const keys = result[1];
          if (keys.length > 0) {
            await redis?.del(...keys);
          }
        }
      } while (cursor !== '0');

      this.log.info('Search caches invalidated');
    } catch (error) {
      this.log.warn({ error }, 'Error invalidating search caches');
    }
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    const total = this.stats.hits + this.stats.misses;
    return {
      hits: this.stats.hits,
      misses: this.stats.misses,
      hitRate: total > 0 ? this.stats.hits / total : 0,
    };
  }

  /**
   * Reset cache statistics
   */
  resetStats(): void {
    this.stats = { hits: 0, misses: 0 };
  }

  // ==========================================================================
  // GENERIC CACHING METHODS
  // ==========================================================================

  /**
   * Get a cached value by key (generic)
   */
  async getCachedValue<T>(key: string): Promise<T | null> {
    if (!isRedisAvailable()) {
      return null;
    }

    const redis = getRedisClient();

    try {
      const cached = await redis?.get(key);
      if (cached) {
        this.stats.hits++;
        return JSON.parse(cached) as T;
      }
      this.stats.misses++;
      return null;
    } catch (error) {
      this.log.warn({ error, key }, 'Error reading cache');
      return null;
    }
  }

  /**
   * Cache a value with TTL (generic)
   */
  async cacheValue<T>(key: string, value: T, ttlSeconds: number): Promise<void> {
    if (!isRedisAvailable()) {
      return;
    }

    const redis = getRedisClient();

    try {
      await redis?.setex(key, ttlSeconds, JSON.stringify(value));
      this.log.debug({ key, ttl: ttlSeconds }, 'Value cached');
    } catch (error) {
      this.log.warn({ error, key }, 'Error caching value');
    }
  }

  /**
   * Delete a cached value by key
   */
  async deleteCache(key: string): Promise<void> {
    if (!isRedisAvailable()) {
      return;
    }

    const redis = getRedisClient();

    try {
      await redis?.del(key);
      this.log.debug({ key }, 'Cache deleted');
    } catch (error) {
      this.log.warn({ error, key }, 'Error deleting cache');
    }
  }
}

// Export singleton instance
export const foodCacheService = new FoodCacheService();
