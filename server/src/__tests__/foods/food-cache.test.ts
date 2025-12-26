/**
 * Food Cache Service Tests
 *
 * Tests for Redis-based caching of USDA food data:
 * - Cache hits and misses
 * - TTL expiration behavior
 * - Cache key generation
 * - User recent foods tracking
 * - Classification caching
 * - Cache statistics
 */

// Mock Redis before imports
const mockRedisGet = jest.fn();
const mockRedisSetex = jest.fn();
const mockRedisDel = jest.fn();
const mockRedisZadd = jest.fn();
const mockRedisZrevrange = jest.fn();
const mockRedisExpire = jest.fn();
const mockRedisZremrangebyrank = jest.fn();
const mockRedisScan = jest.fn();

const mockRedisClient = {
  get: mockRedisGet,
  setex: mockRedisSetex,
  del: mockRedisDel,
  zadd: mockRedisZadd,
  zrevrange: mockRedisZrevrange,
  expire: mockRedisExpire,
  zremrangebyrank: mockRedisZremrangebyrank,
  scan: mockRedisScan,
};

let mockRedisAvailable = true;

jest.mock('../../config/redis', () => ({
  getRedisClient: jest.fn(() => mockRedisAvailable ? mockRedisClient : null),
  isRedisAvailable: jest.fn(() => mockRedisAvailable),
}));

// Mock logger
jest.mock('../../config/logger', () => ({
  logger: {
    warn: jest.fn(),
    debug: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
  },
  createChildLogger: jest.fn(() => ({
    warn: jest.fn(),
    debug: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
  })),
}));

import { FoodCacheService } from '../../services/foodCacheService';
import type { TransformedUSDAFood, TransformedNutrients } from '../../types/usda';

// ============================================================================
// TEST DATA
// ============================================================================

const mockFood: TransformedUSDAFood = {
  fdcId: 171688,
  name: 'Apple',
  description: 'Apples, raw, with skin',
  dataType: 'Foundation',
  nutrients: {
    calories: 52,
    protein: 0.3,
    carbs: 13.8,
    fat: 0.2,
    fiber: 2.4,
    sugar: 10.4,
    sodium: 1,
    potassium: 107,
    vitaminC: 4.6,
  },
  servingSize: 100,
  servingSizeUnit: 'g',
  category: 'Fruits and Fruit Juices',
};

const mockNutrients: TransformedNutrients = {
  calories: 52,
  protein: 0.3,
  carbs: 13.8,
  fat: 0.2,
  fiber: 2.4,
  sugar: 10.4,
};

const mockSearchResult = {
  foods: [mockFood],
  pagination: {
    page: 1,
    limit: 25,
    total: 100,
    totalPages: 4,
    hasNextPage: true,
    hasPrevPage: false,
  },
  cachedAt: Date.now(),
};

// ============================================================================
// SEARCH CACHE TESTS
// ============================================================================

describe('FoodCacheService - Search Caching', () => {
  let cacheService: FoodCacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    cacheService = new FoodCacheService();
    cacheService.resetStats();
  });

  describe('getCachedSearch', () => {
    it('should return cached search results when available', async () => {
      mockRedisGet.mockResolvedValueOnce(JSON.stringify(mockSearchResult));

      const result = await cacheService.getCachedSearch('apple', 1, 25);

      expect(result).not.toBeNull();
      expect(result?.foods).toHaveLength(1);
      expect(result?.foods[0].name).toBe('Apple');
      expect(result?.pagination.total).toBe(100);
    });

    it('should return null on cache miss', async () => {
      mockRedisGet.mockResolvedValueOnce(null);

      const result = await cacheService.getCachedSearch('unknown food', 1, 25);

      expect(result).toBeNull();
    });

    it('should track cache hits and misses', async () => {
      // Hit
      mockRedisGet.mockResolvedValueOnce(JSON.stringify(mockSearchResult));
      await cacheService.getCachedSearch('apple', 1, 25);

      // Miss
      mockRedisGet.mockResolvedValueOnce(null);
      await cacheService.getCachedSearch('banana', 1, 25);

      const stats = cacheService.getStats();
      expect(stats.hits).toBe(1);
      expect(stats.misses).toBe(1);
      expect(stats.hitRate).toBe(0.5);
    });

    it('should generate consistent cache keys for same query', async () => {
      mockRedisGet.mockResolvedValue(null);

      await cacheService.getCachedSearch('apple', 1, 25);
      await cacheService.getCachedSearch('apple', 1, 25);

      // Both calls should use the same key
      const firstCall = mockRedisGet.mock.calls[0][0];
      const secondCall = mockRedisGet.mock.calls[1][0];
      expect(firstCall).toBe(secondCall);
    });

    it('should generate different keys for different pages', async () => {
      mockRedisGet.mockResolvedValue(null);

      await cacheService.getCachedSearch('apple', 1, 25);
      await cacheService.getCachedSearch('apple', 2, 25);

      const firstCall = mockRedisGet.mock.calls[0][0];
      const secondCall = mockRedisGet.mock.calls[1][0];
      expect(firstCall).not.toBe(secondCall);
    });

    it('should generate different keys for different data types', async () => {
      mockRedisGet.mockResolvedValue(null);

      await cacheService.getCachedSearch('apple', 1, 25, ['Foundation']);
      await cacheService.getCachedSearch('apple', 1, 25, ['Branded']);

      const firstCall = mockRedisGet.mock.calls[0][0];
      const secondCall = mockRedisGet.mock.calls[1][0];
      expect(firstCall).not.toBe(secondCall);
    });

    it('should return null when Redis is unavailable', async () => {
      mockRedisAvailable = false;
      const newService = new FoodCacheService();

      const result = await newService.getCachedSearch('apple', 1, 25);

      expect(result).toBeNull();
      expect(mockRedisGet).not.toHaveBeenCalled();
    });

    it('should handle Redis errors gracefully', async () => {
      mockRedisGet.mockRejectedValueOnce(new Error('Redis connection error'));

      const result = await cacheService.getCachedSearch('apple', 1, 25);

      expect(result).toBeNull();
    });
  });

  describe('cacheSearchResults', () => {
    it('should cache search results with correct TTL', async () => {
      mockRedisSetex.mockResolvedValueOnce('OK');

      await cacheService.cacheSearchResults(
        'apple',
        1,
        25,
        undefined,
        [mockFood],
        mockSearchResult.pagination
      );

      expect(mockRedisSetex).toHaveBeenCalledWith(
        expect.any(String),
        3600, // 1 hour TTL
        expect.any(String)
      );
    });

    it('should not attempt to cache when Redis is unavailable', async () => {
      mockRedisAvailable = false;
      const newService = new FoodCacheService();

      await newService.cacheSearchResults(
        'apple',
        1,
        25,
        undefined,
        [mockFood],
        mockSearchResult.pagination
      );

      expect(mockRedisSetex).not.toHaveBeenCalled();
    });

    it('should include cachedAt timestamp', async () => {
      mockRedisSetex.mockResolvedValueOnce('OK');

      await cacheService.cacheSearchResults(
        'apple',
        1,
        25,
        undefined,
        [mockFood],
        mockSearchResult.pagination
      );

      const cachedData = JSON.parse(mockRedisSetex.mock.calls[0][2]);
      expect(cachedData).toHaveProperty('cachedAt');
      expect(typeof cachedData.cachedAt).toBe('number');
    });
  });
});

// ============================================================================
// FOOD ITEM CACHE TESTS
// ============================================================================

describe('FoodCacheService - Food Item Caching', () => {
  let cacheService: FoodCacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    cacheService = new FoodCacheService();
    cacheService.resetStats();
  });

  describe('getCachedFood', () => {
    it('should return cached food when available', async () => {
      mockRedisGet.mockResolvedValueOnce(JSON.stringify(mockFood));

      const result = await cacheService.getCachedFood(171688);

      expect(result).not.toBeNull();
      expect(result?.fdcId).toBe(171688);
      expect(result?.name).toBe('Apple');
    });

    it('should return null on cache miss', async () => {
      mockRedisGet.mockResolvedValueOnce(null);

      const result = await cacheService.getCachedFood(999999);

      expect(result).toBeNull();
    });

    it('should track cache statistics', async () => {
      mockRedisGet
        .mockResolvedValueOnce(JSON.stringify(mockFood))
        .mockResolvedValueOnce(null);

      await cacheService.getCachedFood(171688);
      await cacheService.getCachedFood(999999);

      const stats = cacheService.getStats();
      expect(stats.hits).toBe(1);
      expect(stats.misses).toBe(1);
    });
  });

  describe('cacheFood', () => {
    it('should cache food with 24-hour TTL', async () => {
      mockRedisSetex.mockResolvedValueOnce('OK');

      await cacheService.cacheFood(mockFood);

      expect(mockRedisSetex).toHaveBeenCalledWith(
        expect.stringContaining('nutri:food:item:171688'),
        86400, // 24 hours
        expect.any(String)
      );
    });

    it('should correctly serialize food data', async () => {
      mockRedisSetex.mockResolvedValueOnce('OK');

      await cacheService.cacheFood(mockFood);

      const cachedData = JSON.parse(mockRedisSetex.mock.calls[0][2]);
      expect(cachedData.fdcId).toBe(171688);
      expect(cachedData.nutrients.calories).toBe(52);
    });
  });

  describe('getCachedNutrients', () => {
    it('should return cached nutrients when available', async () => {
      mockRedisGet.mockResolvedValueOnce(JSON.stringify(mockNutrients));

      const result = await cacheService.getCachedNutrients(171688);

      expect(result).not.toBeNull();
      expect(result?.calories).toBe(52);
      expect(result?.protein).toBe(0.3);
    });
  });

  describe('cacheNutrients', () => {
    it('should cache nutrients with correct key format', async () => {
      mockRedisSetex.mockResolvedValueOnce('OK');

      await cacheService.cacheNutrients(171688, mockNutrients);

      expect(mockRedisSetex).toHaveBeenCalledWith(
        expect.stringContaining('nutri:food:nutrients:171688'),
        86400,
        expect.any(String)
      );
    });
  });
});

// ============================================================================
// USER RECENT FOODS TESTS
// ============================================================================

describe('FoodCacheService - User Recent Foods', () => {
  let cacheService: FoodCacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    cacheService = new FoodCacheService();
  });

  describe('addUserRecentFood', () => {
    it('should add food to user recent set', async () => {
      mockRedisZadd.mockResolvedValueOnce(1);
      mockRedisExpire.mockResolvedValueOnce(1);
      mockRedisZremrangebyrank.mockResolvedValueOnce(0);

      await cacheService.addUserRecentFood('user123', 171688);

      expect(mockRedisZadd).toHaveBeenCalledWith(
        expect.stringContaining('nutri:food:user:user123:recent'),
        expect.any(Number), // timestamp
        '171688'
      );
    });

    it('should set expiry on the sorted set', async () => {
      mockRedisZadd.mockResolvedValueOnce(1);
      mockRedisExpire.mockResolvedValueOnce(1);
      mockRedisZremrangebyrank.mockResolvedValueOnce(0);

      await cacheService.addUserRecentFood('user123', 171688);

      expect(mockRedisExpire).toHaveBeenCalledWith(
        expect.stringContaining('nutri:food:user:user123:recent'),
        2592000 // 30 days
      );
    });

    it('should trim to keep only last 100 items', async () => {
      mockRedisZadd.mockResolvedValueOnce(1);
      mockRedisExpire.mockResolvedValueOnce(1);
      mockRedisZremrangebyrank.mockResolvedValueOnce(0);

      await cacheService.addUserRecentFood('user123', 171688);

      expect(mockRedisZremrangebyrank).toHaveBeenCalledWith(
        expect.any(String),
        0,
        -101
      );
    });
  });

  describe('getUserRecentFoods', () => {
    it('should return user recent foods', async () => {
      mockRedisZrevrange.mockResolvedValueOnce(['171688', '171689', '171690']);

      const result = await cacheService.getUserRecentFoods('user123', 20);

      expect(result).toEqual([171688, 171689, 171690]);
    });

    it('should return empty array when no recent foods', async () => {
      mockRedisZrevrange.mockResolvedValueOnce([]);

      const result = await cacheService.getUserRecentFoods('user123');

      expect(result).toEqual([]);
    });

    it('should respect limit parameter', async () => {
      mockRedisZrevrange.mockResolvedValueOnce(['171688', '171689']);

      await cacheService.getUserRecentFoods('user123', 10);

      expect(mockRedisZrevrange).toHaveBeenCalledWith(
        expect.any(String),
        0,
        9 // limit - 1
      );
    });

    it('should return empty array when Redis is unavailable', async () => {
      mockRedisAvailable = false;
      const newService = new FoodCacheService();

      const result = await newService.getUserRecentFoods('user123');

      expect(result).toEqual([]);
    });
  });
});

// ============================================================================
// CLASSIFICATION CACHE TESTS
// ============================================================================

describe('FoodCacheService - Classification Caching', () => {
  let cacheService: FoodCacheService;

  const mockClassification = {
    category: 'fruits_fresh',
    confidence: 0.85,
    suggestions: ['apple', 'fruit'],
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    cacheService = new FoodCacheService();
    cacheService.resetStats();
  });

  describe('getCachedClassification', () => {
    it('should return cached classification when available', async () => {
      mockRedisGet.mockResolvedValueOnce(JSON.stringify(mockClassification));

      const result = await cacheService.getCachedClassification('abc123hash');

      expect(result).not.toBeNull();
      expect(result?.category).toBe('fruits_fresh');
      expect(result?.confidence).toBe(0.85);
    });

    it('should return null on cache miss', async () => {
      mockRedisGet.mockResolvedValueOnce(null);

      const result = await cacheService.getCachedClassification('unknownhash');

      expect(result).toBeNull();
    });

    it('should track statistics for classification cache', async () => {
      mockRedisGet
        .mockResolvedValueOnce(JSON.stringify(mockClassification))
        .mockResolvedValueOnce(null);

      await cacheService.getCachedClassification('abc123');
      await cacheService.getCachedClassification('xyz789');

      const stats = cacheService.getStats();
      expect(stats.hits).toBe(1);
      expect(stats.misses).toBe(1);
    });
  });

  describe('cacheClassification', () => {
    it('should cache classification with 1-hour TTL', async () => {
      mockRedisSetex.mockResolvedValueOnce('OK');

      await cacheService.cacheClassification('abc123hash', mockClassification);

      expect(mockRedisSetex).toHaveBeenCalledWith(
        expect.stringContaining('nutri:food:classify:abc123hash'),
        3600, // 1 hour
        expect.any(String)
      );
    });
  });
});

// ============================================================================
// POPULAR FOODS CACHE TESTS
// ============================================================================

describe('FoodCacheService - Popular Foods Caching', () => {
  let cacheService: FoodCacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    cacheService = new FoodCacheService();
  });

  describe('getCachedPopularFoods', () => {
    it('should return cached popular foods when available', async () => {
      const popularFoods = [mockFood, { ...mockFood, fdcId: 171689, name: 'Banana' }];
      mockRedisGet.mockResolvedValueOnce(JSON.stringify(popularFoods));

      const result = await cacheService.getCachedPopularFoods();

      expect(result).not.toBeNull();
      expect(result).toHaveLength(2);
    });

    it('should return null on cache miss', async () => {
      mockRedisGet.mockResolvedValueOnce(null);

      const result = await cacheService.getCachedPopularFoods();

      expect(result).toBeNull();
    });
  });

  describe('cachePopularFoods', () => {
    it('should cache popular foods with 24-hour TTL', async () => {
      mockRedisSetex.mockResolvedValueOnce('OK');
      const popularFoods = [mockFood];

      await cacheService.cachePopularFoods(popularFoods);

      expect(mockRedisSetex).toHaveBeenCalledWith(
        expect.stringContaining('nutri:food:popular'),
        86400,
        expect.any(String)
      );
    });
  });
});

// ============================================================================
// CACHE MANAGEMENT TESTS
// ============================================================================

describe('FoodCacheService - Cache Management', () => {
  let cacheService: FoodCacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    cacheService = new FoodCacheService();
  });

  describe('invalidateFood', () => {
    it('should delete food and nutrient cache entries', async () => {
      mockRedisDel.mockResolvedValueOnce(2);

      await cacheService.invalidateFood(171688);

      expect(mockRedisDel).toHaveBeenCalledWith(
        expect.stringContaining('nutri:food:item:171688'),
        expect.stringContaining('nutri:food:nutrients:171688')
      );
    });

    it('should not attempt deletion when Redis unavailable', async () => {
      mockRedisAvailable = false;
      const newService = new FoodCacheService();

      await newService.invalidateFood(171688);

      expect(mockRedisDel).not.toHaveBeenCalled();
    });
  });

  describe('invalidateSearchCaches', () => {
    it('should scan and delete all search cache keys', async () => {
      mockRedisScan
        .mockResolvedValueOnce(['100', ['nutri:food:search:key1', 'nutri:food:search:key2']])
        .mockResolvedValueOnce(['0', ['nutri:food:search:key3']]);
      mockRedisDel.mockResolvedValue(1);

      await cacheService.invalidateSearchCaches();

      expect(mockRedisScan).toHaveBeenCalled();
      expect(mockRedisDel).toHaveBeenCalledTimes(2);
    });
  });

  describe('getStats', () => {
    it('should return accurate cache statistics', async () => {
      // Generate some hits and misses
      mockRedisGet
        .mockResolvedValueOnce(JSON.stringify(mockFood))
        .mockResolvedValueOnce(JSON.stringify(mockFood))
        .mockResolvedValueOnce(null);

      await cacheService.getCachedFood(1);
      await cacheService.getCachedFood(2);
      await cacheService.getCachedFood(3);

      const stats = cacheService.getStats();

      expect(stats.hits).toBe(2);
      expect(stats.misses).toBe(1);
      expect(stats.hitRate).toBeCloseTo(0.667, 2);
    });

    it('should handle zero total requests', () => {
      const stats = cacheService.getStats();

      expect(stats.hits).toBe(0);
      expect(stats.misses).toBe(0);
      expect(stats.hitRate).toBe(0);
    });
  });

  describe('resetStats', () => {
    it('should reset all statistics to zero', async () => {
      mockRedisGet.mockResolvedValue(JSON.stringify(mockFood));

      await cacheService.getCachedFood(1);
      await cacheService.getCachedFood(2);

      cacheService.resetStats();

      const stats = cacheService.getStats();
      expect(stats.hits).toBe(0);
      expect(stats.misses).toBe(0);
    });
  });
});

// ============================================================================
// GENERIC CACHE METHODS TESTS
// ============================================================================

describe('FoodCacheService - Generic Cache Methods', () => {
  let cacheService: FoodCacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    cacheService = new FoodCacheService();
    cacheService.resetStats();
  });

  describe('getCachedValue', () => {
    it('should retrieve and parse cached JSON value', async () => {
      const testData = { foo: 'bar', count: 42 };
      mockRedisGet.mockResolvedValueOnce(JSON.stringify(testData));

      const result = await cacheService.getCachedValue<typeof testData>('custom:key');

      expect(result).toEqual(testData);
    });

    it('should track statistics', async () => {
      mockRedisGet.mockResolvedValueOnce(null);

      await cacheService.getCachedValue('missing:key');

      const stats = cacheService.getStats();
      expect(stats.misses).toBe(1);
    });
  });

  describe('cacheValue', () => {
    it('should cache value with custom TTL', async () => {
      mockRedisSetex.mockResolvedValueOnce('OK');

      await cacheService.cacheValue('custom:key', { data: 'test' }, 7200);

      expect(mockRedisSetex).toHaveBeenCalledWith(
        'custom:key',
        7200,
        JSON.stringify({ data: 'test' })
      );
    });
  });

  describe('deleteCache', () => {
    it('should delete cache entry by key', async () => {
      mockRedisDel.mockResolvedValueOnce(1);

      await cacheService.deleteCache('custom:key');

      expect(mockRedisDel).toHaveBeenCalledWith('custom:key');
    });
  });
});

// ============================================================================
// CACHE HIT RATE BENCHMARK
// ============================================================================

describe('Cache Hit Rate Benchmarks', () => {
  let cacheService: FoodCacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    cacheService = new FoodCacheService();
    cacheService.resetStats();
  });

  it('should achieve >80% hit rate for repeated popular queries', async () => {
    const popularFoods = [171688, 171689, 171690, 171691, 171692];
    const queryCount = 100;

    // Simulate cache: first access misses, subsequent hits
    const seenFoods = new Set<number>();

    for (let i = 0; i < queryCount; i++) {
      const foodId = popularFoods[i % popularFoods.length];

      if (seenFoods.has(foodId)) {
        mockRedisGet.mockResolvedValueOnce(JSON.stringify({ ...mockFood, fdcId: foodId }));
      } else {
        mockRedisGet.mockResolvedValueOnce(null);
        seenFoods.add(foodId);
      }

      await cacheService.getCachedFood(foodId);
    }

    const stats = cacheService.getStats();

    // First 5 queries are misses, remaining 95 are hits
    expect(stats.hits).toBe(95);
    expect(stats.misses).toBe(5);
    expect(stats.hitRate).toBeGreaterThan(0.8);
  });

  it('should track per-query-type statistics', async () => {
    // Test that different cache types are tracked together
    mockRedisGet
      .mockResolvedValueOnce(JSON.stringify(mockFood)) // Food hit
      .mockResolvedValueOnce(null) // Search miss
      .mockResolvedValueOnce(JSON.stringify({ category: 'fruit' })); // Classification hit

    await cacheService.getCachedFood(171688);
    await cacheService.getCachedSearch('apple', 1, 25);
    await cacheService.getCachedClassification('hash123');

    const stats = cacheService.getStats();
    expect(stats.hits).toBe(2);
    expect(stats.misses).toBe(1);
    expect(stats.hitRate).toBeCloseTo(0.667, 2);
  });
});
