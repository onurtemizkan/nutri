/**
 * Performance Benchmark Tests for Food Search Pipeline
 *
 * Performance targets:
 * - Search response time (with cache): <500ms
 * - Classification + search (total): <2s
 * - Cache hit rate for popular queries: >80%
 * - Transform + rank pipeline: <100ms for 100 results
 */

// Mock axios
const mockPost = jest.fn();
const mockGet = jest.fn();
const mockAxiosInstance = {
  post: mockPost,
  get: mockGet,
  interceptors: {
    request: { use: jest.fn() },
    response: { use: jest.fn() },
  },
};

jest.mock('axios', () => ({
  ...jest.requireActual('axios'),
  create: jest.fn(() => mockAxiosInstance),
  isAxiosError: jest.fn((error): boolean => {
    return error && typeof error === 'object' && 'isAxiosError' in error;
  }),
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

// Mock Redis
const mockRedisGet = jest.fn();
const mockRedisSetex = jest.fn();
let mockRedisAvailable = true;

jest.mock('../../config/redis', () => ({
  getRedisClient: jest.fn(() =>
    mockRedisAvailable
      ? { get: mockRedisGet, setex: mockRedisSetex, del: jest.fn() }
      : null
  ),
  isRedisAvailable: jest.fn(() => mockRedisAvailable),
}));

import { USDAApiService } from '../../services/usdaApiService';
import { NutrientMappingService } from '../../services/nutrientMappingService';
import { FoodCacheService } from '../../services/foodCacheService';
import { rankSearchResults, type RankingHints } from '../../utils/foodRanking';
import type { USDASearchResponse, USDASearchResultFood, TransformedUSDAFood } from '../../types/usda';

// ============================================================================
// TEST DATA FACTORIES
// ============================================================================

/**
 * Create mock USDA response with configurable size
 */
function createLargeUSDAResponse(size: number): USDASearchResponse {
  const foods: USDASearchResultFood[] = Array.from({ length: size }, (_, i) => ({
    fdcId: 170000 + i,
    description: `Test food item ${i}, prepared form`,
    dataType: i % 4 === 0 ? 'Foundation' : i % 4 === 1 ? 'SR Legacy' : i % 4 === 2 ? 'Survey (FNDDS)' : 'Branded',
    publishedDate: '2023-01-01',
    foodCategory: 'Test Category',
    foodNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 50 + i },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 0.5 + i * 0.1 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 10 + i * 0.5 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 0.2 + i * 0.05 },
      { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 2 + i * 0.1 },
      { nutrientId: 2000, nutrientName: 'Sugars', nutrientNumber: '269', unitName: 'g', value: 5 + i * 0.2 },
      { nutrientId: 1093, nutrientName: 'Sodium', nutrientNumber: '307', unitName: 'mg', value: 100 + i * 2 },
      { nutrientId: 1092, nutrientName: 'Potassium', nutrientNumber: '306', unitName: 'mg', value: 200 + i * 3 },
    ],
    foodMeasures: [
      { id: 1, gramWeight: 100, disseminationText: '1 serving' },
      { id: 2, gramWeight: 150, disseminationText: '1 cup' },
    ],
  }));

  return {
    totalHits: size,
    currentPage: 1,
    totalPages: Math.ceil(size / 25),
    foodSearchCriteria: {
      query: 'test',
      pageNumber: 1,
      numberOfResultsPerPage: size,
      requireAllWords: false,
    },
    foods,
  };
}

/**
 * Create mock transformed foods for ranking tests
 */
function createTransformedFoods(size: number): TransformedUSDAFood[] {
  return Array.from({ length: size }, (_, i) => ({
    fdcId: 170000 + i,
    name: `Test Food ${i}`,
    description: `Test food item ${i}, prepared form`,
    dataType: (i % 4 === 0 ? 'Foundation' : i % 4 === 1 ? 'SR Legacy' : i % 4 === 2 ? 'Survey (FNDDS)' : 'Branded') as TransformedUSDAFood['dataType'],
    category: 'Test Category',
    nutrients: {
      calories: 50 + i,
      protein: 0.5 + i * 0.1,
      carbs: 10 + i * 0.5,
      fat: 0.2 + i * 0.05,
      fiber: 2 + i * 0.1,
      sugar: 5 + i * 0.2,
      sodium: 100 + i * 2,
      potassium: 200 + i * 3,
    },
    servingSize: 100,
    servingSizeUnit: 'g',
  }));
}

// ============================================================================
// PERFORMANCE BENCHMARK TESTS
// ============================================================================

describe('Performance Benchmarks', () => {
  let nutrientService: NutrientMappingService;
  let cacheService: FoodCacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    mockRedisGet.mockResolvedValue(null);

    nutrientService = new NutrientMappingService();
    cacheService = new FoodCacheService();
    cacheService.resetStats();
  });

  describe('transform performance', () => {
    const testCases = [
      { size: 25, maxTime: 10 },
      { size: 50, maxTime: 25 },
      { size: 100, maxTime: 50 },
      { size: 200, maxTime: 100 },
    ];

    testCases.forEach(({ size, maxTime }) => {
      it(`should transform ${size} results in under ${maxTime}ms`, () => {
        const mockResponse = createLargeUSDAResponse(size);

        const startTime = performance.now();

        const transformedFoods = mockResponse.foods.map(food =>
          nutrientService.transformSearchResultFood(food)
        );

        const duration = performance.now() - startTime;

        expect(transformedFoods.length).toBe(size);
        expect(duration).toBeLessThan(maxTime);
      });
    });
  });

  describe('ranking performance', () => {
    const testCases = [
      { size: 25, maxTime: 5 },
      { size: 50, maxTime: 10 },
      { size: 100, maxTime: 20 },
      { size: 200, maxTime: 40 },
    ];

    testCases.forEach(({ size, maxTime }) => {
      it(`should rank ${size} results in under ${maxTime}ms`, () => {
        const transformedFoods = createTransformedFoods(size);
        const hints: RankingHints = {
          query: 'test food',
          category: 'fruit',
          isWholeFoodQuery: true,
        };

        const startTime = performance.now();

        const rankedFoods = rankSearchResults(transformedFoods, hints);

        const duration = performance.now() - startTime;

        expect(rankedFoods.length).toBe(size);
        expect(duration).toBeLessThan(maxTime);
      });
    });
  });

  describe('full pipeline performance', () => {
    it('should complete full pipeline (transform + rank) for 100 results in under 100ms', () => {
      const mockResponse = createLargeUSDAResponse(100);
      const hints: RankingHints = {
        query: 'test',
        category: 'fruits_fresh',
        subcategoryHints: ['appears fresh'],
        isWholeFoodQuery: true,
      };

      const startTime = performance.now();

      // Transform
      const transformedFoods = mockResponse.foods.map(food =>
        nutrientService.transformSearchResultFood(food)
      );

      // Rank
      const rankedFoods = rankSearchResults(transformedFoods, hints);

      const duration = performance.now() - startTime;

      expect(rankedFoods.length).toBe(100);
      expect(duration).toBeLessThan(100);
    });

    it('should complete full pipeline for 50 results in under 50ms', () => {
      const mockResponse = createLargeUSDAResponse(50);
      const hints: RankingHints = {
        query: 'apple',
        category: 'fruits_fresh',
        isWholeFoodQuery: true,
      };

      const startTime = performance.now();

      const transformedFoods = mockResponse.foods.map(food =>
        nutrientService.transformSearchResultFood(food)
      );
      const rankedFoods = rankSearchResults(transformedFoods, hints);

      const duration = performance.now() - startTime;

      expect(rankedFoods.length).toBe(50);
      expect(duration).toBeLessThan(50);
    });
  });

  describe('concurrent operations', () => {
    it('should handle 10 concurrent transform operations efficiently', async () => {
      const responses = Array.from({ length: 10 }, () =>
        createLargeUSDAResponse(50)
      );

      const startTime = performance.now();

      const results = await Promise.all(
        responses.map(async response => {
          const transformed = response.foods.map(food =>
            nutrientService.transformSearchResultFood(food)
          );
          return rankSearchResults(transformed, { query: 'test' });
        })
      );

      const duration = performance.now() - startTime;

      expect(results.length).toBe(10);
      results.forEach(result => {
        expect(result.length).toBe(50);
      });
      expect(duration).toBeLessThan(500);
    });

    it('should handle 50 concurrent transform operations', async () => {
      const responses = Array.from({ length: 50 }, () =>
        createLargeUSDAResponse(25)
      );

      const startTime = performance.now();

      const results = await Promise.all(
        responses.map(async response => {
          const transformed = response.foods.map(food =>
            nutrientService.transformSearchResultFood(food)
          );
          return rankSearchResults(transformed, { query: 'test' });
        })
      );

      const duration = performance.now() - startTime;

      expect(results.length).toBe(50);
      expect(duration).toBeLessThan(1000);
    });
  });

  describe('memory efficiency', () => {
    it('should not leak memory across multiple transform batches', () => {
      const initialMemory = process.memoryUsage().heapUsed;

      // Run 100 batches
      for (let i = 0; i < 100; i++) {
        const mockResponse = createLargeUSDAResponse(100);
        const transformed = mockResponse.foods.map(food =>
          nutrientService.transformSearchResultFood(food)
        );
        rankSearchResults(transformed, { query: 'test' });
      }

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = (finalMemory - initialMemory) / 1024 / 1024; // MB

      // Memory increase should be reasonable (less than 50MB for this test)
      expect(memoryIncrease).toBeLessThan(50);
    });
  });
});

// ============================================================================
// CACHE HIT RATE BENCHMARKS
// ============================================================================

describe('Cache Hit Rate Benchmarks', () => {
  let cacheService: FoodCacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    mockRedisAvailable = true;
    mockRedisGet.mockResolvedValue(null);

    cacheService = new FoodCacheService();
    cacheService.resetStats();
  });

  describe('popular query simulation', () => {
    it('should achieve >80% hit rate for repeated popular queries', async () => {
      // Popular queries (20 common foods)
      const popularQueries = [
        'apple', 'banana', 'chicken', 'rice', 'bread',
        'milk', 'egg', 'salmon', 'spinach', 'carrot',
        'orange', 'beef', 'pasta', 'cheese', 'yogurt',
        'tomato', 'potato', 'broccoli', 'oatmeal', 'almond',
      ];

      const cachedResults = new Map<string, string>();

      // Simulate 100 queries with Zipf distribution (popular items queried more)
      const queryCount = 100;
      for (let i = 0; i < queryCount; i++) {
        // Weighted selection favoring earlier (more popular) items
        const weightedIndex = Math.floor(Math.pow(Math.random(), 2) * popularQueries.length);
        const query = popularQueries[weightedIndex];

        // Cache key would be: `nutri:food:search:${query}:1:25`
        if (cachedResults.has(query)) {
          mockRedisGet.mockResolvedValueOnce(cachedResults.get(query)!);
        } else {
          mockRedisGet.mockResolvedValueOnce(null);
          cachedResults.set(query, JSON.stringify({ foods: [], cachedAt: Date.now() }));
        }

        await cacheService.getCachedSearch(query, 1, 25);
      }

      const stats = cacheService.getStats();

      // After warm-up, hit rate should be >=80%
      expect(stats.hitRate).toBeGreaterThanOrEqual(0.8);
    });

    it('should achieve >90% hit rate for food item cache', async () => {
      // Simulate looking up the same food items repeatedly
      const popularFoods = [171688, 171689, 171690, 171691, 171692];
      const cachedFoods = new Map<number, string>();

      // First pass: populate cache
      for (const fdcId of popularFoods) {
        mockRedisGet.mockResolvedValueOnce(null);
        await cacheService.getCachedFood(fdcId);
        cachedFoods.set(fdcId, JSON.stringify({ fdcId, name: `Food ${fdcId}` }));
      }

      // Reset stats after warm-up
      cacheService.resetStats();

      // Second pass: 100 lookups, all should be cache hits
      for (let i = 0; i < 100; i++) {
        const fdcId = popularFoods[i % popularFoods.length];
        mockRedisGet.mockResolvedValueOnce(cachedFoods.get(fdcId)!);
        await cacheService.getCachedFood(fdcId);
      }

      const stats = cacheService.getStats();

      expect(stats.hits).toBe(100);
      expect(stats.misses).toBe(0);
      expect(stats.hitRate).toBe(1.0);
    });
  });

  describe('classification cache hit rate', () => {
    it('should cache classification results effectively', async () => {
      const imageHashes = ['hash1', 'hash2', 'hash3', 'hash4', 'hash5'];
      const cachedClassifications = new Map<string, string>();

      // First pass: populate cache
      for (const hash of imageHashes) {
        mockRedisGet.mockResolvedValueOnce(null);
        await cacheService.getCachedClassification(hash);
        cachedClassifications.set(hash, JSON.stringify({
          category: 'fruits_fresh',
          confidence: 0.9,
        }));
      }

      cacheService.resetStats();

      // Second pass: all hits
      for (let i = 0; i < 50; i++) {
        const hash = imageHashes[i % imageHashes.length];
        mockRedisGet.mockResolvedValueOnce(cachedClassifications.get(hash)!);
        await cacheService.getCachedClassification(hash);
      }

      const stats = cacheService.getStats();

      expect(stats.hitRate).toBe(1.0);
    });
  });
});

// ============================================================================
// SEARCH RESPONSE TIME BENCHMARKS
// ============================================================================

describe('Search Response Time Benchmarks', () => {
  let usdaService: USDAApiService;
  let nutrientService: NutrientMappingService;
  let cacheService: FoodCacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env.USDA_API_KEY = 'test-api-key';
    mockRedisAvailable = true;

    usdaService = new USDAApiService();
    nutrientService = new NutrientMappingService();
    cacheService = new FoodCacheService();
    cacheService.resetStats();
  });

  afterEach(() => {
    delete process.env.USDA_API_KEY;
  });

  describe('cached search response time', () => {
    it('should return cached results in under 10ms', async () => {
      const cachedResult = {
        foods: createTransformedFoods(25),
        pagination: { page: 1, limit: 25, total: 100, totalPages: 4 },
        cachedAt: Date.now(),
      };

      mockRedisGet.mockResolvedValueOnce(JSON.stringify(cachedResult));

      const startTime = performance.now();

      const result = await cacheService.getCachedSearch('apple', 1, 25);

      const duration = performance.now() - startTime;

      expect(result).not.toBeNull();
      expect(duration).toBeLessThan(10);
    });
  });

  describe('end-to-end latency simulation', () => {
    it('should complete search with mock API in under 200ms', async () => {
      const mockResponse = createLargeUSDAResponse(25);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      const startTime = performance.now();

      // Simulate full pipeline
      const searchResult = await usdaService.searchFoods({ query: 'apple' });
      const transformedFoods = searchResult.foods.map(food =>
        nutrientService.transformSearchResultFood(food)
      );
      const rankedFoods = rankSearchResults(transformedFoods, {
        query: 'apple',
        isWholeFoodQuery: true,
      });

      const duration = performance.now() - startTime;

      expect(rankedFoods.length).toBe(25);
      expect(duration).toBeLessThan(200);
    });
  });
});

// ============================================================================
// SCALABILITY TESTS
// ============================================================================

describe('Scalability Tests', () => {
  let nutrientService: NutrientMappingService;

  beforeEach(() => {
    nutrientService = new NutrientMappingService();
  });

  describe('large result set handling', () => {
    it('should handle 500 results without significant degradation', () => {
      const mockResponse = createLargeUSDAResponse(500);

      const startTime = performance.now();

      const transformedFoods = mockResponse.foods.map(food =>
        nutrientService.transformSearchResultFood(food)
      );
      const rankedFoods = rankSearchResults(transformedFoods, { query: 'test' });

      const duration = performance.now() - startTime;

      expect(rankedFoods.length).toBe(500);
      expect(duration).toBeLessThan(500); // 500 results in 500ms
    });

    it('should maintain linear scaling for ranking', () => {
      const sizes = [100, 200, 400];
      const durations: number[] = [];

      sizes.forEach(size => {
        const foods = createTransformedFoods(size);

        const startTime = performance.now();
        rankSearchResults(foods, { query: 'test' });
        durations.push(performance.now() - startTime);
      });

      // Check that 2x size doesn't result in >3x time (sub-linear or linear)
      const ratio1 = durations[1] / durations[0];
      const ratio2 = durations[2] / durations[1];

      expect(ratio1).toBeLessThan(3);
      expect(ratio2).toBeLessThan(3);
    });
  });
});
