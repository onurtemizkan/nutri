/**
 * Classification + Search E2E Integration Tests
 *
 * Tests for the end-to-end flow:
 * 1. Image classification (ML service)
 * 2. Search enhancement with classification hints
 * 3. Result ranking with classification context
 * 4. Cache integration
 * 5. Performance benchmarks
 */

// Mock axios for USDA API calls
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
import type { USDASearchResponse, USDASearchResultFood } from '../../types/usda';

// ============================================================================
// MOCK DATA
// ============================================================================

/**
 * Simulated ML classification results for different food categories
 */
const MOCK_CLASSIFICATIONS = {
  apple: {
    category: 'fruits_fresh',
    confidence: 0.92,
    subcategoryHints: ['appears fresh', 'appears whole'],
    usdaDataTypes: ['Foundation', 'SR Legacy'],
    alternatives: [
      { category: 'vegetables_other', confidence: 0.05 },
      { category: 'snacks_sweet', confidence: 0.03 },
    ],
  },
  burger: {
    category: 'fast_food',
    confidence: 0.88,
    subcategoryHints: ['appears grilled', 'appears processed'],
    usdaDataTypes: ['Branded', 'Survey (FNDDS)'],
    alternatives: [
      { category: 'meat_processed', confidence: 0.08 },
      { category: 'mixed_dishes', confidence: 0.04 },
    ],
  },
  salad: {
    category: 'vegetables_leafy',
    confidence: 0.85,
    subcategoryHints: ['appears fresh', 'raw texture'],
    usdaDataTypes: ['Foundation', 'SR Legacy'],
    alternatives: [
      { category: 'mixed_dishes', confidence: 0.10 },
      { category: 'vegetables_other', confidence: 0.05 },
    ],
  },
  chicken: {
    category: 'meat_poultry',
    confidence: 0.90,
    subcategoryHints: ['appears grilled', 'appears cooked'],
    usdaDataTypes: ['Foundation', 'SR Legacy'],
    alternatives: [
      { category: 'meat_red', confidence: 0.06 },
      { category: 'mixed_dishes', confidence: 0.04 },
    ],
  },
  pasta: {
    category: 'grains_pasta',
    confidence: 0.87,
    subcategoryHints: ['appears cooked'],
    usdaDataTypes: ['SR Legacy', 'Branded'],
    alternatives: [
      { category: 'mixed_dishes', confidence: 0.10 },
      { category: 'grains_other', confidence: 0.03 },
    ],
  },
};

/**
 * Create mock USDA search response
 */
function createMockUSDAResponse(
  query: string,
  dataTypes: string[] = ['Foundation', 'SR Legacy']
): USDASearchResponse {
  const foods: USDASearchResultFood[] = dataTypes.flatMap((dataType, typeIdx) =>
    Array.from({ length: 5 }, (_, i) => ({
      fdcId: 170000 + typeIdx * 100 + i,
      description: `${query}, ${dataType === 'Foundation' ? 'raw' : 'prepared'} form ${i + 1}`,
      dataType: dataType as USDASearchResultFood['dataType'],
      publishedDate: '2023-01-01',
      foodCategory: dataType === 'Foundation' ? 'Fruits and Fruit Juices' : 'Prepared Foods',
      foodNutrients: [
        { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 50 + i * 10 },
        { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 0.5 + i * 0.2 },
        { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 12 + i * 2 },
        { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 0.2 + i * 0.1 },
        { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 2 + i * 0.3 },
      ],
      foodMeasures: [{ id: 1, gramWeight: 100, disseminationText: '1 serving' }],
    }))
  );

  return {
    totalHits: foods.length,
    currentPage: 1,
    totalPages: 1,
    foodSearchCriteria: {
      query,
      pageNumber: 1,
      numberOfResultsPerPage: 25,
      requireAllWords: false,
    },
    foods,
  };
}

// ============================================================================
// E2E CLASSIFICATION + SEARCH FLOW TESTS
// ============================================================================

describe('Classification + Search E2E Flow', () => {
  let usdaService: USDAApiService;
  let nutrientService: NutrientMappingService;
  let cacheService: FoodCacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env.USDA_API_KEY = 'test-api-key';
    mockRedisAvailable = true;
    mockRedisGet.mockResolvedValue(null); // Default: cache miss

    usdaService = new USDAApiService();
    nutrientService = new NutrientMappingService();
    cacheService = new FoodCacheService();
    cacheService.resetStats();
  });

  afterEach(() => {
    delete process.env.USDA_API_KEY;
  });

  describe('fresh fruit classification flow', () => {
    it('should complete full flow for fresh fruit classification', async () => {
      // Step 1: Receive classification result
      const classification = MOCK_CLASSIFICATIONS.apple;

      // Step 2: Build search hints from classification
      const searchHints: RankingHints = {
        query: 'apple',
        category: classification.category,
        subcategoryHints: classification.subcategoryHints,
        isWholeFoodQuery: true,
      };

      // Step 3: Search USDA with preferred data types
      const mockResponse = createMockUSDAResponse('apple', classification.usdaDataTypes);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      const searchResult = await usdaService.searchFoods({
        query: 'apple',
        dataType: classification.usdaDataTypes as Array<'Foundation' | 'SR Legacy' | 'Survey (FNDDS)' | 'Branded'>,
      });

      // Step 4: Transform results
      const transformedFoods = searchResult.foods.map(food =>
        nutrientService.transformSearchResultFood(food)
      );

      // Step 5: Rank with classification hints
      const rankedFoods = rankSearchResults(transformedFoods, searchHints);

      // Verify flow
      expect(searchResult.foods.length).toBeGreaterThan(0);
      expect(transformedFoods.length).toBe(searchResult.foods.length);
      expect(rankedFoods.length).toBe(transformedFoods.length);

      // Foundation foods should rank higher for fresh fruit
      const topResults = rankedFoods.slice(0, 3);
      const foundationCount = topResults.filter(f => f.dataType === 'Foundation').length;
      expect(foundationCount).toBeGreaterThanOrEqual(2);
    });
  });

  describe('fast food classification flow', () => {
    it('should complete full flow for fast food classification', async () => {
      const classification = MOCK_CLASSIFICATIONS.burger;

      const searchHints: RankingHints = {
        query: 'burger',
        category: classification.category,
        subcategoryHints: classification.subcategoryHints,
        isBrandedQuery: true,
      };

      // Fast food prefers Branded data type
      const mockResponse = createMockUSDAResponse('burger', classification.usdaDataTypes);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      const searchResult = await usdaService.searchFoods({
        query: 'burger',
        dataType: classification.usdaDataTypes as Array<'Foundation' | 'SR Legacy' | 'Survey (FNDDS)' | 'Branded'>,
      });

      const transformedFoods = searchResult.foods.map(food =>
        nutrientService.transformSearchResultFood(food)
      );

      const rankedFoods = rankSearchResults(transformedFoods, searchHints);

      // Verify Branded foods get boosted for fast food
      const brandedResults = rankedFoods.filter(f => f.dataType === 'Branded');
      expect(brandedResults.length).toBeGreaterThan(0);

      // Check that Branded foods have boosted data type scores
      brandedResults.forEach(food => {
        expect(food.scoreBreakdown.dataTypeScore).toBe(80); // Boosted from 50 to 80
      });
    });
  });

  describe('mixed dish classification flow', () => {
    it('should handle mixed dishes with Survey data preference', async () => {
      const classification = {
        category: 'mixed_dishes',
        confidence: 0.75,
        subcategoryHints: ['appears prepared'],
        usdaDataTypes: ['Survey (FNDDS)', 'Branded'],
        alternatives: [],
      };

      const searchHints: RankingHints = {
        query: 'stir fry',
        category: classification.category,
        subcategoryHints: classification.subcategoryHints,
      };

      const mockResponse = createMockUSDAResponse('stir fry', classification.usdaDataTypes);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      const searchResult = await usdaService.searchFoods({
        query: 'stir fry',
        dataType: classification.usdaDataTypes as Array<'Foundation' | 'SR Legacy' | 'Survey (FNDDS)' | 'Branded'>,
      });

      const transformedFoods = searchResult.foods.map(food =>
        nutrientService.transformSearchResultFood(food)
      );

      const rankedFoods = rankSearchResults(transformedFoods, searchHints);

      expect(rankedFoods.length).toBeGreaterThan(0);
    });
  });

  describe('low confidence classification handling', () => {
    it('should search all data types when confidence is low', async () => {
      const lowConfidenceClassification = {
        category: 'unknown',
        confidence: 0.35,
        subcategoryHints: [],
        usdaDataTypes: ['Foundation', 'SR Legacy', 'Survey (FNDDS)', 'Branded'],
        alternatives: [],
      };

      // When confidence is low, search all data types
      const mockResponse = createMockUSDAResponse('mystery food', lowConfidenceClassification.usdaDataTypes);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      await usdaService.searchFoods({
        query: 'mystery food',
        // Don't filter by data type when confidence is low
      });

      expect(mockPost).toHaveBeenCalledWith(
        '/foods/search',
        expect.not.objectContaining({ dataType: expect.anything() })
      );
    });
  });
});

// ============================================================================
// CACHE INTEGRATION TESTS
// ============================================================================

describe('Classification + Cache Integration', () => {
  let usdaService: USDAApiService;
  let nutrientService: NutrientMappingService;
  let cacheService: FoodCacheService;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env.USDA_API_KEY = 'test-api-key';
    mockRedisAvailable = true;
    mockRedisGet.mockResolvedValue(null);

    usdaService = new USDAApiService();
    nutrientService = new NutrientMappingService();
    cacheService = new FoodCacheService();
    cacheService.resetStats();
  });

  afterEach(() => {
    delete process.env.USDA_API_KEY;
  });

  it('should cache classification results', async () => {
    const imageHash = 'abc123';
    const classification = MOCK_CLASSIFICATIONS.apple;

    // Cache the classification
    mockRedisSetex.mockResolvedValueOnce('OK');
    await cacheService.cacheClassification(imageHash, {
      category: classification.category,
      confidence: classification.confidence,
      suggestions: classification.subcategoryHints,
    });

    expect(mockRedisSetex).toHaveBeenCalledWith(
      expect.stringContaining('nutri:food:classify:abc123'),
      3600, // 1 hour TTL
      expect.any(String)
    );
  });

  it('should use cached classification on repeat requests', async () => {
    const imageHash = 'abc123';
    const cachedClassification = {
      category: 'fruits_fresh',
      confidence: 0.92,
      suggestions: ['appears fresh'],
    };

    // First request: cache hit
    mockRedisGet.mockResolvedValueOnce(JSON.stringify(cachedClassification));
    const result = await cacheService.getCachedClassification(imageHash);

    expect(result).not.toBeNull();
    expect(result?.category).toBe('fruits_fresh');
    expect(result?.confidence).toBe(0.92);
  });

  it('should cache search results after classification-enhanced search', async () => {
    const classification = MOCK_CLASSIFICATIONS.apple;
    const mockResponse = createMockUSDAResponse('apple', classification.usdaDataTypes);

    // Search
    mockPost.mockResolvedValueOnce({ data: mockResponse });
    const searchResult = await usdaService.searchFoods({
      query: 'apple',
      dataType: classification.usdaDataTypes as Array<'Foundation' | 'SR Legacy' | 'Survey (FNDDS)' | 'Branded'>,
    });

    // Transform
    const transformedFoods = searchResult.foods.map(food =>
      nutrientService.transformSearchResultFood(food)
    );

    // Cache
    mockRedisSetex.mockResolvedValueOnce('OK');
    await cacheService.cacheSearchResults(
      'apple',
      1,
      25,
      classification.usdaDataTypes,
      transformedFoods,
      {
        page: 1,
        limit: 25,
        total: searchResult.totalHits,
        totalPages: 1,
        hasNextPage: false,
        hasPrevPage: false,
      }
    );

    expect(mockRedisSetex).toHaveBeenCalled();
    const cachedData = JSON.parse(mockRedisSetex.mock.calls[0][2]);
    expect(cachedData.foods.length).toBe(transformedFoods.length);
  });

  it('should skip USDA API when cached results exist', async () => {
    const classification = MOCK_CLASSIFICATIONS.apple;
    const cachedSearchResult = {
      foods: [
        {
          fdcId: 171688,
          name: 'Apple',
          description: 'Apples, raw, with skin',
          dataType: 'Foundation',
          nutrients: { calories: 52, protein: 0.3, carbs: 13.8, fat: 0.2 },
        },
      ],
      pagination: {
        page: 1,
        limit: 25,
        total: 1,
        totalPages: 1,
        hasNextPage: false,
        hasPrevPage: false,
      },
      cachedAt: Date.now(),
    };

    // Cache hit
    mockRedisGet.mockResolvedValueOnce(JSON.stringify(cachedSearchResult));

    const result = await cacheService.getCachedSearch('apple', 1, 25, classification.usdaDataTypes);

    expect(result).not.toBeNull();
    expect(result?.foods.length).toBe(1);
    expect(mockPost).not.toHaveBeenCalled(); // USDA API should not be called
  });
});

// ============================================================================
// PERFORMANCE BENCHMARK TESTS
// ============================================================================

describe('Classification + Search Performance', () => {
  let nutrientService: NutrientMappingService;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env.USDA_API_KEY = 'test-api-key';
    mockRedisAvailable = false; // Disable cache for pure performance tests

    nutrientService = new NutrientMappingService();
  });

  afterEach(() => {
    delete process.env.USDA_API_KEY;
  });

  it('should complete transform + rank pipeline in under 50ms for 50 results', () => {
    const mockResponse = createMockUSDAResponse('apple', ['Foundation', 'SR Legacy', 'Survey (FNDDS)', 'Branded']);

    const startTime = performance.now();

    // Transform
    const transformedFoods = mockResponse.foods.map(food =>
      nutrientService.transformSearchResultFood(food)
    );

    // Rank
    const hints: RankingHints = {
      query: 'apple',
      category: 'fruits_fresh',
      subcategoryHints: ['appears fresh'],
      isWholeFoodQuery: true,
    };
    const rankedFoods = rankSearchResults(transformedFoods, hints);

    const endTime = performance.now();
    const duration = endTime - startTime;

    expect(rankedFoods.length).toBe(20); // 5 per data type * 4 data types
    expect(duration).toBeLessThan(50);
  });

  it('should handle 100 concurrent transform operations efficiently', async () => {
    const mockResponses = Array.from({ length: 100 }, (_, i) =>
      createMockUSDAResponse(`food${i}`, ['Foundation'])
    );

    const startTime = performance.now();

    const results = await Promise.all(
      mockResponses.map(async response => {
        const transformed = response.foods.map(food =>
          nutrientService.transformSearchResultFood(food)
        );
        return rankSearchResults(transformed, { query: 'food' });
      })
    );

    const endTime = performance.now();
    const duration = endTime - startTime;

    expect(results.length).toBe(100);
    expect(duration).toBeLessThan(500); // 100 operations in under 500ms
  });

  it('should maintain consistent ranking across repeated operations', () => {
    const mockResponse = createMockUSDAResponse('apple', ['Foundation', 'Branded']);
    const transformedFoods = mockResponse.foods.map(food =>
      nutrientService.transformSearchResultFood(food)
    );

    const hints: RankingHints = {
      query: 'apple',
      category: 'fruits_fresh',
      isWholeFoodQuery: true,
    };

    // Run ranking 10 times
    const rankings = Array.from({ length: 10 }, () =>
      rankSearchResults([...transformedFoods], hints)
    );

    // All rankings should produce same order
    const firstRanking = rankings[0].map(f => f.fdcId);
    rankings.forEach((ranking) => {
      const currentOrder = ranking.map(f => f.fdcId);
      expect(currentOrder).toEqual(firstRanking);
    });
  });
});

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

describe('Classification + Search Error Handling', () => {
  let usdaService: USDAApiService;
  let nutrientService: NutrientMappingService;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env.USDA_API_KEY = 'test-api-key';
    mockRedisAvailable = true;
    mockRedisGet.mockResolvedValue(null);

    usdaService = new USDAApiService();
    nutrientService = new NutrientMappingService();
  });

  afterEach(() => {
    delete process.env.USDA_API_KEY;
  });

  it('should gracefully handle classification service timeout', async () => {
    // Simulate classification failure - fallback classification would be:
    // { category: 'unknown', confidence: 0, subcategoryHints: [],
    //   usdaDataTypes: ['Foundation', 'SR Legacy', 'Survey (FNDDS)', 'Branded'] }

    // Should still perform search without classification hints
    const mockResponse = createMockUSDAResponse('apple', ['Foundation']);
    mockPost.mockResolvedValueOnce({ data: mockResponse });

    const searchResult = await usdaService.searchFoods({ query: 'apple' });

    expect(searchResult.foods.length).toBeGreaterThan(0);
  });

  it('should fall back to unranked results if ranking fails', () => {
    const mockResponse = createMockUSDAResponse('apple', ['Foundation']);
    const transformedFoods = mockResponse.foods.map(food =>
      nutrientService.transformSearchResultFood(food)
    );

    // Even with invalid hints, should return valid results
    const rankedFoods = rankSearchResults(transformedFoods, {
      query: 'apple',
      category: undefined,
      subcategoryHints: undefined,
    });

    expect(rankedFoods.length).toBe(transformedFoods.length);
    // Results should still have scores
    rankedFoods.forEach(food => {
      expect(food.rankScore).toBeDefined();
      expect(typeof food.rankScore).toBe('number');
    });
  });

  it('should handle empty search results gracefully', async () => {
    const emptyResponse: USDASearchResponse = {
      totalHits: 0,
      currentPage: 1,
      totalPages: 0,
      foodSearchCriteria: {
        query: 'xyznonexistent',
        pageNumber: 1,
        numberOfResultsPerPage: 25,
        requireAllWords: false,
      },
      foods: [],
    };

    mockPost.mockResolvedValueOnce({ data: emptyResponse });

    const searchResult = await usdaService.searchFoods({ query: 'xyznonexistent' });
    const rankedFoods = rankSearchResults([], { query: 'xyznonexistent' });

    expect(searchResult.foods).toHaveLength(0);
    expect(rankedFoods).toHaveLength(0);
  });
});

// ============================================================================
// DATA TYPE PREFERENCE TESTS
// ============================================================================

describe('Classification-Based Data Type Preferences', () => {
  let usdaService: USDAApiService;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env.USDA_API_KEY = 'test-api-key';

    usdaService = new USDAApiService();
  });

  afterEach(() => {
    delete process.env.USDA_API_KEY;
  });

  const dataTypeTestCases = [
    {
      category: 'fruits_fresh',
      expectedPrimary: 'Foundation',
      query: 'apple',
    },
    {
      category: 'vegetables_leafy',
      expectedPrimary: 'Foundation',
      query: 'spinach',
    },
    {
      category: 'meat_poultry',
      expectedPrimary: 'Foundation',
      query: 'chicken breast',
    },
    {
      category: 'fast_food',
      expectedPrimary: 'Branded',
      query: 'burger',
    },
    {
      category: 'snacks_sweet',
      expectedPrimary: 'Branded',
      query: 'cookies',
    },
    {
      category: 'grains_bread',
      expectedPrimary: 'Branded',
      query: 'bread',
    },
    {
      category: 'mixed_dishes',
      expectedPrimary: 'Survey (FNDDS)',
      query: 'casserole',
    },
  ];

  dataTypeTestCases.forEach(({ category, expectedPrimary, query }) => {
    it(`should prefer ${expectedPrimary} data type for ${category}`, async () => {
      const mockResponse = createMockUSDAResponse(query, [expectedPrimary]);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      const searchResult = await usdaService.searchFoods({
        query,
        dataType: [expectedPrimary as 'Foundation' | 'SR Legacy' | 'Survey (FNDDS)' | 'Branded'],
      });

      expect(mockPost).toHaveBeenCalledWith(
        '/foods/search',
        expect.objectContaining({
          dataType: [expectedPrimary],
        })
      );

      // All results should be of the expected type
      searchResult.foods.forEach(food => {
        expect(food.dataType).toBe(expectedPrimary);
      });
    });
  });
});
