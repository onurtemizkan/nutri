/**
 * USDA Search Integration Tests
 *
 * Tests for search accuracy, pagination, filtering, and ranking:
 * - Search query accuracy
 * - Pagination handling
 * - Data type filtering
 * - Result ranking quality
 * - Search performance benchmarks
 */

// Create mock axios instance before anything else
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

// Mock axios before importing the service
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
jest.mock('../../config/redis', () => ({
  getRedisClient: jest.fn(() => null),
  isRedisAvailable: jest.fn(() => false),
}));

import { USDAApiService } from '../../services/usdaApiService';
import { NutrientMappingService } from '../../services/nutrientMappingService';
import { rankSearchResults, type RankingHints } from '../../utils/foodRanking';
import type { USDASearchResponse, USDASearchResultFood, TransformedUSDAFood } from '../../types/usda';

// ============================================================================
// TEST DATA
// ============================================================================

/**
 * Golden dataset of common foods with expected USDA search results
 * Used for search accuracy validation
 */
/**
 * Golden dataset for future live API integration tests
 * Currently mocked but preserved for documentation purposes
 */
void [
  { query: 'apple', expectedDataTypes: ['Foundation', 'SR Legacy'], minResults: 5 },
  { query: 'chicken breast', expectedDataTypes: ['Foundation', 'SR Legacy'], minResults: 3 },
  { query: 'banana', expectedDataTypes: ['Foundation', 'SR Legacy'], minResults: 5 },
  { query: 'brown rice', expectedDataTypes: ['Foundation', 'SR Legacy'], minResults: 3 },
  { query: 'salmon', expectedDataTypes: ['Foundation', 'SR Legacy'], minResults: 3 },
  { query: 'broccoli', expectedDataTypes: ['Foundation', 'SR Legacy'], minResults: 3 },
  { query: 'egg', expectedDataTypes: ['Foundation', 'SR Legacy'], minResults: 5 },
  { query: 'milk', expectedDataTypes: ['Foundation', 'SR Legacy'], minResults: 5 },
  { query: 'orange', expectedDataTypes: ['Foundation', 'SR Legacy'], minResults: 5 },
  { query: 'spinach', expectedDataTypes: ['Foundation', 'SR Legacy'], minResults: 3 },
];

/**
 * Mock USDA search response with complete data
 */
function createMockSearchResponse(
  query: string,
  totalHits: number = 100,
  pageNumber: number = 1,
  pageSize: number = 25,
  dataType: string = 'Foundation'
): USDASearchResponse {
  const foods: USDASearchResultFood[] = Array.from({ length: Math.min(pageSize, totalHits) }, (_, i) => ({
    fdcId: 170000 + i,
    description: `${query}, raw, form ${i + 1}`,
    dataType: dataType as USDASearchResultFood['dataType'],
    publishedDate: '2023-01-01',
    foodNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 50 + i * 5 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 0.3 + i * 0.1 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 12 + i },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 0.2 + i * 0.05 },
      { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 2 + i * 0.2 },
      { nutrientId: 2000, nutrientName: 'Sugars', nutrientNumber: '269', unitName: 'g', value: 8 + i * 0.5 },
      { nutrientId: 1093, nutrientName: 'Sodium', nutrientNumber: '307', unitName: 'mg', value: 1 + i },
    ],
    foodMeasures: [
      { id: 1, gramWeight: 100 + i * 10, disseminationText: '1 serving' },
    ],
  }));

  return {
    totalHits,
    currentPage: pageNumber,
    totalPages: Math.ceil(totalHits / pageSize),
    foodSearchCriteria: {
      query,
      pageNumber,
      numberOfResultsPerPage: pageSize,
      requireAllWords: false,
    },
    foods,
  };
}

// ============================================================================
// SEARCH ACCURACY TESTS
// ============================================================================

describe('USDA Search Accuracy', () => {
  let usdaService: USDAApiService;
  let nutrientService: NutrientMappingService;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env.USDA_API_KEY = 'test-api-key';
    usdaService = new USDAApiService();
    nutrientService = new NutrientMappingService();
  });

  afterEach(() => {
    delete process.env.USDA_API_KEY;
  });

  describe('basic search functionality', () => {
    it('should return relevant results for common food queries', async () => {
      const mockResponse = createMockSearchResponse('apple', 50);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      const result = await usdaService.searchFoods({ query: 'apple' });

      expect(result.foods.length).toBeGreaterThan(0);
      expect(result.totalHits).toBeGreaterThan(0);
      result.foods.forEach(food => {
        expect(food.description.toLowerCase()).toContain('apple');
      });
    });

    it('should return results with complete nutrient data', async () => {
      const mockResponse = createMockSearchResponse('banana', 30);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      const result = await usdaService.searchFoods({ query: 'banana' });

      result.foods.forEach(food => {
        // Core nutrients should be present
        const nutrientIds = food.foodNutrients.map(n => n.nutrientId);
        expect(nutrientIds).toContain(1008); // Calories
        expect(nutrientIds).toContain(1003); // Protein
        expect(nutrientIds).toContain(1005); // Carbs
        expect(nutrientIds).toContain(1004); // Fat
      });
    });

    it('should handle multi-word queries correctly', async () => {
      const mockResponse = createMockSearchResponse('chicken breast', 25);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      const result = await usdaService.searchFoods({
        query: 'chicken breast',
        requireAllWords: true
      });

      expect(result.foods.length).toBeGreaterThan(0);
    });
  });

  describe('data type filtering', () => {
    it('should filter by Foundation data type', async () => {
      const mockResponse = createMockSearchResponse('apple', 20, 1, 25, 'Foundation');
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      await usdaService.searchFoods({
        query: 'apple',
        dataType: ['Foundation'],
      });

      expect(mockPost).toHaveBeenCalledWith(
        '/foods/search',
        expect.objectContaining({
          dataType: ['Foundation'],
        })
      );
    });

    it('should filter by multiple data types', async () => {
      const mockResponse = createMockSearchResponse('apple', 20);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      await usdaService.searchFoods({
        query: 'apple',
        dataType: ['Foundation', 'SR Legacy'],
      });

      expect(mockPost).toHaveBeenCalledWith(
        '/foods/search',
        expect.objectContaining({
          dataType: ['Foundation', 'SR Legacy'],
        })
      );
    });

    it('should filter by Branded data type for packaged products', async () => {
      const mockResponse = createMockSearchResponse('protein bar', 50, 1, 25, 'Branded');
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      await usdaService.searchFoods({
        query: 'protein bar',
        dataType: ['Branded'],
      });

      expect(mockPost).toHaveBeenCalledWith(
        '/foods/search',
        expect.objectContaining({
          dataType: ['Branded'],
        })
      );
    });
  });

  describe('search result transformation', () => {
    it('should correctly transform search results to app format', async () => {
      const mockResponse = createMockSearchResponse('apple', 10);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      const result = await usdaService.searchFoods({ query: 'apple' });

      // Transform each result
      const transformedFoods = result.foods.map(food =>
        nutrientService.transformSearchResultFood(food)
      );

      transformedFoods.forEach(food => {
        expect(food).toHaveProperty('fdcId');
        expect(food).toHaveProperty('name');
        expect(food).toHaveProperty('description');
        expect(food).toHaveProperty('dataType');
        expect(food).toHaveProperty('nutrients');
        expect(food.nutrients).toHaveProperty('calories');
        expect(food.nutrients).toHaveProperty('protein');
        expect(food.nutrients).toHaveProperty('carbs');
        expect(food.nutrients).toHaveProperty('fat');
      });
    });

    it('should extract clean food names from descriptions', async () => {
      const mockResponse = createMockSearchResponse('apple', 5);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      const result = await usdaService.searchFoods({ query: 'apple' });
      const transformed = nutrientService.transformSearchResultFood(result.foods[0]);

      // Name should be extracted (first part before comma)
      expect(transformed.name).toBeTruthy();
      expect(typeof transformed.name).toBe('string');
    });
  });
});

// ============================================================================
// PAGINATION TESTS
// ============================================================================

describe('USDA Search Pagination', () => {
  let usdaService: USDAApiService;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env.USDA_API_KEY = 'test-api-key';
    usdaService = new USDAApiService();
  });

  afterEach(() => {
    delete process.env.USDA_API_KEY;
  });

  it('should handle first page correctly', async () => {
    const mockResponse = createMockSearchResponse('apple', 100, 1, 25);
    mockPost.mockResolvedValueOnce({ data: mockResponse });

    const result = await usdaService.searchFoods({
      query: 'apple',
      pageNumber: 1,
      pageSize: 25,
    });

    expect(result.currentPage).toBe(1);
    expect(result.foods.length).toBe(25);
    expect(result.totalPages).toBe(4);
  });

  it('should handle middle pages correctly', async () => {
    const mockResponse = createMockSearchResponse('apple', 100, 2, 25);
    mockPost.mockResolvedValueOnce({ data: mockResponse });

    const result = await usdaService.searchFoods({
      query: 'apple',
      pageNumber: 2,
      pageSize: 25,
    });

    expect(result.currentPage).toBe(2);
  });

  it('should handle last page with fewer results', async () => {
    // Total 100 items, page 4 with 25 per page should have the remaining items
    const mockResponse = createMockSearchResponse('apple', 100, 4, 25);
    mockPost.mockResolvedValueOnce({ data: mockResponse });

    const result = await usdaService.searchFoods({
      query: 'apple',
      pageNumber: 4,
      pageSize: 25,
    });

    expect(result.currentPage).toBe(4);
    expect(result.totalPages).toBe(4);
  });

  it('should respect custom page sizes', async () => {
    const mockResponse = createMockSearchResponse('apple', 100, 1, 50);
    mockPost.mockResolvedValueOnce({ data: mockResponse });

    const result = await usdaService.searchFoods({
      query: 'apple',
      pageSize: 50,
    });

    expect(result.foods.length).toBe(50);
    expect(result.totalPages).toBe(2);
  });

  it('should cap page size at maximum allowed', async () => {
    mockPost.mockResolvedValueOnce({ data: createMockSearchResponse('apple', 500, 1, 200) });

    await usdaService.searchFoods({
      query: 'apple',
      pageSize: 500, // Over limit
    });

    expect(mockPost).toHaveBeenCalledWith(
      '/foods/search',
      expect.objectContaining({
        pageSize: 200, // Capped at MAX_PAGE_SIZE
      })
    );
  });

  it('should calculate total pages correctly', async () => {
    // Test various total hits / page size combinations
    const testCases = [
      { totalHits: 100, pageSize: 25, expectedPages: 4 },
      { totalHits: 101, pageSize: 25, expectedPages: 5 },
      { totalHits: 50, pageSize: 50, expectedPages: 1 },
      { totalHits: 1, pageSize: 25, expectedPages: 1 },
    ];

    for (const tc of testCases) {
      jest.clearAllMocks();
      const mockResponse = createMockSearchResponse('test', tc.totalHits, 1, tc.pageSize);
      mockPost.mockResolvedValueOnce({ data: mockResponse });

      const result = await usdaService.searchFoods({
        query: 'test',
        pageSize: tc.pageSize,
      });

      expect(result.totalPages).toBe(tc.expectedPages);
    }
  });
});

// ============================================================================
// RANKING INTEGRATION TESTS
// ============================================================================

describe('Search Result Ranking', () => {
  beforeEach(() => {
    // NutrientMappingService is instantiated to verify ranking works
    // but not directly used in current tests
  });

  it('should rank Foundation foods higher than Branded for whole food queries', () => {
    const foods: TransformedUSDAFood[] = [
      {
        fdcId: 1,
        name: 'Apple Chips',
        description: 'Apple Chips, packaged snack',
        dataType: 'Branded',
        nutrients: { calories: 150, protein: 0, carbs: 30, fat: 5 },
      },
      {
        fdcId: 2,
        name: 'Apple',
        description: 'Apple, raw, with skin',
        dataType: 'Foundation',
        nutrients: { calories: 52, protein: 0.3, carbs: 14, fat: 0.2, fiber: 2.4 },
        servingSize: 100,
      },
    ];

    const hints: RankingHints = {
      query: 'apple',
      isWholeFoodQuery: true,
    };

    const ranked = rankSearchResults(foods, hints);

    expect(ranked[0].dataType).toBe('Foundation');
    expect(ranked[0].rankScore).toBeGreaterThan(ranked[1].rankScore);
  });

  it('should rank Branded foods higher for branded product queries', () => {
    const foods: TransformedUSDAFood[] = [
      {
        fdcId: 1,
        name: 'Cola',
        description: 'Cola nut, raw',
        dataType: 'Foundation',
        nutrients: { calories: 50, protein: 1, carbs: 10, fat: 0 },
      },
      {
        fdcId: 2,
        name: 'Coca-Cola',
        description: 'Coca-Cola, classic',
        dataType: 'Branded',
        nutrients: { calories: 140, protein: 0, carbs: 39, fat: 0 },
        brand: 'Coca-Cola',
      },
    ];

    const hints: RankingHints = {
      query: 'coca-cola',
      isBrandedQuery: true,
    };

    const ranked = rankSearchResults(foods, hints);

    // Branded should get boosted score
    const brandedResult = ranked.find(r => r.dataType === 'Branded');
    expect(brandedResult?.scoreBreakdown.dataTypeScore).toBe(80);
  });

  it('should prioritize exact name matches', () => {
    const foods: TransformedUSDAFood[] = [
      {
        fdcId: 1,
        name: 'Apple pie',
        description: 'Apple pie, homemade',
        dataType: 'Foundation',
        nutrients: { calories: 250, protein: 2, carbs: 40, fat: 10 },
      },
      {
        fdcId: 2,
        name: 'apple',
        description: 'apple',
        dataType: 'Foundation',
        nutrients: { calories: 52, protein: 0.3, carbs: 14, fat: 0.2 },
      },
    ];

    const hints: RankingHints = { query: 'apple' };
    const ranked = rankSearchResults(foods, hints);

    expect(ranked[0].description.toLowerCase()).toBe('apple');
    expect(ranked[0].scoreBreakdown.nameMatchScore).toBe(100);
  });

  it('should prioritize foods with complete nutrition data', () => {
    // Test with incomplete data - using type assertion since real USDA data
    // may sometimes be missing fields
    const foods: TransformedUSDAFood[] = [
      {
        fdcId: 1,
        name: 'Apple',
        description: 'Apple, incomplete data',
        dataType: 'Foundation',
        // Missing carbs and fat - truly incomplete core nutrients
        nutrients: { calories: 52, protein: 0 } as TransformedUSDAFood['nutrients'],
      },
      {
        fdcId: 2,
        name: 'Apple',
        description: 'Apple, complete data',
        dataType: 'Foundation',
        nutrients: {
          calories: 52,
          protein: 0.3,
          carbs: 14,
          fat: 0.2,
          fiber: 2.4,
          sugar: 10,
          sodium: 1,
          saturatedFat: 0.03,
        },
        servingSize: 100,
      },
    ];

    const ranked = rankSearchResults(foods);

    expect(ranked[0].fdcId).toBe(2); // Complete data should rank higher
    expect(ranked[0].scoreBreakdown.completenessScore).toBeGreaterThan(
      ranked[1].scoreBreakdown.completenessScore
    );
  });

  it('should use category hints from classification', () => {
    const foods: TransformedUSDAFood[] = [
      {
        fdcId: 1,
        name: 'Apple sauce',
        description: 'Apple sauce, baby food',
        dataType: 'Branded',
        category: 'Baby Foods',
        nutrients: { calories: 40, protein: 0, carbs: 10, fat: 0 },
      },
      {
        fdcId: 2,
        name: 'Apple',
        description: 'Apple, fresh fruit',
        dataType: 'Foundation',
        category: 'Fruits and Fruit Juices',
        nutrients: { calories: 52, protein: 0.3, carbs: 14, fat: 0.2 },
      },
    ];

    const hints: RankingHints = {
      query: 'apple',
      category: 'fruit',
    };

    const ranked = rankSearchResults(foods, hints);

    expect(ranked[0].fdcId).toBe(2); // Fruit category match should rank higher
  });
});

// ============================================================================
// PERFORMANCE BENCHMARKS
// ============================================================================

describe('Search Performance Benchmarks', () => {
  let nutrientService: NutrientMappingService;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env.USDA_API_KEY = 'test-api-key';
    nutrientService = new NutrientMappingService();
  });

  afterEach(() => {
    delete process.env.USDA_API_KEY;
  });

  it('should transform 100 search results in under 50ms', () => {
    const mockResponse = createMockSearchResponse('apple', 100, 1, 100);

    const startTime = performance.now();

    const transformedFoods = mockResponse.foods.map(food =>
      nutrientService.transformSearchResultFood(food)
    );

    const endTime = performance.now();
    const duration = endTime - startTime;

    expect(transformedFoods.length).toBe(100);
    expect(duration).toBeLessThan(50); // Should transform in under 50ms
  });

  it('should rank 100 search results in under 20ms', () => {
    const mockResponse = createMockSearchResponse('apple', 100, 1, 100);
    const transformedFoods = mockResponse.foods.map(food =>
      nutrientService.transformSearchResultFood(food)
    );

    const hints: RankingHints = {
      query: 'apple',
      isWholeFoodQuery: true,
    };

    const startTime = performance.now();

    const rankedFoods = rankSearchResults(transformedFoods, hints);

    const endTime = performance.now();
    const duration = endTime - startTime;

    expect(rankedFoods.length).toBe(100);
    expect(duration).toBeLessThan(20); // Should rank in under 20ms
  });

  it('should complete full search -> transform -> rank pipeline in under 100ms', () => {
    const mockResponse = createMockSearchResponse('apple', 50, 1, 50);

    const hints: RankingHints = {
      query: 'apple',
      isWholeFoodQuery: true,
      category: 'fruit',
    };

    const startTime = performance.now();

    // Transform
    const transformedFoods = mockResponse.foods.map(food =>
      nutrientService.transformSearchResultFood(food)
    );

    // Rank
    const rankedFoods = rankSearchResults(transformedFoods, hints);

    const endTime = performance.now();
    const duration = endTime - startTime;

    expect(rankedFoods.length).toBe(50);
    expect(duration).toBeLessThan(100); // Full pipeline under 100ms
  });
});

// ============================================================================
// EDGE CASES
// ============================================================================

describe('Search Edge Cases', () => {
  let usdaService: USDAApiService;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env.USDA_API_KEY = 'test-api-key';
    usdaService = new USDAApiService();
  });

  afterEach(() => {
    delete process.env.USDA_API_KEY;
  });

  it('should handle empty search results gracefully', async () => {
    const emptyResponse: USDASearchResponse = {
      totalHits: 0,
      currentPage: 1,
      totalPages: 0,
      foodSearchCriteria: {
        query: 'xyznonexistent123',
        pageNumber: 1,
        numberOfResultsPerPage: 25,
        requireAllWords: false,
      },
      foods: [],
    };
    mockPost.mockResolvedValueOnce({ data: emptyResponse });

    const result = await usdaService.searchFoods({ query: 'xyznonexistent123' });

    expect(result.foods).toHaveLength(0);
    expect(result.totalHits).toBe(0);
  });

  it('should handle special characters in search query', async () => {
    const mockResponse = createMockSearchResponse("McDonald's", 5);
    mockPost.mockResolvedValueOnce({ data: mockResponse });

    const result = await usdaService.searchFoods({ query: "McDonald's burger" });

    expect(mockPost).toHaveBeenCalledWith(
      '/foods/search',
      expect.objectContaining({
        query: "McDonald's burger",
      })
    );
    expect(result.foods.length).toBeGreaterThan(0);
  });

  it('should handle unicode characters in search query', async () => {
    const mockResponse = createMockSearchResponse('tofu', 10);
    mockPost.mockResolvedValueOnce({ data: mockResponse });

    await usdaService.searchFoods({ query: 'tofu (豆腐)' });

    expect(mockPost).toHaveBeenCalled();
  });

  it('should trim whitespace from queries', async () => {
    const mockResponse = createMockSearchResponse('apple', 10);
    mockPost.mockResolvedValueOnce({ data: mockResponse });

    await usdaService.searchFoods({ query: '  apple  ' });

    expect(mockPost).toHaveBeenCalledWith(
      '/foods/search',
      expect.objectContaining({
        query: 'apple', // Trimmed
      })
    );
  });
});
