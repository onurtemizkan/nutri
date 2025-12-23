/**
 * USDA API Service Tests
 *
 * Tests for the USDA FoodData Central API client with:
 * - Rate limiting behavior
 * - Error handling and retries
 * - Response transformation
 */

// Create the mock axios instance before anything else
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
  },
  createChildLogger: jest.fn(() => ({
    warn: jest.fn(),
    debug: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
  })),
}));

import { AxiosError } from 'axios';
import {
  USDASearchResponse,
  USDAFoodItem,
  USDAApiError,
  USDAApiRateLimitError,
  USDAApiTimeoutError,
} from '../../types/usda';
import { USDAApiService } from '../../services/usdaApiService';

describe('USDAApiService', () => {
  let service: USDAApiService;

  const mockSearchResponse: USDASearchResponse = {
    totalHits: 100,
    currentPage: 1,
    totalPages: 4,
    foodSearchCriteria: {
      query: 'apple',
      pageNumber: 1,
      numberOfResultsPerPage: 25,
      requireAllWords: false,
    },
    foods: [
      {
        fdcId: 171688,
        description: 'Apples, raw, with skin',
        dataType: 'Foundation',
        publishedDate: '2019-04-01',
        foodNutrients: [
          {
            nutrientId: 1008,
            nutrientName: 'Energy',
            nutrientNumber: '208',
            unitName: 'kcal',
            value: 52,
          },
          {
            nutrientId: 1003,
            nutrientName: 'Protein',
            nutrientNumber: '203',
            unitName: 'g',
            value: 0.26,
          },
        ],
        foodMeasures: [],
      },
    ],
  };

  const mockFoodItem: USDAFoodItem = {
    fdcId: 171688,
    description: 'Apples, raw, with skin',
    dataType: 'Foundation',
    publicationDate: '2019-04-01',
    foodNutrients: [
      {
        nutrientId: 1008,
        nutrientName: 'Energy',
        nutrientNumber: '208',
        unitName: 'kcal',
        value: 52,
        derivationCode: 'NC',
        derivationDescription: 'Calculated',
      },
      {
        nutrientId: 1003,
        nutrientName: 'Protein',
        nutrientNumber: '203',
        unitName: 'g',
        value: 0.26,
        derivationCode: 'A',
        derivationDescription: 'Analytical',
      },
    ],
    foodPortions: [
      {
        id: 1,
        measureUnit: { id: 1, name: 'cup, sliced', abbreviation: 'cup' },
        gramWeight: 110,
        sequenceNumber: 1,
      },
    ],
  };

  beforeEach(() => {
    jest.clearAllMocks();

    // Set USDA API key for tests
    process.env.USDA_API_KEY = 'test-api-key';

    service = new USDAApiService();
  });

  afterEach(() => {
    delete process.env.USDA_API_KEY;
  });

  describe('searchFoods', () => {
    it('should search foods successfully', async () => {
      mockPost.mockResolvedValueOnce({ data: mockSearchResponse });

      const result = await service.searchFoods({ query: 'apple' });

      expect(result).toEqual(mockSearchResponse);
      expect(mockPost).toHaveBeenCalledWith(
        '/foods/search',
        expect.objectContaining({
          query: 'apple',
          pageNumber: 1,
          pageSize: 25,
          sortBy: 'dataType.keyword',
          sortOrder: 'asc',
          requireAllWords: false,
        })
      );
    });

    it('should throw error for empty query', async () => {
      await expect(service.searchFoods({ query: '' })).rejects.toThrow(
        USDAApiError
      );
      await expect(service.searchFoods({ query: '   ' })).rejects.toThrow(
        USDAApiError
      );
    });

    it('should respect pageSize limit', async () => {
      mockPost.mockResolvedValueOnce({ data: mockSearchResponse });

      await service.searchFoods({ query: 'apple', pageSize: 500 });

      expect(mockPost).toHaveBeenCalledWith(
        '/foods/search',
        expect.objectContaining({
          pageSize: 200, // Should be capped at MAX_PAGE_SIZE
        })
      );
    });

    it('should support dataType filtering', async () => {
      mockPost.mockResolvedValueOnce({ data: mockSearchResponse });

      await service.searchFoods({
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

    it('should handle pagination options', async () => {
      mockPost.mockResolvedValueOnce({ data: mockSearchResponse });

      await service.searchFoods({
        query: 'apple',
        pageNumber: 3,
        pageSize: 50,
      });

      expect(mockPost).toHaveBeenCalledWith(
        '/foods/search',
        expect.objectContaining({
          pageNumber: 3,
          pageSize: 50,
        })
      );
    });
  });

  describe('getFoodById', () => {
    it('should get food by ID successfully', async () => {
      mockGet.mockResolvedValueOnce({ data: mockFoodItem });

      const result = await service.getFoodById(171688);

      expect(result).toEqual(mockFoodItem);
      expect(mockGet).toHaveBeenCalledWith('/food/171688', {
        params: { format: 'full' },
      });
    });

    it('should throw error for invalid FDC ID', async () => {
      await expect(service.getFoodById(0)).rejects.toThrow(USDAApiError);
      await expect(service.getFoodById(-1)).rejects.toThrow(USDAApiError);
    });

    it('should support format option', async () => {
      mockGet.mockResolvedValueOnce({ data: mockFoodItem });

      await service.getFoodById(171688, { format: 'abridged' });

      expect(mockGet).toHaveBeenCalledWith('/food/171688', {
        params: { format: 'abridged' },
      });
    });

    it('should support nutrient filtering', async () => {
      mockGet.mockResolvedValueOnce({ data: mockFoodItem });

      await service.getFoodById(171688, { nutrients: [1003, 1004, 1005] });

      expect(mockGet).toHaveBeenCalledWith('/food/171688', {
        params: expect.objectContaining({
          format: 'full',
          nutrients: ['1003', '1004', '1005'],
        }),
      });
    });
  });

  describe('getFoodsByIds', () => {
    it('should get multiple foods by IDs', async () => {
      const multipleFoods = [mockFoodItem, { ...mockFoodItem, fdcId: 171689 }];
      mockPost.mockResolvedValueOnce({ data: multipleFoods });

      const result = await service.getFoodsByIds([171688, 171689]);

      expect(result).toEqual(multipleFoods);
      expect(mockPost).toHaveBeenCalledWith('/foods', {
        fdcIds: [171688, 171689],
        format: 'full',
      });
    });

    it('should throw error for empty ID array', async () => {
      await expect(service.getFoodsByIds([])).rejects.toThrow(USDAApiError);
    });

    it('should filter out invalid IDs', async () => {
      mockPost.mockResolvedValueOnce({ data: [mockFoodItem] });

      await service.getFoodsByIds([171688, 0, -1, 171689]);

      expect(mockPost).toHaveBeenCalledWith('/foods', {
        fdcIds: [171688, 171689],
        format: 'full',
      });
    });

    it('should batch requests for more than 20 items', async () => {
      const ids = Array.from({ length: 25 }, (_, i) => i + 1);
      const batch1 = [mockFoodItem];
      const batch2 = [{ ...mockFoodItem, fdcId: 21 }];

      mockPost
        .mockResolvedValueOnce({ data: batch1 })
        .mockResolvedValueOnce({ data: batch2 });

      const result = await service.getFoodsByIds(ids);

      expect(mockPost).toHaveBeenCalledTimes(2);
      expect(result).toHaveLength(2);
    });
  });

  describe('error handling', () => {
    it('should handle rate limit errors', async () => {
      const rateLimitError = {
        isAxiosError: true,
        response: {
          status: 429,
          data: { error: 'Rate limit exceeded' },
          headers: { 'retry-after': '60' },
        },
        message: 'Request failed with status code 429',
        config: { url: '/foods/search' },
      } as unknown as AxiosError;

      mockPost.mockRejectedValue(rateLimitError);

      await expect(service.searchFoods({ query: 'apple' })).rejects.toThrow(
        USDAApiRateLimitError
      );
    });

    it('should handle timeout errors', async () => {
      const timeoutError = {
        isAxiosError: true,
        code: 'ECONNABORTED',
        message: 'timeout of 10000ms exceeded',
        config: { url: '/foods/search' },
      } as unknown as AxiosError;

      mockPost.mockRejectedValue(timeoutError);

      await expect(service.searchFoods({ query: 'apple' })).rejects.toThrow(
        USDAApiTimeoutError
      );
    });

    it('should retry on server errors (5xx)', async () => {
      const serverError = {
        isAxiosError: true,
        response: {
          status: 503,
          data: { error: 'Service Unavailable' },
        },
        message: 'Request failed with status code 503',
        config: { url: '/foods/search' },
      } as unknown as AxiosError;

      // Fail twice, then succeed
      mockPost
        .mockRejectedValueOnce(serverError)
        .mockRejectedValueOnce(serverError)
        .mockResolvedValueOnce({ data: mockSearchResponse });

      const result = await service.searchFoods({ query: 'apple' });

      expect(result).toEqual(mockSearchResponse);
      expect(mockPost).toHaveBeenCalledTimes(3);
    }, 15000);

    it('should not retry on client errors (4xx)', async () => {
      const clientError = {
        isAxiosError: true,
        response: {
          status: 400,
          data: { error: 'Bad Request' },
        },
        message: 'Request failed with status code 400',
        config: { url: '/foods/search' },
      } as unknown as AxiosError;

      mockPost.mockRejectedValue(clientError);

      await expect(service.searchFoods({ query: 'apple' })).rejects.toThrow(
        USDAApiError
      );
      expect(mockPost).toHaveBeenCalledTimes(1);
    });

    it('should fail after max retries', async () => {
      const serverError = {
        isAxiosError: true,
        response: {
          status: 500,
          data: { error: 'Internal Server Error' },
        },
        message: 'Request failed with status code 500',
        config: { url: '/foods/search' },
      } as unknown as AxiosError;

      mockPost.mockRejectedValue(serverError);

      await expect(service.searchFoods({ query: 'apple' })).rejects.toThrow(
        USDAApiError
      );
      expect(mockPost).toHaveBeenCalledTimes(3); // MAX_RETRIES
    }, 30000);
  });

  describe('healthCheck', () => {
    it('should return healthy when API is accessible', async () => {
      mockPost.mockResolvedValueOnce({ data: mockSearchResponse });

      const result = await service.healthCheck();

      expect(result).toEqual({
        healthy: true,
        message: 'USDA API is accessible',
      });
    });

    it('should return unhealthy when API key is not configured', async () => {
      delete process.env.USDA_API_KEY;
      const newService = new USDAApiService();

      const result = await newService.healthCheck();

      expect(result).toEqual({
        healthy: false,
        message: 'USDA_API_KEY not configured',
      });
    });

    it('should return unhealthy when API call fails', async () => {
      const error = new Error('Network error');
      mockPost.mockRejectedValue(error);

      const result = await service.healthCheck();

      expect(result.healthy).toBe(false);
      expect(result.message).toContain('error');
    });
  });

  describe('getRateLimitStatus', () => {
    it('should return rate limit status', () => {
      const status = service.getRateLimitStatus();

      expect(status).toHaveProperty('available');
      expect(status).toHaveProperty('max');
      expect(typeof status.available).toBe('number');
      expect(typeof status.max).toBe('number');
      expect(status.max).toBe(1000); // Default rate limit
    });
  });
});

describe('USDA Error Classes', () => {
  describe('USDAApiError', () => {
    it('should create error with correct properties', () => {
      const error = new USDAApiError('Test error', 404, 'NOT_FOUND', false);

      expect(error.message).toBe('Test error');
      expect(error.statusCode).toBe(404);
      expect(error.errorCode).toBe('NOT_FOUND');
      expect(error.retryable).toBe(false);
      expect(error.name).toBe('USDAApiError');
    });
  });

  describe('USDAApiRateLimitError', () => {
    it('should create rate limit error', () => {
      const error = new USDAApiRateLimitError('Rate limited', 60000);

      expect(error.message).toBe('Rate limited');
      expect(error.statusCode).toBe(429);
      expect(error.errorCode).toBe('USDA_RATE_LIMIT');
      expect(error.retryable).toBe(true);
      expect(error.retryAfter).toBe(60000);
      expect(error.name).toBe('USDAApiRateLimitError');
    });
  });

  describe('USDAApiTimeoutError', () => {
    it('should create timeout error', () => {
      const error = new USDAApiTimeoutError();

      expect(error.message).toBe('USDA API request timed out');
      expect(error.statusCode).toBe(408);
      expect(error.errorCode).toBe('USDA_TIMEOUT');
      expect(error.retryable).toBe(true);
      expect(error.name).toBe('USDAApiTimeoutError');
    });
  });
});
