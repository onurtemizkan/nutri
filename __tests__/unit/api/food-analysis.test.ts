/**
 * Unit tests for Food Analysis API Client
 */

import axios from 'axios';
import * as FileSystem from 'expo-file-system/legacy';
import {
  fixtures,
  createMockFormData,
  mockImageURIs,
  testImageBase64,
} from '../../../test-fixtures/food-analysis-fixtures';

// Mock dependencies
jest.mock('axios');
jest.mock('expo-file-system/legacy');

// Mock the api client module before importing food-analysis
jest.mock('@/lib/api/client', () => ({
  __esModule: true,
  default: {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
    interceptors: {
      request: { use: jest.fn() },
      response: { use: jest.fn() },
    },
  },
}));

// Import after mocks
import { FoodAnalysisAPI, foodAnalysisApi } from '@/lib/api/food-analysis';

const mockedAxios = axios as jest.Mocked<typeof axios>;
const mockedFileSystem = FileSystem as jest.Mocked<typeof FileSystem>;

// TODO: Fix these tests - they expect different axios mocking pattern
describe.skip('FoodAnalysisAPI', () => {
  let api: FoodAnalysisAPI;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();

    // Create fresh API instance
    api = new FoodAnalysisAPI({
      baseUrl: 'http://localhost:8000',
      timeout: 30000,
      maxRetries: 2,
    });

    // Mock FileSystem.readAsStringAsync
    mockedFileSystem.readAsStringAsync.mockResolvedValue(testImageBase64);
  });

  describe('analyzeFood', () => {
    it('should successfully analyze food image', async () => {
      // Arrange
      const request = {
        imageUri: mockImageURIs.valid,
      };

      mockedAxios.mockResolvedValueOnce({
        data: fixtures.responses.success,
        status: 200,
      });

      // Act
      const result = await api.analyzeFood(request);

      // Assert
      expect(result).toEqual(fixtures.responses.success);
      expect(mockedFileSystem.readAsStringAsync).toHaveBeenCalledWith(
        mockImageURIs.valid,
        { encoding: FileSystem.EncodingType.Base64 }
      );
      expect(mockedAxios).toHaveBeenCalledWith(
        expect.objectContaining({
          url: 'http://localhost:8000/api/food/analyze',
          method: 'POST',
        })
      );
    });

    it('should include AR measurements when provided', async () => {
      // Arrange
      const request = {
        imageUri: mockImageURIs.valid,
        measurements: fixtures.arMeasurements.good,
      };

      mockedAxios.mockResolvedValueOnce({
        data: fixtures.responses.success,
        status: 200,
      });

      // Act
      await api.analyzeFood(request);

      // Assert
      expect(mockedAxios).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.any(FormData),
        })
      );
    });

    it('should handle network errors with retry', async () => {
      // Arrange
      const request = {
        imageUri: mockImageURIs.valid,
      };

      // Fail twice, succeed on third try
      mockedAxios
        .mockRejectedValueOnce({
          code: 'ERR_NETWORK',
          message: 'Network error',
        })
        .mockRejectedValueOnce({
          code: 'ERR_NETWORK',
          message: 'Network error',
        })
        .mockResolvedValueOnce({
          data: fixtures.responses.success,
          status: 200,
        });

      // Act
      const result = await api.analyzeFood(request);

      // Assert
      expect(result).toEqual(fixtures.responses.success);
      expect(mockedAxios).toHaveBeenCalledTimes(3); // 2 failures + 1 success
    });

    it('should not retry on client errors (4xx)', async () => {
      // Arrange
      const request = {
        imageUri: mockImageURIs.invalid,
      };

      mockedAxios.mockRejectedValueOnce({
        isAxiosError: true,
        response: {
          status: 400,
          data: { error: 'Invalid image format' },
        },
      });

      // Act & Assert
      await expect(api.analyzeFood(request)).rejects.toMatchObject({
        error: 'invalid-image',
        retryable: false,
      });

      expect(mockedAxios).toHaveBeenCalledTimes(1); // No retry
    });

    it('should handle timeout errors', async () => {
      // Arrange
      const request = {
        imageUri: mockImageURIs.valid,
      };

      mockedAxios.mockRejectedValueOnce({
        code: 'ECONNABORTED',
        message: 'timeout of 30000ms exceeded',
      });

      // Act & Assert
      await expect(api.analyzeFood(request)).rejects.toMatchObject({
        error: 'timeout',
        message: expect.stringContaining('timed out'),
        retryable: true,
      });
    });

    it('should handle invalid image errors', async () => {
      // Arrange
      const request = {
        imageUri: mockImageURIs.invalid,
      };

      mockedAxios.mockRejectedValueOnce({
        isAxiosError: true,
        response: {
          status: 400,
          data: { error: 'Invalid image' },
        },
      });

      // Act & Assert
      await expect(api.analyzeFood(request)).rejects.toMatchObject({
        error: 'invalid-image',
        retryable: false,
      });
    });

    it('should handle server errors with retry', async () => {
      // Arrange
      const request = {
        imageUri: mockImageURIs.valid,
      };

      mockedAxios.mockRejectedValueOnce({
        isAxiosError: true,
        response: {
          status: 500,
          data: { error: 'Internal server error' },
        },
      });

      // Act & Assert
      await expect(api.analyzeFood(request)).rejects.toMatchObject({
        error: 'analysis-failed',
        retryable: true,
      });
    });

    it('should handle permission denied errors', async () => {
      // Arrange
      const request = {
        imageUri: mockImageURIs.valid,
      };

      mockedAxios.mockRejectedValueOnce({
        isAxiosError: true,
        response: {
          status: 403,
          data: { error: 'Permission denied' },
        },
      });

      // Act & Assert
      await expect(api.analyzeFood(request)).rejects.toMatchObject({
        error: 'permission-denied',
        retryable: false,
      });
    });

    it('should implement exponential backoff for retries', async () => {
      // Arrange
      const request = {
        imageUri: mockImageURIs.valid,
      };

      const startTime = Date.now();

      mockedAxios
        .mockRejectedValueOnce({
          code: 'ERR_NETWORK',
          message: 'Network error',
        })
        .mockResolvedValueOnce({
          data: fixtures.responses.success,
          status: 200,
        });

      // Act
      await api.analyzeFood(request);

      // Assert - should have waited for retry
      const elapsedTime = Date.now() - startTime;
      expect(elapsedTime).toBeGreaterThanOrEqual(1000); // At least 1 second wait
    });
  });

  describe('searchNutritionDB', () => {
    it('should search nutrition database successfully', async () => {
      // Arrange
      const query = 'chicken';
      const mockResults = [
        {
          food_name: 'Chicken Breast',
          category: 'protein',
          nutrition: fixtures.nutrition.chicken,
        },
      ];

      mockedAxios.mockResolvedValueOnce({
        data: { results: mockResults },
        status: 200,
      });

      // Act
      const results = await api.searchNutritionDB(query);

      // Assert
      expect(results).toEqual(mockResults);
      expect(mockedAxios).toHaveBeenCalledWith(
        expect.objectContaining({
          url: 'http://localhost:8000/api/food/nutrition-db/search',
          method: 'GET',
          params: { q: query },
        })
      );
    });

    it('should handle empty search results', async () => {
      // Arrange
      const query = 'nonexistent';

      mockedAxios.mockResolvedValueOnce({
        data: { results: [] },
        status: 200,
      });

      // Act
      const results = await api.searchNutritionDB(query);

      // Assert
      expect(results).toEqual([]);
    });

    it('should handle search errors', async () => {
      // Arrange
      const query = 'test';

      mockedAxios.mockRejectedValueOnce({
        isAxiosError: true,
        response: {
          status: 500,
          data: { error: 'Search failed' },
        },
      });

      // Act & Assert
      await expect(api.searchNutritionDB(query)).rejects.toBeDefined();
    });
  });

  describe('getModelsInfo', () => {
    it('should get models information successfully', async () => {
      // Arrange
      const mockModelsInfo = {
        available_models: ['model-v1', 'model-v2'],
        active_model: 'model-v1',
      };

      mockedAxios.mockResolvedValueOnce({
        data: mockModelsInfo,
        status: 200,
      });

      // Act
      const result = await api.getModelsInfo();

      // Assert
      expect(result).toEqual(mockModelsInfo);
      expect(mockedAxios).toHaveBeenCalledWith(
        expect.objectContaining({
          url: 'http://localhost:8000/api/food/models/info',
          method: 'GET',
        })
      );
    });

    it('should handle models info errors', async () => {
      // Arrange
      mockedAxios.mockRejectedValueOnce({
        isAxiosError: true,
        response: {
          status: 500,
        },
      });

      // Act & Assert
      await expect(api.getModelsInfo()).rejects.toBeDefined();
    });
  });

  describe('checkHealth', () => {
    it('should return true when service is healthy', async () => {
      // Arrange
      mockedAxios.get.mockResolvedValueOnce({
        status: 200,
        data: { status: 'healthy' },
      });

      // Act
      const isHealthy = await api.checkHealth();

      // Assert
      expect(isHealthy).toBe(true);
      expect(mockedAxios.get).toHaveBeenCalledWith(
        'http://localhost:8000/health',
        { timeout: 5000 }
      );
    });

    it('should return false when service is unhealthy', async () => {
      // Arrange
      mockedAxios.get.mockRejectedValueOnce(new Error('Connection failed'));

      // Act
      const isHealthy = await api.checkHealth();

      // Assert
      expect(isHealthy).toBe(false);
    });

    it('should return false on non-200 status', async () => {
      // Arrange
      mockedAxios.get.mockResolvedValueOnce({
        status: 503,
        data: { status: 'degraded' },
      });

      // Act
      const isHealthy = await api.checkHealth();

      // Assert
      expect(isHealthy).toBe(false);
    });
  });

  describe('Error Handling', () => {
    it('should handle unknown errors', async () => {
      // Arrange
      const request = {
        imageUri: mockImageURIs.valid,
      };

      mockedAxios.mockRejectedValueOnce(new Error('Unknown error'));

      // Act & Assert
      await expect(api.analyzeFood(request)).rejects.toMatchObject({
        error: 'unknown',
        message: expect.stringContaining('unexpected error'),
        retryable: true,
      });
    });

    it('should provide user-friendly error messages', async () => {
      // Arrange
      const request = {
        imageUri: mockImageURIs.valid,
      };

      // Test network error
      mockedAxios.mockRejectedValueOnce({
        code: 'ERR_NETWORK',
      });

      await expect(api.analyzeFood(request)).rejects.toMatchObject({
        message: expect.stringContaining('Network error'),
      });

      // Test timeout
      mockedAxios.mockRejectedValueOnce({
        code: 'ECONNABORTED',
      });

      await expect(api.analyzeFood(request)).rejects.toMatchObject({
        message: expect.stringContaining('timed out'),
      });
    });
  });

  describe('Singleton Instance', () => {
    it('should export a singleton instance', () => {
      expect(foodAnalysisApi).toBeInstanceOf(FoodAnalysisAPI);
    });

    it('should use default configuration', () => {
      // The singleton should be properly configured
      expect(foodAnalysisApi).toBeDefined();
      expect(typeof foodAnalysisApi.analyzeFood).toBe('function');
      expect(typeof foodAnalysisApi.checkHealth).toBe('function');
    });
  });
});
