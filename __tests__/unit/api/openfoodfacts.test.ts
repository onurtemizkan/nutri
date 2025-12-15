/**
 * Open Food Facts API Client Tests
 */

import {
  fetchProductByBarcode,
  calculateServingNutrition,
  formatNutritionGrade,
  getNutritionGradeColor,
  isValidBarcode,
  clearProductCache,
  getCacheStats,
} from '@/lib/api/openfoodfacts';
import type { BarcodeProduct } from '@/lib/types/barcode';

// Mock AsyncStorage
jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(() => Promise.resolve(null)),
  setItem: jest.fn(() => Promise.resolve()),
  removeItem: jest.fn(() => Promise.resolve()),
  getAllKeys: jest.fn(() => Promise.resolve([])),
  multiRemove: jest.fn(() => Promise.resolve()),
}));

// Mock axios
jest.mock('axios', () => {
  const mockAxiosInstance = {
    get: jest.fn(),
  };
  return {
    create: jest.fn(() => mockAxiosInstance),
    isAxiosError: jest.fn((error: unknown) => error && typeof error === 'object' && 'isAxiosError' in error && (error as { isAxiosError: boolean }).isAxiosError === true),
  };
});

import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

// Get the mock axios instance
const mockAxiosInstance = (axios.create as jest.Mock)() as jest.Mocked<{ get: jest.Mock }>;

describe('Open Food Facts API Client', () => {
  beforeEach(async () => {
    jest.clearAllMocks();
    // Clear any in-memory cache between tests
    await clearProductCache();
  });

  describe('isValidBarcode', () => {
    it('should validate EAN-13 barcodes (13 digits)', () => {
      expect(isValidBarcode('5449000000996')).toBe(true);
      expect(isValidBarcode('3017620422003')).toBe(true);
    });

    it('should validate EAN-8 barcodes (8 digits)', () => {
      expect(isValidBarcode('96385074')).toBe(true);
    });

    it('should validate UPC-A barcodes (12 digits)', () => {
      expect(isValidBarcode('012345678905')).toBe(true);
    });

    it('should reject invalid barcodes', () => {
      expect(isValidBarcode('')).toBe(false);
      expect(isValidBarcode('abc')).toBe(false);
      expect(isValidBarcode('123')).toBe(false); // Too short
      expect(isValidBarcode('12345678901234')).toBe(false); // Too long
      expect(isValidBarcode('123abc456')).toBe(false); // Contains letters
    });

    it('should trim whitespace before validation', () => {
      expect(isValidBarcode('  5449000000996  ')).toBe(true);
    });
  });

  describe('fetchProductByBarcode', () => {
    it('should return product data for valid barcode', async () => {
      const mockResponse = {
        data: {
          status: 1,
          status_verbose: 'product found',
          code: '5449000000996',
          product: {
            product_name: 'Coca Cola',
            brands: 'Coca-Cola',
            nutriments: {
              'energy-kcal_100g': 42,
              proteins_100g: 0,
              carbohydrates_100g: 10.6,
              fat_100g: 0,
              sugars_100g: 10.6,
            },
            serving_size: '330ml',
            nutriscore_grade: 'e',
          },
        },
      };

      mockAxiosInstance.get.mockResolvedValueOnce(mockResponse);

      const result = await fetchProductByBarcode('5449000000996');

      expect(result.success).toBe(true);
      expect(result.product).toBeDefined();
      expect(result.product?.name).toBe('Coca Cola');
      expect(result.product?.brand).toBe('Coca-Cola');
      expect(result.product?.nutrition.calories).toBe(42);
      expect(result.product?.nutrition.carbs).toBe(10.6);
    });

    it('should return error for product not found', async () => {
      const mockResponse = {
        data: {
          status: 0,
          status_verbose: 'product not found',
          code: '0000000000000',
        },
      };

      mockAxiosInstance.get.mockResolvedValueOnce(mockResponse);

      const result = await fetchProductByBarcode('0000000000000');

      expect(result.success).toBe(false);
      expect(result.error?.type).toBe('PRODUCT_NOT_FOUND');
    });

    it('should return error for invalid barcode format', async () => {
      const result = await fetchProductByBarcode('abc');

      expect(result.success).toBe(false);
      expect(result.error?.type).toBe('INVALID_BARCODE');
    });

    it('should handle network errors', async () => {
      const networkError = {
        isAxiosError: true,
        code: 'ECONNABORTED',
        response: undefined,
      };

      mockAxiosInstance.get.mockRejectedValueOnce(networkError);
      (axios.isAxiosError as jest.Mock).mockReturnValueOnce(true);

      const result = await fetchProductByBarcode('5449000000996');

      expect(result.success).toBe(false);
      expect(result.error?.type).toBe('NETWORK_ERROR');
    });

    it('should handle API errors', async () => {
      const apiError = {
        isAxiosError: true,
        response: {
          status: 500,
        },
      };

      mockAxiosInstance.get.mockRejectedValueOnce(apiError);
      (axios.isAxiosError as jest.Mock).mockReturnValueOnce(true);

      const result = await fetchProductByBarcode('5449000000996');

      expect(result.success).toBe(false);
      expect(result.error?.type).toBe('API_ERROR');
    });

    it('should map energy from kJ when kcal not available', async () => {
      const mockResponse = {
        data: {
          status: 1,
          code: '1234567890123',
          product: {
            product_name: 'Test Product',
            nutriments: {
              'energy-kj_100g': 418.4, // 100 kcal in kJ
              proteins_100g: 5,
              carbohydrates_100g: 10,
              fat_100g: 3,
            },
          },
        },
      };

      mockAxiosInstance.get.mockResolvedValueOnce(mockResponse);

      const result = await fetchProductByBarcode('1234567890123');

      expect(result.success).toBe(true);
      expect(result.product?.nutrition.calories).toBe(100); // Converted from kJ
    });
  });

  describe('calculateServingNutrition', () => {
    const mockProduct: BarcodeProduct = {
      barcode: '1234567890123',
      name: 'Test Product',
      nutrition: {
        calories: 100,
        protein: 10,
        carbs: 20,
        fat: 5,
        fiber: 2,
        sugar: 8,
      },
    };

    it('should calculate nutrition for 100g serving (no change)', () => {
      const result = calculateServingNutrition(mockProduct, 100);

      expect(result.calories).toBe(100);
      expect(result.protein).toBe(10);
      expect(result.carbs).toBe(20);
      expect(result.fat).toBe(5);
    });

    it('should calculate nutrition for 50g serving (half)', () => {
      const result = calculateServingNutrition(mockProduct, 50);

      expect(result.calories).toBe(50);
      expect(result.protein).toBe(5);
      expect(result.carbs).toBe(10);
      expect(result.fat).toBe(2.5);
    });

    it('should calculate nutrition for 200g serving (double)', () => {
      const result = calculateServingNutrition(mockProduct, 200);

      expect(result.calories).toBe(200);
      expect(result.protein).toBe(20);
      expect(result.carbs).toBe(40);
      expect(result.fat).toBe(10);
    });

    it('should handle optional fiber and sugar', () => {
      const productWithoutOptional: BarcodeProduct = {
        barcode: '1234567890123',
        name: 'Test Product',
        nutrition: {
          calories: 100,
          protein: 10,
          carbs: 20,
          fat: 5,
        },
      };

      const result = calculateServingNutrition(productWithoutOptional, 100);

      expect(result.fiber).toBeUndefined();
      expect(result.sugar).toBeUndefined();
    });
  });

  describe('formatNutritionGrade', () => {
    it('should format grade to uppercase', () => {
      expect(formatNutritionGrade('a')).toBe('A');
      expect(formatNutritionGrade('b')).toBe('B');
      expect(formatNutritionGrade('e')).toBe('E');
    });

    it('should return null for undefined grade', () => {
      expect(formatNutritionGrade(undefined)).toBeNull();
    });
  });

  describe('getNutritionGradeColor', () => {
    it('should return correct colors for each grade', () => {
      expect(getNutritionGradeColor('a')).toBe('#038141'); // Dark green
      expect(getNutritionGradeColor('b')).toBe('#85bb2f'); // Light green
      expect(getNutritionGradeColor('c')).toBe('#fecb02'); // Yellow
      expect(getNutritionGradeColor('d')).toBe('#ee8100'); // Orange
      expect(getNutritionGradeColor('e')).toBe('#e63e11'); // Red
    });

    it('should return gray for unknown grade', () => {
      expect(getNutritionGradeColor('x')).toBe('#888888');
      expect(getNutritionGradeColor(undefined)).toBe('#888888');
    });

    it('should be case insensitive', () => {
      expect(getNutritionGradeColor('A')).toBe('#038141');
      expect(getNutritionGradeColor('E')).toBe('#e63e11');
    });
  });

  describe('caching', () => {
    it('should cache product after successful fetch', async () => {
      const mockResponse = {
        data: {
          status: 1,
          code: '5449000000996',
          product: {
            product_name: 'Coca Cola',
            brands: 'Coca-Cola',
            nutriments: {
              'energy-kcal_100g': 42,
              proteins_100g: 0,
              carbohydrates_100g: 10.6,
              fat_100g: 0,
            },
          },
        },
      };

      mockAxiosInstance.get.mockResolvedValueOnce(mockResponse);

      // First call - should hit API
      const result1 = await fetchProductByBarcode('5449000000996');
      expect(result1.success).toBe(true);
      expect(mockAxiosInstance.get).toHaveBeenCalledTimes(1);

      // Verify setItem was called to cache the product
      expect(AsyncStorage.setItem).toHaveBeenCalled();
    });

    it('should return cached product on subsequent calls', async () => {
      const mockProduct = {
        barcode: '5449000000996',
        name: 'Coca Cola',
        brand: 'Coca-Cola',
        nutrition: {
          calories: 42,
          protein: 0,
          carbs: 10.6,
          fat: 0,
        },
      };

      const cachedEntry = {
        product: mockProduct,
        timestamp: Date.now(),
      };

      // Mock AsyncStorage to return cached product
      (AsyncStorage.getItem as jest.Mock).mockResolvedValueOnce(
        JSON.stringify(cachedEntry)
      );

      const result = await fetchProductByBarcode('5449000000996');

      expect(result.success).toBe(true);
      expect(result.product?.name).toBe('Coca Cola');
      // API should NOT be called since we have cached data
      expect(mockAxiosInstance.get).not.toHaveBeenCalled();
    });

    it('should skip cache when skipCache=true', async () => {
      const mockResponse = {
        data: {
          status: 1,
          code: '5449000000996',
          product: {
            product_name: 'Coca Cola Fresh',
            brands: 'Coca-Cola',
            nutriments: {
              'energy-kcal_100g': 42,
              proteins_100g: 0,
              carbohydrates_100g: 10.6,
              fat_100g: 0,
            },
          },
        },
      };

      const cachedEntry = {
        product: {
          barcode: '5449000000996',
          name: 'Coca Cola Old',
          brand: 'Coca-Cola',
          nutrition: { calories: 42, protein: 0, carbs: 10.6, fat: 0 },
        },
        timestamp: Date.now(),
      };

      (AsyncStorage.getItem as jest.Mock).mockResolvedValueOnce(
        JSON.stringify(cachedEntry)
      );
      mockAxiosInstance.get.mockResolvedValueOnce(mockResponse);

      const result = await fetchProductByBarcode('5449000000996', true);

      expect(result.success).toBe(true);
      expect(result.product?.name).toBe('Coca Cola Fresh');
      // API should be called despite cached data
      expect(mockAxiosInstance.get).toHaveBeenCalled();
    });

    it('should return initial cache stats', async () => {
      const stats = await getCacheStats();
      expect(stats.itemCount).toBe(0);
      expect(stats.memoryCount).toBe(0);
    });

    it('should clear product cache', async () => {
      (AsyncStorage.getAllKeys as jest.Mock).mockResolvedValueOnce([
        'off_product_123',
        'off_product_456',
      ]);

      await clearProductCache();

      expect(AsyncStorage.multiRemove).toHaveBeenCalledWith([
        'off_product_123',
        'off_product_456',
      ]);
    });
  });
});
