/**
 * Open Food Facts API Client
 *
 * A client for fetching product information from the Open Food Facts database.
 * API documentation: https://world.openfoodfacts.org/data
 *
 * The API is free to use and contains over 2 million products worldwide.
 * Rate limits: ~10 requests/second for search, single product lookups less restricted
 */

import axios, { AxiosError } from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  OpenFoodFactsResponse,
  OpenFoodFactsProduct,
  BarcodeProduct,
  BarcodeScanResult,
  BarcodeError,
} from '../types/barcode';

// Base URL for Open Food Facts API
const OPEN_FOOD_FACTS_API_BASE = 'https://world.openfoodfacts.org/api/v2';

// Request timeout in milliseconds
const REQUEST_TIMEOUT = 10000;

// User agent header (required by Open Food Facts API)
const USER_AGENT = 'Nutri-App/1.0 (https://github.com/nutri-app)';

// Cache configuration
const CACHE_PREFIX = 'off_product_';
const CACHE_EXPIRY_MS = 7 * 24 * 60 * 60 * 1000; // 7 days
const MAX_CACHE_ITEMS = 100; // Maximum cached products

/**
 * In-memory cache for faster repeated lookups in same session
 */
const memoryCache = new Map<string, { product: BarcodeProduct; timestamp: number }>();

/**
 * Cache entry structure for persistent storage
 */
interface CacheEntry {
  product: BarcodeProduct;
  timestamp: number;
}

/**
 * Axios instance configured for Open Food Facts API
 */
const openFoodFactsApi = axios.create({
  baseURL: OPEN_FOOD_FACTS_API_BASE,
  timeout: REQUEST_TIMEOUT,
  headers: {
    'User-Agent': USER_AGENT,
    Accept: 'application/json',
  },
});

/**
 * Maps Open Food Facts product data to our app's BarcodeProduct schema
 */
function mapProductToAppSchema(
  product: OpenFoodFactsProduct,
  barcode: string
): BarcodeProduct {
  const nutriments = product.nutriments || {};

  // Get calories - prefer kcal, fall back to kJ conversion
  let calories = nutriments['energy-kcal_100g'] || 0;
  if (!calories && nutriments['energy-kj_100g']) {
    // Convert kJ to kcal (1 kcal = 4.184 kJ)
    calories = Math.round(nutriments['energy-kj_100g'] / 4.184);
  }
  if (!calories && nutriments.energy_100g) {
    // Check if unit is kJ or kcal
    if (nutriments.energy_unit === 'kJ') {
      calories = Math.round(nutriments.energy_100g / 4.184);
    } else {
      calories = nutriments.energy_100g;
    }
  }

  // Get product name - try multiple fields
  const name =
    product.product_name ||
    product.product_name_en ||
    product.generic_name ||
    'Unknown Product';

  // Get serving size info
  const servingSize = product.serving_size || '100g';
  const servingQuantity = product.serving_quantity || 100;

  // Get image URL - prefer front image
  const imageUrl =
    product.image_front_small_url ||
    product.image_front_url ||
    product.image_url;

  // Get allergens as array
  const allergens = product.allergens_tags
    ? product.allergens_tags.map((tag) => tag.replace('en:', ''))
    : undefined;

  // Get categories as array
  const categories = product.categories_tags
    ? product.categories_tags
        .slice(0, 5)
        .map((tag) => tag.replace('en:', '').replace(/-/g, ' '))
    : undefined;

  return {
    barcode,
    name,
    brand: product.brands || product.brand_owner,
    servingSize,
    servingQuantity,
    imageUrl,
    nutrition: {
      calories: Math.round(calories),
      protein: Math.round((nutriments.proteins_100g || 0) * 10) / 10,
      carbs: Math.round((nutriments.carbohydrates_100g || 0) * 10) / 10,
      fat: Math.round((nutriments.fat_100g || 0) * 10) / 10,
      fiber:
        nutriments.fiber_100g !== undefined
          ? Math.round(nutriments.fiber_100g * 10) / 10
          : undefined,
      sugar:
        nutriments.sugars_100g !== undefined
          ? Math.round(nutriments.sugars_100g * 10) / 10
          : undefined,
    },
    nutriscoreGrade:
      product.nutriscore_grade || product.nutrition_grades || undefined,
    categories,
    ingredients: product.ingredients_text_en || product.ingredients_text,
    allergens: allergens?.length ? allergens : undefined,
  };
}

/**
 * Creates a BarcodeError object
 */
function createBarcodeError(
  type: BarcodeError['type'],
  message: string,
  barcode?: string
): BarcodeError {
  return { type, message, barcode };
}

// ============================================================================
// CACHING FUNCTIONS
// ============================================================================

/**
 * Gets a product from cache (memory first, then persistent)
 */
async function getFromCache(barcode: string): Promise<BarcodeProduct | null> {
  const now = Date.now();

  // Check memory cache first (fastest)
  const memCached = memoryCache.get(barcode);
  if (memCached && now - memCached.timestamp < CACHE_EXPIRY_MS) {
    return memCached.product;
  }

  // Check persistent cache
  try {
    const stored = await AsyncStorage.getItem(`${CACHE_PREFIX}${barcode}`);
    if (stored) {
      const entry: CacheEntry = JSON.parse(stored);
      if (now - entry.timestamp < CACHE_EXPIRY_MS) {
        // Update memory cache
        memoryCache.set(barcode, entry);
        return entry.product;
      } else {
        // Remove expired entry
        await AsyncStorage.removeItem(`${CACHE_PREFIX}${barcode}`);
      }
    }
  } catch (error) {
    console.warn('Cache read error:', error);
  }

  return null;
}

/**
 * Saves a product to cache (both memory and persistent)
 */
async function saveToCache(product: BarcodeProduct): Promise<void> {
  const entry: CacheEntry = {
    product,
    timestamp: Date.now(),
  };

  // Save to memory cache
  memoryCache.set(product.barcode, entry);

  // Save to persistent cache
  try {
    await AsyncStorage.setItem(
      `${CACHE_PREFIX}${product.barcode}`,
      JSON.stringify(entry)
    );

    // Clean up old entries if we have too many
    await cleanupCache();
  } catch (error) {
    console.warn('Cache write error:', error);
  }
}

/**
 * Cleans up expired and excess cache entries
 */
async function cleanupCache(): Promise<void> {
  try {
    const keys = await AsyncStorage.getAllKeys();
    const cacheKeys = keys.filter((k: string) => k.startsWith(CACHE_PREFIX));

    if (cacheKeys.length <= MAX_CACHE_ITEMS) {
      return;
    }

    // Get all cache entries with timestamps
    const entries: { key: string; timestamp: number }[] = [];
    const now = Date.now();

    for (const key of cacheKeys) {
      const stored = await AsyncStorage.getItem(key);
      if (stored) {
        const entry: CacheEntry = JSON.parse(stored);
        // Remove expired entries
        if (now - entry.timestamp >= CACHE_EXPIRY_MS) {
          await AsyncStorage.removeItem(key);
        } else {
          entries.push({ key, timestamp: entry.timestamp });
        }
      }
    }

    // If still too many, remove oldest entries
    if (entries.length > MAX_CACHE_ITEMS) {
      entries.sort((a, b) => a.timestamp - b.timestamp);
      const toRemove = entries.slice(0, entries.length - MAX_CACHE_ITEMS);
      for (const { key } of toRemove) {
        await AsyncStorage.removeItem(key);
      }
    }
  } catch (error) {
    console.warn('Cache cleanup error:', error);
  }
}

/**
 * Clears all cached products
 */
export async function clearProductCache(): Promise<void> {
  try {
    const keys = await AsyncStorage.getAllKeys();
    const cacheKeys = keys.filter((k: string) => k.startsWith(CACHE_PREFIX));
    await AsyncStorage.multiRemove(cacheKeys);
    memoryCache.clear();
  } catch (error) {
    console.warn('Cache clear error:', error);
  }
}

/**
 * Gets cache statistics
 */
export async function getCacheStats(): Promise<{
  itemCount: number;
  memoryCount: number;
}> {
  try {
    const keys = await AsyncStorage.getAllKeys();
    const cacheKeys = keys.filter((k: string) => k.startsWith(CACHE_PREFIX));
    return {
      itemCount: cacheKeys.length,
      memoryCount: memoryCache.size,
    };
  } catch {
    return { itemCount: 0, memoryCount: memoryCache.size };
  }
}

/**
 * Validates a barcode format
 * Supports EAN-13, EAN-8, UPC-A, UPC-E
 */
export function isValidBarcode(barcode: string): boolean {
  // Remove any whitespace
  const cleanBarcode = barcode.trim();

  // Check if it's numeric
  if (!/^\d+$/.test(cleanBarcode)) {
    return false;
  }

  // Valid lengths: 8 (EAN-8, UPC-E), 12 (UPC-A), 13 (EAN-13)
  const validLengths = [8, 12, 13];
  return validLengths.includes(cleanBarcode.length);
}

/**
 * Fetches product information from Open Food Facts by barcode
 * Uses a two-tier cache (memory + persistent) for fast lookups and offline support
 *
 * @param barcode - The product barcode (EAN-13, EAN-8, UPC-A, or UPC-E)
 * @param skipCache - If true, bypasses cache and fetches fresh data
 * @returns A BarcodeScanResult object with product data or error information
 */
export async function fetchProductByBarcode(
  barcode: string,
  skipCache = false
): Promise<BarcodeScanResult> {
  // Clean and validate barcode
  const cleanBarcode = barcode.trim();

  if (!isValidBarcode(cleanBarcode)) {
    return {
      success: false,
      barcode: cleanBarcode,
      error: createBarcodeError(
        'INVALID_BARCODE',
        'Invalid barcode format. Please scan a valid EAN-13, EAN-8, UPC-A, or UPC-E barcode.',
        cleanBarcode
      ),
    };
  }

  // Check cache first (unless skipped)
  if (!skipCache) {
    const cachedProduct = await getFromCache(cleanBarcode);
    if (cachedProduct) {
      return {
        success: true,
        barcode: cleanBarcode,
        product: cachedProduct,
      };
    }
  }

  try {
    // Fetch product from Open Food Facts API
    const response = await openFoodFactsApi.get<OpenFoodFactsResponse>(
      `/product/${cleanBarcode}.json`,
      {
        params: {
          fields:
            'code,product_name,product_name_en,generic_name,brands,brand_owner,categories_tags,serving_size,serving_quantity,quantity,nutriments,nutriscore_grade,nutrition_grades,nova_group,image_url,image_front_url,image_front_small_url,ingredients_text,ingredients_text_en,allergens_tags',
        },
      }
    );

    const data = response.data;

    // Check if product was found
    if (data.status === 0 || !data.product) {
      return {
        success: false,
        barcode: cleanBarcode,
        error: createBarcodeError(
          'PRODUCT_NOT_FOUND',
          'Product not found in database. You can add it manually.',
          cleanBarcode
        ),
      };
    }

    // Map product to app schema
    const product = mapProductToAppSchema(data.product, cleanBarcode);

    // Save to cache for future lookups
    await saveToCache(product);

    return {
      success: true,
      barcode: cleanBarcode,
      product,
    };
  } catch (error) {
    // Handle network errors
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;

      if (axiosError.code === 'ECONNABORTED') {
        return {
          success: false,
          barcode: cleanBarcode,
          error: createBarcodeError(
            'NETWORK_ERROR',
            'Request timed out. Please check your internet connection.',
            cleanBarcode
          ),
        };
      }

      if (!axiosError.response) {
        return {
          success: false,
          barcode: cleanBarcode,
          error: createBarcodeError(
            'NETWORK_ERROR',
            'Unable to connect to the server. Please check your internet connection.',
            cleanBarcode
          ),
        };
      }

      // Handle API errors
      return {
        success: false,
        barcode: cleanBarcode,
        error: createBarcodeError(
          'API_ERROR',
          `Server error (${axiosError.response.status}). Please try again later.`,
          cleanBarcode
        ),
      };
    }

    // Handle unexpected errors
    return {
      success: false,
      barcode: cleanBarcode,
      error: createBarcodeError(
        'API_ERROR',
        'An unexpected error occurred. Please try again.',
        cleanBarcode
      ),
    };
  }
}

/**
 * Calculates nutrition values for a given serving amount
 *
 * @param product - The BarcodeProduct with per-100g nutrition values
 * @param servingGrams - The serving size in grams
 * @returns Nutrition values adjusted for the serving size
 */
export function calculateServingNutrition(
  product: BarcodeProduct,
  servingGrams: number
): BarcodeProduct['nutrition'] {
  const multiplier = servingGrams / 100;

  return {
    calories: Math.round(product.nutrition.calories * multiplier),
    protein: Math.round(product.nutrition.protein * multiplier * 10) / 10,
    carbs: Math.round(product.nutrition.carbs * multiplier * 10) / 10,
    fat: Math.round(product.nutrition.fat * multiplier * 10) / 10,
    fiber: product.nutrition.fiber
      ? Math.round(product.nutrition.fiber * multiplier * 10) / 10
      : undefined,
    sugar: product.nutrition.sugar
      ? Math.round(product.nutrition.sugar * multiplier * 10) / 10
      : undefined,
  };
}

/**
 * Formats nutrition grade for display
 */
export function formatNutritionGrade(grade?: string): string | null {
  if (!grade) return null;
  return grade.toUpperCase();
}

/**
 * Gets a color for the nutrition grade
 */
export function getNutritionGradeColor(grade?: string): string {
  if (!grade) return '#888888';

  switch (grade.toLowerCase()) {
    case 'a':
      return '#038141'; // Dark green
    case 'b':
      return '#85bb2f'; // Light green
    case 'c':
      return '#fecb02'; // Yellow
    case 'd':
      return '#ee8100'; // Orange
    case 'e':
      return '#e63e11'; // Red
    default:
      return '#888888'; // Gray
  }
}
