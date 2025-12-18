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
import Constants from 'expo-constants';
import { Platform } from 'react-native';
import {
  OpenFoodFactsResponse,
  OpenFoodFactsProduct,
  BarcodeProduct,
  BarcodeScanResult,
  BarcodeError,
} from '../types/barcode';

// Base URL for Open Food Facts API
const OPEN_FOOD_FACTS_API_BASE = 'https://world.openfoodfacts.org/api/v2';

// ==============================================================================
// ML SERVICE CONFIGURATION (for micronutrient estimation)
// ==============================================================================

/**
 * Get the ML service URL based on environment and platform
 *
 * Priority:
 * 1. Custom mlServiceUrl from app.config.js (highest priority)
 * 2. Derived from apiUrl (for physical devices using same host)
 * 3. Platform-specific defaults (localhost for simulator)
 */
function getMlServiceUrl(): string {
  // Check for custom ML URL in app.config.js
  const customMlUrl = Constants.expoConfig?.extra?.mlServiceUrl;
  if (customMlUrl && typeof customMlUrl === 'string' && customMlUrl.trim() !== '') {
    return customMlUrl;
  }

  // Production environment
  if (!__DEV__) {
    const productionMlUrl = Constants.expoConfig?.extra?.productionMlServiceUrl;
    if (productionMlUrl && typeof productionMlUrl === 'string' && productionMlUrl.trim() !== '') {
      return productionMlUrl;
    }
    // Return placeholder if not configured
    return 'https://ml-url-not-configured.invalid';
  }

  // Development environment - try to derive from apiUrl for physical devices
  const apiUrl = Constants.expoConfig?.extra?.apiUrl;
  if (apiUrl && typeof apiUrl === 'string') {
    // Extract host from API URL (e.g., "http://192.168.1.68:3000/api" -> "192.168.1.68")
    const match = apiUrl.match(/^(https?:\/\/)([^:/]+)/);
    if (match) {
      const protocol = match[1];
      const host = match[2];
      // If it's not localhost, assume physical device - use same host with port 8000
      if (host !== 'localhost' && host !== '127.0.0.1') {
        return `${protocol}${host}:8000`;
      }
    }
  }

  // Fallback: platform-specific defaults for simulators/emulators
  if (Platform.OS === 'ios') {
    return 'http://localhost:8000';
  } else if (Platform.OS === 'android') {
    return 'http://10.0.2.2:8000';
  }

  return 'http://localhost:8000';
}

const ML_SERVICE_URL = getMlServiceUrl();

// Log ML service URL in development for debugging
if (__DEV__) {
  console.log(`[OpenFoodFacts] ML Service URL: ${ML_SERVICE_URL}`);
}

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
 * Helper to round a value with outlier protection
 * Returns undefined for extreme outliers that are likely data errors
 */
function safeRound(value: number | undefined, decimals: number = 1, maxValue?: number): number | undefined {
  if (value === undefined || value === null) return undefined;
  // Filter outliers (likely data entry errors)
  if (maxValue && value > maxValue) return undefined;
  if (value < 0) return undefined;
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
}

/**
 * Maps Open Food Facts product data to our app's BarcodeProduct schema
 * Includes micronutrient extraction with outlier protection
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

  // Extract micronutrients with outlier protection
  // Maximum reasonable values per 100g to filter data entry errors:
  // - Vitamins: based on fortified food limits
  // - Minerals: based on supplements/fortified food max
  const nutrition: BarcodeProduct['nutrition'] = {
    // Core macros
    calories: Math.round(calories),
    protein: Math.round((nutriments.proteins_100g || 0) * 10) / 10,
    carbs: Math.round((nutriments.carbohydrates_100g || 0) * 10) / 10,
    fat: Math.round((nutriments.fat_100g || 0) * 10) / 10,
    fiber: safeRound(nutriments.fiber_100g, 1, 100), // max 100g fiber per 100g
    sugar: safeRound(nutriments.sugars_100g, 1, 100), // max 100g sugar per 100g

    // Fat breakdown
    saturatedFat: safeRound(nutriments['saturated-fat_100g'], 1, 100),
    transFat: safeRound(nutriments['trans-fat_100g'], 1, 50),
    cholesterol: safeRound(nutriments.cholesterol_100g, 0, 2000), // mg, max 2000mg

    // Minerals (mg) - with reasonable max values
    sodium: safeRound(nutriments.sodium_100g, 0, 50000), // mg
    potassium: safeRound(nutriments.potassium_100g, 0, 10000), // mg
    calcium: safeRound(nutriments.calcium_100g, 0, 5000), // mg
    iron: safeRound(nutriments.iron_100g, 1, 200), // mg
    magnesium: safeRound(nutriments.magnesium_100g, 0, 2000), // mg
    zinc: safeRound(nutriments.zinc_100g, 1, 100), // mg
    phosphorus: safeRound(nutriments.phosphorus_100g, 0, 5000), // mg

    // Vitamins - with fortification-based limits
    vitaminA: safeRound(nutriments['vitamin-a_100g'], 0, 30000), // mcg RAE
    vitaminC: safeRound(nutriments['vitamin-c_100g'], 0, 5000), // mg
    vitaminD: safeRound(nutriments['vitamin-d_100g'], 1, 500), // mcg
    vitaminE: safeRound(nutriments['vitamin-e_100g'], 1, 500), // mg
    vitaminK: safeRound(nutriments['vitamin-k_100g'], 0, 1000), // mcg
    vitaminB6: safeRound(nutriments['vitamin-b6_100g'], 2, 100), // mg
    vitaminB12: safeRound(nutriments['vitamin-b12_100g'], 2, 500), // mcg
    folate: safeRound(nutriments.folate_100g || nutriments['folic-acid_100g'], 0, 5000), // mcg
    thiamin: safeRound(nutriments.thiamin_100g, 2, 100), // mg
    riboflavin: safeRound(nutriments.riboflavin_100g, 2, 100), // mg
    niacin: safeRound(nutriments.niacin_100g, 1, 200), // mg
  };

  return {
    barcode,
    name,
    brand: product.brands || product.brand_owner,
    servingSize,
    servingQuantity,
    imageUrl,
    nutrition,
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

// ==============================================================================
// MICRONUTRIENT ESTIMATION
// ==============================================================================

/**
 * ML service response for micronutrient estimation
 */
interface MicronutrientEstimationResponse {
  estimated: {
    potassium?: number;
    calcium?: number;
    iron?: number;
    magnesium?: number;
    zinc?: number;
    phosphorus?: number;
    vitamin_a?: number;
    vitamin_c?: number;
    vitamin_d?: number;
    vitamin_e?: number;
    vitamin_k?: number;
    vitamin_b6?: number;
    vitamin_b12?: number;
    folate?: number;
    thiamin?: number;
    riboflavin?: number;
    niacin?: number;
  };
  category_used: string;
  confidence: 'high' | 'medium' | 'low';
  source: string;
}

/**
 * Checks if a product is missing most micronutrients
 * Returns true if at least 5 key micronutrients are missing
 */
function isMissingMicronutrients(nutrition: BarcodeProduct['nutrition']): boolean {
  const micronutrientKeys: (keyof BarcodeProduct['nutrition'])[] = [
    'potassium', 'calcium', 'iron', 'magnesium', 'zinc',
    'vitaminA', 'vitaminC', 'vitaminD', 'vitaminB6', 'vitaminB12',
  ];

  const missingCount = micronutrientKeys.filter(
    (key) => nutrition[key] === undefined || nutrition[key] === null
  ).length;

  // Consider missing if at least half of key micronutrients are undefined
  return missingCount >= 5;
}

/**
 * Estimates missing micronutrients for a product using the ML service
 *
 * Uses sophisticated multi-signal estimation:
 * 1. Ingredient parsing - identifies nutrient-rich ingredients (spinach, salmon, etc.)
 * 2. Food name matching - matches to our food database
 * 3. Macronutrient inference - high protein → likely B12 rich, high fiber → magnesium rich
 * 4. Category baseline - fallback when other signals weak
 *
 * @param product - The barcode product with potentially incomplete nutrition
 * @param categories - Open Food Facts category tags (original format with "en:" prefix)
 * @param ingredientsText - Raw ingredients list from product label
 * @returns Product with estimated micronutrients filled in
 */
async function estimateMissingMicronutrients(
  product: BarcodeProduct,
  categories?: string[],
  ingredientsText?: string
): Promise<BarcodeProduct> {
  // Skip if nutrition data is already complete
  if (!isMissingMicronutrients(product.nutrition)) {
    return product;
  }

  try {
    // Build existing micronutrients object (only include non-undefined values)
    const existing: Record<string, number> = {};
    const nutrition = product.nutrition;

    if (nutrition.potassium !== undefined) existing.potassium = nutrition.potassium;
    if (nutrition.calcium !== undefined) existing.calcium = nutrition.calcium;
    if (nutrition.iron !== undefined) existing.iron = nutrition.iron;
    if (nutrition.magnesium !== undefined) existing.magnesium = nutrition.magnesium;
    if (nutrition.zinc !== undefined) existing.zinc = nutrition.zinc;
    if (nutrition.phosphorus !== undefined) existing.phosphorus = nutrition.phosphorus;
    if (nutrition.vitaminA !== undefined) existing.vitamin_a = nutrition.vitaminA;
    if (nutrition.vitaminC !== undefined) existing.vitamin_c = nutrition.vitaminC;
    if (nutrition.vitaminD !== undefined) existing.vitamin_d = nutrition.vitaminD;
    if (nutrition.vitaminE !== undefined) existing.vitamin_e = nutrition.vitaminE;
    if (nutrition.vitaminK !== undefined) existing.vitamin_k = nutrition.vitaminK;
    if (nutrition.vitaminB6 !== undefined) existing.vitamin_b6 = nutrition.vitaminB6;
    if (nutrition.vitaminB12 !== undefined) existing.vitamin_b12 = nutrition.vitaminB12;
    if (nutrition.folate !== undefined) existing.folate = nutrition.folate;
    if (nutrition.thiamin !== undefined) existing.thiamin = nutrition.thiamin;
    if (nutrition.riboflavin !== undefined) existing.riboflavin = nutrition.riboflavin;
    if (nutrition.niacin !== undefined) existing.niacin = nutrition.niacin;

    // Call ML service with sophisticated multi-signal estimation
    // Includes ingredients text and macros for better accuracy
    const response = await axios.post<MicronutrientEstimationResponse>(
      `${ML_SERVICE_URL}/api/food/estimate-micronutrients`,
      {
        food_name: product.name,
        categories: categories, // Pass original format for ML service mapping
        portion_weight: 100, // Per 100g (we scale in calculateServingNutrition)
        // NEW: Send ingredients for ingredient-based estimation
        ingredients_text: ingredientsText,
        // NEW: Send macros for macro-informed estimation
        protein: nutrition.protein,
        fiber: nutrition.fiber,
        fat: nutrition.fat,
        existing: Object.keys(existing).length > 0 ? existing : undefined,
      },
      {
        timeout: 5000, // 5 second timeout for ML calls
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    const estimates = response.data.estimated;

    // Only fill in undefined values with estimates
    const updatedNutrition = { ...product.nutrition };

    if (updatedNutrition.potassium === undefined && estimates.potassium !== undefined) {
      updatedNutrition.potassium = estimates.potassium;
    }
    if (updatedNutrition.calcium === undefined && estimates.calcium !== undefined) {
      updatedNutrition.calcium = estimates.calcium;
    }
    if (updatedNutrition.iron === undefined && estimates.iron !== undefined) {
      updatedNutrition.iron = estimates.iron;
    }
    if (updatedNutrition.magnesium === undefined && estimates.magnesium !== undefined) {
      updatedNutrition.magnesium = estimates.magnesium;
    }
    if (updatedNutrition.zinc === undefined && estimates.zinc !== undefined) {
      updatedNutrition.zinc = estimates.zinc;
    }
    if (updatedNutrition.phosphorus === undefined && estimates.phosphorus !== undefined) {
      updatedNutrition.phosphorus = estimates.phosphorus;
    }
    if (updatedNutrition.vitaminA === undefined && estimates.vitamin_a !== undefined) {
      updatedNutrition.vitaminA = estimates.vitamin_a;
    }
    if (updatedNutrition.vitaminC === undefined && estimates.vitamin_c !== undefined) {
      updatedNutrition.vitaminC = estimates.vitamin_c;
    }
    if (updatedNutrition.vitaminD === undefined && estimates.vitamin_d !== undefined) {
      updatedNutrition.vitaminD = estimates.vitamin_d;
    }
    if (updatedNutrition.vitaminE === undefined && estimates.vitamin_e !== undefined) {
      updatedNutrition.vitaminE = estimates.vitamin_e;
    }
    if (updatedNutrition.vitaminK === undefined && estimates.vitamin_k !== undefined) {
      updatedNutrition.vitaminK = estimates.vitamin_k;
    }
    if (updatedNutrition.vitaminB6 === undefined && estimates.vitamin_b6 !== undefined) {
      updatedNutrition.vitaminB6 = estimates.vitamin_b6;
    }
    if (updatedNutrition.vitaminB12 === undefined && estimates.vitamin_b12 !== undefined) {
      updatedNutrition.vitaminB12 = estimates.vitamin_b12;
    }
    if (updatedNutrition.folate === undefined && estimates.folate !== undefined) {
      updatedNutrition.folate = estimates.folate;
    }
    if (updatedNutrition.thiamin === undefined && estimates.thiamin !== undefined) {
      updatedNutrition.thiamin = estimates.thiamin;
    }
    if (updatedNutrition.riboflavin === undefined && estimates.riboflavin !== undefined) {
      updatedNutrition.riboflavin = estimates.riboflavin;
    }
    if (updatedNutrition.niacin === undefined && estimates.niacin !== undefined) {
      updatedNutrition.niacin = estimates.niacin;
    }

    if (__DEV__) {
      console.log(
        `📊 Estimated ${Object.keys(estimates).length} micronutrients for "${product.name}" ` +
        `(category: ${response.data.category_used}, confidence: ${response.data.confidence})`
      );
    }

    return {
      ...product,
      nutrition: updatedNutrition,
    };
  } catch (error) {
    // Silently fail and return original product - estimation is optional enhancement
    if (__DEV__) {
      console.warn('Micronutrient estimation failed (non-critical):', error);
    }
    return product;
  }
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
      // Check if cached product needs micronutrient estimation
      // (for products cached before the estimation feature was added)
      if (isMissingMicronutrients(cachedProduct.nutrition)) {
        // Re-estimate micronutrients for old cached products
        // We don't have ingredients/categories in cache, so use basic estimation
        const updatedProduct = await estimateMissingMicronutrients(
          cachedProduct,
          undefined, // No categories available from cache
          cachedProduct.ingredients // Use cached ingredients if available
        );
        // Update cache with estimated micronutrients
        await saveToCache(updatedProduct);
        return {
          success: true,
          barcode: cleanBarcode,
          product: updatedProduct,
        };
      }
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
    let product = mapProductToAppSchema(data.product, cleanBarcode);

    // Estimate missing micronutrients using sophisticated ML service
    // Pass:
    // - Original category tags (with "en:" prefix) for category mapping
    // - Ingredients text for ingredient-based estimation (e.g., "spinach" → high iron)
    // The ML service also uses the product's macros (protein, fiber, fat) for inference
    product = await estimateMissingMicronutrients(
      product,
      data.product.categories_tags,
      data.product.ingredients_text_en || data.product.ingredients_text
    );

    // Save to cache for future lookups (with estimated micronutrients)
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

      // Handle 404 as product not found
      if (axiosError.response.status === 404) {
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

      // Handle other API errors with user-friendly messages
      const statusCode = axiosError.response.status;
      let errorMessage = 'Something went wrong. Please try again.';
      if (statusCode >= 500) {
        errorMessage = 'The product database is temporarily unavailable. Please try again later.';
      } else if (statusCode === 429) {
        errorMessage = 'Too many requests. Please wait a moment and try again.';
      }

      return {
        success: false,
        barcode: cleanBarcode,
        error: createBarcodeError(
          'API_ERROR',
          errorMessage,
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
  const n = product.nutrition;

  // Helper to scale optional values
  const scale = (val: number | undefined, decimals: number = 1): number | undefined => {
    if (val === undefined) return undefined;
    const factor = Math.pow(10, decimals);
    return Math.round(val * multiplier * factor) / factor;
  };

  return {
    // Core macros
    calories: Math.round(n.calories * multiplier),
    protein: Math.round(n.protein * multiplier * 10) / 10,
    carbs: Math.round(n.carbs * multiplier * 10) / 10,
    fat: Math.round(n.fat * multiplier * 10) / 10,
    fiber: scale(n.fiber),
    sugar: scale(n.sugar),

    // Fat breakdown
    saturatedFat: scale(n.saturatedFat),
    transFat: scale(n.transFat),
    cholesterol: scale(n.cholesterol, 0),

    // Minerals
    sodium: scale(n.sodium, 0),
    potassium: scale(n.potassium, 0),
    calcium: scale(n.calcium, 0),
    iron: scale(n.iron),
    magnesium: scale(n.magnesium, 0),
    zinc: scale(n.zinc),
    phosphorus: scale(n.phosphorus, 0),

    // Vitamins
    vitaminA: scale(n.vitaminA, 0),
    vitaminC: scale(n.vitaminC, 0),
    vitaminD: scale(n.vitaminD),
    vitaminE: scale(n.vitaminE),
    vitaminK: scale(n.vitaminK, 0),
    vitaminB6: scale(n.vitaminB6, 2),
    vitaminB12: scale(n.vitaminB12, 2),
    folate: scale(n.folate, 0),
    thiamin: scale(n.thiamin, 2),
    riboflavin: scale(n.riboflavin, 2),
    niacin: scale(n.niacin),
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
