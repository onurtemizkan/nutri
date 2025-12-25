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

// ==============================================================================
// FOOD CATEGORY DETECTION
// ==============================================================================

/**
 * Food categories for nutrient sanity checking
 */
type FoodCategory =
  | 'dairy'
  | 'meat'
  | 'poultry'
  | 'fish'
  | 'seafood'
  | 'eggs'
  | 'citrus'
  | 'fruit'
  | 'leafy_greens'
  | 'vegetable'
  | 'nuts_seeds'
  | 'legumes'
  | 'whole_grains'
  | 'fortified'
  | 'unknown';

/**
 * Category detection patterns
 */
const CATEGORY_PATTERNS: Record<FoodCategory, { categories: string[]; ingredients: string[] }> = {
  dairy: {
    categories: ['dairy', 'yogurt', 'yoghurt', 'milk', 'cheese', 'quark', 'skyr', 'fromage', 'lait', 'yaourt', 'kefir', 'cream', 'butter', 'curd'],
    ingredients: ['milk', 'quark', 'yogurt', 'yoghurt', 'cream', 'skyr', 'fromage frais', 'cheese', 'whey', 'casein', 'lactose'],
  },
  meat: {
    categories: ['meat', 'beef', 'pork', 'lamb', 'veal', 'venison', 'bison', 'game'],
    ingredients: ['beef', 'pork', 'lamb', 'veal', 'meat', 'steak', 'ground beef', 'bacon', 'ham', 'sausage'],
  },
  poultry: {
    categories: ['poultry', 'chicken', 'turkey', 'duck', 'goose'],
    ingredients: ['chicken', 'turkey', 'duck', 'poultry'],
  },
  fish: {
    categories: ['fish', 'salmon', 'tuna', 'cod', 'mackerel', 'sardine', 'herring', 'trout', 'halibut', 'tilapia'],
    ingredients: ['salmon', 'tuna', 'cod', 'mackerel', 'sardine', 'herring', 'trout', 'fish', 'halibut', 'tilapia', 'anchovies'],
  },
  seafood: {
    categories: ['seafood', 'shrimp', 'prawn', 'crab', 'lobster', 'oyster', 'mussel', 'clam', 'scallop', 'squid', 'octopus'],
    ingredients: ['shrimp', 'prawn', 'crab', 'lobster', 'oyster', 'mussel', 'clam', 'scallop', 'squid', 'octopus', 'shellfish'],
  },
  eggs: {
    categories: ['egg', 'eggs', 'omelette', 'omelet', 'frittata'],
    ingredients: ['egg', 'eggs', 'egg white', 'egg yolk', 'whole egg'],
  },
  citrus: {
    categories: ['citrus', 'orange', 'lemon', 'lime', 'grapefruit', 'tangerine', 'mandarin', 'clementine'],
    ingredients: ['orange', 'lemon', 'lime', 'grapefruit', 'tangerine', 'mandarin', 'citrus'],
  },
  fruit: {
    categories: ['fruit', 'berry', 'berries', 'apple', 'banana', 'grape', 'melon', 'mango', 'papaya', 'kiwi', 'pineapple'],
    ingredients: ['strawberry', 'blueberry', 'raspberry', 'apple', 'banana', 'mango', 'papaya', 'kiwi', 'pineapple', 'grape'],
  },
  leafy_greens: {
    categories: ['salad', 'leafy', 'spinach', 'kale', 'lettuce', 'arugula', 'chard', 'collard'],
    ingredients: ['spinach', 'kale', 'lettuce', 'arugula', 'chard', 'collard', 'cabbage', 'bok choy', 'watercress'],
  },
  vegetable: {
    categories: ['vegetable', 'broccoli', 'carrot', 'tomato', 'pepper', 'onion', 'potato', 'squash', 'zucchini'],
    ingredients: ['broccoli', 'carrot', 'tomato', 'pepper', 'onion', 'potato', 'squash', 'zucchini', 'cauliflower', 'asparagus'],
  },
  nuts_seeds: {
    categories: ['nut', 'nuts', 'seed', 'seeds', 'almond', 'walnut', 'cashew', 'peanut', 'pistachio', 'sunflower', 'pumpkin seed', 'chia', 'flax'],
    ingredients: ['almond', 'walnut', 'cashew', 'peanut', 'pistachio', 'hazelnut', 'pecan', 'sunflower seed', 'pumpkin seed', 'chia', 'flax', 'sesame'],
  },
  legumes: {
    categories: ['legume', 'bean', 'lentil', 'chickpea', 'pea', 'soy', 'tofu', 'tempeh', 'edamame'],
    ingredients: ['bean', 'lentil', 'chickpea', 'pea', 'soy', 'tofu', 'tempeh', 'edamame', 'hummus', 'falafel'],
  },
  whole_grains: {
    categories: ['whole grain', 'whole wheat', 'oat', 'oats', 'quinoa', 'brown rice', 'barley', 'buckwheat', 'millet', 'farro'],
    ingredients: ['whole wheat', 'oat', 'oats', 'quinoa', 'brown rice', 'barley', 'buckwheat', 'millet', 'whole grain'],
  },
  fortified: {
    categories: ['fortified', 'enriched', 'breakfast cereal', 'cereal'],
    ingredients: ['fortified', 'enriched', 'added vitamins', 'vitamin d added'],
  },
  unknown: {
    categories: [],
    ingredients: [],
  },
};

/**
 * Minimum expected nutrient values per 100g for each food category
 * Values below these thresholds are flagged as suspicious (likely data errors)
 * Thresholds are set conservatively low to avoid false positives
 */
const CATEGORY_NUTRIENT_MINIMUMS: Record<FoodCategory, Partial<Record<keyof BarcodeProduct['nutrition'], number>>> = {
  dairy: {
    calcium: 50,        // mg - dairy is calcium-rich (typical: 100-300mg)
    vitaminB12: 0.2,    // mcg - dairy has B12 (typical: 0.4-1.5mcg)
    riboflavin: 0.05,   // mg - dairy is riboflavin source (typical: 0.1-0.4mg)
    phosphorus: 30,     // mg - dairy has phosphorus (typical: 80-200mg)
  },
  meat: {
    vitaminB12: 0.3,    // mcg - meat is B12-rich (typical: 1-6mcg)
    iron: 0.3,          // mg - red meat has iron (typical: 1-3mg)
    zinc: 0.5,          // mg - meat has zinc (typical: 2-6mg)
    niacin: 1,          // mg - meat has niacin (typical: 3-8mg)
    vitaminB6: 0.1,     // mg - meat has B6 (typical: 0.3-0.6mg)
  },
  poultry: {
    vitaminB12: 0.1,    // mcg - poultry has some B12 (typical: 0.2-0.5mcg)
    niacin: 2,          // mg - poultry is niacin-rich (typical: 5-12mg)
    vitaminB6: 0.2,     // mg - poultry has B6 (typical: 0.4-0.6mg)
    zinc: 0.5,          // mg - poultry has zinc (typical: 1-3mg)
  },
  fish: {
    vitaminB12: 1,      // mcg - fish is B12-rich (typical: 2-10mcg)
    vitaminD: 1,        // mcg - fatty fish is vitamin D source (typical: 5-20mcg)
    niacin: 2,          // mg - fish has niacin (typical: 3-10mg)
    phosphorus: 50,     // mg - fish has phosphorus (typical: 150-300mg)
  },
  seafood: {
    vitaminB12: 0.5,    // mcg - shellfish has B12 (typical: 1-20mcg for clams/oysters)
    zinc: 0.5,          // mg - oysters are zinc-rich (typical: 1-75mg)
    iron: 0.3,          // mg - shellfish has iron (typical: 1-5mg)
  },
  eggs: {
    vitaminB12: 0.3,    // mcg - eggs have B12 (typical: 0.5-1mcg)
    vitaminD: 0.5,      // mcg - eggs have vitamin D (typical: 1-2mcg)
    riboflavin: 0.1,    // mg - eggs have riboflavin (typical: 0.2-0.5mg)
  },
  citrus: {
    vitaminC: 20,       // mg - citrus is vitamin C-rich (typical: 30-60mg)
    folate: 10,         // mcg - citrus has folate (typical: 20-40mcg)
    potassium: 50,      // mg - citrus has potassium (typical: 100-200mg)
  },
  fruit: {
    vitaminC: 3,        // mg - most fruits have some C (typical: 5-20mg)
    potassium: 50,      // mg - fruits have potassium (typical: 100-400mg)
  },
  leafy_greens: {
    vitaminK: 50,       // mcg - leafy greens are K-rich (typical: 100-500mcg)
    vitaminA: 100,      // mcg RAE - dark greens have A (typical: 200-1000mcg)
    folate: 30,         // mcg - greens have folate (typical: 50-200mcg)
    iron: 0.5,          // mg - greens have iron (typical: 1-4mg)
  },
  vegetable: {
    potassium: 50,      // mg - vegetables have potassium (typical: 100-400mg)
    vitaminC: 3,        // mg - most vegetables have some C (typical: 5-30mg)
  },
  nuts_seeds: {
    vitaminE: 1,        // mg - nuts/seeds are E-rich (typical: 2-25mg)
    magnesium: 30,      // mg - nuts/seeds have magnesium (typical: 50-300mg)
    zinc: 0.5,          // mg - nuts/seeds have zinc (typical: 1-5mg)
    phosphorus: 50,     // mg - nuts/seeds have phosphorus (typical: 200-600mg)
  },
  legumes: {
    folate: 50,         // mcg - legumes are folate-rich (typical: 100-400mcg)
    iron: 1,            // mg - legumes have iron (typical: 2-6mg)
    magnesium: 20,      // mg - legumes have magnesium (typical: 40-100mg)
    potassium: 100,     // mg - legumes have potassium (typical: 300-600mg)
    zinc: 0.5,          // mg - legumes have zinc (typical: 1-3mg)
  },
  whole_grains: {
    magnesium: 20,      // mg - whole grains have magnesium (typical: 30-150mg)
    thiamin: 0.1,       // mg - grains have thiamin (typical: 0.2-0.5mg)
    iron: 0.5,          // mg - grains have iron (typical: 1-4mg)
    zinc: 0.5,          // mg - grains have zinc (typical: 1-3mg)
  },
  fortified: {
    // Fortified products should have added vitamins
    vitaminD: 0.5,      // mcg - fortified foods have D (typical: 1-3mcg)
  },
  unknown: {
    // No minimums for unknown category
  },
};

/**
 * Detects the primary food category based on categories and ingredients
 * Returns the most specific category that matches
 */
function detectFoodCategory(categories?: string[], ingredientsText?: string): FoodCategory {
  const categoriesLower = categories?.map(c => c.toLowerCase()) || [];
  const ingredientsLower = ingredientsText?.toLowerCase() || '';

  // Check each category in order of specificity
  // More specific categories (dairy, fish) before general ones (fruit, vegetable)
  const categoryOrder: FoodCategory[] = [
    'dairy', 'fish', 'seafood', 'meat', 'poultry', 'eggs',
    'citrus', 'leafy_greens', 'nuts_seeds', 'legumes', 'whole_grains',
    'fortified', 'fruit', 'vegetable',
  ];

  for (const category of categoryOrder) {
    const patterns = CATEGORY_PATTERNS[category];

    // Check category tags
    if (patterns.categories.some(p => categoriesLower.some(c => c.includes(p)))) {
      return category;
    }

    // Check ingredients (only for primary ingredients at the start)
    // First 100 chars typically contain main ingredients
    const primaryIngredients = ingredientsLower.slice(0, 150);
    if (patterns.ingredients.some(p => primaryIngredients.includes(p))) {
      return category;
    }
  }

  return 'unknown';
}

/**
 * Gets all suspicious nutrient values for a product
 * Returns a record of nutrient keys that have implausibly low values for the detected category
 */
function getSuspiciousNutrients(
  nutrition: BarcodeProduct['nutrition'],
  categories?: string[],
  ingredientsText?: string
): Set<keyof BarcodeProduct['nutrition']> {
  const suspicious = new Set<keyof BarcodeProduct['nutrition']>();
  const category = detectFoodCategory(categories, ingredientsText);

  if (category === 'unknown') {
    return suspicious; // No sanity checks for unknown categories
  }

  const minimums = CATEGORY_NUTRIENT_MINIMUMS[category];

  // Check each nutrient against its minimum
  for (const [nutrient, minimum] of Object.entries(minimums)) {
    const key = nutrient as keyof BarcodeProduct['nutrition'];
    const value = nutrition[key];

    // If value exists and is below minimum, it's suspicious
    if (value !== undefined && value !== null && typeof value === 'number' && value < minimum) {
      suspicious.add(key);
    }
  }

  return suspicious;
}

/**
 * Checks if any micronutrient values are suspiciously low for the product category
 */
function hasSuspiciousNutrients(
  nutrition: BarcodeProduct['nutrition'],
  categories?: string[],
  ingredientsText?: string
): boolean {
  return getSuspiciousNutrients(nutrition, categories, ingredientsText).size > 0;
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
 * Also detects and corrects suspiciously low values based on food category:
 * - Dairy with <50mg calcium
 * - Fish with <1mcg B12
 * - Citrus with <20mg vitamin C
 * - etc.
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
  // Detect the food category and find all suspicious nutrient values
  const detectedCategory = detectFoodCategory(categories, ingredientsText);
  const suspiciousNutrients = getSuspiciousNutrients(product.nutrition, categories, ingredientsText);
  const hasSuspiciousValues = suspiciousNutrients.size > 0;

  // Skip if nutrition data is already complete AND no suspicious values
  if (!isMissingMicronutrients(product.nutrition) && !hasSuspiciousValues) {
    return product;
  }

  try {
    // Build existing micronutrients object
    // Exclude suspicious values so ML can re-estimate them
    const existing: Record<string, number> = {};
    const nutrition = product.nutrition;

    // Mapping from nutrition keys to ML service keys
    const nutrientMapping: [keyof BarcodeProduct['nutrition'], string][] = [
      ['potassium', 'potassium'],
      ['calcium', 'calcium'],
      ['iron', 'iron'],
      ['magnesium', 'magnesium'],
      ['zinc', 'zinc'],
      ['phosphorus', 'phosphorus'],
      ['vitaminA', 'vitamin_a'],
      ['vitaminC', 'vitamin_c'],
      ['vitaminD', 'vitamin_d'],
      ['vitaminE', 'vitamin_e'],
      ['vitaminK', 'vitamin_k'],
      ['vitaminB6', 'vitamin_b6'],
      ['vitaminB12', 'vitamin_b12'],
      ['folate', 'folate'],
      ['thiamin', 'thiamin'],
      ['riboflavin', 'riboflavin'],
      ['niacin', 'niacin'],
    ];

    // Only include non-suspicious values in existing
    for (const [nutritionKey, mlKey] of nutrientMapping) {
      const value = nutrition[nutritionKey];
      if (value !== undefined && !suspiciousNutrients.has(nutritionKey)) {
        existing[mlKey] = value;
      }
    }

    // Call ML service with sophisticated multi-signal estimation
    const response = await axios.post<MicronutrientEstimationResponse>(
      `${ML_SERVICE_URL}/api/food/estimate-micronutrients`,
      {
        food_name: product.name,
        categories: categories,
        portion_weight: 100, // Per 100g (we scale in calculateServingNutrition)
        ingredients_text: ingredientsText,
        protein: nutrition.protein,
        fiber: nutrition.fiber,
        fat: nutrition.fat,
        existing: Object.keys(existing).length > 0 ? existing : undefined,
      },
      {
        timeout: 5000,
        headers: { 'Content-Type': 'application/json' },
      }
    );

    const estimates = response.data.estimated;
    const updatedNutrition = { ...product.nutrition };

    // Track what we override for logging
    const overrides: string[] = [];

    // Apply estimates - override if undefined OR suspicious
    const applyEstimate = (
      nutritionKey: keyof BarcodeProduct['nutrition'],
      estimateKey: keyof MicronutrientEstimationResponse['estimated']
    ) => {
      const estimate = estimates[estimateKey];
      const isSuspicious = suspiciousNutrients.has(nutritionKey);
      const isUndefined = updatedNutrition[nutritionKey] === undefined;

      if ((isUndefined || isSuspicious) && estimate !== undefined) {
        if (isSuspicious) {
          overrides.push(`${nutritionKey}: ${updatedNutrition[nutritionKey]}→${estimate}`);
        }
        (updatedNutrition as Record<string, number | undefined>)[nutritionKey] = estimate;
      }
    };

    applyEstimate('potassium', 'potassium');
    applyEstimate('calcium', 'calcium');
    applyEstimate('iron', 'iron');
    applyEstimate('magnesium', 'magnesium');
    applyEstimate('zinc', 'zinc');
    applyEstimate('phosphorus', 'phosphorus');
    applyEstimate('vitaminA', 'vitamin_a');
    applyEstimate('vitaminC', 'vitamin_c');
    applyEstimate('vitaminD', 'vitamin_d');
    applyEstimate('vitaminE', 'vitamin_e');
    applyEstimate('vitaminK', 'vitamin_k');
    applyEstimate('vitaminB6', 'vitamin_b6');
    applyEstimate('vitaminB12', 'vitamin_b12');
    applyEstimate('folate', 'folate');
    applyEstimate('thiamin', 'thiamin');
    applyEstimate('riboflavin', 'riboflavin');
    applyEstimate('niacin', 'niacin');

    if (__DEV__) {
      const overrideNote = overrides.length > 0
        ? `, ⚠️ corrected ${overrides.length} suspicious values: ${overrides.join(', ')}`
        : '';
      console.log(
        `📊 Estimated micronutrients for "${product.name}" ` +
        `(detected: ${detectedCategory}, category: ${response.data.category_used}, ` +
        `confidence: ${response.data.confidence}${overrideNote})`
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
      // (for products cached before the estimation feature was added,
      // or with suspiciously low values for the detected category)
      const needsEstimation = isMissingMicronutrients(cachedProduct.nutrition) ||
        hasSuspiciousNutrients(cachedProduct.nutrition, undefined, cachedProduct.ingredients);

      if (needsEstimation) {
        // Re-estimate micronutrients for old cached products or suspicious values
        // We don't have categories in cache, so use ingredients-based estimation
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
