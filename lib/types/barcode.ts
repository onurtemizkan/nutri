/**
 * Barcode Scanner Types
 *
 * Types for barcode scanning and Open Food Facts API integration.
 * Open Food Facts API documentation: https://world.openfoodfacts.org/data
 */

/**
 * Supported barcode formats for food products
 */
export type BarcodeFormat = 'ean13' | 'ean8' | 'upc_a' | 'upc_e';

/**
 * Open Food Facts API nutriments structure
 * Values are typically per 100g/100ml
 */
export interface OpenFoodFactsNutriments {
  // Energy
  'energy-kcal_100g'?: number;
  'energy-kj_100g'?: number;
  energy_100g?: number;
  energy_unit?: string;

  // Macronutrients
  proteins_100g?: number;
  carbohydrates_100g?: number;
  fat_100g?: number;
  fiber_100g?: number;
  sugars_100g?: number;

  // Fats breakdown
  'saturated-fat_100g'?: number;
  'monounsaturated-fat_100g'?: number;
  'polyunsaturated-fat_100g'?: number;
  'trans-fat_100g'?: number;
  cholesterol_100g?: number;

  // Minerals (all in mg per 100g unless noted)
  sodium_100g?: number;
  salt_100g?: number;
  calcium_100g?: number;
  iron_100g?: number;
  potassium_100g?: number;
  magnesium_100g?: number;
  zinc_100g?: number;
  phosphorus_100g?: number;

  // Vitamins
  'vitamin-a_100g'?: number; // mcg RAE
  'vitamin-c_100g'?: number; // mg
  'vitamin-d_100g'?: number; // mcg
  'vitamin-e_100g'?: number; // mg
  'vitamin-k_100g'?: number; // mcg
  'vitamin-b6_100g'?: number; // mg
  'vitamin-b12_100g'?: number; // mcg
  folate_100g?: number; // mcg
  'folic-acid_100g'?: number; // mcg (alternative key)
  thiamin_100g?: number; // mg (B1)
  riboflavin_100g?: number; // mg (B2)
  niacin_100g?: number; // mg (B3)

  // Serving info
  'energy-kcal_serving'?: number;
  proteins_serving?: number;
  carbohydrates_serving?: number;
  fat_serving?: number;
  fiber_serving?: number;
  sugars_serving?: number;
}

/**
 * Product images from Open Food Facts
 */
export interface OpenFoodFactsImages {
  front?: {
    display?: {
      en?: string;
    };
    small?: {
      en?: string;
    };
    thumb?: {
      en?: string;
    };
  };
  nutrition?: {
    display?: {
      en?: string;
    };
  };
}

/**
 * Open Food Facts product structure
 */
export interface OpenFoodFactsProduct {
  // Identifiers
  code?: string;
  _id?: string;
  id?: string;

  // Product info
  product_name?: string;
  product_name_en?: string;
  generic_name?: string;
  brands?: string;
  brand_owner?: string;
  categories?: string;
  categories_tags?: string[];

  // Serving info
  serving_size?: string;
  serving_quantity?: number;
  quantity?: string;
  product_quantity?: number;

  // Nutrition
  nutriments?: OpenFoodFactsNutriments;
  nutrition_grades?: string; // A, B, C, D, E
  nutriscore_grade?: string;
  nova_group?: number; // 1-4 food processing score

  // Images
  image_url?: string;
  image_front_url?: string;
  image_front_small_url?: string;
  image_front_thumb_url?: string;
  image_nutrition_url?: string;
  selected_images?: OpenFoodFactsImages;

  // Origin
  countries?: string;
  countries_tags?: string[];
  origins?: string;

  // Ingredients
  ingredients_text?: string;
  ingredients_text_en?: string;
  allergens?: string;
  allergens_tags?: string[];
  traces?: string;
  traces_tags?: string[];

  // Status
  completeness?: number;
  data_quality_errors_tags?: string[];
}

/**
 * Open Food Facts API response
 */
export interface OpenFoodFactsResponse {
  status: 0 | 1; // 0 = not found, 1 = found
  status_verbose: string;
  code: string;
  product?: OpenFoodFactsProduct;
}

/**
 * Simplified product data mapped to app schema
 */
export interface BarcodeProduct {
  barcode: string;
  name: string;
  brand?: string;
  servingSize?: string;
  servingQuantity?: number;
  imageUrl?: string;

  // Nutrition per 100g (mapped to app schema)
  nutrition: {
    // Core macros
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
    fiber?: number;
    sugar?: number;

    // Fat breakdown
    saturatedFat?: number;
    transFat?: number;
    cholesterol?: number; // mg

    // Minerals (mg)
    sodium?: number;
    potassium?: number;
    calcium?: number;
    iron?: number;
    magnesium?: number;
    zinc?: number;
    phosphorus?: number;

    // Vitamins
    vitaminA?: number; // mcg RAE
    vitaminC?: number; // mg
    vitaminD?: number; // mcg
    vitaminE?: number; // mg
    vitaminK?: number; // mcg
    vitaminB6?: number; // mg
    vitaminB12?: number; // mcg
    folate?: number; // mcg DFE
    thiamin?: number; // mg (B1)
    riboflavin?: number; // mg (B2)
    niacin?: number; // mg (B3)
  };

  // Additional info
  nutriscoreGrade?: string;
  categories?: string[];
  ingredients?: string;
  allergens?: string[];
}

/**
 * Barcode scan result
 */
export interface BarcodeScanResult {
  success: boolean;
  barcode?: string;
  product?: BarcodeProduct;
  error?: BarcodeError;
}

/**
 * Barcode error types
 */
export type BarcodeErrorType =
  | 'PRODUCT_NOT_FOUND'
  | 'NETWORK_ERROR'
  | 'INVALID_BARCODE'
  | 'API_ERROR'
  | 'CAMERA_PERMISSION_DENIED'
  | 'CAMERA_UNAVAILABLE';

/**
 * Barcode error structure
 */
export interface BarcodeError {
  type: BarcodeErrorType;
  message: string;
  barcode?: string;
}

/**
 * Barcode scanner state
 */
export interface BarcodeScannerState {
  isScanning: boolean;
  hasPermission: boolean | null;
  lastScannedBarcode: string | null;
  isLoading: boolean;
  error: BarcodeError | null;
}
