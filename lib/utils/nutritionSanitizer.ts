/**
 * Nutrition Data Sanitizer
 *
 * Provides validation and sanitization for nutrition data to prevent
 * outliers from mislabeled data, data entry errors, and estimation errors.
 *
 * All limits are based on:
 * - USDA DRI (Dietary Reference Intakes)
 * - FDA Daily Values for food labeling
 * - Maximum safe levels from scientific literature
 * - Practical limits for single serving sizes
 */

// Maximum values per 100g of food (realistic food limits)
// These catch data entry errors from databases
const MAX_VALUES_PER_100G = {
  // Core macros - can't exceed 100g per 100g
  calories: 900, // Pure fat is ~900 kcal/100g
  protein: 100,
  carbs: 100,
  fat: 100,
  fiber: 100,
  sugar: 100,

  // Fat breakdown
  saturatedFat: 100, // Can't exceed total fat
  transFat: 50, // Rare to exceed this even in worst foods
  cholesterol: 2000, // mg - eggs have ~400mg/100g, liver ~500mg

  // Minerals (mg per 100g)
  sodium: 50000, // Pure salt is ~39,000mg/100g
  potassium: 10000, // Dried apricots have ~1,160mg/100g
  calcium: 5000, // Cheese has ~1,200mg/100g max
  iron: 200, // Fortified cereals can have high amounts
  magnesium: 2000, // Seeds can have ~500mg/100g
  zinc: 100, // Oysters have ~75mg/100g
  phosphorus: 5000, // Cheese can have ~700mg/100g

  // Vitamins (various units, per 100g)
  vitaminA: 30000, // mcg RAE - liver has ~6,000-25,000
  vitaminC: 5000, // mg - acerola cherries have ~1,680mg
  vitaminD: 500, // mcg - fortified foods up to ~100mcg
  vitaminE: 500, // mg - oils can have ~20-40mg
  vitaminK: 1000, // mcg - kale has ~700mcg
  vitaminB6: 100, // mg - fortified cereals up to ~50mg
  vitaminB12: 500, // mcg - liver has ~80mcg
  folate: 5000, // mcg DFE - fortified foods up to ~500
  thiamin: 100, // mg - fortified cereals up to ~50mg
  riboflavin: 100, // mg - fortified foods up to ~10mg
  niacin: 200, // mg - fortified cereals up to ~50mg
};

// Maximum reasonable values per single serving (what a person would eat at once)
// This is a sanity check for serving-based calculations
const MAX_VALUES_PER_SERVING = {
  calories: 3000, // A massive single meal
  protein: 200, // Very large protein shake
  carbs: 500, // Extreme carb loading
  fat: 200, // Very fatty meal
  fiber: 100, // Would cause GI distress
  sugar: 300, // Huge dessert

  // Fat breakdown
  saturatedFat: 100,
  transFat: 20,
  cholesterol: 3000, // mg

  // Minerals
  sodium: 10000, // mg - very salty meal
  potassium: 5000, // mg
  calcium: 3000, // mg
  iron: 100, // mg
  magnesium: 1500, // mg
  zinc: 50, // mg
  phosphorus: 3000, // mg

  // Vitamins
  vitaminA: 50000, // mcg - single serving of liver
  vitaminC: 5000, // mg - supplement levels
  vitaminD: 250, // mcg - 10,000 IU supplement
  vitaminE: 300, // mg
  vitaminK: 2000, // mcg
  vitaminB6: 200, // mg
  vitaminB12: 10000, // mcg - B12 supplements can be very high
  folate: 5000, // mcg
  thiamin: 200, // mg
  riboflavin: 200, // mg
  niacin: 500, // mg
};

// Minimum detectable/sensible values (below this, treat as 0)
const MIN_DETECTABLE = {
  calories: 0.5,
  protein: 0.1,
  carbs: 0.1,
  fat: 0.1,
  fiber: 0.1,
  sugar: 0.1,
  saturatedFat: 0.1,
  transFat: 0.01,
  cholesterol: 0.5, // mg
  sodium: 0.5, // mg
  potassium: 0.5,
  calcium: 0.5,
  iron: 0.01,
  magnesium: 0.5,
  zinc: 0.01,
  phosphorus: 0.5,
  vitaminA: 0.5, // mcg
  vitaminC: 0.1, // mg
  vitaminD: 0.01, // mcg
  vitaminE: 0.01, // mg
  vitaminK: 0.1, // mcg
  vitaminB6: 0.001, // mg
  vitaminB12: 0.01, // mcg
  folate: 0.5, // mcg
  thiamin: 0.001, // mg
  riboflavin: 0.001, // mg
  niacin: 0.01, // mg
};

export type NutrientKey = keyof typeof MAX_VALUES_PER_100G;

export interface NutritionData {
  calories?: number;
  protein?: number;
  carbs?: number;
  fat?: number;
  fiber?: number;
  sugar?: number;
  saturatedFat?: number;
  transFat?: number;
  cholesterol?: number;
  sodium?: number;
  potassium?: number;
  calcium?: number;
  iron?: number;
  magnesium?: number;
  zinc?: number;
  phosphorus?: number;
  vitaminA?: number;
  vitaminC?: number;
  vitaminD?: number;
  vitaminE?: number;
  vitaminK?: number;
  vitaminB6?: number;
  vitaminB12?: number;
  folate?: number;
  thiamin?: number;
  riboflavin?: number;
  niacin?: number;
}

export interface SanitizeResult {
  data: NutritionData;
  warnings: string[];
  removed: NutrientKey[];
}

/**
 * Sanitizes a single nutrition value
 * Returns undefined if the value is an outlier
 */
function sanitizeValue(
  key: NutrientKey,
  value: number | undefined,
  mode: 'per100g' | 'perServing'
): number | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== 'number' || isNaN(value)) return undefined;

  // Negative values are always invalid
  if (value < 0) return undefined;

  // Below minimum detectable - treat as 0/undefined
  const minVal = MIN_DETECTABLE[key];
  if (value < minVal) return undefined;

  // Check against maximum limits
  const maxLimits = mode === 'per100g' ? MAX_VALUES_PER_100G : MAX_VALUES_PER_SERVING;
  const maxVal = maxLimits[key];
  if (value > maxVal) return undefined;

  return value;
}

/**
 * Performs logical consistency checks on nutrition data
 * e.g., saturated fat can't exceed total fat
 */
function checkConsistency(data: NutritionData): string[] {
  const warnings: string[] = [];

  // Fat breakdown can't exceed total fat
  if (data.fat !== undefined) {
    const fatBreakdown = (data.saturatedFat || 0) + (data.transFat || 0);
    if (fatBreakdown > data.fat * 1.1) { // 10% tolerance for rounding
      warnings.push('Fat breakdown exceeds total fat');
    }
  }

  // Sugar can't exceed total carbs
  if (data.carbs !== undefined && data.sugar !== undefined) {
    if (data.sugar > data.carbs * 1.1) {
      warnings.push('Sugar exceeds total carbs');
    }
  }

  // Calories sanity check: should roughly match macros
  if (data.calories !== undefined && data.protein !== undefined &&
      data.carbs !== undefined && data.fat !== undefined) {
    const calculatedCals = (data.protein * 4) + (data.carbs * 4) + (data.fat * 9);
    const diff = Math.abs(data.calories - calculatedCals);
    // Allow 20% tolerance (alcohol, fiber, rounding)
    if (diff > calculatedCals * 0.3 && calculatedCals > 50) {
      warnings.push('Calories do not match macronutrient calculation');
    }
  }

  return warnings;
}

/**
 * Sanitizes nutrition data to remove outliers and invalid values
 *
 * @param data Raw nutrition data
 * @param mode Whether data is per 100g or per serving
 * @returns Sanitized data with warnings and list of removed fields
 */
export function sanitizeNutritionData(
  data: NutritionData,
  mode: 'per100g' | 'perServing' = 'per100g'
): SanitizeResult {
  const sanitized: NutritionData = {};
  const warnings: string[] = [];
  const removed: NutrientKey[] = [];

  // Sanitize each nutrient
  const nutrientKeys: NutrientKey[] = [
    'calories', 'protein', 'carbs', 'fat', 'fiber', 'sugar',
    'saturatedFat', 'transFat', 'cholesterol',
    'sodium', 'potassium', 'calcium', 'iron', 'magnesium', 'zinc', 'phosphorus',
    'vitaminA', 'vitaminC', 'vitaminD', 'vitaminE', 'vitaminK',
    'vitaminB6', 'vitaminB12', 'folate', 'thiamin', 'riboflavin', 'niacin',
  ];

  for (const key of nutrientKeys) {
    const rawValue = data[key];
    const sanitizedValue = sanitizeValue(key, rawValue, mode);

    if (sanitizedValue !== undefined) {
      sanitized[key] = sanitizedValue;
    } else if (rawValue !== undefined && rawValue !== null) {
      // Value was removed as an outlier
      removed.push(key);
      warnings.push(`Removed outlier ${key}: ${rawValue}`);
    }
  }

  // Check logical consistency
  const consistencyWarnings = checkConsistency(sanitized);
  warnings.push(...consistencyWarnings);

  return { data: sanitized, warnings, removed };
}

/**
 * Formats a nutrition value for display with appropriate precision
 */
export function formatNutrientValue(key: NutrientKey, value: number | undefined): string {
  if (value === undefined || value === null) return '-';

  // Different precision for different nutrients
  const integerNutrients: NutrientKey[] = [
    'calories', 'sodium', 'potassium', 'calcium', 'magnesium',
    'phosphorus', 'vitaminA', 'vitaminK', 'folate', 'cholesterol'
  ];

  const twoDecimalNutrients: NutrientKey[] = [
    'vitaminB6', 'vitaminB12', 'thiamin', 'riboflavin'
  ];

  if (integerNutrients.includes(key)) {
    return Math.round(value).toString();
  } else if (twoDecimalNutrients.includes(key)) {
    return value.toFixed(2);
  } else {
    // One decimal for most nutrients
    return value.toFixed(1);
  }
}

/**
 * Gets the unit for a nutrient
 */
export function getNutrientUnit(key: NutrientKey): string {
  const units: Record<NutrientKey, string> = {
    calories: 'kcal',
    protein: 'g',
    carbs: 'g',
    fat: 'g',
    fiber: 'g',
    sugar: 'g',
    saturatedFat: 'g',
    transFat: 'g',
    cholesterol: 'mg',
    sodium: 'mg',
    potassium: 'mg',
    calcium: 'mg',
    iron: 'mg',
    magnesium: 'mg',
    zinc: 'mg',
    phosphorus: 'mg',
    vitaminA: 'mcg',
    vitaminC: 'mg',
    vitaminD: 'mcg',
    vitaminE: 'mg',
    vitaminK: 'mcg',
    vitaminB6: 'mg',
    vitaminB12: 'mcg',
    folate: 'mcg',
    thiamin: 'mg',
    riboflavin: 'mg',
    niacin: 'mg',
  };
  return units[key];
}

/**
 * Check if a meal's nutrition data has any micronutrients
 */
export function hasMicronutrients(data: NutritionData): boolean {
  const microKeys: NutrientKey[] = [
    'saturatedFat', 'transFat', 'cholesterol',
    'sodium', 'potassium', 'calcium', 'iron', 'magnesium', 'zinc', 'phosphorus',
    'vitaminA', 'vitaminC', 'vitaminD', 'vitaminE', 'vitaminK',
    'vitaminB6', 'vitaminB12', 'folate', 'thiamin', 'riboflavin', 'niacin',
  ];

  return microKeys.some(key => data[key] !== undefined && data[key] !== null);
}
