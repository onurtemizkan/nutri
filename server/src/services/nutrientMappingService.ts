/**
 * Nutrient Mapping Service
 *
 * Maps USDA FoodData Central nutrient IDs to application schema.
 * USDA uses numeric nutrient IDs (e.g., 1003 for protein), while our app
 * uses named fields (e.g., 'protein').
 *
 * Reference: https://fdc.nal.usda.gov/api-spec/fdc_api.html#/Nutrient
 */

import {
  USDANutrient,
  USDASearchNutrient,
  USDAFoodItem,
  USDASearchResultFood,
  TransformedNutrients,
  TransformedUSDAFood,
  TransformedPortion,
} from '../types/usda';
import { createChildLogger, Logger } from '../config/logger';

// ============================================================================
// USDA NUTRIENT ID MAPPINGS
// ============================================================================

/**
 * USDA nutrient ID to app field mapping
 * These IDs are standardized by USDA and consistent across all food data types
 */
export const USDA_NUTRIENT_ID_MAP: Record<number, keyof TransformedNutrients> = {
  // Core Macronutrients
  1008: 'calories',        // Energy (kcal)
  1003: 'protein',         // Protein (g)
  1005: 'carbs',           // Carbohydrate, by difference (g)
  1004: 'fat',             // Total lipid (fat) (g)

  // Fiber & Sugars
  1079: 'fiber',           // Fiber, total dietary (g)
  2000: 'sugar',           // Sugars, total including NLEA (g)
  1235: 'addedSugar',      // Sugars, added (g)

  // Fat Breakdown
  1258: 'saturatedFat',       // Fatty acids, total saturated (g)
  1257: 'transFat',           // Fatty acids, total trans (g)
  1253: 'cholesterol',        // Cholesterol (mg)
  1292: 'monounsaturatedFat', // Fatty acids, total monounsaturated (g)
  1293: 'polyunsaturatedFat', // Fatty acids, total polyunsaturated (g)

  // Minerals
  1093: 'sodium',          // Sodium, Na (mg)
  1092: 'potassium',       // Potassium, K (mg)
  1087: 'calcium',         // Calcium, Ca (mg)
  1089: 'iron',            // Iron, Fe (mg)
  1090: 'magnesium',       // Magnesium, Mg (mg)
  1095: 'zinc',            // Zinc, Zn (mg)
  1091: 'phosphorus',      // Phosphorus, P (mg)

  // Vitamins
  1106: 'vitaminA',        // Vitamin A, RAE (mcg)
  1162: 'vitaminC',        // Vitamin C, total ascorbic acid (mg)
  1114: 'vitaminD',        // Vitamin D (D2 + D3) (mcg)
  1109: 'vitaminE',        // Vitamin E (alpha-tocopherol) (mg)
  1185: 'vitaminK',        // Vitamin K (phylloquinone) (mcg)
  1175: 'vitaminB6',       // Vitamin B-6 (mg)
  1178: 'vitaminB12',      // Vitamin B-12 (mcg)
  1177: 'folate',          // Folate, total (mcg)
  1165: 'thiamin',         // Thiamin (mg)
  1166: 'riboflavin',      // Riboflavin (mg)
  1167: 'niacin',          // Niacin (mg)

  // Amino Acids (for Lysine/Arginine tracking)
  1213: 'lysine',          // Lysine (g)
  1220: 'arginine',        // Arginine (g)
};

/**
 * Alternate nutrient IDs used in some USDA data types
 * These map to the same app fields but have different USDA IDs
 */
export const USDA_ALTERNATE_NUTRIENT_IDS: Record<number, keyof TransformedNutrients> = {
  // Energy alternates
  1062: 'calories',        // Energy (ATWATER) - sometimes used instead of 1008
  2047: 'calories',        // Energy (Atwater General Factors)
  2048: 'calories',        // Energy (Atwater Specific Factors)

  // Carbs alternates
  1050: 'carbs',           // Carbohydrate, by summation (g)
  1072: 'fiber',           // Fiber, total dietary (AOAC 2011.25) (g)

  // Sugar alternates
  1063: 'sugar',           // Sugars, Total (NLEA) (g)
  1011: 'sugar',           // Sugars, total (g) - older format
};

/**
 * Combined nutrient ID map (primary + alternates)
 */
const COMBINED_NUTRIENT_MAP: Record<number, keyof TransformedNutrients> = {
  ...USDA_NUTRIENT_ID_MAP,
  ...USDA_ALTERNATE_NUTRIENT_IDS,
};

// ============================================================================
// NUTRIENT MAPPING SERVICE
// ============================================================================

export class NutrientMappingService {
  private readonly log: Logger;

  constructor() {
    this.log = createChildLogger({ service: 'NutrientMappingService' });
  }

  // ==========================================================================
  // PUBLIC METHODS
  // ==========================================================================

  /**
   * Map USDA nutrients array to app nutrient schema
   * Values are per 100g as standardized by USDA
   */
  mapUSDANutrients(
    nutrients: (USDANutrient | USDASearchNutrient)[]
  ): TransformedNutrients {
    const result: TransformedNutrients = {
      calories: 0,
      protein: 0,
      carbs: 0,
      fat: 0,
    };

    if (!nutrients || !Array.isArray(nutrients)) {
      return result;
    }

    for (const nutrient of nutrients) {
      const fieldName = COMBINED_NUTRIENT_MAP[nutrient.nutrientId];

      if (fieldName && typeof nutrient.value === 'number') {
        // If field already has a value (from primary ID), don't overwrite with alternate
        if (result[fieldName] === 0 || result[fieldName] === undefined) {
          result[fieldName] = this.roundNutrientValue(nutrient.value, fieldName);
        }
      }
    }

    return result;
  }

  /**
   * Transform full USDA food item to app format
   */
  transformUSDAFood(usdaFood: USDAFoodItem): TransformedUSDAFood {
    return {
      fdcId: usdaFood.fdcId,
      name: this.extractFoodName(usdaFood.description),
      description: usdaFood.description,
      dataType: usdaFood.dataType,
      brand: usdaFood.brandName,
      brandOwner: usdaFood.brandOwner,
      category: usdaFood.foodCategory?.description,
      servingSize: usdaFood.servingSize,
      servingSizeUnit: usdaFood.servingSizeUnit,
      householdServing: usdaFood.householdServingFullText,
      ingredients: usdaFood.ingredients,
      upc: usdaFood.gtinUpc,
      nutrients: this.mapUSDANutrients(usdaFood.foodNutrients),
      portions: this.transformPortions(usdaFood.foodPortions),
    };
  }

  /**
   * Transform search result food to app format
   */
  transformSearchResultFood(food: USDASearchResultFood): TransformedUSDAFood {
    return {
      fdcId: food.fdcId,
      name: this.extractFoodName(food.description),
      description: food.description,
      dataType: food.dataType,
      brand: food.brandName,
      brandOwner: food.brandOwner,
      category: food.foodCategory,
      servingSize: food.servingSize,
      servingSizeUnit: food.servingSizeUnit,
      householdServing: food.householdServingFullText,
      ingredients: food.ingredients,
      upc: food.gtinUpc,
      nutrients: this.mapUSDANutrients(food.foodNutrients),
      portions: this.transformFoodMeasures(food.foodMeasures),
    };
  }

  /**
   * Scale nutrients to a specific serving size
   * USDA nutrients are per 100g, this scales to the desired amount
   */
  scaleNutrients(
    nutrients: TransformedNutrients,
    targetGrams: number
  ): TransformedNutrients {
    if (targetGrams <= 0) {
      this.log.warn({ targetGrams }, 'Invalid serving size, returning zero nutrients');
      return this.getZeroNutrients();
    }

    const scale = targetGrams / 100;
    const scaled: TransformedNutrients = {
      calories: 0,
      protein: 0,
      carbs: 0,
      fat: 0,
    };

    // Scale all present nutrients
    for (const [key, value] of Object.entries(nutrients)) {
      if (typeof value === 'number') {
        const fieldName = key as keyof TransformedNutrients;
        scaled[fieldName] = this.roundNutrientValue(value * scale, fieldName);
      }
    }

    return scaled;
  }

  /**
   * Get a specific nutrient value from USDA nutrients array
   */
  getNutrientValue(
    nutrients: (USDANutrient | USDASearchNutrient)[],
    nutrientId: number
  ): number | undefined {
    const nutrient = nutrients.find((n) => n.nutrientId === nutrientId);
    return nutrient?.value;
  }

  /**
   * Get all mapped nutrient IDs
   */
  getSupportedNutrientIds(): number[] {
    return Object.keys(COMBINED_NUTRIENT_MAP).map(Number);
  }

  /**
   * Check if a nutrient ID is supported by our mapping
   */
  isNutrientSupported(nutrientId: number): boolean {
    return nutrientId in COMBINED_NUTRIENT_MAP;
  }

  /**
   * Get the app field name for a USDA nutrient ID
   */
  getFieldName(nutrientId: number): keyof TransformedNutrients | undefined {
    return COMBINED_NUTRIENT_MAP[nutrientId];
  }

  // ==========================================================================
  // PRIVATE METHODS
  // ==========================================================================

  /**
   * Extract a cleaner food name from USDA description
   * USDA descriptions are often verbose (e.g., "Apples, raw, with skin")
   */
  private extractFoodName(description: string): string {
    if (!description) return 'Unknown Food';

    // For now, just clean up the description slightly
    // More sophisticated name extraction could be added later
    const cleanName = description
      .split(',')[0] // Take first part before comma
      .trim()
      .replace(/^[^a-zA-Z]+/, '') // Remove leading non-letters
      .trim();

    return cleanName || description;
  }

  /**
   * Transform USDA food portions to app format
   */
  private transformPortions(
    portions?: USDAFoodItem['foodPortions']
  ): TransformedPortion[] | undefined {
    if (!portions || portions.length === 0) {
      return undefined;
    }

    return portions.map((portion) => ({
      id: portion.id,
      name: portion.modifier || portion.measureUnit?.name || 'serving',
      gramWeight: portion.gramWeight,
      amount: portion.amount,
    }));
  }

  /**
   * Transform USDA food measures (from search results) to portions
   */
  private transformFoodMeasures(
    measures?: USDASearchResultFood['foodMeasures']
  ): TransformedPortion[] | undefined {
    if (!measures || measures.length === 0) {
      return undefined;
    }

    return measures
      .filter((m) => m.gramWeight > 0)
      .map((measure) => ({
        id: measure.id,
        name: measure.disseminationText || measure.modifier || 'serving',
        gramWeight: measure.gramWeight,
        amount: 1,
      }));
  }

  /**
   * Round nutrient values appropriately based on type
   */
  private roundNutrientValue(
    value: number,
    fieldName: keyof TransformedNutrients
  ): number {
    // Calories are whole numbers
    if (fieldName === 'calories') {
      return Math.round(value);
    }

    // Micronutrients (vitamins, minerals in mcg) keep more precision
    const micronutrients: (keyof TransformedNutrients)[] = [
      'vitaminA',
      'vitaminD',
      'vitaminK',
      'vitaminB12',
      'folate',
    ];

    if (micronutrients.includes(fieldName)) {
      return Math.round(value * 10) / 10; // One decimal place
    }

    // Amino acids are typically small values
    if (fieldName === 'lysine' || fieldName === 'arginine') {
      return Math.round(value * 100) / 100; // Two decimal places
    }

    // Most macros and minerals: one decimal place
    return Math.round(value * 10) / 10;
  }

  /**
   * Get zero nutrients object
   */
  private getZeroNutrients(): TransformedNutrients {
    return {
      calories: 0,
      protein: 0,
      carbs: 0,
      fat: 0,
    };
  }
}

// Export singleton instance
export const nutrientMappingService = new NutrientMappingService();
