/**
 * Supplement Micronutrient Extractor and Estimator
 *
 * Extracts micronutrient values from barcode-scanned supplements
 * and estimates micronutrients from supplement names and dosages.
 *
 * This is particularly useful for:
 * 1. Barcode-scanned supplements from Open Food Facts
 * 2. Manual supplement entry (estimate from name + dosage)
 */

import type { NutritionData } from './nutritionSanitizer';
import { sanitizeNutritionData } from './nutritionSanitizer';

// Maximum reasonable values for supplements (per serving/dose)
// These prevent outliers from bad data
const MAX_SUPPLEMENT_VALUES = {
  // Vitamins
  vitaminA: 10000, // mcg - 10,000 mcg is 1000% DV
  vitaminC: 10000, // mg - high dose supplements can be 10g
  vitaminD: 500, // mcg - 20,000 IU = 500 mcg
  vitaminE: 1000, // mg
  vitaminK: 5000, // mcg
  vitaminB6: 500, // mg
  vitaminB12: 50000, // mcg - B12 supplements can be very high
  folate: 10000, // mcg
  thiamin: 500, // mg
  riboflavin: 500, // mg
  niacin: 2000, // mg
  biotin: 50000, // mcg

  // Minerals
  calcium: 5000, // mg
  iron: 200, // mg
  magnesium: 2000, // mg
  zinc: 200, // mg
  potassium: 5000, // mg
  sodium: 2000, // mg
  phosphorus: 3000, // mg
  selenium: 1000, // mcg
  copper: 20, // mg
  manganese: 50, // mg
  chromium: 2000, // mcg
  iodine: 5000, // mcg
  molybdenum: 2000, // mcg

  // Other
  omega3: 10000, // mg (fish oil)
  omega6: 5000, // mg
  coq10: 1000, // mg
  probiotics: 1000000000000, // CFU (billions are common)
};

/**
 * Supplement nutrient info that can be extracted or estimated
 */
export interface SupplementNutrients {
  // Vitamins
  vitaminA?: number; // mcg RAE
  vitaminC?: number; // mg
  vitaminD?: number; // mcg (not IU - convert IU to mcg)
  vitaminE?: number; // mg
  vitaminK?: number; // mcg
  vitaminB6?: number; // mg
  vitaminB12?: number; // mcg
  folate?: number; // mcg DFE
  thiamin?: number; // mg (B1)
  riboflavin?: number; // mg (B2)
  niacin?: number; // mg (B3)

  // Minerals
  calcium?: number; // mg
  iron?: number; // mg
  magnesium?: number; // mg
  zinc?: number; // mg
  potassium?: number; // mg
  sodium?: number; // mg
  phosphorus?: number; // mg

  // Special
  omega3?: number; // mg (total EPA + DHA)
}

/**
 * Known supplement name patterns and their primary nutrient
 * Used to estimate micronutrient content from supplement names
 */
const SUPPLEMENT_PATTERNS: Array<{
  patterns: RegExp[];
  nutrient: keyof SupplementNutrients;
  defaultUnit: string;
  conversionFactor?: number; // e.g., IU to mcg
}> = [
  // Vitamin D
  {
    patterns: [/vitamin\s*d3?/i, /cholecalciferol/i, /d3?\s*(\d)/i],
    nutrient: 'vitaminD',
    defaultUnit: 'IU',
    conversionFactor: 0.025, // 1 IU = 0.025 mcg
  },
  // Vitamin C
  {
    patterns: [/vitamin\s*c/i, /ascorbic\s*acid/i],
    nutrient: 'vitaminC',
    defaultUnit: 'mg',
  },
  // Vitamin A
  {
    patterns: [/vitamin\s*a/i, /retinol/i, /beta[\s-]?carotene/i],
    nutrient: 'vitaminA',
    defaultUnit: 'IU',
    conversionFactor: 0.3, // 1 IU = 0.3 mcg RAE for retinol
  },
  // Vitamin E
  {
    patterns: [/vitamin\s*e/i, /tocopherol/i],
    nutrient: 'vitaminE',
    defaultUnit: 'IU',
    conversionFactor: 0.67, // 1 IU = 0.67 mg for d-alpha-tocopherol
  },
  // Vitamin K
  {
    patterns: [/vitamin\s*k2?/i, /menaquinone/i, /phylloquinone/i],
    nutrient: 'vitaminK',
    defaultUnit: 'mcg',
  },
  // B Vitamins
  {
    patterns: [/vitamin\s*b6/i, /pyridoxine/i],
    nutrient: 'vitaminB6',
    defaultUnit: 'mg',
  },
  {
    patterns: [/vitamin\s*b12/i, /cobalamin/i, /methylcobalamin/i, /cyanocobalamin/i],
    nutrient: 'vitaminB12',
    defaultUnit: 'mcg',
  },
  {
    patterns: [/folate/i, /folic\s*acid/i, /methylfolate/i],
    nutrient: 'folate',
    defaultUnit: 'mcg',
  },
  {
    patterns: [/thiamin/i, /vitamin\s*b1/i],
    nutrient: 'thiamin',
    defaultUnit: 'mg',
  },
  {
    patterns: [/riboflavin/i, /vitamin\s*b2/i],
    nutrient: 'riboflavin',
    defaultUnit: 'mg',
  },
  {
    patterns: [/niacin/i, /vitamin\s*b3/i, /nicotinic\s*acid/i],
    nutrient: 'niacin',
    defaultUnit: 'mg',
  },
  // Minerals
  {
    patterns: [/calcium/i, /\bcal\b/i],
    nutrient: 'calcium',
    defaultUnit: 'mg',
  },
  {
    patterns: [/\biron\b/i, /ferrous/i, /ferric/i],
    nutrient: 'iron',
    defaultUnit: 'mg',
  },
  {
    patterns: [/magnesium/i, /\bmag\b/i],
    nutrient: 'magnesium',
    defaultUnit: 'mg',
  },
  {
    patterns: [/\bzinc\b/i],
    nutrient: 'zinc',
    defaultUnit: 'mg',
  },
  {
    patterns: [/potassium/i],
    nutrient: 'potassium',
    defaultUnit: 'mg',
  },
  // Special
  {
    patterns: [/omega[\s-]?3/i, /fish\s*oil/i, /epa/i, /dha/i],
    nutrient: 'omega3',
    defaultUnit: 'mg',
  },
];

/**
 * Common multivitamin profile (typical amounts per serving)
 * Used when supplement name suggests it's a multivitamin
 */
const MULTIVITAMIN_PROFILE: SupplementNutrients = {
  vitaminA: 900, // mcg (100% DV)
  vitaminC: 90, // mg (100% DV)
  vitaminD: 20, // mcg (100% DV)
  vitaminE: 15, // mg (100% DV)
  vitaminK: 120, // mcg (100% DV)
  vitaminB6: 1.7, // mg (100% DV)
  vitaminB12: 2.4, // mcg (100% DV)
  folate: 400, // mcg (100% DV)
  thiamin: 1.2, // mg (100% DV)
  riboflavin: 1.3, // mg (100% DV)
  niacin: 16, // mg (100% DV)
  calcium: 200, // mg (partial)
  iron: 18, // mg (100% DV)
  magnesium: 100, // mg (partial)
  zinc: 11, // mg (100% DV)
};

/**
 * Extracts micronutrients from Open Food Facts nutriments data
 */
export function extractMicronutrientsFromBarcode(
  nutriments: Record<string, number | string | undefined>,
  servingSize?: string
): SupplementNutrients {
  const nutrients: SupplementNutrients = {};

  // Helper to safely extract numeric values
  const getValue = (
    servingKey: string,
    per100gKey: string,
    maxValue: number
  ): number | undefined => {
    // Prefer per-serving values
    let value = nutriments[servingKey];
    if (typeof value === 'number' && value > 0 && value <= maxValue) {
      return Math.round(value * 100) / 100;
    }

    // Fall back to per-100g (supplements are often single servings)
    value = nutriments[per100gKey];
    if (typeof value === 'number' && value > 0 && value <= maxValue) {
      return Math.round(value * 100) / 100;
    }

    return undefined;
  };

  // Extract vitamins
  nutrients.vitaminA = getValue(
    'vitamin-a_serving',
    'vitamin-a_100g',
    MAX_SUPPLEMENT_VALUES.vitaminA
  );
  nutrients.vitaminC = getValue(
    'vitamin-c_serving',
    'vitamin-c_100g',
    MAX_SUPPLEMENT_VALUES.vitaminC
  );
  nutrients.vitaminD = getValue(
    'vitamin-d_serving',
    'vitamin-d_100g',
    MAX_SUPPLEMENT_VALUES.vitaminD
  );
  nutrients.vitaminE = getValue(
    'vitamin-e_serving',
    'vitamin-e_100g',
    MAX_SUPPLEMENT_VALUES.vitaminE
  );
  nutrients.vitaminK = getValue(
    'vitamin-k_serving',
    'vitamin-k_100g',
    MAX_SUPPLEMENT_VALUES.vitaminK
  );
  nutrients.vitaminB6 = getValue(
    'vitamin-b6_serving',
    'vitamin-b6_100g',
    MAX_SUPPLEMENT_VALUES.vitaminB6
  );
  nutrients.vitaminB12 = getValue(
    'vitamin-b12_serving',
    'vitamin-b12_100g',
    MAX_SUPPLEMENT_VALUES.vitaminB12
  );
  nutrients.folate = getValue(
    'folates_serving',
    'folates_100g',
    MAX_SUPPLEMENT_VALUES.folate
  );
  nutrients.thiamin = getValue(
    'vitamin-b1_serving',
    'vitamin-b1_100g',
    MAX_SUPPLEMENT_VALUES.thiamin
  );
  nutrients.riboflavin = getValue(
    'vitamin-b2_serving',
    'vitamin-b2_100g',
    MAX_SUPPLEMENT_VALUES.riboflavin
  );
  nutrients.niacin = getValue(
    'vitamin-pp_serving',
    'vitamin-pp_100g',
    MAX_SUPPLEMENT_VALUES.niacin
  );

  // Extract minerals
  nutrients.calcium = getValue(
    'calcium_serving',
    'calcium_100g',
    MAX_SUPPLEMENT_VALUES.calcium
  );
  nutrients.iron = getValue(
    'iron_serving',
    'iron_100g',
    MAX_SUPPLEMENT_VALUES.iron
  );
  nutrients.magnesium = getValue(
    'magnesium_serving',
    'magnesium_100g',
    MAX_SUPPLEMENT_VALUES.magnesium
  );
  nutrients.zinc = getValue(
    'zinc_serving',
    'zinc_100g',
    MAX_SUPPLEMENT_VALUES.zinc
  );
  nutrients.potassium = getValue(
    'potassium_serving',
    'potassium_100g',
    MAX_SUPPLEMENT_VALUES.potassium
  );
  nutrients.sodium = getValue(
    'sodium_serving',
    'sodium_100g',
    MAX_SUPPLEMENT_VALUES.sodium
  );
  nutrients.phosphorus = getValue(
    'phosphorus_serving',
    'phosphorus_100g',
    MAX_SUPPLEMENT_VALUES.phosphorus
  );

  // Clean up undefined values
  return Object.fromEntries(
    Object.entries(nutrients).filter(([, v]) => v !== undefined)
  ) as SupplementNutrients;
}

/**
 * Estimates micronutrients from supplement name and dosage
 */
export function estimateMicronutrientsFromName(
  supplementName: string,
  dosageAmount?: number,
  dosageUnit?: string
): SupplementNutrients {
  const nutrients: SupplementNutrients = {};
  const lowerName = supplementName.toLowerCase();

  // Check for multivitamin first
  if (
    lowerName.includes('multivitamin') ||
    lowerName.includes('multi-vitamin') ||
    lowerName.includes('one a day') ||
    lowerName.includes('daily vitamin') ||
    lowerName.includes('complete vitamin')
  ) {
    return { ...MULTIVITAMIN_PROFILE };
  }

  // Match against known supplement patterns
  for (const pattern of SUPPLEMENT_PATTERNS) {
    const matches = pattern.patterns.some((p) => p.test(supplementName));
    if (matches && dosageAmount && dosageUnit) {
      let value = dosageAmount;

      // Convert units if necessary (e.g., IU to mcg)
      const lowerUnit = dosageUnit.toLowerCase();
      if (
        pattern.conversionFactor &&
        (lowerUnit === 'iu' || lowerUnit === 'i.u.')
      ) {
        value = dosageAmount * pattern.conversionFactor;
      }

      // Apply max limits
      const maxValue =
        MAX_SUPPLEMENT_VALUES[
          pattern.nutrient as keyof typeof MAX_SUPPLEMENT_VALUES
        ];
      if (maxValue && value > maxValue) {
        value = maxValue;
      }

      nutrients[pattern.nutrient] = Math.round(value * 100) / 100;

      // For single-nutrient supplements, we typically only have one nutrient
      // but some may have related nutrients
      break;
    }
  }

  return nutrients;
}

/**
 * Combines barcode data and name-based estimation
 * Prefers barcode data when available, fills in with estimates
 */
export function getSupplementMicronutrients(
  supplementName: string,
  dosageAmount?: number,
  dosageUnit?: string,
  barcodeNutriments?: Record<string, number | string | undefined>
): SupplementNutrients {
  // Start with barcode extraction if available
  let nutrients: SupplementNutrients = {};

  if (barcodeNutriments) {
    nutrients = extractMicronutrientsFromBarcode(barcodeNutriments);
  }

  // If no barcode data or incomplete, estimate from name
  const hasNutrients = Object.keys(nutrients).length > 0;
  if (!hasNutrients) {
    nutrients = estimateMicronutrientsFromName(
      supplementName,
      dosageAmount,
      dosageUnit
    );
  }

  return nutrients;
}

/**
 * Sanitizes supplement micronutrient data to ensure reasonable values
 */
export function sanitizeSupplementNutrients(
  nutrients: SupplementNutrients
): SupplementNutrients {
  // Convert to NutritionData format for sanitization
  const nutritionData: NutritionData = {
    vitaminA: nutrients.vitaminA,
    vitaminC: nutrients.vitaminC,
    vitaminD: nutrients.vitaminD,
    vitaminE: nutrients.vitaminE,
    vitaminK: nutrients.vitaminK,
    vitaminB6: nutrients.vitaminB6,
    vitaminB12: nutrients.vitaminB12,
    folate: nutrients.folate,
    thiamin: nutrients.thiamin,
    riboflavin: nutrients.riboflavin,
    niacin: nutrients.niacin,
    calcium: nutrients.calcium,
    iron: nutrients.iron,
    magnesium: nutrients.magnesium,
    zinc: nutrients.zinc,
    potassium: nutrients.potassium,
    sodium: nutrients.sodium,
    phosphorus: nutrients.phosphorus,
  };

  // Use per-serving mode since supplements are typically single servings
  const result = sanitizeNutritionData(nutritionData, 'perServing');

  // Convert back to SupplementNutrients
  const sanitized: SupplementNutrients = {};

  if (result.data.vitaminA !== undefined) sanitized.vitaminA = result.data.vitaminA;
  if (result.data.vitaminC !== undefined) sanitized.vitaminC = result.data.vitaminC;
  if (result.data.vitaminD !== undefined) sanitized.vitaminD = result.data.vitaminD;
  if (result.data.vitaminE !== undefined) sanitized.vitaminE = result.data.vitaminE;
  if (result.data.vitaminK !== undefined) sanitized.vitaminK = result.data.vitaminK;
  if (result.data.vitaminB6 !== undefined) sanitized.vitaminB6 = result.data.vitaminB6;
  if (result.data.vitaminB12 !== undefined) sanitized.vitaminB12 = result.data.vitaminB12;
  if (result.data.folate !== undefined) sanitized.folate = result.data.folate;
  if (result.data.thiamin !== undefined) sanitized.thiamin = result.data.thiamin;
  if (result.data.riboflavin !== undefined) sanitized.riboflavin = result.data.riboflavin;
  if (result.data.niacin !== undefined) sanitized.niacin = result.data.niacin;
  if (result.data.calcium !== undefined) sanitized.calcium = result.data.calcium;
  if (result.data.iron !== undefined) sanitized.iron = result.data.iron;
  if (result.data.magnesium !== undefined) sanitized.magnesium = result.data.magnesium;
  if (result.data.zinc !== undefined) sanitized.zinc = result.data.zinc;
  if (result.data.potassium !== undefined) sanitized.potassium = result.data.potassium;
  if (result.data.sodium !== undefined) sanitized.sodium = result.data.sodium;
  if (result.data.phosphorus !== undefined) sanitized.phosphorus = result.data.phosphorus;

  // Keep omega3 if present (not in standard nutrition data)
  if (nutrients.omega3 !== undefined) {
    sanitized.omega3 = Math.min(nutrients.omega3, MAX_SUPPLEMENT_VALUES.omega3);
  }

  return sanitized;
}

/**
 * Formats supplement micronutrients for display
 */
export function formatSupplementNutrient(
  nutrient: keyof SupplementNutrients,
  value: number | undefined
): string {
  if (value === undefined) return '-';

  // Units for each nutrient
  const units: Record<keyof SupplementNutrients, string> = {
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
    calcium: 'mg',
    iron: 'mg',
    magnesium: 'mg',
    zinc: 'mg',
    potassium: 'mg',
    sodium: 'mg',
    phosphorus: 'mg',
    omega3: 'mg',
  };

  const unit = units[nutrient] || '';

  // Format with appropriate decimal places
  const integerNutrients: Array<keyof SupplementNutrients> = [
    'vitaminA',
    'vitaminK',
    'folate',
    'vitaminB12',
    'calcium',
    'magnesium',
    'potassium',
    'sodium',
    'phosphorus',
    'omega3',
  ];

  if (integerNutrients.includes(nutrient)) {
    return `${Math.round(value)}${unit}`;
  }

  return `${value.toFixed(1)}${unit}`;
}

/**
 * Gets human-readable label for a supplement nutrient
 */
export function getSupplementNutrientLabel(
  nutrient: keyof SupplementNutrients
): string {
  const labels: Record<keyof SupplementNutrients, string> = {
    vitaminA: 'Vitamin A',
    vitaminC: 'Vitamin C',
    vitaminD: 'Vitamin D',
    vitaminE: 'Vitamin E',
    vitaminK: 'Vitamin K',
    vitaminB6: 'Vitamin B6',
    vitaminB12: 'Vitamin B12',
    folate: 'Folate',
    thiamin: 'Thiamin (B1)',
    riboflavin: 'Riboflavin (B2)',
    niacin: 'Niacin (B3)',
    calcium: 'Calcium',
    iron: 'Iron',
    magnesium: 'Magnesium',
    zinc: 'Zinc',
    potassium: 'Potassium',
    sodium: 'Sodium',
    phosphorus: 'Phosphorus',
    omega3: 'Omega-3',
  };

  return labels[nutrient] || nutrient;
}

/**
 * Checks if a supplement has any estimated micronutrients
 */
export function hasSupplementMicronutrients(
  nutrients: SupplementNutrients | undefined
): boolean {
  if (!nutrients) return false;
  return Object.values(nutrients).some((v) => v !== undefined && v > 0);
}
