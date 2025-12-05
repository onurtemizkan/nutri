/**
 * Portion Estimation Utilities
 *
 * Client-side utilities for volume calculation, food density lookup,
 * and weight estimation from AR measurements.
 */

import type { ARMeasurement } from '@/lib/types/food-analysis';

/**
 * Food density data in g/cm³
 * Sources: USDA, food science literature
 */
export interface FoodDensityEntry {
  name: string;
  density: number; // g/cm³
  category: FoodCategory;
  shapeFactor: number; // Accounts for non-cuboid shapes (0-1)
}

export type FoodCategory =
  | 'fruit'
  | 'vegetable'
  | 'protein'
  | 'grain'
  | 'dairy'
  | 'beverage'
  | 'baked'
  | 'snack'
  | 'mixed'
  | 'unknown';

/**
 * Food density lookup table
 * Values are approximate and based on typical food densities
 */
export const FOOD_DENSITY_TABLE: Record<string, FoodDensityEntry> = {
  // Fruits
  apple: { name: 'Apple', density: 0.7, category: 'fruit', shapeFactor: 0.52 },
  banana: { name: 'Banana', density: 0.9, category: 'fruit', shapeFactor: 0.45 },
  orange: { name: 'Orange', density: 0.75, category: 'fruit', shapeFactor: 0.52 },
  strawberry: { name: 'Strawberry', density: 0.65, category: 'fruit', shapeFactor: 0.5 },
  grape: { name: 'Grape', density: 0.85, category: 'fruit', shapeFactor: 0.52 },
  watermelon: { name: 'Watermelon', density: 0.95, category: 'fruit', shapeFactor: 0.6 },
  mango: { name: 'Mango', density: 0.8, category: 'fruit', shapeFactor: 0.55 },
  pineapple: { name: 'Pineapple', density: 0.6, category: 'fruit', shapeFactor: 0.5 },
  blueberry: { name: 'Blueberry', density: 0.7, category: 'fruit', shapeFactor: 0.52 },
  avocado: { name: 'Avocado', density: 0.85, category: 'fruit', shapeFactor: 0.55 },

  // Vegetables
  broccoli: { name: 'Broccoli', density: 0.3, category: 'vegetable', shapeFactor: 0.35 },
  carrot: { name: 'Carrot', density: 0.85, category: 'vegetable', shapeFactor: 0.7 },
  potato: { name: 'Potato', density: 1.05, category: 'vegetable', shapeFactor: 0.6 },
  tomato: { name: 'Tomato', density: 0.95, category: 'vegetable', shapeFactor: 0.52 },
  cucumber: { name: 'Cucumber', density: 0.95, category: 'vegetable', shapeFactor: 0.75 },
  lettuce: { name: 'Lettuce', density: 0.15, category: 'vegetable', shapeFactor: 0.25 },
  spinach: { name: 'Spinach', density: 0.2, category: 'vegetable', shapeFactor: 0.2 },
  onion: { name: 'Onion', density: 0.9, category: 'vegetable', shapeFactor: 0.52 },
  pepper: { name: 'Pepper', density: 0.35, category: 'vegetable', shapeFactor: 0.4 },
  mushroom: { name: 'Mushroom', density: 0.4, category: 'vegetable', shapeFactor: 0.45 },
  corn: { name: 'Corn', density: 0.75, category: 'vegetable', shapeFactor: 0.7 },
  celery: { name: 'Celery', density: 0.6, category: 'vegetable', shapeFactor: 0.65 },

  // Proteins
  'chicken breast': { name: 'Chicken Breast', density: 1.05, category: 'protein', shapeFactor: 0.7 },
  'chicken thigh': { name: 'Chicken Thigh', density: 1.0, category: 'protein', shapeFactor: 0.65 },
  beef: { name: 'Beef', density: 1.05, category: 'protein', shapeFactor: 0.75 },
  steak: { name: 'Steak', density: 1.05, category: 'protein', shapeFactor: 0.8 },
  pork: { name: 'Pork', density: 1.0, category: 'protein', shapeFactor: 0.75 },
  salmon: { name: 'Salmon', density: 1.0, category: 'protein', shapeFactor: 0.8 },
  tuna: { name: 'Tuna', density: 1.05, category: 'protein', shapeFactor: 0.8 },
  shrimp: { name: 'Shrimp', density: 0.9, category: 'protein', shapeFactor: 0.5 },
  egg: { name: 'Egg', density: 1.03, category: 'protein', shapeFactor: 0.52 },
  tofu: { name: 'Tofu', density: 1.0, category: 'protein', shapeFactor: 0.85 },

  // Grains
  rice: { name: 'Rice (cooked)', density: 0.8, category: 'grain', shapeFactor: 0.9 },
  pasta: { name: 'Pasta (cooked)', density: 0.85, category: 'grain', shapeFactor: 0.7 },
  bread: { name: 'Bread', density: 0.3, category: 'grain', shapeFactor: 0.85 },
  oatmeal: { name: 'Oatmeal', density: 0.75, category: 'grain', shapeFactor: 0.9 },
  quinoa: { name: 'Quinoa', density: 0.8, category: 'grain', shapeFactor: 0.9 },
  cereal: { name: 'Cereal', density: 0.25, category: 'grain', shapeFactor: 0.85 },

  // Dairy
  cheese: { name: 'Cheese', density: 1.1, category: 'dairy', shapeFactor: 0.9 },
  yogurt: { name: 'Yogurt', density: 1.05, category: 'dairy', shapeFactor: 0.95 },
  milk: { name: 'Milk', density: 1.03, category: 'beverage', shapeFactor: 1.0 },
  butter: { name: 'Butter', density: 0.91, category: 'dairy', shapeFactor: 0.95 },

  // Baked goods
  cake: { name: 'Cake', density: 0.5, category: 'baked', shapeFactor: 0.85 },
  cookie: { name: 'Cookie', density: 0.6, category: 'baked', shapeFactor: 0.8 },
  muffin: { name: 'Muffin', density: 0.45, category: 'baked', shapeFactor: 0.6 },
  donut: { name: 'Donut', density: 0.4, category: 'baked', shapeFactor: 0.5 },
  pizza: { name: 'Pizza', density: 0.7, category: 'baked', shapeFactor: 0.85 },

  // Snacks
  chips: { name: 'Chips', density: 0.15, category: 'snack', shapeFactor: 0.3 },
  popcorn: { name: 'Popcorn', density: 0.05, category: 'snack', shapeFactor: 0.2 },
  nuts: { name: 'Nuts', density: 0.65, category: 'snack', shapeFactor: 0.6 },

  // Mixed/Prepared
  salad: { name: 'Salad', density: 0.25, category: 'mixed', shapeFactor: 0.3 },
  soup: { name: 'Soup', density: 1.0, category: 'mixed', shapeFactor: 1.0 },
  sandwich: { name: 'Sandwich', density: 0.55, category: 'mixed', shapeFactor: 0.8 },
  burrito: { name: 'Burrito', density: 0.9, category: 'mixed', shapeFactor: 0.75 },
  burger: { name: 'Burger', density: 0.7, category: 'mixed', shapeFactor: 0.7 },
};

/**
 * Default density values by category (g/cm³)
 */
export const DEFAULT_DENSITY_BY_CATEGORY: Record<FoodCategory, number> = {
  fruit: 0.75,
  vegetable: 0.55,
  protein: 1.0,
  grain: 0.6,
  dairy: 1.0,
  beverage: 1.0,
  baked: 0.45,
  snack: 0.3,
  mixed: 0.7,
  unknown: 0.7,
};

/**
 * Default shape factors by category
 */
export const DEFAULT_SHAPE_FACTOR_BY_CATEGORY: Record<FoodCategory, number> = {
  fruit: 0.52,
  vegetable: 0.5,
  protein: 0.75,
  grain: 0.85,
  dairy: 0.9,
  beverage: 1.0,
  baked: 0.75,
  snack: 0.4,
  mixed: 0.7,
  unknown: 0.7,
};

// Weight bounds
export const MIN_WEIGHT_GRAMS = 1;
export const MAX_WEIGHT_GRAMS = 5000;

// ============================================================================
// Volume Calculations
// ============================================================================

/**
 * Calculate volume from dimensions (cuboid approximation)
 * @param width Width in cm
 * @param height Height in cm
 * @param depth Depth in cm
 * @returns Volume in cm³
 */
export function calculateVolume(width: number, height: number, depth: number): number {
  if (width <= 0 || height <= 0 || depth <= 0) {
    return 0;
  }
  return width * height * depth;
}

/**
 * Calculate volume from ARMeasurement
 * @param measurement ARMeasurement object
 * @returns Volume in cm³
 */
export function volumeFromMeasurement(measurement: ARMeasurement): number {
  return calculateVolume(measurement.width, measurement.height, measurement.depth);
}

/**
 * Apply shape factor to volume
 * Shape factor accounts for the fact that food items are not perfect cuboids
 * @param volume Volume in cm³
 * @param shapeFactor Factor between 0 and 1
 * @returns Adjusted volume in cm³
 */
export function applyShapeFactor(volume: number, shapeFactor: number): number {
  const clampedFactor = Math.max(0, Math.min(1, shapeFactor));
  return volume * clampedFactor;
}

// ============================================================================
// Density Lookup
// ============================================================================

/**
 * Get food density entry by name (case-insensitive fuzzy match)
 * @param foodName Name of the food
 * @returns FoodDensityEntry or undefined if not found
 */
export function lookupFoodDensity(foodName: string): FoodDensityEntry | undefined {
  const normalizedName = foodName.toLowerCase().trim();

  // Exact match
  if (FOOD_DENSITY_TABLE[normalizedName]) {
    return FOOD_DENSITY_TABLE[normalizedName];
  }

  // Partial match (food name contains or is contained by query)
  for (const [key, entry] of Object.entries(FOOD_DENSITY_TABLE)) {
    if (key.includes(normalizedName) || normalizedName.includes(key)) {
      return entry;
    }
  }

  return undefined;
}

/**
 * Get density for a food, falling back to category or default
 * @param foodName Name of the food
 * @param category Optional category hint
 * @returns Density in g/cm³
 */
export function getDensity(foodName: string, category?: FoodCategory): number {
  const entry = lookupFoodDensity(foodName);
  if (entry) {
    return entry.density;
  }

  if (category) {
    return DEFAULT_DENSITY_BY_CATEGORY[category];
  }

  return DEFAULT_DENSITY_BY_CATEGORY.unknown;
}

/**
 * Get shape factor for a food, falling back to category or default
 * @param foodName Name of the food
 * @param category Optional category hint
 * @returns Shape factor between 0 and 1
 */
export function getShapeFactor(foodName: string, category?: FoodCategory): number {
  const entry = lookupFoodDensity(foodName);
  if (entry) {
    return entry.shapeFactor;
  }

  if (category) {
    return DEFAULT_SHAPE_FACTOR_BY_CATEGORY[category];
  }

  return DEFAULT_SHAPE_FACTOR_BY_CATEGORY.unknown;
}

// ============================================================================
// Weight Estimation
// ============================================================================

export interface WeightEstimate {
  weight: number; // grams
  confidence: number; // 0-1
  method: 'density-lookup' | 'category-default' | 'generic-default';
  densityUsed: number; // g/cm³
  shapeFactorUsed: number;
  volumeRaw: number; // cm³
  volumeAdjusted: number; // cm³ after shape factor
}

/**
 * Estimate weight from dimensions and food type
 * @param width Width in cm
 * @param height Height in cm
 * @param depth Depth in cm
 * @param foodName Name of the food item
 * @param category Optional category hint
 * @returns WeightEstimate with details
 */
export function estimateWeight(
  width: number,
  height: number,
  depth: number,
  foodName: string,
  category?: FoodCategory
): WeightEstimate {
  // Calculate raw volume
  const volumeRaw = calculateVolume(width, height, depth);

  // Lookup density and shape factor
  const entry = lookupFoodDensity(foodName);
  let density: number;
  let shapeFactor: number;
  let method: WeightEstimate['method'];

  if (entry) {
    density = entry.density;
    shapeFactor = entry.shapeFactor;
    method = 'density-lookup';
  } else if (category) {
    density = DEFAULT_DENSITY_BY_CATEGORY[category];
    shapeFactor = DEFAULT_SHAPE_FACTOR_BY_CATEGORY[category];
    method = 'category-default';
  } else {
    density = DEFAULT_DENSITY_BY_CATEGORY.unknown;
    shapeFactor = DEFAULT_SHAPE_FACTOR_BY_CATEGORY.unknown;
    method = 'generic-default';
  }

  // Apply shape factor to volume
  const volumeAdjusted = applyShapeFactor(volumeRaw, shapeFactor);

  // Calculate weight
  let weight = volumeAdjusted * density;

  // Apply bounds
  weight = Math.max(MIN_WEIGHT_GRAMS, Math.min(MAX_WEIGHT_GRAMS, weight));

  // Calculate confidence based on method and measurement quality
  let confidence: number;
  switch (method) {
    case 'density-lookup':
      confidence = 0.8;
      break;
    case 'category-default':
      confidence = 0.6;
      break;
    case 'generic-default':
    default:
      confidence = 0.4;
      break;
  }

  return {
    weight: Math.round(weight * 10) / 10, // Round to 1 decimal
    confidence,
    method,
    densityUsed: density,
    shapeFactorUsed: shapeFactor,
    volumeRaw,
    volumeAdjusted,
  };
}

/**
 * Estimate weight from ARMeasurement
 * @param measurement ARMeasurement object
 * @param foodName Name of the food item
 * @param category Optional category hint
 * @returns WeightEstimate with details
 */
export function estimateWeightFromMeasurement(
  measurement: ARMeasurement,
  foodName: string,
  category?: FoodCategory
): WeightEstimate {
  const estimate = estimateWeight(
    measurement.width,
    measurement.height,
    measurement.depth,
    foodName,
    category
  );

  // Adjust confidence based on AR measurement confidence
  const arConfidenceMultiplier =
    measurement.confidence === 'high' ? 1.0 :
    measurement.confidence === 'medium' ? 0.85 :
    0.7;

  // Further adjust if plane wasn't detected
  const planeMultiplier = measurement.planeDetected ? 1.0 : 0.8;

  estimate.confidence = Math.round(
    estimate.confidence * arConfidenceMultiplier * planeMultiplier * 100
  ) / 100;

  return estimate;
}

// ============================================================================
// Unit Conversions
// ============================================================================

/**
 * Convert centimeters to inches
 */
export function cmToInches(cm: number): number {
  return cm / 2.54;
}

/**
 * Convert inches to centimeters
 */
export function inchesToCm(inches: number): number {
  return inches * 2.54;
}

/**
 * Convert grams to ounces
 */
export function gramsToOz(grams: number): number {
  return grams / 28.3495;
}

/**
 * Convert ounces to grams
 */
export function ozToGrams(oz: number): number {
  return oz * 28.3495;
}

/**
 * Convert grams to pounds
 */
export function gramsToLbs(grams: number): number {
  return grams / 453.592;
}

/**
 * Convert pounds to grams
 */
export function lbsToGrams(lbs: number): number {
  return lbs * 453.592;
}

/**
 * Convert cubic centimeters to milliliters (1:1)
 */
export function volumeCm3ToMl(cm3: number): number {
  return cm3; // 1 cm³ = 1 mL
}

/**
 * Convert milliliters to cubic centimeters (1:1)
 */
export function mlToVolumeCm3(ml: number): number {
  return ml; // 1 mL = 1 cm³
}

/**
 * Convert cubic centimeters to cups (US)
 */
export function volumeCm3ToCups(cm3: number): number {
  return cm3 / 236.588;
}

/**
 * Convert cups (US) to cubic centimeters
 */
export function cupsToVolumeCm3(cups: number): number {
  return cups * 236.588;
}

// ============================================================================
// Preset Sizes for Manual Selection
// ============================================================================

export interface PresetSize {
  name: string;
  displayName: string;
  width: number; // cm
  height: number; // cm
  depth: number; // cm
  description: string;
  referenceObject: string;
}

export const PRESET_SIZES: PresetSize[] = [
  {
    name: 'extra-small',
    displayName: 'Extra Small',
    width: 3,
    height: 3,
    depth: 3,
    description: 'About 27 cm³ / 1 tablespoon',
    referenceObject: 'Golf ball or walnut',
  },
  {
    name: 'small',
    displayName: 'Small',
    width: 5,
    height: 5,
    depth: 5,
    description: 'About 125 cm³ / 1/2 cup',
    referenceObject: 'Tennis ball or small apple',
  },
  {
    name: 'medium',
    displayName: 'Medium',
    width: 8,
    height: 8,
    depth: 6,
    description: 'About 384 cm³ / 1.5 cups',
    referenceObject: 'Baseball or medium potato',
  },
  {
    name: 'large',
    displayName: 'Large',
    width: 12,
    height: 10,
    depth: 8,
    description: 'About 960 cm³ / 4 cups',
    referenceObject: 'Softball or large apple',
  },
  {
    name: 'extra-large',
    displayName: 'Extra Large',
    width: 15,
    height: 12,
    depth: 10,
    description: 'About 1800 cm³ / 7.5 cups',
    referenceObject: 'Grapefruit or small melon',
  },
];

/**
 * Convert preset size to ARMeasurement
 * @param preset Preset size object
 * @returns ARMeasurement with low confidence (manual estimate)
 */
export function presetToMeasurement(preset: PresetSize): ARMeasurement {
  return {
    width: preset.width,
    height: preset.height,
    depth: preset.depth,
    distance: 30, // Assumed typical viewing distance
    confidence: 'low',
    planeDetected: false,
    timestamp: new Date(),
  };
}

/**
 * Get preset size by name
 */
export function getPresetSize(name: string): PresetSize | undefined {
  return PRESET_SIZES.find((p) => p.name === name);
}

// ============================================================================
// Validation Helpers
// ============================================================================

/**
 * Validate dimension value (must be positive and reasonable for food)
 * @param value Dimension value in cm
 * @param maxValue Maximum allowed value (default 100 cm)
 * @returns true if valid
 */
export function isValidDimension(value: number, maxValue: number = 100): boolean {
  return typeof value === 'number' && value > 0 && value <= maxValue && !isNaN(value);
}

/**
 * Validate ARMeasurement object
 * @param measurement ARMeasurement to validate
 * @returns Object with valid flag and optional error message
 */
export function validateMeasurement(measurement: ARMeasurement): {
  valid: boolean;
  error?: string;
} {
  if (!isValidDimension(measurement.width)) {
    return { valid: false, error: 'Invalid width: must be between 0 and 100 cm' };
  }
  if (!isValidDimension(measurement.height)) {
    return { valid: false, error: 'Invalid height: must be between 0 and 100 cm' };
  }
  if (!isValidDimension(measurement.depth)) {
    return { valid: false, error: 'Invalid depth: must be between 0 and 100 cm' };
  }
  if (measurement.distance !== undefined && measurement.distance < 0) {
    return { valid: false, error: 'Invalid distance: must be non-negative' };
  }

  return { valid: true };
}

// ============================================================================
// Formatting Helpers
// ============================================================================

/**
 * Format weight for display
 * @param grams Weight in grams
 * @param includeUnits Whether to include unit suffix
 * @returns Formatted string
 */
export function formatWeight(grams: number, includeUnits: boolean = true): string {
  if (grams < 1) {
    return includeUnits ? '<1g' : '<1';
  }
  if (grams >= 1000) {
    const kg = Math.round(grams / 100) / 10;
    return includeUnits ? `${kg}kg` : `${kg}`;
  }
  const rounded = Math.round(grams);
  return includeUnits ? `${rounded}g` : `${rounded}`;
}

/**
 * Format dimensions for display
 * @param width Width in cm
 * @param height Height in cm
 * @param depth Depth in cm
 * @returns Formatted string like "10 × 8 × 5 cm"
 */
export function formatDimensions(
  width: number,
  height: number,
  depth: number
): string {
  const w = Math.round(width * 10) / 10;
  const h = Math.round(height * 10) / 10;
  const d = Math.round(depth * 10) / 10;
  return `${w} × ${h} × ${d} cm`;
}

/**
 * Format volume for display
 * @param cm3 Volume in cubic centimeters
 * @returns Formatted string
 */
export function formatVolume(cm3: number): string {
  if (cm3 < 1) {
    return '<1 cm³';
  }
  if (cm3 >= 1000) {
    const liters = Math.round(cm3 / 100) / 10;
    return `${liters} L`;
  }
  return `${Math.round(cm3)} cm³`;
}
