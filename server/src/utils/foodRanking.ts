/**
 * Food Ranking Utility
 *
 * Provides ranking algorithms for USDA food search results
 * Prioritizes data quality and relevance based on:
 * - Data type (Foundation > SR Legacy > Survey > Branded)
 * - Name match quality
 * - Data completeness
 * - Category hints from classification
 */

import type { TransformedUSDAFood, USDADataType } from '../types/usda';

// ============================================================================
// TYPES
// ============================================================================

export interface RankingHints {
  /** Classification category from ML model */
  category?: string;
  /** Subcategory hints from classification */
  subcategoryHints?: string[];
  /** Whether this is a whole food query */
  isWholeFoodQuery?: boolean;
  /** Whether this appears to be a branded product query */
  isBrandedQuery?: boolean;
  /** Original search query */
  query?: string;
}

export interface RankedFood extends TransformedUSDAFood {
  /** Computed ranking score (0-100) */
  rankScore: number;
  /** Breakdown of score components */
  scoreBreakdown: {
    dataTypeScore: number;
    nameMatchScore: number;
    completenessScore: number;
    categoryMatchScore: number;
  };
  /** Data quality indicators */
  dataQuality: {
    isComplete: boolean;
    missingFields: string[];
    hasServingSize: boolean;
  };
}

// ============================================================================
// CONSTANTS
// ============================================================================

/** Data type ranking weights (higher = better quality) */
const DATA_TYPE_WEIGHTS: Record<USDADataType, number> = {
  Foundation: 100,
  'SR Legacy': 90,
  'Survey (FNDDS)': 70,
  Branded: 50,
  Experimental: 30,
};

/** Score component weights */
const SCORE_WEIGHTS = {
  dataType: 0.35,
  nameMatch: 0.30,
  completeness: 0.20,
  categoryMatch: 0.15,
};

// Core nutrients: calories, protein, carbs, fat
// Extended nutrients: fiber, sugar, sodium, saturatedFat
// (Checked directly in completeness scoring functions)

// ============================================================================
// MAIN RANKING FUNCTION
// ============================================================================

/**
 * Rank search results based on data quality and relevance
 *
 * @param results - Array of transformed USDA foods
 * @param hints - Optional classification hints for improved ranking
 * @returns Sorted array of ranked foods (highest score first)
 */
export function rankSearchResults(
  results: TransformedUSDAFood[],
  hints?: RankingHints
): RankedFood[] {
  if (!results || results.length === 0) {
    return [];
  }

  const rankedResults = results.map((food) => {
    const dataTypeScore = calculateDataTypeScore(food, hints);
    const nameMatchScore = calculateNameMatchScore(food, hints);
    const completenessScore = calculateCompletenessScore(food);
    const categoryMatchScore = calculateCategoryMatchScore(food, hints);

    // Calculate weighted total score
    const rankScore = Math.round(
      dataTypeScore * SCORE_WEIGHTS.dataType +
      nameMatchScore * SCORE_WEIGHTS.nameMatch +
      completenessScore * SCORE_WEIGHTS.completeness +
      categoryMatchScore * SCORE_WEIGHTS.categoryMatch
    );

    // Determine data quality
    const dataQuality = assessDataQuality(food);

    return {
      ...food,
      rankScore,
      scoreBreakdown: {
        dataTypeScore,
        nameMatchScore,
        completenessScore,
        categoryMatchScore,
      },
      dataQuality,
    } as RankedFood;
  });

  // Sort by rank score (descending)
  return rankedResults.sort((a, b) => b.rankScore - a.rankScore);
}

// ============================================================================
// SCORE CALCULATION FUNCTIONS
// ============================================================================

/**
 * Calculate score based on data type
 * Foundation foods are highest quality, Branded lowest
 */
function calculateDataTypeScore(
  food: TransformedUSDAFood,
  hints?: RankingHints
): number {
  const baseScore = DATA_TYPE_WEIGHTS[food.dataType] || 40;

  // Adjust based on query type hints
  if (hints?.isWholeFoodQuery) {
    // Boost Foundation and SR Legacy for whole food queries
    if (food.dataType === 'Foundation' || food.dataType === 'SR Legacy') {
      return Math.min(100, baseScore + 10);
    }
    // Penalize branded for whole food queries
    if (food.dataType === 'Branded') {
      return Math.max(0, baseScore - 20);
    }
  }

  if (hints?.isBrandedQuery) {
    // Boost Branded for branded product queries
    if (food.dataType === 'Branded') {
      return Math.min(100, baseScore + 30);
    }
  }

  return baseScore;
}

/**
 * Calculate score based on how well the food name matches the query
 */
function calculateNameMatchScore(
  food: TransformedUSDAFood,
  hints?: RankingHints
): number {
  if (!hints?.query) {
    return 50; // Default neutral score
  }

  const query = hints.query.toLowerCase();
  const description = food.description.toLowerCase();

  // Exact match
  if (description === query) {
    return 100;
  }

  // Starts with query
  if (description.startsWith(query)) {
    return 90;
  }

  // Query is a complete word in description
  const words = description.split(/[\s,]+/);
  const queryWords = query.split(/\s+/);

  // All query words found in description
  const allWordsFound = queryWords.every((qw) =>
    words.some((w) => w.startsWith(qw) || w === qw)
  );

  if (allWordsFound) {
    return 80;
  }

  // Some query words found
  const foundWords = queryWords.filter((qw) =>
    words.some((w) => w.includes(qw))
  );
  const matchRatio = foundWords.length / queryWords.length;

  return Math.round(50 + matchRatio * 30);
}

/**
 * Calculate score based on data completeness
 * Uses TransformedUSDAFood.nutrients object
 */
function calculateCompletenessScore(food: TransformedUSDAFood): number {
  if (!food.nutrients) {
    return 0;
  }

  const { nutrients } = food;

  // Check core nutrients (required): calories, protein, carbs, fat
  let coreScore = 0;
  if (nutrients.calories !== undefined && nutrients.calories >= 0) coreScore += 25;
  if (nutrients.protein !== undefined && nutrients.protein >= 0) coreScore += 25;
  if (nutrients.carbs !== undefined && nutrients.carbs >= 0) coreScore += 25;
  if (nutrients.fat !== undefined && nutrients.fat >= 0) coreScore += 25;

  // Cap at 100 and reduce if core nutrients missing
  if (coreScore < 100) {
    return coreScore * 0.7; // Penalize incomplete core data
  }

  // Check extended nutrients (bonus): fiber, sugar, sodium, saturatedFat
  let extendedCount = 0;
  if (nutrients.fiber !== undefined) extendedCount++;
  if (nutrients.sugar !== undefined) extendedCount++;
  if (nutrients.sodium !== undefined) extendedCount++;
  if (nutrients.saturatedFat !== undefined) extendedCount++;

  // Bonus for extended nutrients (up to 20 extra points)
  const extendedBonus = Math.min(20, extendedCount * 5);

  // Check serving size (bonus)
  const servingSizeBonus = food.servingSize ? 10 : 0;

  return Math.min(100, coreScore + extendedBonus + servingSizeBonus);
}

/**
 * Calculate score based on category match from classification
 */
function calculateCategoryMatchScore(
  food: TransformedUSDAFood,
  hints?: RankingHints
): number {
  if (!hints?.category && !hints?.subcategoryHints) {
    return 50; // Default neutral score
  }

  const description = food.description.toLowerCase();
  let score = 50;

  // Check main category match
  if (hints.category) {
    const category = hints.category.toLowerCase();
    if (description.includes(category)) {
      score += 30;
    } else if (categoryMatchesFoodGroup(category, food.category)) {
      score += 20;
    }
  }

  // Check subcategory hints
  if (hints.subcategoryHints && hints.subcategoryHints.length > 0) {
    const matchedHints = hints.subcategoryHints.filter((hint) =>
      description.includes(hint.toLowerCase())
    );
    score += matchedHints.length * 10;
  }

  return Math.min(100, score);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Check if a classification category matches a food's category
 */
function categoryMatchesFoodGroup(
  classificationCategory: string,
  foodCategory?: string
): boolean {
  if (!foodCategory) return false;

  const categoryLower = classificationCategory.toLowerCase();
  const foodCategoryLower = foodCategory.toLowerCase();

  // Common mappings between classification categories and USDA food categories
  const categoryMappings: Record<string, string[]> = {
    fruit: ['fruits', 'fruit juice', 'citrus'],
    vegetable: ['vegetables', 'legumes', 'greens'],
    meat: ['beef', 'pork', 'poultry', 'lamb', 'game'],
    seafood: ['fish', 'shellfish', 'finfish'],
    dairy: ['milk', 'cheese', 'yogurt', 'dairy'],
    grain: ['cereals', 'grains', 'bread', 'pasta', 'rice'],
    beverage: ['beverages', 'drinks', 'juice', 'coffee', 'tea'],
    snack: ['snacks', 'chips', 'crackers'],
    dessert: ['sweets', 'candy', 'desserts', 'baked products'],
  };

  const mappedCategories = categoryMappings[categoryLower] || [categoryLower];
  return mappedCategories.some((mapped) =>
    foodCategoryLower.includes(mapped)
  );
}

/**
 * Assess the overall data quality of a food item
 */
function assessDataQuality(food: TransformedUSDAFood): RankedFood['dataQuality'] {
  const missingFields: string[] = [];

  if (!food.nutrients) {
    return {
      isComplete: false,
      missingFields: ['calories', 'protein', 'carbs', 'fat'],
      hasServingSize: !!food.servingSize,
    };
  }

  const { nutrients } = food;

  // Check core nutrients
  if (nutrients.calories === undefined) missingFields.push('calories');
  if (nutrients.protein === undefined) missingFields.push('protein');
  if (nutrients.carbs === undefined) missingFields.push('carbs');
  if (nutrients.fat === undefined) missingFields.push('fat');

  return {
    isComplete: missingFields.length === 0,
    missingFields,
    hasServingSize: !!food.servingSize,
  };
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Determine if a query is likely for whole foods
 */
export function isWholeFoodQuery(query: string): boolean {
  const wholeFoodIndicators = [
    'raw',
    'fresh',
    'whole',
    'organic',
    'plain',
    'unprocessed',
  ];

  const queryLower = query.toLowerCase();

  // Check for whole food indicators
  if (wholeFoodIndicators.some((ind) => queryLower.includes(ind))) {
    return true;
  }

  // Check for simple single-word food queries (likely whole foods)
  const words = queryLower.split(/\s+/);
  if (words.length === 1) {
    const commonWholeFoods = [
      'apple',
      'banana',
      'orange',
      'chicken',
      'beef',
      'salmon',
      'rice',
      'broccoli',
      'spinach',
      'egg',
      'milk',
      'potato',
      'carrot',
      'tomato',
      'avocado',
    ];
    if (commonWholeFoods.includes(words[0])) {
      return true;
    }
  }

  return false;
}

/**
 * Determine if a query is likely for branded products
 */
export function isBrandedQuery(query: string): boolean {
  const queryLower = query.toLowerCase();

  // Check for brand indicators
  const brandIndicators = [
    'brand',
    'store',
    'restaurant',
    "'s", // possessive (McDonald's, Trader Joe's)
    'inc',
    'corp',
    'co.',
    'llc',
  ];

  if (brandIndicators.some((ind) => queryLower.includes(ind))) {
    return true;
  }

  // Check for common brand patterns (capitalized words, ®, ™)
  if (/[®™]/.test(query) || /\b[A-Z]{2,}\b/.test(query)) {
    return true;
  }

  // Check for product-like queries (with numbers, sizes)
  if (/\d+\s*(oz|g|ml|l)\b/i.test(query)) {
    return true;
  }

  return false;
}

/**
 * Get display label for data type (for UI badges)
 */
export function getDataTypeLabel(dataType: USDADataType): string {
  switch (dataType) {
    case 'Foundation':
    case 'SR Legacy':
      return 'USDA';
    case 'Branded':
      return 'Brand';
    case 'Survey (FNDDS)':
      return 'Recipe';
    case 'Experimental':
      return 'Exp';
    default:
      return 'Food';
  }
}

/**
 * Get styling info for data type badge
 */
export function getDataTypeBadgeStyle(dataType: USDADataType): {
  label: string;
  color: string;
  priority: number;
} {
  switch (dataType) {
    case 'Foundation':
      return { label: 'USDA Foundation', color: '#22c55e', priority: 1 };
    case 'SR Legacy':
      return { label: 'USDA Standard', color: '#22c55e', priority: 2 };
    case 'Survey (FNDDS)':
      return { label: 'Recipe/Meal', color: '#f59e0b', priority: 3 };
    case 'Branded':
      return { label: 'Branded', color: '#6366f1', priority: 4 };
    case 'Experimental':
      return { label: 'Experimental', color: '#6b7280', priority: 5 };
    default:
      return { label: 'Food', color: '#6b7280', priority: 6 };
  }
}

export default {
  rankSearchResults,
  isWholeFoodQuery,
  isBrandedQuery,
  getDataTypeLabel,
  getDataTypeBadgeStyle,
};
