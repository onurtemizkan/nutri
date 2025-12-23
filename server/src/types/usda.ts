/**
 * USDA FoodData Central API Types
 * Based on: https://fdc.nal.usda.gov/api-spec/fdc_api.html
 */

// ============================================================================
// USDA DATA TYPES
// ============================================================================

/**
 * USDA FoodData Central data types
 */
export type USDADataType =
  | 'Foundation'      // Unprocessed/lightly processed foods
  | 'SR Legacy'       // Standard Reference Legacy (final release 2018)
  | 'Survey (FNDDS)'  // Food and Nutrient Database for Dietary Studies
  | 'Branded'         // Commercial branded products
  | 'Experimental';   // Research data

// ============================================================================
// API RESPONSE TYPES
// ============================================================================

/**
 * USDA Nutrient data structure
 */
export interface USDANutrient {
  nutrientId: number;
  nutrientName: string;
  nutrientNumber: string;
  unitName: string;
  derivationCode?: string;
  derivationDescription?: string;
  derivationId?: number;
  value: number;
  foodNutrientSourceId?: number;
  foodNutrientSourceCode?: string;
  foodNutrientSourceDescription?: string;
  rank?: number;
  indentLevel?: number;
  foodNutrientId?: number;
  percentDailyValue?: number;
}

/**
 * USDA Food portion/measure information
 */
export interface USDAFoodPortion {
  id: number;
  measureUnit: {
    id: number;
    name: string;
    abbreviation: string;
  };
  modifier?: string;
  gramWeight: number;
  sequenceNumber?: number;
  amount?: number;
  portionDescription?: string;
}

/**
 * USDA Nutrient conversion factor
 */
export interface USDANutrientConversionFactor {
  type: string;
  value: number;
}

/**
 * USDA Food category
 */
export interface USDAFoodCategory {
  id: number;
  code?: string;
  description: string;
}

/**
 * USDA Food item from search results
 */
export interface USDASearchResultFood {
  fdcId: number;
  description: string;
  dataType: USDADataType;
  gtinUpc?: string;
  publishedDate?: string;
  brandOwner?: string;
  brandName?: string;
  ingredients?: string;
  marketCountry?: string;
  foodCategory?: string;
  modifiedDate?: string;
  dataSource?: string;
  packageWeight?: string;
  servingSizeUnit?: string;
  servingSize?: number;
  householdServingFullText?: string;
  tradeChannels?: string[];
  allHighlightFields?: string;
  score?: number;
  microbes?: unknown[];
  foodNutrients: USDASearchNutrient[];
  finalFoodInputFoods?: unknown[];
  foodMeasures?: USDAFoodMeasure[];
  foodAttributes?: unknown[];
  foodAttributeTypes?: unknown[];
  foodVersionIds?: unknown[];
}

/**
 * Nutrient in search results (simplified)
 */
export interface USDASearchNutrient {
  nutrientId: number;
  nutrientName: string;
  nutrientNumber: string;
  unitName: string;
  value: number;
  derivationCode?: string;
  derivationDescription?: string;
  derivationId?: number;
  foodNutrientId?: number;
  foodNutrientSourceId?: number;
  foodNutrientSourceCode?: string;
  foodNutrientSourceDescription?: string;
  rank?: number;
  indentLevel?: number;
  percentDailyValue?: number;
}

/**
 * Food measure in search results
 */
export interface USDAFoodMeasure {
  disseminationText?: string;
  gramWeight: number;
  id: number;
  modifier?: string;
  measureUnitAbbreviation?: string;
  measureUnitName?: string;
  measureUnitId?: number;
  rank?: number;
}

/**
 * Full USDA Food item (from get by ID)
 */
export interface USDAFoodItem {
  fdcId: number;
  description: string;
  dataType: USDADataType;
  publicationDate?: string;
  foodCode?: string;
  foodNutrients: USDANutrient[];
  foodPortions?: USDAFoodPortion[];
  scientificName?: string;
  foodCategory?: USDAFoodCategory;
  foodComponents?: unknown[];
  nutrientConversionFactors?: USDANutrientConversionFactor[];
  isHistoricalReference?: boolean;
  ndbNumber?: number;
  // Branded food specific
  brandOwner?: string;
  brandName?: string;
  gtinUpc?: string;
  ingredients?: string;
  servingSize?: number;
  servingSizeUnit?: string;
  householdServingFullText?: string;
  labelNutrients?: Record<string, { value: number }>;
  foodUpdateLog?: unknown[];
}

// ============================================================================
// SEARCH TYPES
// ============================================================================

/**
 * USDA Search API response
 */
export interface USDASearchResponse {
  totalHits: number;
  currentPage: number;
  totalPages: number;
  pageList?: number[];
  foodSearchCriteria: {
    query: string;
    generalSearchInput?: string;
    pageNumber: number;
    numberOfResultsPerPage: number;
    pageSize?: number;
    requireAllWords: boolean;
    foodTypes?: string[];
    sortBy?: string;
    sortOrder?: string;
    brandOwner?: string;
    tradeChannel?: string[];
    startDate?: string;
    endDate?: string;
  };
  foods: USDASearchResultFood[];
  aggregations?: {
    dataType?: Record<string, number>;
    nutrients?: Record<string, unknown>;
  };
}

/**
 * Search options for USDA API
 */
export interface USDASearchOptions {
  query: string;
  dataType?: USDADataType[];
  pageNumber?: number;
  pageSize?: number;
  sortBy?: 'dataType.keyword' | 'description' | 'fdcId' | 'publishedDate';
  sortOrder?: 'asc' | 'desc';
  brandOwner?: string;
  requireAllWords?: boolean;
}

/**
 * Get food options
 */
export interface USDAGetFoodOptions {
  format?: 'abridged' | 'full';
  nutrients?: number[];
}

// ============================================================================
// APP-SPECIFIC TYPES
// ============================================================================

/**
 * Transformed USDA food for app consumption
 */
export interface TransformedUSDAFood {
  fdcId: number;
  name: string;
  description: string;
  dataType: USDADataType;
  brand?: string;
  brandOwner?: string;
  category?: string;
  servingSize?: number;
  servingSizeUnit?: string;
  householdServing?: string;
  ingredients?: string;
  upc?: string;
  nutrients: TransformedNutrients;
  portions?: TransformedPortion[];
}

/**
 * Transformed nutrients matching app schema
 */
export interface TransformedNutrients {
  // Core macros (per 100g)
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber?: number;
  sugar?: number;
  addedSugar?: number;
  // Fat breakdown
  saturatedFat?: number;
  transFat?: number;
  cholesterol?: number;
  monounsaturatedFat?: number;
  polyunsaturatedFat?: number;
  // Minerals
  sodium?: number;
  potassium?: number;
  calcium?: number;
  iron?: number;
  magnesium?: number;
  zinc?: number;
  phosphorus?: number;
  // Vitamins
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
  // Amino acids
  lysine?: number;
  arginine?: number;
}

/**
 * Transformed portion information
 */
export interface TransformedPortion {
  id: number;
  name: string;
  gramWeight: number;
  amount?: number;
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/**
 * USDA API Error
 */
export class USDAApiError extends Error {
  public readonly statusCode: number;
  public readonly errorCode: string;
  public readonly retryable: boolean;

  constructor(
    message: string,
    statusCode: number,
    errorCode: string = 'USDA_API_ERROR',
    retryable: boolean = false
  ) {
    super(message);
    this.name = 'USDAApiError';
    this.statusCode = statusCode;
    this.errorCode = errorCode;
    this.retryable = retryable;
    Object.setPrototypeOf(this, USDAApiError.prototype);
  }
}

/**
 * Rate limit error
 */
export class USDAApiRateLimitError extends USDAApiError {
  public readonly retryAfter?: number;

  constructor(message: string, retryAfter?: number) {
    super(message, 429, 'USDA_RATE_LIMIT', true);
    this.name = 'USDAApiRateLimitError';
    this.retryAfter = retryAfter;
    Object.setPrototypeOf(this, USDAApiRateLimitError.prototype);
  }
}

/**
 * API timeout error
 */
export class USDAApiTimeoutError extends USDAApiError {
  constructor(message: string = 'USDA API request timed out') {
    super(message, 408, 'USDA_TIMEOUT', true);
    this.name = 'USDAApiTimeoutError';
    Object.setPrototypeOf(this, USDAApiTimeoutError.prototype);
  }
}
