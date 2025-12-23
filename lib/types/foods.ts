/**
 * Food Search Types
 *
 * Types for USDA FoodData Central integration
 */

// USDA Data Types
export type USDADataType =
  | 'Foundation'
  | 'SR Legacy'
  | 'Survey (FNDDS)'
  | 'Branded'
  | 'Experimental';

// Transformed USDA Food (from backend)
export interface USDAFood {
  fdcId: number;
  description: string;
  brandOwner?: string;
  brandName?: string;
  dataType: USDADataType;
  publishedDate?: string;
  servingSize?: number;
  servingSizeUnit?: string;
  ingredients?: string;
  // Core nutrients
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber?: number;
  sugar?: number;
  // Extended nutrients (optional)
  saturatedFat?: number;
  transFat?: number;
  cholesterol?: number;
  sodium?: number;
  potassium?: number;
  calcium?: number;
  iron?: number;
  vitaminA?: number;
  vitaminC?: number;
  vitaminD?: number;
}

// Nutrient info (same as meal nutrients)
export interface FoodNutrients {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
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

// Search pagination
export interface SearchPagination {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
  hasNextPage: boolean;
  hasPrevPage: boolean;
}

// Food search result
export interface FoodSearchResult {
  foods: USDAFood[];
  pagination: SearchPagination;
}

// Search options
export interface FoodSearchOptions {
  query: string;
  page?: number;
  limit?: number;
  dataType?: USDADataType[];
  sortBy?: 'dataType.keyword' | 'description' | 'fdcId' | 'publishedDate';
  sortOrder?: 'asc' | 'desc';
  brandOwner?: string;
}

// Classification result (from ML service)
export interface FoodClassification {
  category: string;
  confidence: number;
  usda_datatypes: string[];
  search_hints: {
    subcategory_hints: string[];
    suggested_query_enhancement: string;
  };
  alternatives: Array<{
    category: string;
    confidence: number;
  }>;
}

// Classify and search response
export interface ClassifyAndSearchResult {
  classification: FoodClassification;
  searchResults: FoodSearchResult;
  portionEstimate?: {
    estimated_grams: number;
    dimensions?: {
      width: number;
      height: number;
      depth?: number;
    };
    quality: 'low' | 'medium' | 'high';
  };
  query: string;
}

// Food feedback submission
export interface FoodFeedbackInput {
  imageHash: string;
  classificationId?: string;
  originalPrediction: string;
  originalConfidence: number;
  originalCategory?: string;
  selectedFdcId: number;
  selectedFoodName: string;
  wasCorrect: boolean;
  classificationHints?: Record<string, unknown>;
  userDescription?: string;
}

// Feedback statistics
export interface FoodFeedbackStats {
  totalFeedback: number;
  pendingFeedback: number;
  approvedFeedback: number;
  rejectedFeedback: number;
  topMisclassifications: Array<{
    original: string;
    corrected: string;
    count: number;
  }>;
  problemFoods: Array<{
    food: string;
    correctionCount: number;
    avgConfidence: number;
  }>;
  patternsNeedingReview: number;
}

// Filter tab options for UI
export type FoodFilterTab = 'all' | 'whole' | 'branded' | 'meals';

// Data type filter mapping
export const DATA_TYPE_FILTERS: Record<FoodFilterTab, USDADataType[] | undefined> = {
  all: undefined,
  whole: ['Foundation', 'SR Legacy'],
  branded: ['Branded'],
  meals: ['Survey (FNDDS)'],
};
