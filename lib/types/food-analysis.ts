// Food Analysis Types for AR-powered food scanning

// Cooking method enum - must match ML service CookingMethod
export type CookingMethod =
  | 'raw'
  | 'cooked'
  | 'boiled'
  | 'steamed'
  | 'grilled'
  | 'fried'
  | 'baked'
  | 'roasted'
  | 'sauteed'
  | 'poached';

export const COOKING_METHODS: CookingMethod[] = [
  'raw',
  'cooked',
  'boiled',
  'steamed',
  'grilled',
  'fried',
  'baked',
  'roasted',
  'sauteed',
  'poached',
];

export interface ARMeasurement {
  width: number; // cm
  height: number; // cm
  depth: number; // cm
  distance: number; // cm from camera
  confidence: 'high' | 'medium' | 'low';
  planeDetected: boolean;
  timestamp: Date;
}

export interface FoodAnalysisRequest {
  imageUri: string;
  measurements?: ARMeasurement;
  cookingMethod?: CookingMethod;
  userId?: string;
}

export interface NutritionInfo {
  calories: number;
  protein: number; // grams
  carbs: number; // grams
  fat: number; // grams
  fiber?: number; // grams
  sugar?: number; // grams
  sodium?: number; // mg
  saturatedFat?: number; // grams
  lysine?: number; // mg - essential amino acid
  arginine?: number; // mg - conditionally essential amino acid
}

export interface FoodItem {
  name: string;
  confidence: number; // 0-1
  portionSize: string; // e.g., "1 cup", "150g", "1 medium apple"
  portionWeight: number; // grams
  nutrition: NutritionInfo;
  category?: string; // e.g., "fruit", "vegetable", "protein", "grain"
  alternatives?: {
    name: string;
    confidence: number;
  }[];
}

export interface FoodAnalysisResponse {
  foodItems: FoodItem[];
  measurementQuality: 'high' | 'medium' | 'low';
  processingTime: number; // milliseconds
  suggestions?: string[];
  error?: string;
}

export interface FoodScanResult extends FoodAnalysisResponse {
  imageUri: string;
  timestamp: Date;
}

// Camera state
export type CameraMode = 'photo' | 'ar-measure';

export interface CameraState {
  hasPermission: boolean | null;
  mode: CameraMode;
  isCapturing: boolean;
  flashEnabled: boolean;
}

// AR measurement state
export interface ARState {
  isInitialized: boolean;
  planeDetected: boolean;
  measurement: ARMeasurement | null;
  isRecording: boolean;
  error?: string;
}

// For ML service API
export interface MLServiceConfig {
  baseUrl: string;
  timeout: number;
  maxRetries: number;
}

// Nutrition database types
export interface NutritionDBEntry {
  foodName: string;
  fdcId?: string; // USDA FoodData Central ID
  category: string;
  servingSize: string;
  servingWeight: number; // grams
  nutrition: NutritionInfo;
  commonPortions: {
    description: string;
    weight: number; // grams
  }[];
}

// Food classification types
export interface FoodClassificationResult {
  className: string;
  confidence: number;
  boundingBox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

// Portion size estimation
export interface PortionSizeEstimate {
  weight: number; // grams
  volume?: number; // ml
  confidence: number;
  method: 'ar-measurement' | 'reference-object' | 'visual-estimation';
  referenceUsed?: string; // e.g., "credit card", "hand", "coin"
}

// Error types
export type FoodAnalysisError =
  | 'network-error'
  | 'permission-denied'
  | 'analysis-failed'
  | 'invalid-image'
  | 'timeout'
  | 'unknown';

export interface FoodAnalysisErrorResponse {
  error: FoodAnalysisError;
  message: string;
  retryable: boolean;
}
