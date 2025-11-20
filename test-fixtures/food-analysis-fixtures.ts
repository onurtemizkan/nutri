/**
 * Test fixtures and mock data for food analysis tests
 */

import type {
  FoodScanResult,
  FoodAnalysisResponse,
  FoodItem,
  NutritionInfo,
  ARMeasurement,
  FoodAnalysisErrorResponse,
} from '@/lib/types/food-analysis';

// Mock nutrition info
export const mockNutritionApple: NutritionInfo = {
  calories: 95,
  protein: 0.5,
  carbs: 25,
  fat: 0.3,
  fiber: 4.4,
  sugar: 19,
};

export const mockNutritionChicken: NutritionInfo = {
  calories: 165,
  protein: 31,
  carbs: 0,
  fat: 3.6,
  fiber: 0,
};

// Mock food items
export const mockFoodItemApple: FoodItem = {
  name: 'Apple',
  confidence: 0.92,
  portionSize: '1 medium (182g)',
  portionWeight: 182,
  nutrition: mockNutritionApple,
  category: 'fruit',
  alternatives: [
    { name: 'Pear', confidence: 0.65 },
    { name: 'Peach', confidence: 0.52 },
  ],
};

export const mockFoodItemChicken: FoodItem = {
  name: 'Chicken Breast',
  confidence: 0.88,
  portionSize: '100g',
  portionWeight: 100,
  nutrition: mockNutritionChicken,
  category: 'protein',
  alternatives: [
    { name: 'Turkey', confidence: 0.72 },
  ],
};

export const mockFoodItemLowConfidence: FoodItem = {
  name: 'Unknown Food',
  confidence: 0.45,
  portionSize: '100g',
  portionWeight: 100,
  nutrition: {
    calories: 100,
    protein: 5,
    carbs: 15,
    fat: 2,
  },
  category: 'unknown',
};

// Mock AR measurements
export const mockARMeasurementGood: ARMeasurement = {
  width: 10.5,
  height: 8.2,
  depth: 6.0,
  distance: 40,
  confidence: 'high',
  planeDetected: true,
  timestamp: new Date('2024-01-15T12:00:00Z'),
};

export const mockARMeasurementPoor: ARMeasurement = {
  width: 5.0,
  height: 50.0, // Unrealistic ratio
  depth: 2.0,
  distance: 80,
  confidence: 'low',
  planeDetected: false,
  timestamp: new Date('2024-01-15T12:00:00Z'),
};

// Mock analysis responses
export const mockAnalysisResponseSuccess: FoodAnalysisResponse = {
  foodItems: [mockFoodItemApple],
  measurementQuality: 'high',
  processingTime: 1234,
  suggestions: [
    'Great photo! Clear view of the food item.',
  ],
};

export const mockAnalysisResponseMultipleFoods: FoodAnalysisResponse = {
  foodItems: [mockFoodItemApple, mockFoodItemChicken],
  measurementQuality: 'medium',
  processingTime: 1856,
  suggestions: [
    'Multiple food items detected. Using the primary item.',
  ],
};

export const mockAnalysisResponseLowConfidence: FoodAnalysisResponse = {
  foodItems: [mockFoodItemLowConfidence],
  measurementQuality: 'low',
  processingTime: 982,
  suggestions: [
    'Take a clearer photo with better lighting for improved classification',
    'Include a reference object (hand, credit card) for better size estimation',
  ],
};

export const mockAnalysisResponseNoMeasurements: FoodAnalysisResponse = {
  foodItems: [mockFoodItemApple],
  measurementQuality: 'low',
  processingTime: 1100,
  suggestions: [
    'Use AR measurements for better portion size accuracy',
  ],
};

// Mock scan results
export const mockScanResult: FoodScanResult = {
  ...mockAnalysisResponseSuccess,
  imageUri: 'file:///path/to/test/image.jpg',
  timestamp: new Date('2024-01-15T12:00:00Z'),
};

// Mock error responses
export const mockNetworkError: FoodAnalysisErrorResponse = {
  error: 'network-error',
  message: 'Network error. Please check your connection and try again.',
  retryable: true,
};

export const mockTimeoutError: FoodAnalysisErrorResponse = {
  error: 'timeout',
  message: 'Food analysis request timed out. Please try again.',
  retryable: true,
};

export const mockInvalidImageError: FoodAnalysisErrorResponse = {
  error: 'invalid-image',
  message: 'Invalid image. Please try taking a clearer photo.',
  retryable: false,
};

export const mockAnalysisFailedError: FoodAnalysisErrorResponse = {
  error: 'analysis-failed',
  message: 'Food analysis failed. Please try again later.',
  retryable: true,
};

// Mock image URIs
export const mockImageURIs = {
  valid: 'file:///mock/image.jpg',
  large: 'file:///mock/large-image.jpg',
  invalid: 'file:///mock/invalid.txt',
  corrupted: 'file:///mock/corrupted.jpg',
};

// Mock FormData for API requests
export const createMockFormData = (
  imageUri: string,
  measurements?: ARMeasurement
) => {
  const formData = new FormData();

  formData.append('image', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'food.jpg',
  } as any);

  if (measurements) {
    formData.append(
      'dimensions',
      JSON.stringify({
        width: measurements.width,
        height: measurements.height,
        depth: measurements.depth,
      })
    );
  }

  return formData;
};

// Helper to create custom mock responses
export const createMockFoodItem = (
  overrides: Partial<FoodItem> = {}
): FoodItem => ({
  ...mockFoodItemApple,
  ...overrides,
});

export const createMockAnalysisResponse = (
  overrides: Partial<FoodAnalysisResponse> = {}
): FoodAnalysisResponse => ({
  ...mockAnalysisResponseSuccess,
  ...overrides,
});

export const createMockARMeasurement = (
  overrides: Partial<ARMeasurement> = {}
): ARMeasurement => ({
  ...mockARMeasurementGood,
  ...overrides,
});

// Mock ML service responses for MSW
export const mlServiceMockHandlers = {
  success: {
    path: '/api/food/analyze',
    response: mockAnalysisResponseSuccess,
    status: 200,
  },
  multipleFoods: {
    path: '/api/food/analyze',
    response: mockAnalysisResponseMultipleFoods,
    status: 200,
  },
  lowConfidence: {
    path: '/api/food/analyze',
    response: mockAnalysisResponseLowConfidence,
    status: 200,
  },
  networkError: {
    path: '/api/food/analyze',
    response: { error: 'Network error' },
    status: 500,
  },
  timeout: {
    path: '/api/food/analyze',
    delay: 31000, // Longer than timeout
    response: {},
    status: 200,
  },
  invalidImage: {
    path: '/api/food/analyze',
    response: { error: 'Invalid image format' },
    status: 400,
  },
};

// Test image data (base64 1x1 transparent PNG)
export const testImageBase64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

// Mock camera permissions
export const mockCameraPermissions = {
  granted: {
    status: 'granted',
    expires: 'never',
    canAskAgain: true,
    granted: true,
  },
  denied: {
    status: 'denied',
    expires: 'never',
    canAskAgain: true,
    granted: false,
  },
  undetermined: {
    status: 'undetermined',
    expires: 'never',
    canAskAgain: true,
    granted: false,
  },
};

// Export all fixtures
export const fixtures = {
  nutrition: {
    apple: mockNutritionApple,
    chicken: mockNutritionChicken,
  },
  foodItems: {
    apple: mockFoodItemApple,
    chicken: mockFoodItemChicken,
    lowConfidence: mockFoodItemLowConfidence,
  },
  arMeasurements: {
    good: mockARMeasurementGood,
    poor: mockARMeasurementPoor,
  },
  responses: {
    success: mockAnalysisResponseSuccess,
    multipleFoods: mockAnalysisResponseMultipleFoods,
    lowConfidence: mockAnalysisResponseLowConfidence,
    noMeasurements: mockAnalysisResponseNoMeasurements,
  },
  errors: {
    network: mockNetworkError,
    timeout: mockTimeoutError,
    invalidImage: mockInvalidImageError,
    analysisFailed: mockAnalysisFailedError,
  },
  images: mockImageURIs,
  scanResult: mockScanResult,
  permissions: mockCameraPermissions,
};
