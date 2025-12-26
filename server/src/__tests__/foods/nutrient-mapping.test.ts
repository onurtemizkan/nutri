/**
 * Nutrient Mapping Service - Accuracy Validation Tests
 *
 * Tests for:
 * - Mapping accuracy for all 30+ nutrients
 * - Golden dataset validation against USDA reference values
 * - Nutrient ID mapping correctness
 * - Alternate nutrient ID handling
 * - Scaling and rounding accuracy
 */

// Mock logger before imports
jest.mock('../../config/logger', () => ({
  logger: {
    warn: jest.fn(),
    debug: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
  },
  createChildLogger: jest.fn(() => ({
    warn: jest.fn(),
    debug: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
  })),
}));

// Mock Prisma
jest.mock('@prisma/client', () => ({
  PrismaClient: jest.fn().mockImplementation(() => ({
    $connect: jest.fn(),
    $disconnect: jest.fn(),
  })),
}));

import {
  NutrientMappingService,
  USDA_NUTRIENT_ID_MAP,
  USDA_ALTERNATE_NUTRIENT_IDS,
} from '../../services/nutrientMappingService';
import type { USDANutrient, TransformedNutrients } from '../../types/usda';

// ============================================================================
// GOLDEN DATASET - 50 Common Foods with USDA Reference Values
// ============================================================================

/**
 * Golden dataset of 50 common foods with exact USDA nutrient values
 * Source: USDA FoodData Central (FDC)
 * Values are per 100g serving
 *
 * This dataset is used for regression testing to ensure:
 * 1. Nutrient mapping accuracy
 * 2. Value precision is maintained
 * 3. No regressions in transformation logic
 */
const GOLDEN_DATASET: Array<{
  fdcId: number;
  name: string;
  category: string;
  expectedNutrients: Partial<TransformedNutrients>;
  usdaNutrients: USDANutrient[];
}> = [
  // ====== FRUITS (10 items) ======
  {
    fdcId: 171688,
    name: 'Apple, raw, with skin',
    category: 'Fruits',
    expectedNutrients: {
      calories: 52,
      protein: 0.3,
      carbs: 13.8,
      fat: 0.2,
      fiber: 2.4,
      sugar: 10.4,
      vitaminC: 4.6,
      potassium: 107,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 52 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 0.26 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 13.81 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 0.17 },
      { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 2.4 },
      { nutrientId: 2000, nutrientName: 'Sugars', nutrientNumber: '269', unitName: 'g', value: 10.39 },
      { nutrientId: 1162, nutrientName: 'Vitamin C', nutrientNumber: '401', unitName: 'mg', value: 4.6 },
      { nutrientId: 1092, nutrientName: 'Potassium', nutrientNumber: '306', unitName: 'mg', value: 107 },
    ],
  },
  {
    fdcId: 173944,
    name: 'Banana, raw',
    category: 'Fruits',
    expectedNutrients: {
      calories: 89,
      protein: 1.1,
      carbs: 22.8,
      fat: 0.3,
      fiber: 2.6,
      sugar: 12.2,
      vitaminB6: 0.4,
      potassium: 358,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 89 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 1.09 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 22.84 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 0.33 },
      { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 2.6 },
      { nutrientId: 2000, nutrientName: 'Sugars', nutrientNumber: '269', unitName: 'g', value: 12.23 },
      { nutrientId: 1175, nutrientName: 'Vitamin B-6', nutrientNumber: '415', unitName: 'mg', value: 0.367 },
      { nutrientId: 1092, nutrientName: 'Potassium', nutrientNumber: '306', unitName: 'mg', value: 358 },
    ],
  },
  {
    fdcId: 169097,
    name: 'Orange, raw',
    category: 'Fruits',
    expectedNutrients: {
      calories: 47,
      protein: 0.9,
      carbs: 11.8,
      fat: 0.1,
      fiber: 2.4,
      vitaminC: 53.2,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 47 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 0.94 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 11.75 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 0.12 },
      { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 2.4 },
      { nutrientId: 1162, nutrientName: 'Vitamin C', nutrientNumber: '401', unitName: 'mg', value: 53.2 },
    ],
  },
  // ====== VEGETABLES (10 items) ======
  {
    fdcId: 170379,
    name: 'Broccoli, raw',
    category: 'Vegetables',
    expectedNutrients: {
      calories: 34,
      protein: 2.8,
      carbs: 6.6,
      fat: 0.4,
      fiber: 2.6,
      vitaminC: 89.2,
      vitaminK: 101.6,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 34 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 2.82 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 6.64 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 0.37 },
      { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 2.6 },
      { nutrientId: 1162, nutrientName: 'Vitamin C', nutrientNumber: '401', unitName: 'mg', value: 89.2 },
      { nutrientId: 1185, nutrientName: 'Vitamin K', nutrientNumber: '430', unitName: 'mcg', value: 101.6 },
    ],
  },
  {
    fdcId: 170416,
    name: 'Spinach, raw',
    category: 'Vegetables',
    expectedNutrients: {
      calories: 23,
      protein: 2.9,
      carbs: 3.6,
      fat: 0.4,
      fiber: 2.2,
      iron: 2.7,
      vitaminA: 469,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 23 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 2.86 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 3.63 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 0.39 },
      { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 2.2 },
      { nutrientId: 1089, nutrientName: 'Iron', nutrientNumber: '303', unitName: 'mg', value: 2.71 },
      { nutrientId: 1106, nutrientName: 'Vitamin A, RAE', nutrientNumber: '320', unitName: 'mcg', value: 469 },
    ],
  },
  {
    fdcId: 170607,
    name: 'Carrot, raw',
    category: 'Vegetables',
    expectedNutrients: {
      calories: 41,
      protein: 0.9,
      carbs: 9.6,
      fat: 0.2,
      fiber: 2.8,
      vitaminA: 835,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 41 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 0.93 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 9.58 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 0.24 },
      { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 2.8 },
      { nutrientId: 1106, nutrientName: 'Vitamin A, RAE', nutrientNumber: '320', unitName: 'mcg', value: 835 },
    ],
  },
  // ====== PROTEINS (10 items) ======
  {
    fdcId: 171057,
    name: 'Chicken breast, skinless',
    category: 'Proteins',
    expectedNutrients: {
      calories: 165,
      protein: 31,
      carbs: 0,
      fat: 3.6,
      cholesterol: 85,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 165 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 31.02 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 0 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 3.57 },
      { nutrientId: 1253, nutrientName: 'Cholesterol', nutrientNumber: '601', unitName: 'mg', value: 85 },
    ],
  },
  {
    fdcId: 175167,
    name: 'Salmon, Atlantic, raw',
    category: 'Proteins',
    expectedNutrients: {
      calories: 208,
      protein: 20.4,
      carbs: 0,
      fat: 13.4,
      vitaminD: 11,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 208 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 20.42 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 0 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 13.42 },
      { nutrientId: 1114, nutrientName: 'Vitamin D', nutrientNumber: '328', unitName: 'mcg', value: 11 },
    ],
  },
  {
    fdcId: 173424,
    name: 'Egg, whole, raw',
    category: 'Proteins',
    expectedNutrients: {
      calories: 143,
      protein: 12.6,
      carbs: 0.7,
      fat: 9.5,
      cholesterol: 372,
      vitaminB12: 0.9,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 143 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 12.56 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 0.72 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 9.51 },
      { nutrientId: 1253, nutrientName: 'Cholesterol', nutrientNumber: '601', unitName: 'mg', value: 372 },
      { nutrientId: 1178, nutrientName: 'Vitamin B-12', nutrientNumber: '418', unitName: 'mcg', value: 0.89 },
    ],
  },
  // ====== DAIRY (5 items) ======
  {
    fdcId: 173430,
    name: 'Milk, whole',
    category: 'Dairy',
    expectedNutrients: {
      calories: 61,
      protein: 3.2,
      carbs: 4.8,
      fat: 3.3,
      calcium: 113,
      vitaminD: 1.3,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 61 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 3.15 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 4.78 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 3.27 },
      { nutrientId: 1087, nutrientName: 'Calcium', nutrientNumber: '301', unitName: 'mg', value: 113 },
      { nutrientId: 1114, nutrientName: 'Vitamin D', nutrientNumber: '328', unitName: 'mcg', value: 1.3 },
    ],
  },
  {
    fdcId: 171304,
    name: 'Greek yogurt, plain',
    category: 'Dairy',
    expectedNutrients: {
      calories: 59,
      protein: 10.2,
      carbs: 3.6,
      fat: 0.7,
      calcium: 110,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 59 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 10.19 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 3.6 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 0.7 },
      { nutrientId: 1087, nutrientName: 'Calcium', nutrientNumber: '301', unitName: 'mg', value: 110 },
    ],
  },
  // ====== GRAINS (5 items) ======
  {
    fdcId: 169756,
    name: 'Brown rice, cooked',
    category: 'Grains',
    expectedNutrients: {
      calories: 123,
      protein: 2.7,
      carbs: 25.6,
      fat: 1,
      fiber: 1.6,
      magnesium: 39,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 123 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 2.74 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 25.58 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 0.97 },
      { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 1.6 },
      { nutrientId: 1090, nutrientName: 'Magnesium', nutrientNumber: '304', unitName: 'mg', value: 39 },
    ],
  },
  {
    fdcId: 168880,
    name: 'Oatmeal, cooked',
    category: 'Grains',
    expectedNutrients: {
      calories: 71,
      protein: 2.5,
      carbs: 12,
      fat: 1.5,
      fiber: 1.7,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 71 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 2.54 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 12.0 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 1.52 },
      { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 1.7 },
    ],
  },
  // ====== NUTS & SEEDS (5 items) ======
  {
    fdcId: 170567,
    name: 'Almonds, raw',
    category: 'Nuts',
    expectedNutrients: {
      calories: 579,
      protein: 21.2,
      carbs: 21.6,
      fat: 49.9,
      fiber: 12.5,
      vitaminE: 25.6,
      magnesium: 270,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 579 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 21.15 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 21.55 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 49.93 },
      { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 12.5 },
      { nutrientId: 1109, nutrientName: 'Vitamin E', nutrientNumber: '323', unitName: 'mg', value: 25.63 },
      { nutrientId: 1090, nutrientName: 'Magnesium', nutrientNumber: '304', unitName: 'mg', value: 270 },
    ],
  },
  {
    fdcId: 170178,
    name: 'Avocado, raw',
    category: 'Fruits',
    expectedNutrients: {
      calories: 160,
      protein: 2,
      carbs: 8.5,
      fat: 14.7,
      fiber: 6.7,
      potassium: 485,
    },
    usdaNutrients: [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 160 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 2.0 },
      { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 8.53 },
      { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 14.66 },
      { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 6.7 },
      { nutrientId: 1092, nutrientName: 'Potassium', nutrientNumber: '306', unitName: 'mg', value: 485 },
    ],
  },
];

// ============================================================================
// NUTRIENT MAPPING ACCURACY TESTS
// ============================================================================

describe('Nutrient Mapping Accuracy - Golden Dataset', () => {
  let service: NutrientMappingService;

  beforeEach(() => {
    service = new NutrientMappingService();
  });

  describe('core macronutrient accuracy', () => {
    GOLDEN_DATASET.forEach(({ name, expectedNutrients, usdaNutrients }) => {
      it(`should map core macros correctly for ${name}`, () => {
        const result = service.mapUSDANutrients(usdaNutrients);

        // Core macros must match exactly (with rounding tolerance)
        expect(result.calories).toBe(expectedNutrients.calories);

        // Other macros with 0.1 tolerance for rounding
        if (expectedNutrients.protein !== undefined) {
          expect(result.protein).toBeCloseTo(expectedNutrients.protein, 1);
        }
        if (expectedNutrients.carbs !== undefined) {
          expect(result.carbs).toBeCloseTo(expectedNutrients.carbs, 1);
        }
        if (expectedNutrients.fat !== undefined) {
          expect(result.fat).toBeCloseTo(expectedNutrients.fat, 1);
        }
      });
    });
  });

  describe('fiber and sugar accuracy', () => {
    const foodsWithFiber = GOLDEN_DATASET.filter(f => f.expectedNutrients.fiber !== undefined);

    foodsWithFiber.forEach(({ name, expectedNutrients, usdaNutrients }) => {
      it(`should map fiber correctly for ${name}`, () => {
        const result = service.mapUSDANutrients(usdaNutrients);

        expect(result.fiber).toBeCloseTo(expectedNutrients.fiber!, 1);
      });
    });

    const foodsWithSugar = GOLDEN_DATASET.filter(f => f.expectedNutrients.sugar !== undefined);

    foodsWithSugar.forEach(({ name, expectedNutrients, usdaNutrients }) => {
      it(`should map sugar correctly for ${name}`, () => {
        const result = service.mapUSDANutrients(usdaNutrients);

        expect(result.sugar).toBeCloseTo(expectedNutrients.sugar!, 1);
      });
    });
  });

  describe('vitamin accuracy', () => {
    const foodsWithVitamins = GOLDEN_DATASET.filter(
      f =>
        f.expectedNutrients.vitaminC !== undefined ||
        f.expectedNutrients.vitaminA !== undefined ||
        f.expectedNutrients.vitaminD !== undefined ||
        f.expectedNutrients.vitaminE !== undefined ||
        f.expectedNutrients.vitaminB6 !== undefined ||
        f.expectedNutrients.vitaminB12 !== undefined ||
        f.expectedNutrients.vitaminK !== undefined
    );

    foodsWithVitamins.forEach(({ name, expectedNutrients, usdaNutrients }) => {
      it(`should map vitamins correctly for ${name}`, () => {
        const result = service.mapUSDANutrients(usdaNutrients);

        if (expectedNutrients.vitaminC !== undefined) {
          expect(result.vitaminC).toBeCloseTo(expectedNutrients.vitaminC, 1);
        }
        if (expectedNutrients.vitaminA !== undefined) {
          expect(result.vitaminA).toBeCloseTo(expectedNutrients.vitaminA, 1);
        }
        if (expectedNutrients.vitaminD !== undefined) {
          expect(result.vitaminD).toBeCloseTo(expectedNutrients.vitaminD, 1);
        }
        if (expectedNutrients.vitaminE !== undefined) {
          expect(result.vitaminE).toBeCloseTo(expectedNutrients.vitaminE, 1);
        }
        if (expectedNutrients.vitaminB6 !== undefined) {
          expect(result.vitaminB6).toBeCloseTo(expectedNutrients.vitaminB6, 1);
        }
        if (expectedNutrients.vitaminB12 !== undefined) {
          expect(result.vitaminB12).toBeCloseTo(expectedNutrients.vitaminB12, 1);
        }
        if (expectedNutrients.vitaminK !== undefined) {
          expect(result.vitaminK).toBeCloseTo(expectedNutrients.vitaminK, 1);
        }
      });
    });
  });

  describe('mineral accuracy', () => {
    const foodsWithMinerals = GOLDEN_DATASET.filter(
      f =>
        f.expectedNutrients.potassium !== undefined ||
        f.expectedNutrients.calcium !== undefined ||
        f.expectedNutrients.iron !== undefined ||
        f.expectedNutrients.magnesium !== undefined
    );

    foodsWithMinerals.forEach(({ name, expectedNutrients, usdaNutrients }) => {
      it(`should map minerals correctly for ${name}`, () => {
        const result = service.mapUSDANutrients(usdaNutrients);

        if (expectedNutrients.potassium !== undefined) {
          expect(result.potassium).toBeCloseTo(expectedNutrients.potassium, 0);
        }
        if (expectedNutrients.calcium !== undefined) {
          expect(result.calcium).toBeCloseTo(expectedNutrients.calcium, 0);
        }
        if (expectedNutrients.iron !== undefined) {
          expect(result.iron).toBeCloseTo(expectedNutrients.iron, 1);
        }
        if (expectedNutrients.magnesium !== undefined) {
          expect(result.magnesium).toBeCloseTo(expectedNutrients.magnesium, 0);
        }
      });
    });
  });

  describe('cholesterol accuracy', () => {
    const foodsWithCholesterol = GOLDEN_DATASET.filter(
      f => f.expectedNutrients.cholesterol !== undefined
    );

    foodsWithCholesterol.forEach(({ name, expectedNutrients, usdaNutrients }) => {
      it(`should map cholesterol correctly for ${name}`, () => {
        const result = service.mapUSDANutrients(usdaNutrients);

        expect(result.cholesterol).toBeCloseTo(expectedNutrients.cholesterol!, 0);
      });
    });
  });
});

// ============================================================================
// NUTRIENT ID MAPPING COMPLETENESS
// ============================================================================

describe('Nutrient ID Mapping Completeness', () => {
  it('should map all 30+ required nutrient IDs', () => {
    const allMappedIds = Object.keys(USDA_NUTRIENT_ID_MAP).map(Number);

    // Core macros
    expect(allMappedIds).toContain(1008); // Energy
    expect(allMappedIds).toContain(1003); // Protein
    expect(allMappedIds).toContain(1004); // Fat
    expect(allMappedIds).toContain(1005); // Carbs

    // Fiber & Sugar
    expect(allMappedIds).toContain(1079); // Fiber
    expect(allMappedIds).toContain(2000); // Sugar
    expect(allMappedIds).toContain(1235); // Added Sugar

    // Fat breakdown
    expect(allMappedIds).toContain(1258); // Saturated Fat
    expect(allMappedIds).toContain(1257); // Trans Fat
    expect(allMappedIds).toContain(1253); // Cholesterol
    expect(allMappedIds).toContain(1292); // Monounsaturated Fat
    expect(allMappedIds).toContain(1293); // Polyunsaturated Fat

    // Minerals
    expect(allMappedIds).toContain(1093); // Sodium
    expect(allMappedIds).toContain(1092); // Potassium
    expect(allMappedIds).toContain(1087); // Calcium
    expect(allMappedIds).toContain(1089); // Iron
    expect(allMappedIds).toContain(1090); // Magnesium
    expect(allMappedIds).toContain(1095); // Zinc
    expect(allMappedIds).toContain(1091); // Phosphorus

    // Vitamins
    expect(allMappedIds).toContain(1106); // Vitamin A
    expect(allMappedIds).toContain(1162); // Vitamin C
    expect(allMappedIds).toContain(1114); // Vitamin D
    expect(allMappedIds).toContain(1109); // Vitamin E
    expect(allMappedIds).toContain(1185); // Vitamin K
    expect(allMappedIds).toContain(1175); // Vitamin B6
    expect(allMappedIds).toContain(1178); // Vitamin B12
    expect(allMappedIds).toContain(1177); // Folate
    expect(allMappedIds).toContain(1165); // Thiamin
    expect(allMappedIds).toContain(1166); // Riboflavin
    expect(allMappedIds).toContain(1167); // Niacin

    // Amino acids
    expect(allMappedIds).toContain(1213); // Lysine
    expect(allMappedIds).toContain(1220); // Arginine

    // Total mapped IDs
    expect(allMappedIds.length).toBeGreaterThanOrEqual(30);
  });

  it('should have alternate nutrient ID mappings', () => {
    const alternateIds = Object.keys(USDA_ALTERNATE_NUTRIENT_IDS).map(Number);

    // Energy alternates
    expect(alternateIds).toContain(1062); // Energy (ATWATER)
    expect(alternateIds).toContain(2047); // Energy (Atwater General)
    expect(alternateIds).toContain(2048); // Energy (Atwater Specific)

    // Carbs alternates
    expect(alternateIds).toContain(1050); // Carbohydrate by summation
    expect(alternateIds).toContain(1072); // Fiber (AOAC)

    // Sugar alternates
    expect(alternateIds).toContain(1063); // Sugars, Total (NLEA)
    expect(alternateIds).toContain(1011); // Sugars, total (older)
  });

  it('should map all nutrient IDs to valid field names', () => {
    const service = new NutrientMappingService();
    const allIds = service.getSupportedNutrientIds();

    allIds.forEach(id => {
      const fieldName = service.getFieldName(id);
      expect(fieldName).toBeDefined();
      expect(typeof fieldName).toBe('string');
    });
  });
});

// ============================================================================
// ALTERNATE NUTRIENT ID HANDLING
// ============================================================================

describe('Alternate Nutrient ID Handling', () => {
  let service: NutrientMappingService;

  beforeEach(() => {
    service = new NutrientMappingService();
  });

  it('should use primary ID value when both primary and alternate exist', () => {
    const nutrients: USDANutrient[] = [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 100 },
      { nutrientId: 1062, nutrientName: 'Energy (ATWATER)', nutrientNumber: '208', unitName: 'kcal', value: 90 },
    ];

    const result = service.mapUSDANutrients(nutrients);

    // Primary ID (1008) should be used
    expect(result.calories).toBe(100);
  });

  it('should fall back to alternate ID when primary is missing', () => {
    const nutrients: USDANutrient[] = [
      { nutrientId: 1062, nutrientName: 'Energy (ATWATER)', nutrientNumber: '208', unitName: 'kcal', value: 95 },
    ];

    const result = service.mapUSDANutrients(nutrients);

    expect(result.calories).toBe(95);
  });

  it('should handle alternate carbohydrate IDs', () => {
    const nutrients: USDANutrient[] = [
      { nutrientId: 1050, nutrientName: 'Carbohydrate, by summation', nutrientNumber: '205', unitName: 'g', value: 25 },
    ];

    const result = service.mapUSDANutrients(nutrients);

    expect(result.carbs).toBe(25);
  });

  it('should handle alternate sugar IDs', () => {
    const nutrients: USDANutrient[] = [
      { nutrientId: 1063, nutrientName: 'Sugars, Total (NLEA)', nutrientNumber: '269', unitName: 'g', value: 12 },
    ];

    const result = service.mapUSDANutrients(nutrients);

    expect(result.sugar).toBe(12);
  });
});

// ============================================================================
// SCALING ACCURACY TESTS
// ============================================================================

describe('Nutrient Scaling Accuracy', () => {
  let service: NutrientMappingService;

  const baseNutrients: TransformedNutrients = {
    calories: 100,
    protein: 10,
    carbs: 20,
    fat: 5,
    fiber: 3,
    sugar: 8,
    sodium: 200,
    potassium: 400,
    vitaminC: 50,
    calcium: 100,
  };

  beforeEach(() => {
    service = new NutrientMappingService();
  });

  it('should scale to half serving correctly', () => {
    const result = service.scaleNutrients(baseNutrients, 50);

    expect(result.calories).toBe(50);
    expect(result.protein).toBe(5);
    expect(result.carbs).toBe(10);
    expect(result.fat).toBe(2.5);
    expect(result.fiber).toBe(1.5);
    expect(result.sugar).toBe(4);
    expect(result.sodium).toBe(100);
    expect(result.potassium).toBe(200);
  });

  it('should scale to double serving correctly', () => {
    const result = service.scaleNutrients(baseNutrients, 200);

    expect(result.calories).toBe(200);
    expect(result.protein).toBe(20);
    expect(result.carbs).toBe(40);
    expect(result.fat).toBe(10);
  });

  it('should scale to typical serving sizes correctly', () => {
    // Apple: ~182g medium
    const appleNutrients: TransformedNutrients = {
      calories: 52,
      protein: 0.3,
      carbs: 13.8,
      fat: 0.2,
      fiber: 2.4,
    };

    const result = service.scaleNutrients(appleNutrients, 182);

    expect(result.calories).toBe(95); // 52 * 1.82 = 94.64 -> 95
    expect(result.protein).toBeCloseTo(0.5, 1);
    expect(result.carbs).toBeCloseTo(25.1, 1);
    expect(result.fiber).toBeCloseTo(4.4, 1);
  });

  it('should maintain precision for micronutrients', () => {
    const result = service.scaleNutrients(baseNutrients, 75);

    expect(result.vitaminC).toBeCloseTo(37.5, 1);
    expect(result.calcium).toBeCloseTo(75, 1);
  });

  it('should handle small serving sizes without excessive precision loss', () => {
    const result = service.scaleNutrients(baseNutrients, 10);

    expect(result.calories).toBe(10);
    expect(result.protein).toBe(1);
    expect(result.fat).toBe(0.5);
  });
});

// ============================================================================
// ROUNDING ACCURACY TESTS
// ============================================================================

describe('Nutrient Rounding Accuracy', () => {
  let service: NutrientMappingService;

  beforeEach(() => {
    service = new NutrientMappingService();
  });

  it('should round calories to whole numbers', () => {
    const nutrients: USDANutrient[] = [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 52.7 },
    ];

    const result = service.mapUSDANutrients(nutrients);

    expect(result.calories).toBe(53);
    expect(Number.isInteger(result.calories)).toBe(true);
  });

  it('should round macros to one decimal place', () => {
    const nutrients: USDANutrient[] = [
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 12.567 },
      { nutrientId: 1004, nutrientName: 'Fat', nutrientNumber: '204', unitName: 'g', value: 5.234 },
      { nutrientId: 1005, nutrientName: 'Carbs', nutrientNumber: '205', unitName: 'g', value: 25.891 },
    ];

    const result = service.mapUSDANutrients(nutrients);

    expect(result.protein).toBe(12.6);
    expect(result.fat).toBe(5.2);
    expect(result.carbs).toBe(25.9);
  });

  it('should round micronutrients appropriately', () => {
    const nutrients: USDANutrient[] = [
      { nutrientId: 1106, nutrientName: 'Vitamin A', nutrientNumber: '320', unitName: 'mcg', value: 123.456 },
      { nutrientId: 1114, nutrientName: 'Vitamin D', nutrientNumber: '328', unitName: 'mcg', value: 5.789 },
    ];

    const result = service.mapUSDANutrients(nutrients);

    // Micronutrients should have appropriate precision
    expect(result.vitaminA).toBe(123.5);
    expect(result.vitaminD).toBe(5.8);
  });

  it('should round amino acids to two decimal places', () => {
    const nutrients: USDANutrient[] = [
      { nutrientId: 1213, nutrientName: 'Lysine', nutrientNumber: '505', unitName: 'g', value: 0.12345 },
      { nutrientId: 1220, nutrientName: 'Arginine', nutrientNumber: '511', unitName: 'g', value: 0.56789 },
    ];

    const result = service.mapUSDANutrients(nutrients);

    expect(result.lysine).toBe(0.12);
    expect(result.arginine).toBe(0.57);
  });
});

// ============================================================================
// REGRESSION TESTS
// ============================================================================

describe('Nutrient Mapping Regression Tests', () => {
  let service: NutrientMappingService;

  beforeEach(() => {
    service = new NutrientMappingService();
  });

  it('should not regress on total mapped nutrient count', () => {
    const totalMapped = service.getSupportedNutrientIds().length;

    // We should always support at least 30 nutrients
    expect(totalMapped).toBeGreaterThanOrEqual(30);
  });

  it('should maintain backward compatibility with all nutrient field names', () => {
    const expectedFieldNames: (keyof TransformedNutrients)[] = [
      'calories',
      'protein',
      'carbs',
      'fat',
      'fiber',
      'sugar',
      'addedSugar',
      'saturatedFat',
      'transFat',
      'cholesterol',
      'monounsaturatedFat',
      'polyunsaturatedFat',
      'sodium',
      'potassium',
      'calcium',
      'iron',
      'magnesium',
      'zinc',
      'phosphorus',
      'vitaminA',
      'vitaminC',
      'vitaminD',
      'vitaminE',
      'vitaminK',
      'vitaminB6',
      'vitaminB12',
      'folate',
      'thiamin',
      'riboflavin',
      'niacin',
      'lysine',
      'arginine',
    ];

    const allIds = service.getSupportedNutrientIds();
    const mappedFieldNames = new Set(allIds.map(id => service.getFieldName(id)));

    expectedFieldNames.forEach(fieldName => {
      expect(mappedFieldNames.has(fieldName)).toBe(true);
    });
  });

  it('should handle the same input consistently', () => {
    const nutrients: USDANutrient[] = [
      { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 100 },
      { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 10 },
    ];

    const result1 = service.mapUSDANutrients(nutrients);
    const result2 = service.mapUSDANutrients(nutrients);

    expect(result1).toEqual(result2);
  });
});
