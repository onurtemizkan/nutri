/**
 * Nutrient Mapping Service Tests
 */

// Mock logger before imports
jest.mock('../../config/logger', () => ({
  logger: {
    warn: jest.fn(),
    debug: jest.fn(),
    error: jest.fn(),
  },
  createChildLogger: jest.fn(() => ({
    warn: jest.fn(),
    debug: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
  })),
}));

// Mock Prisma to avoid database connection
jest.mock('@prisma/client', () => ({
  PrismaClient: jest.fn().mockImplementation(() => ({
    $connect: jest.fn(),
    $disconnect: jest.fn(),
  })),
}));

import {
  NutrientMappingService,
  USDA_NUTRIENT_ID_MAP,
  nutrientMappingService,
} from '../../services/nutrientMappingService';
import {
  USDANutrient,
  USDAFoodItem,
  USDASearchResultFood,
  TransformedNutrients,
} from '../../types/usda';

describe('NutrientMappingService', () => {
  let service: NutrientMappingService;

  // Sample USDA nutrients data
  const sampleNutrients: USDANutrient[] = [
    { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 52 },
    { nutrientId: 1003, nutrientName: 'Protein', nutrientNumber: '203', unitName: 'g', value: 0.26 },
    { nutrientId: 1005, nutrientName: 'Carbohydrate', nutrientNumber: '205', unitName: 'g', value: 13.81 },
    { nutrientId: 1004, nutrientName: 'Total lipid (fat)', nutrientNumber: '204', unitName: 'g', value: 0.17 },
    { nutrientId: 1079, nutrientName: 'Fiber', nutrientNumber: '291', unitName: 'g', value: 2.4 },
    { nutrientId: 2000, nutrientName: 'Sugars', nutrientNumber: '269', unitName: 'g', value: 10.39 },
    { nutrientId: 1093, nutrientName: 'Sodium', nutrientNumber: '307', unitName: 'mg', value: 1 },
    { nutrientId: 1092, nutrientName: 'Potassium', nutrientNumber: '306', unitName: 'mg', value: 107 },
    { nutrientId: 1087, nutrientName: 'Calcium', nutrientNumber: '301', unitName: 'mg', value: 6 },
    { nutrientId: 1089, nutrientName: 'Iron', nutrientNumber: '303', unitName: 'mg', value: 0.12 },
    { nutrientId: 1162, nutrientName: 'Vitamin C', nutrientNumber: '401', unitName: 'mg', value: 4.6 },
    { nutrientId: 1106, nutrientName: 'Vitamin A, RAE', nutrientNumber: '320', unitName: 'mcg', value: 3 },
  ];

  const sampleUSDAFood: USDAFoodItem = {
    fdcId: 171688,
    description: 'Apples, raw, with skin',
    dataType: 'Foundation',
    publicationDate: '2019-04-01',
    foodNutrients: sampleNutrients,
    foodCategory: { id: 1, description: 'Fruits and Fruit Juices' },
    foodPortions: [
      {
        id: 1,
        measureUnit: { id: 1, name: 'cup, sliced', abbreviation: 'cup' },
        gramWeight: 110,
        sequenceNumber: 1,
        amount: 1,
      },
      {
        id: 2,
        measureUnit: { id: 2, name: 'medium', abbreviation: 'med' },
        gramWeight: 182,
        sequenceNumber: 2,
        amount: 1,
      },
    ],
  };

  const sampleSearchResultFood: USDASearchResultFood = {
    fdcId: 171688,
    description: 'Apples, raw, with skin',
    dataType: 'Foundation',
    publishedDate: '2019-04-01',
    foodNutrients: sampleNutrients.map(n => ({
      nutrientId: n.nutrientId,
      nutrientName: n.nutrientName,
      nutrientNumber: n.nutrientNumber,
      unitName: n.unitName,
      value: n.value,
    })),
    foodMeasures: [
      { id: 1, gramWeight: 110, modifier: 'cup, sliced', disseminationText: '1 cup sliced' },
      { id: 2, gramWeight: 182, modifier: 'medium', disseminationText: '1 medium apple' },
    ],
  };

  beforeEach(() => {
    service = new NutrientMappingService();
  });

  describe('mapUSDANutrients', () => {
    it('should map core macronutrients correctly', () => {
      const result = service.mapUSDANutrients(sampleNutrients);

      expect(result.calories).toBe(52);
      expect(result.protein).toBe(0.3); // Rounded to 1 decimal
      expect(result.carbs).toBe(13.8);
      expect(result.fat).toBe(0.2);
    });

    it('should map fiber and sugar correctly', () => {
      const result = service.mapUSDANutrients(sampleNutrients);

      expect(result.fiber).toBe(2.4);
      expect(result.sugar).toBe(10.4);
    });

    it('should map minerals correctly', () => {
      const result = service.mapUSDANutrients(sampleNutrients);

      expect(result.sodium).toBe(1);
      expect(result.potassium).toBe(107);
      expect(result.calcium).toBe(6);
      expect(result.iron).toBe(0.1);
    });

    it('should map vitamins correctly', () => {
      const result = service.mapUSDANutrients(sampleNutrients);

      expect(result.vitaminC).toBe(4.6);
      expect(result.vitaminA).toBe(3);
    });

    it('should return default nutrients for empty array', () => {
      const result = service.mapUSDANutrients([]);

      expect(result.calories).toBe(0);
      expect(result.protein).toBe(0);
      expect(result.carbs).toBe(0);
      expect(result.fat).toBe(0);
    });

    it('should handle null/undefined nutrients', () => {
      const result = service.mapUSDANutrients(null as unknown as USDANutrient[]);

      expect(result).toEqual({
        calories: 0,
        protein: 0,
        carbs: 0,
        fat: 0,
      });
    });

    it('should handle nutrients with missing values', () => {
      const incompleteNutrients = [
        { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 100 },
        // Missing protein, carbs, fat
      ];

      const result = service.mapUSDANutrients(incompleteNutrients as USDANutrient[]);

      expect(result.calories).toBe(100);
      expect(result.protein).toBe(0);
      expect(result.carbs).toBe(0);
      expect(result.fat).toBe(0);
    });

    it('should handle alternate nutrient IDs', () => {
      const alternateNutrients = [
        { nutrientId: 1062, nutrientName: 'Energy (ATWATER)', nutrientNumber: '208', unitName: 'kcal', value: 50 },
        { nutrientId: 1050, nutrientName: 'Carbohydrate, by summation', nutrientNumber: '205', unitName: 'g', value: 15 },
      ];

      const result = service.mapUSDANutrients(alternateNutrients as USDANutrient[]);

      expect(result.calories).toBe(50);
      expect(result.carbs).toBe(15);
    });

    it('should not overwrite primary ID with alternate ID', () => {
      const nutrients = [
        { nutrientId: 1008, nutrientName: 'Energy', nutrientNumber: '208', unitName: 'kcal', value: 100 },
        { nutrientId: 1062, nutrientName: 'Energy (ATWATER)', nutrientNumber: '208', unitName: 'kcal', value: 50 },
      ];

      const result = service.mapUSDANutrients(nutrients as USDANutrient[]);

      // Primary ID (1008) should be used, not alternate (1062)
      expect(result.calories).toBe(100);
    });
  });

  describe('transformUSDAFood', () => {
    it('should transform USDA food item correctly', () => {
      const result = service.transformUSDAFood(sampleUSDAFood);

      expect(result.fdcId).toBe(171688);
      expect(result.name).toBe('Apples');
      expect(result.description).toBe('Apples, raw, with skin');
      expect(result.dataType).toBe('Foundation');
      expect(result.category).toBe('Fruits and Fruit Juices');
    });

    it('should include nutrients in transformed food', () => {
      const result = service.transformUSDAFood(sampleUSDAFood);

      expect(result.nutrients.calories).toBe(52);
      expect(result.nutrients.protein).toBe(0.3);
      expect(result.nutrients.carbs).toBe(13.8);
    });

    it('should include portions in transformed food', () => {
      const result = service.transformUSDAFood(sampleUSDAFood);

      expect(result.portions).toHaveLength(2);
      expect(result.portions![0].name).toBe('cup, sliced');
      expect(result.portions![0].gramWeight).toBe(110);
      expect(result.portions![1].name).toBe('medium');
      expect(result.portions![1].gramWeight).toBe(182);
    });

    it('should handle branded food with brand info', () => {
      const brandedFood: USDAFoodItem = {
        ...sampleUSDAFood,
        dataType: 'Branded',
        brandName: 'Granny Smith',
        brandOwner: 'Apple Co',
        gtinUpc: '123456789012',
        ingredients: 'Fresh apples',
      };

      const result = service.transformUSDAFood(brandedFood);

      expect(result.brand).toBe('Granny Smith');
      expect(result.brandOwner).toBe('Apple Co');
      expect(result.upc).toBe('123456789012');
      expect(result.ingredients).toBe('Fresh apples');
    });

    it('should handle food without portions', () => {
      const foodWithoutPortions: USDAFoodItem = {
        ...sampleUSDAFood,
        foodPortions: undefined,
      };

      const result = service.transformUSDAFood(foodWithoutPortions);

      expect(result.portions).toBeUndefined();
    });
  });

  describe('transformSearchResultFood', () => {
    it('should transform search result food correctly', () => {
      const result = service.transformSearchResultFood(sampleSearchResultFood);

      expect(result.fdcId).toBe(171688);
      expect(result.name).toBe('Apples');
      expect(result.description).toBe('Apples, raw, with skin');
      expect(result.nutrients.calories).toBe(52);
    });

    it('should transform food measures to portions', () => {
      const result = service.transformSearchResultFood(sampleSearchResultFood);

      expect(result.portions).toHaveLength(2);
      expect(result.portions![0].name).toBe('1 cup sliced');
      expect(result.portions![0].gramWeight).toBe(110);
    });
  });

  describe('scaleNutrients', () => {
    const baseNutrients: TransformedNutrients = {
      calories: 100,
      protein: 10,
      carbs: 20,
      fat: 5,
      fiber: 2,
      sodium: 100,
    };

    it('should scale nutrients to 50g (half serving)', () => {
      const result = service.scaleNutrients(baseNutrients, 50);

      expect(result.calories).toBe(50);
      expect(result.protein).toBe(5);
      expect(result.carbs).toBe(10);
      expect(result.fat).toBe(2.5);
      expect(result.fiber).toBe(1);
      expect(result.sodium).toBe(50);
    });

    it('should scale nutrients to 200g (double serving)', () => {
      const result = service.scaleNutrients(baseNutrients, 200);

      expect(result.calories).toBe(200);
      expect(result.protein).toBe(20);
      expect(result.carbs).toBe(40);
      expect(result.fat).toBe(10);
    });

    it('should return zero nutrients for zero grams', () => {
      const result = service.scaleNutrients(baseNutrients, 0);

      expect(result.calories).toBe(0);
      expect(result.protein).toBe(0);
      expect(result.carbs).toBe(0);
      expect(result.fat).toBe(0);
    });

    it('should return zero nutrients for negative grams', () => {
      const result = service.scaleNutrients(baseNutrients, -50);

      expect(result.calories).toBe(0);
      expect(result.protein).toBe(0);
    });

    it('should maintain precision for small servings', () => {
      const result = service.scaleNutrients(baseNutrients, 10);

      expect(result.calories).toBe(10);
      expect(result.protein).toBe(1);
      expect(result.fat).toBe(0.5);
    });
  });

  describe('getSupportedNutrientIds', () => {
    it('should return all mapped nutrient IDs', () => {
      const ids = service.getSupportedNutrientIds();

      expect(ids).toContain(1008); // Calories
      expect(ids).toContain(1003); // Protein
      expect(ids).toContain(1004); // Fat
      expect(ids).toContain(1005); // Carbs
      expect(ids).toContain(1079); // Fiber
      expect(ids.length).toBeGreaterThan(30);
    });
  });

  describe('isNutrientSupported', () => {
    it('should return true for supported nutrients', () => {
      expect(service.isNutrientSupported(1008)).toBe(true); // Calories
      expect(service.isNutrientSupported(1003)).toBe(true); // Protein
      expect(service.isNutrientSupported(1162)).toBe(true); // Vitamin C
    });

    it('should return false for unsupported nutrients', () => {
      expect(service.isNutrientSupported(9999)).toBe(false);
      expect(service.isNutrientSupported(0)).toBe(false);
    });
  });

  describe('getFieldName', () => {
    it('should return correct field name for nutrient ID', () => {
      expect(service.getFieldName(1008)).toBe('calories');
      expect(service.getFieldName(1003)).toBe('protein');
      expect(service.getFieldName(1004)).toBe('fat');
      expect(service.getFieldName(1005)).toBe('carbs');
      expect(service.getFieldName(1162)).toBe('vitaminC');
    });

    it('should return undefined for unknown nutrient ID', () => {
      expect(service.getFieldName(9999)).toBeUndefined();
    });
  });

  describe('getNutrientValue', () => {
    it('should get specific nutrient value', () => {
      const result = service.getNutrientValue(sampleNutrients, 1008);
      expect(result).toBe(52);
    });

    it('should return undefined for non-existent nutrient', () => {
      const result = service.getNutrientValue(sampleNutrients, 9999);
      expect(result).toBeUndefined();
    });
  });

  describe('USDA_NUTRIENT_ID_MAP', () => {
    it('should have all core macronutrients', () => {
      expect(USDA_NUTRIENT_ID_MAP[1008]).toBe('calories');
      expect(USDA_NUTRIENT_ID_MAP[1003]).toBe('protein');
      expect(USDA_NUTRIENT_ID_MAP[1004]).toBe('fat');
      expect(USDA_NUTRIENT_ID_MAP[1005]).toBe('carbs');
    });

    it('should have fiber and sugar mappings', () => {
      expect(USDA_NUTRIENT_ID_MAP[1079]).toBe('fiber');
      expect(USDA_NUTRIENT_ID_MAP[2000]).toBe('sugar');
    });

    it('should have key vitamins', () => {
      expect(USDA_NUTRIENT_ID_MAP[1106]).toBe('vitaminA');
      expect(USDA_NUTRIENT_ID_MAP[1162]).toBe('vitaminC');
      expect(USDA_NUTRIENT_ID_MAP[1114]).toBe('vitaminD');
    });

    it('should have key minerals', () => {
      expect(USDA_NUTRIENT_ID_MAP[1093]).toBe('sodium');
      expect(USDA_NUTRIENT_ID_MAP[1092]).toBe('potassium');
      expect(USDA_NUTRIENT_ID_MAP[1087]).toBe('calcium');
      expect(USDA_NUTRIENT_ID_MAP[1089]).toBe('iron');
    });
  });

  describe('singleton instance', () => {
    it('should export singleton instance', () => {
      expect(nutrientMappingService).toBeInstanceOf(NutrientMappingService);
    });
  });
});
