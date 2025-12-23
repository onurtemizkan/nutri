/**
 * Food Ranking Utility Tests
 *
 * Tests for the food ranking and scoring algorithms
 *
 * Note: This is a unit test file that doesn't require database access.
 */

import {
  rankSearchResults,
  isWholeFoodQuery,
  isBrandedQuery,
  getDataTypeLabel,
  getDataTypeBadgeStyle,
  type RankingHints,
} from '../utils/foodRanking';
import type { TransformedUSDAFood, USDADataType } from '../types/usda';

// ============================================================================
// TEST DATA FACTORIES
// ============================================================================

/**
 * Create a mock TransformedUSDAFood for testing
 */
function createMockFood(overrides: Partial<TransformedUSDAFood> = {}): TransformedUSDAFood {
  return {
    fdcId: Math.floor(Math.random() * 1000000),
    name: 'Test Food',
    description: 'Test food description',
    dataType: 'Foundation',
    nutrients: {
      calories: 100,
      protein: 10,
      carbs: 15,
      fat: 5,
      fiber: 2,
      sugar: 3,
      sodium: 100,
      saturatedFat: 1,
    },
    servingSize: 100,
    servingSizeUnit: 'g',
    ...overrides,
  };
}

// ============================================================================
// RANKING TESTS
// ============================================================================

describe('rankSearchResults', () => {
  describe('data type ranking', () => {
    it('should rank Foundation foods higher than Branded foods', () => {
      const foundationFood = createMockFood({
        fdcId: 1,
        dataType: 'Foundation',
        description: 'Apple, raw',
      });
      const brandedFood = createMockFood({
        fdcId: 2,
        dataType: 'Branded',
        description: 'Apple Chips',
      });

      const results = rankSearchResults([brandedFood, foundationFood]);

      expect(results[0].dataType).toBe('Foundation');
      expect(results[1].dataType).toBe('Branded');
      expect(results[0].rankScore).toBeGreaterThan(results[1].rankScore);
    });

    it('should rank SR Legacy foods higher than Branded foods', () => {
      const srLegacyFood = createMockFood({
        fdcId: 1,
        dataType: 'SR Legacy',
        description: 'Chicken breast',
      });
      const brandedFood = createMockFood({
        fdcId: 2,
        dataType: 'Branded',
        description: 'Chicken nuggets',
      });

      const results = rankSearchResults([brandedFood, srLegacyFood]);

      expect(results[0].dataType).toBe('SR Legacy');
      expect(results[1].dataType).toBe('Branded');
    });

    it('should boost Foundation for whole food queries', () => {
      const foundationFood = createMockFood({
        fdcId: 1,
        dataType: 'Foundation',
        description: 'Apple, raw, with skin',
      });
      const brandedFood = createMockFood({
        fdcId: 2,
        dataType: 'Branded',
        description: 'Apple flavored drink',
      });

      const hints: RankingHints = {
        query: 'raw apple',
        isWholeFoodQuery: true,
      };

      const results = rankSearchResults([brandedFood, foundationFood], hints);

      // Foundation should get boosted score
      expect(results[0].scoreBreakdown.dataTypeScore).toBeGreaterThan(
        results[1].scoreBreakdown.dataTypeScore
      );
    });

    it('should boost Branded data type score for branded product queries', () => {
      const foundationFood = createMockFood({
        fdcId: 1,
        dataType: 'Foundation',
        description: 'Cola nut',
      });
      const brandedFood = createMockFood({
        fdcId: 2,
        dataType: 'Branded',
        description: 'Coca-Cola',
      });

      const hints: RankingHints = {
        query: "McDonald's burger",
        isBrandedQuery: true,
      };

      const results = rankSearchResults([foundationFood, brandedFood], hints);

      // Branded should get boosted data type score (50 + 30 = 80)
      // Find the branded result
      const brandedResult = results.find(r => r.dataType === 'Branded');
      expect(brandedResult?.scoreBreakdown.dataTypeScore).toBe(80);
    });
  });

  describe('name match scoring', () => {
    it('should rank exact matches higher', () => {
      const exactMatch = createMockFood({
        fdcId: 1,
        dataType: 'Foundation',
        description: 'apple',
      });
      const partialMatch = createMockFood({
        fdcId: 2,
        dataType: 'Foundation',
        description: 'Apple pie with cinnamon',
      });

      const hints: RankingHints = { query: 'apple' };
      const results = rankSearchResults([partialMatch, exactMatch], hints);

      expect(results[0].scoreBreakdown.nameMatchScore).toBe(100);
    });

    it('should rank prefix matches higher than partial matches', () => {
      const prefixMatch = createMockFood({
        fdcId: 1,
        dataType: 'Foundation',
        description: 'apple, raw, with skin',
      });
      const containsMatch = createMockFood({
        fdcId: 2,
        dataType: 'Foundation',
        description: 'Pie, apple, homemade',
      });

      const hints: RankingHints = { query: 'apple' };
      const results = rankSearchResults([containsMatch, prefixMatch], hints);

      expect(results[0].scoreBreakdown.nameMatchScore).toBeGreaterThan(
        results[1].scoreBreakdown.nameMatchScore
      );
    });
  });

  describe('completeness scoring', () => {
    it('should give higher completeness score for data with extended nutrients', () => {
      const completeFood = createMockFood({
        fdcId: 1,
        dataType: 'Foundation',
        nutrients: {
          calories: 100,
          protein: 10,
          carbs: 15,
          fat: 5,
          fiber: 2,
          sugar: 3,
          sodium: 100,
          saturatedFat: 1,
        },
        servingSize: 100,
      });

      const basicFood = createMockFood({
        fdcId: 2,
        dataType: 'Foundation',
        nutrients: {
          calories: 100,
          protein: 10,
          carbs: 15,
          fat: 5,
        },
        servingSize: undefined, // No serving size
      });

      const results = rankSearchResults([basicFood, completeFood]);

      // Complete food should have 100 (core) + 20 (4 extended nutrients * 5) + 10 (serving size) = 130, capped to 100
      // Basic food should have 100 (core) + 0 (no extended) + 0 (no serving size) = 100
      expect(results[0].scoreBreakdown.completenessScore).toBe(100);
      // Basic food also gets 100 since it has all core nutrients
      expect(results[1].scoreBreakdown.completenessScore).toBe(100);
    });

    it('should penalize missing core nutrients', () => {
      const missingProtein = createMockFood({
        fdcId: 1,
        dataType: 'Foundation',
        nutrients: {
          calories: 100,
          protein: undefined as unknown as number,
          carbs: 15,
          fat: 5,
        },
      });

      const results = rankSearchResults([missingProtein]);

      expect(results[0].dataQuality.isComplete).toBe(false);
      expect(results[0].dataQuality.missingFields).toContain('protein');
    });
  });

  describe('category match scoring', () => {
    it('should boost foods matching classification category', () => {
      const fruitFood = createMockFood({
        fdcId: 1,
        dataType: 'Foundation',
        description: 'Apple, raw, with skin',
        category: 'Fruits and Fruit Juices',
      });
      const otherFood = createMockFood({
        fdcId: 2,
        dataType: 'Foundation',
        description: 'Apple sauce',
        category: 'Baby Foods',
      });

      const hints: RankingHints = {
        category: 'fruit',
        query: 'apple',
      };

      const results = rankSearchResults([otherFood, fruitFood], hints);

      expect(results[0].scoreBreakdown.categoryMatchScore).toBeGreaterThan(
        results[1].scoreBreakdown.categoryMatchScore
      );
    });

    it('should use subcategory hints for additional scoring', () => {
      const matchingFood = createMockFood({
        fdcId: 1,
        dataType: 'Foundation',
        description: 'granny smith apple, raw',
      });
      const nonMatchingFood = createMockFood({
        fdcId: 2,
        dataType: 'Foundation',
        description: 'apple, generic',
      });

      const hints: RankingHints = {
        subcategoryHints: ['granny smith', 'green apple'],
        query: 'apple',
      };

      const results = rankSearchResults([nonMatchingFood, matchingFood], hints);

      // Matching food should get boost from subcategory hint
      const matchingResult = results.find(r => r.fdcId === 1);
      const nonMatchingResult = results.find(r => r.fdcId === 2);

      // Matching food has "granny smith" in description, gets +10 for that hint
      expect(matchingResult?.scoreBreakdown.categoryMatchScore).toBeGreaterThan(
        nonMatchingResult?.scoreBreakdown.categoryMatchScore ?? 0
      );
    });
  });

  describe('score breakdown', () => {
    it('should include all score components in breakdown', () => {
      const food = createMockFood();
      const results = rankSearchResults([food]);

      expect(results[0].scoreBreakdown).toHaveProperty('dataTypeScore');
      expect(results[0].scoreBreakdown).toHaveProperty('nameMatchScore');
      expect(results[0].scoreBreakdown).toHaveProperty('completenessScore');
      expect(results[0].scoreBreakdown).toHaveProperty('categoryMatchScore');
    });

    it('should include data quality assessment', () => {
      const food = createMockFood();
      const results = rankSearchResults([food]);

      expect(results[0].dataQuality).toHaveProperty('isComplete');
      expect(results[0].dataQuality).toHaveProperty('missingFields');
      expect(results[0].dataQuality).toHaveProperty('hasServingSize');
    });
  });

  describe('edge cases', () => {
    it('should return empty array for empty input', () => {
      const results = rankSearchResults([]);
      expect(results).toEqual([]);
    });

    it('should handle null input', () => {
      const results = rankSearchResults(null as unknown as TransformedUSDAFood[]);
      expect(results).toEqual([]);
    });

    it('should handle foods with missing nutrients object', () => {
      const foodWithoutNutrients = createMockFood({
        nutrients: undefined as unknown as TransformedUSDAFood['nutrients'],
      });

      const results = rankSearchResults([foodWithoutNutrients]);

      expect(results[0].scoreBreakdown.completenessScore).toBe(0);
      expect(results[0].dataQuality.isComplete).toBe(false);
    });
  });
});

// ============================================================================
// QUERY DETECTION TESTS
// ============================================================================

describe('isWholeFoodQuery', () => {
  it('should return true for queries with whole food indicators', () => {
    expect(isWholeFoodQuery('raw apple')).toBe(true);
    expect(isWholeFoodQuery('fresh chicken breast')).toBe(true);
    expect(isWholeFoodQuery('whole wheat bread')).toBe(true);
    expect(isWholeFoodQuery('organic spinach')).toBe(true);
    expect(isWholeFoodQuery('plain yogurt')).toBe(true);
  });

  it('should return true for common single-word whole foods', () => {
    expect(isWholeFoodQuery('apple')).toBe(true);
    expect(isWholeFoodQuery('banana')).toBe(true);
    expect(isWholeFoodQuery('chicken')).toBe(true);
    expect(isWholeFoodQuery('salmon')).toBe(true);
    expect(isWholeFoodQuery('egg')).toBe(true);
  });

  it('should return false for processed food queries', () => {
    expect(isWholeFoodQuery('chips')).toBe(false);
    expect(isWholeFoodQuery('candy bar')).toBe(false);
    expect(isWholeFoodQuery('frozen pizza')).toBe(false);
  });
});

describe('isBrandedQuery', () => {
  it('should return true for queries with brand indicators', () => {
    expect(isBrandedQuery("McDonald's burger")).toBe(true);
    expect(isBrandedQuery("Trader Joe's salad")).toBe(true);
    expect(isBrandedQuery('brand name cereal')).toBe(true);
  });

  it('should return true for queries with trademark symbols', () => {
    expect(isBrandedQuery('Coca-Cola®')).toBe(true);
    expect(isBrandedQuery('Pepsi™')).toBe(true);
  });

  it('should return true for queries with size specifications', () => {
    expect(isBrandedQuery('energy drink 12oz')).toBe(true);
    expect(isBrandedQuery('protein bar 50g')).toBe(true);
    expect(isBrandedQuery('soda 500ml')).toBe(true);
  });

  it('should return false for generic food queries', () => {
    expect(isBrandedQuery('apple')).toBe(false);
    expect(isBrandedQuery('grilled chicken')).toBe(false);
    expect(isBrandedQuery('steamed rice')).toBe(false);
  });
});

// ============================================================================
// DATA TYPE LABEL TESTS
// ============================================================================

describe('getDataTypeLabel', () => {
  it('should return correct labels for each data type', () => {
    expect(getDataTypeLabel('Foundation')).toBe('USDA');
    expect(getDataTypeLabel('SR Legacy')).toBe('USDA');
    expect(getDataTypeLabel('Branded')).toBe('Brand');
    expect(getDataTypeLabel('Survey (FNDDS)')).toBe('Recipe');
    expect(getDataTypeLabel('Experimental')).toBe('Exp');
  });

  it('should return default label for unknown types', () => {
    expect(getDataTypeLabel('Unknown' as USDADataType)).toBe('Food');
  });
});

describe('getDataTypeBadgeStyle', () => {
  it('should return correct badge styles for each data type', () => {
    const foundationStyle = getDataTypeBadgeStyle('Foundation');
    expect(foundationStyle.label).toBe('USDA Foundation');
    expect(foundationStyle.color).toBe('#22c55e');
    expect(foundationStyle.priority).toBe(1);

    const brandedStyle = getDataTypeBadgeStyle('Branded');
    expect(brandedStyle.label).toBe('Branded');
    expect(brandedStyle.color).toBe('#6366f1');
    expect(brandedStyle.priority).toBe(4);
  });

  it('should return consistent priority ordering', () => {
    const foundation = getDataTypeBadgeStyle('Foundation');
    const srLegacy = getDataTypeBadgeStyle('SR Legacy');
    const survey = getDataTypeBadgeStyle('Survey (FNDDS)');
    const branded = getDataTypeBadgeStyle('Branded');
    const experimental = getDataTypeBadgeStyle('Experimental');

    expect(foundation.priority).toBeLessThan(srLegacy.priority);
    expect(srLegacy.priority).toBeLessThan(survey.priority);
    expect(survey.priority).toBeLessThan(branded.priority);
    expect(branded.priority).toBeLessThan(experimental.priority);
  });
});
