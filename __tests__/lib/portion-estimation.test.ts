/**
 * Tests for Portion Estimation Utilities
 */

import {
  // Volume calculations
  calculateVolume,
  volumeFromMeasurement,
  applyShapeFactor,
  // Density lookup
  lookupFoodDensity,
  getDensity,
  getShapeFactor,
  FOOD_DENSITY_TABLE,
  DEFAULT_DENSITY_BY_CATEGORY,
  // Weight estimation
  estimateWeight,
  estimateWeightFromMeasurement,
  MIN_WEIGHT_GRAMS,
  MAX_WEIGHT_GRAMS,
  // Unit conversions
  cmToInches,
  inchesToCm,
  gramsToOz,
  ozToGrams,
  gramsToLbs,
  lbsToGrams,
  volumeCm3ToMl,
  mlToVolumeCm3,
  volumeCm3ToCups,
  cupsToVolumeCm3,
  // Preset sizes
  PRESET_SIZES,
  presetToMeasurement,
  getPresetSize,
  // Validation
  isValidDimension,
  validateMeasurement,
  // Formatting
  formatWeight,
  formatDimensions,
  formatVolume,
} from '@/lib/utils/portion-estimation';

import type { ARMeasurement } from '@/lib/types/food-analysis';

describe('portion-estimation utilities', () => {
  // ===========================================================================
  // Volume Calculations
  // ===========================================================================
  describe('calculateVolume', () => {
    it('calculates volume correctly for positive dimensions', () => {
      expect(calculateVolume(10, 10, 10)).toBe(1000);
      expect(calculateVolume(5, 4, 3)).toBe(60);
      expect(calculateVolume(2.5, 2.5, 2.5)).toBe(15.625);
    });

    it('returns 0 for zero dimensions', () => {
      expect(calculateVolume(0, 10, 10)).toBe(0);
      expect(calculateVolume(10, 0, 10)).toBe(0);
      expect(calculateVolume(10, 10, 0)).toBe(0);
    });

    it('returns 0 for negative dimensions', () => {
      expect(calculateVolume(-5, 10, 10)).toBe(0);
      expect(calculateVolume(10, -5, 10)).toBe(0);
      expect(calculateVolume(10, 10, -5)).toBe(0);
    });
  });

  describe('volumeFromMeasurement', () => {
    it('calculates volume from ARMeasurement object', () => {
      const measurement: ARMeasurement = {
        width: 10,
        height: 8,
        depth: 5,
        distance: 30,
        confidence: 'high',
        planeDetected: true,
        timestamp: new Date(),
      };

      expect(volumeFromMeasurement(measurement)).toBe(400);
    });
  });

  describe('applyShapeFactor', () => {
    it('applies shape factor to volume', () => {
      expect(applyShapeFactor(1000, 0.7)).toBe(700);
      expect(applyShapeFactor(100, 0.5)).toBe(50);
    });

    it('clamps shape factor to [0, 1]', () => {
      expect(applyShapeFactor(100, 1.5)).toBe(100); // Clamped to 1
      expect(applyShapeFactor(100, -0.5)).toBe(0); // Clamped to 0
    });
  });

  // ===========================================================================
  // Density Lookup
  // ===========================================================================
  describe('lookupFoodDensity', () => {
    it('finds exact match (case-insensitive)', () => {
      const result = lookupFoodDensity('Apple');
      expect(result).toBeDefined();
      expect(result?.name).toBe('Apple');
      expect(result?.density).toBe(0.7);
    });

    it('finds partial match', () => {
      const result = lookupFoodDensity('chicken');
      expect(result).toBeDefined();
      expect(result?.category).toBe('protein');
    });

    it('returns undefined for unknown food', () => {
      const result = lookupFoodDensity('xyzzyfood');
      expect(result).toBeUndefined();
    });
  });

  describe('getDensity', () => {
    it('returns density for known food', () => {
      expect(getDensity('apple')).toBe(0.7);
      expect(getDensity('salmon')).toBe(1.0);
    });

    it('returns category default for unknown food with category hint', () => {
      expect(getDensity('mystery fruit', 'fruit')).toBe(DEFAULT_DENSITY_BY_CATEGORY.fruit);
    });

    it('returns generic default for completely unknown food', () => {
      expect(getDensity('unknown item')).toBe(DEFAULT_DENSITY_BY_CATEGORY.unknown);
    });
  });

  describe('getShapeFactor', () => {
    it('returns shape factor for known food', () => {
      expect(getShapeFactor('apple')).toBe(0.52);
      expect(getShapeFactor('bread')).toBe(0.85);
    });

    it('returns category default for unknown food', () => {
      expect(getShapeFactor('mystery', 'protein')).toBe(0.75);
    });
  });

  // ===========================================================================
  // Weight Estimation
  // ===========================================================================
  describe('estimateWeight', () => {
    it('estimates weight for known food', () => {
      // Apple: density 0.7, shapeFactor 0.52
      // 7cm cube: volume = 343, adjusted = 343 * 0.52 = 178.36, weight = 178.36 * 0.7 = 124.85
      const result = estimateWeight(7, 7, 7, 'apple');

      expect(result.method).toBe('density-lookup');
      expect(result.densityUsed).toBe(0.7);
      expect(result.shapeFactorUsed).toBe(0.52);
      expect(result.volumeRaw).toBe(343);
      expect(result.weight).toBeGreaterThan(100);
      expect(result.weight).toBeLessThan(150);
      expect(result.confidence).toBe(0.8);
    });

    it('estimates weight with category fallback', () => {
      const result = estimateWeight(5, 5, 5, 'mystery fruit', 'fruit');

      expect(result.method).toBe('category-default');
      expect(result.densityUsed).toBe(DEFAULT_DENSITY_BY_CATEGORY.fruit);
      expect(result.confidence).toBe(0.6);
    });

    it('estimates weight with generic default', () => {
      const result = estimateWeight(5, 5, 5, 'completely unknown');

      expect(result.method).toBe('generic-default');
      expect(result.confidence).toBe(0.4);
    });

    it('enforces minimum weight', () => {
      const result = estimateWeight(0.1, 0.1, 0.1, 'rice');
      expect(result.weight).toBeGreaterThanOrEqual(MIN_WEIGHT_GRAMS);
    });

    it('enforces maximum weight', () => {
      const result = estimateWeight(100, 100, 100, 'water');
      expect(result.weight).toBeLessThanOrEqual(MAX_WEIGHT_GRAMS);
    });
  });

  describe('estimateWeightFromMeasurement', () => {
    it('adjusts confidence based on AR measurement quality', () => {
      const highConfMeasurement: ARMeasurement = {
        width: 7,
        height: 7,
        depth: 7,
        distance: 30,
        confidence: 'high',
        planeDetected: true,
        timestamp: new Date(),
      };

      const lowConfMeasurement: ARMeasurement = {
        ...highConfMeasurement,
        confidence: 'low',
        planeDetected: false,
      };

      const highResult = estimateWeightFromMeasurement(highConfMeasurement, 'apple');
      const lowResult = estimateWeightFromMeasurement(lowConfMeasurement, 'apple');

      // Same weight (same dimensions)
      expect(highResult.weight).toBe(lowResult.weight);

      // But different confidence
      expect(highResult.confidence).toBeGreaterThan(lowResult.confidence);
    });
  });

  // ===========================================================================
  // Unit Conversions
  // ===========================================================================
  describe('unit conversions', () => {
    describe('length conversions', () => {
      it('converts cm to inches', () => {
        expect(cmToInches(2.54)).toBeCloseTo(1, 5);
        expect(cmToInches(10)).toBeCloseTo(3.937, 2);
      });

      it('converts inches to cm', () => {
        expect(inchesToCm(1)).toBeCloseTo(2.54, 5);
        expect(inchesToCm(12)).toBeCloseTo(30.48, 2);
      });

      it('round-trips correctly', () => {
        const original = 15;
        const roundTrip = inchesToCm(cmToInches(original));
        expect(roundTrip).toBeCloseTo(original, 5);
      });
    });

    describe('weight conversions', () => {
      it('converts grams to ounces', () => {
        expect(gramsToOz(28.3495)).toBeCloseTo(1, 3);
        expect(gramsToOz(100)).toBeCloseTo(3.527, 2);
      });

      it('converts ounces to grams', () => {
        expect(ozToGrams(1)).toBeCloseTo(28.3495, 3);
      });

      it('converts grams to pounds', () => {
        expect(gramsToLbs(453.592)).toBeCloseTo(1, 3);
      });

      it('converts pounds to grams', () => {
        expect(lbsToGrams(1)).toBeCloseTo(453.592, 2);
      });
    });

    describe('volume conversions', () => {
      it('converts cm³ to mL (1:1)', () => {
        expect(volumeCm3ToMl(100)).toBe(100);
        expect(volumeCm3ToMl(250)).toBe(250);
      });

      it('converts mL to cm³ (1:1)', () => {
        expect(mlToVolumeCm3(100)).toBe(100);
      });

      it('converts cm³ to cups', () => {
        expect(volumeCm3ToCups(236.588)).toBeCloseTo(1, 3);
      });

      it('converts cups to cm³', () => {
        expect(cupsToVolumeCm3(1)).toBeCloseTo(236.588, 2);
      });
    });
  });

  // ===========================================================================
  // Preset Sizes
  // ===========================================================================
  describe('preset sizes', () => {
    it('has defined preset sizes', () => {
      expect(PRESET_SIZES.length).toBeGreaterThan(0);
      expect(PRESET_SIZES.map(p => p.name)).toContain('small');
      expect(PRESET_SIZES.map(p => p.name)).toContain('medium');
      expect(PRESET_SIZES.map(p => p.name)).toContain('large');
    });

    describe('presetToMeasurement', () => {
      it('converts preset to ARMeasurement with low confidence', () => {
        const smallPreset = PRESET_SIZES.find(p => p.name === 'small')!;
        const measurement = presetToMeasurement(smallPreset);

        expect(measurement.width).toBe(smallPreset.width);
        expect(measurement.height).toBe(smallPreset.height);
        expect(measurement.depth).toBe(smallPreset.depth);
        expect(measurement.confidence).toBe('low');
        expect(measurement.planeDetected).toBe(false);
      });
    });

    describe('getPresetSize', () => {
      it('finds preset by name', () => {
        const medium = getPresetSize('medium');
        expect(medium).toBeDefined();
        expect(medium?.displayName).toBe('Medium');
      });

      it('returns undefined for unknown preset', () => {
        expect(getPresetSize('gigantic')).toBeUndefined();
      });
    });
  });

  // ===========================================================================
  // Validation
  // ===========================================================================
  describe('isValidDimension', () => {
    it('returns true for valid dimensions', () => {
      expect(isValidDimension(1)).toBe(true);
      expect(isValidDimension(50)).toBe(true);
      expect(isValidDimension(0.5)).toBe(true);
    });

    it('returns false for invalid dimensions', () => {
      expect(isValidDimension(0)).toBe(false);
      expect(isValidDimension(-5)).toBe(false);
      expect(isValidDimension(NaN)).toBe(false);
      expect(isValidDimension(150)).toBe(false); // > 100 default max
    });

    it('respects custom max value', () => {
      expect(isValidDimension(150, 200)).toBe(true);
      expect(isValidDimension(250, 200)).toBe(false);
    });
  });

  describe('validateMeasurement', () => {
    it('validates correct measurement', () => {
      const measurement: ARMeasurement = {
        width: 10,
        height: 8,
        depth: 5,
        distance: 30,
        confidence: 'high',
        planeDetected: true,
        timestamp: new Date(),
      };

      const result = validateMeasurement(measurement);
      expect(result.valid).toBe(true);
      expect(result.error).toBeUndefined();
    });

    it('rejects invalid width', () => {
      const measurement: ARMeasurement = {
        width: -5,
        height: 8,
        depth: 5,
        distance: 30,
        confidence: 'high',
        planeDetected: true,
        timestamp: new Date(),
      };

      const result = validateMeasurement(measurement);
      expect(result.valid).toBe(false);
      expect(result.error).toContain('width');
    });

    it('rejects too large dimensions', () => {
      const measurement: ARMeasurement = {
        width: 150,
        height: 8,
        depth: 5,
        distance: 30,
        confidence: 'high',
        planeDetected: true,
        timestamp: new Date(),
      };

      const result = validateMeasurement(measurement);
      expect(result.valid).toBe(false);
    });
  });

  // ===========================================================================
  // Formatting
  // ===========================================================================
  describe('formatWeight', () => {
    it('formats small weights', () => {
      expect(formatWeight(50)).toBe('50g');
      expect(formatWeight(0.5)).toBe('<1g');
    });

    it('formats large weights in kg', () => {
      expect(formatWeight(1500)).toBe('1.5kg');
      expect(formatWeight(2000)).toBe('2kg');
    });

    it('respects includeUnits flag', () => {
      expect(formatWeight(50, false)).toBe('50');
      expect(formatWeight(1500, false)).toBe('1.5');
    });
  });

  describe('formatDimensions', () => {
    it('formats dimensions correctly', () => {
      expect(formatDimensions(10, 8, 5)).toBe('10 × 8 × 5 cm');
      expect(formatDimensions(10.5, 8.25, 5.1)).toBe('10.5 × 8.3 × 5.1 cm');
    });
  });

  describe('formatVolume', () => {
    it('formats small volumes', () => {
      expect(formatVolume(100)).toBe('100 cm³');
      expect(formatVolume(0.5)).toBe('<1 cm³');
    });

    it('formats large volumes in liters', () => {
      expect(formatVolume(1500)).toBe('1.5 L');
      expect(formatVolume(2000)).toBe('2 L');
    });
  });

  // ===========================================================================
  // Food Density Table Coverage
  // ===========================================================================
  describe('FOOD_DENSITY_TABLE', () => {
    it('has entries for common food categories', () => {
      const categories = new Set(Object.values(FOOD_DENSITY_TABLE).map(e => e.category));

      expect(categories.has('fruit')).toBe(true);
      expect(categories.has('vegetable')).toBe(true);
      expect(categories.has('protein')).toBe(true);
      expect(categories.has('grain')).toBe(true);
    });

    it('all entries have valid density values (0 < density <= 2)', () => {
      for (const [name, entry] of Object.entries(FOOD_DENSITY_TABLE)) {
        expect(entry.density).toBeGreaterThan(0);
        expect(entry.density).toBeLessThanOrEqual(2);
      }
    });

    it('all entries have valid shape factors (0 < shapeFactor <= 1)', () => {
      for (const [name, entry] of Object.entries(FOOD_DENSITY_TABLE)) {
        expect(entry.shapeFactor).toBeGreaterThan(0);
        expect(entry.shapeFactor).toBeLessThanOrEqual(1);
      }
    });
  });
});
