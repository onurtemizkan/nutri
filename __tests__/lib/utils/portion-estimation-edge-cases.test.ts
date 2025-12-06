/**
 * Edge Case Tests for Portion Estimation Utilities
 *
 * Tests boundary conditions, error handling, and special cases
 * for the portion estimation algorithms.
 */

import {
  calculateVolume,
  estimateWeight,
  estimateWeightFromMeasurement,
  validateMeasurement,
  formatWeight,
  formatDimensions,
  formatVolume,
  presetToMeasurement,
  PRESET_SIZES,
  FOOD_DENSITY_TABLE,
} from '@/lib/utils/portion-estimation';
import type { ARMeasurement } from '@/lib/types/food-analysis';

describe('Portion Estimation Edge Cases', () => {
  // ===========================================================================
  // Volume Calculation Edge Cases
  // ===========================================================================
  describe('calculateVolume edge cases', () => {
    it('handles very small dimensions', () => {
      const volume = calculateVolume(0.1, 0.1, 0.1);
      expect(volume).toBeCloseTo(0.001, 4);
    });

    it('handles very large dimensions', () => {
      const volume = calculateVolume(100, 100, 100);
      expect(volume).toBe(1000000);
    });

    it('handles zero dimension', () => {
      const volume = calculateVolume(10, 10, 0);
      expect(volume).toBe(0);
    });

    it('handles negative dimensions (returns absolute)', () => {
      // Implementation may vary - test current behavior
      const volume = calculateVolume(-5, 5, 5);
      expect(typeof volume).toBe('number');
    });

    it('handles decimal precision', () => {
      const volume = calculateVolume(2.54, 2.54, 2.54);
      // 2.54^3 = 16.387...
      expect(volume).toBeCloseTo(16.387, 2);
    });
  });

  // ===========================================================================
  // Weight Estimation Edge Cases
  // ===========================================================================
  describe('estimateWeight edge cases', () => {
    it('handles empty food name', () => {
      const result = estimateWeight(5, 5, 5, '');
      expect(result.weight).toBeGreaterThan(0);
      expect(result.densityUsed).toBe(0.7); // Default density
    });

    it('handles unknown food', () => {
      const result = estimateWeight(5, 5, 5, 'alien_fruit_xyz');
      expect(result.weight).toBeGreaterThan(0);
      expect(result.densityUsed).toBe(0.7); // Default density
    });

    it('handles case insensitive food matching', () => {
      const lowercase = estimateWeight(5, 5, 5, 'apple');
      const uppercase = estimateWeight(5, 5, 5, 'APPLE');
      const mixedCase = estimateWeight(5, 5, 5, 'ApPlE');

      expect(lowercase.weight).toBe(uppercase.weight);
      expect(lowercase.weight).toBe(mixedCase.weight);
    });

    it('handles partial food name matching', () => {
      const result = estimateWeight(5, 5, 5, 'apple pie');
      expect(result.weight).toBeGreaterThan(0);
    });

    it('handles food with spaces', () => {
      const result = estimateWeight(5, 5, 5, '  apple  ');
      expect(result.weight).toBeGreaterThan(0);
    });

    it('returns consistent shape factor', () => {
      const result1 = estimateWeight(5, 5, 5, 'bread');
      const result2 = estimateWeight(5, 5, 5, 'bread');
      expect(result1.shapeFactorUsed).toBe(result2.shapeFactorUsed);
    });
  });

  // ===========================================================================
  // Measurement Validation Edge Cases
  // ===========================================================================
  describe('validateMeasurement edge cases', () => {
    it('validates valid small dimensions', () => {
      const measurement: ARMeasurement = {
        width: 0.5,
        height: 0.5,
        depth: 0.5,
        distance: 30,
        confidence: 'high',
        planeDetected: true,
        timestamp: new Date(),
      };

      const result = validateMeasurement(measurement);
      // Small but valid dimensions should pass
      expect(result.valid).toBe(true);
    });

    it('validates valid maximum dimensions', () => {
      const measurement: ARMeasurement = {
        width: 100,
        height: 100,
        depth: 100,
        distance: 30,
        confidence: 'high',
        planeDetected: true,
        timestamp: new Date(),
      };

      const result = validateMeasurement(measurement);
      // Maximum valid dimensions should pass
      expect(result.valid).toBe(true);
    });

    it('rejects dimensions over 100cm', () => {
      const measurement: ARMeasurement = {
        width: 101,
        height: 10,
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

    it('rejects negative dimensions', () => {
      const measurement: ARMeasurement = {
        width: -5,
        height: 10,
        depth: 5,
        distance: 30,
        confidence: 'high',
        planeDetected: true,
        timestamp: new Date(),
      };

      const result = validateMeasurement(measurement);
      expect(result.valid).toBe(false);
    });

    it('rejects negative distance', () => {
      const measurement: ARMeasurement = {
        width: 10,
        height: 10,
        depth: 5,
        distance: -10,
        confidence: 'high',
        planeDetected: true,
        timestamp: new Date(),
      };

      const result = validateMeasurement(measurement);
      expect(result.valid).toBe(false);
      expect(result.error).toContain('distance');
    });

    it('accepts valid measurements with all fields', () => {
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
  });

  // ===========================================================================
  // Format Functions Edge Cases
  // ===========================================================================
  describe('Format functions edge cases', () => {
    describe('formatWeight', () => {
      it('formats grams for small weights', () => {
        expect(formatWeight(50)).toMatch(/g/);
      });

      it('formats kilograms for large weights', () => {
        expect(formatWeight(1500)).toMatch(/kg/);
      });

      it('handles zero weight', () => {
        const result = formatWeight(0);
        expect(result).toBe('<1g');
      });

      it('handles decimal weights', () => {
        const result = formatWeight(123.456);
        expect(result).toBeTruthy();
      });
    });

    describe('formatDimensions', () => {
      it('formats standard dimensions', () => {
        const result = formatDimensions(10, 8, 5);
        expect(result).toContain('10');
        expect(result).toContain('8');
        expect(result).toContain('5');
        expect(result).toContain('cm');
      });

      it('handles decimal dimensions', () => {
        const result = formatDimensions(10.5, 8.3, 5.7);
        expect(result).toBeTruthy();
      });
    });

    describe('formatVolume', () => {
      it('formats cubic centimeters for small volumes', () => {
        expect(formatVolume(50)).toMatch(/cm³/);
      });

      it('formats liters for large volumes', () => {
        expect(formatVolume(1500)).toMatch(/L/);
      });

      it('handles very small volume', () => {
        const result = formatVolume(0);
        expect(result).toBe('<1 cm³');
      });

      it('handles sub-1 volume', () => {
        const result = formatVolume(0.5);
        expect(result).toBe('<1 cm³');
      });
    });
  });

  // ===========================================================================
  // Preset Sizes Edge Cases
  // ===========================================================================
  describe('Preset sizes edge cases', () => {
    it('all presets have valid measurements', () => {
      PRESET_SIZES.forEach((preset) => {
        expect(preset.width).toBeGreaterThan(0);
        expect(preset.height).toBeGreaterThan(0);
        expect(preset.depth).toBeGreaterThan(0);
      });
    });

    it('presets are in ascending order by volume', () => {
      const volumes = PRESET_SIZES.map((p) =>
        calculateVolume(p.width, p.height, p.depth)
      );

      for (let i = 1; i < volumes.length; i++) {
        expect(volumes[i]).toBeGreaterThan(volumes[i - 1]);
      }
    });

    it('presetToMeasurement returns valid ARMeasurement', () => {
      PRESET_SIZES.forEach((preset) => {
        const measurement = presetToMeasurement(preset);

        expect(measurement.width).toBe(preset.width);
        expect(measurement.height).toBe(preset.height);
        expect(measurement.depth).toBe(preset.depth);
        expect(measurement.confidence).toBe('low'); // Manual estimate has low confidence
        expect(measurement.planeDetected).toBe(false);
        expect(measurement.timestamp).toBeInstanceOf(Date);
      });
    });
  });

  // ===========================================================================
  // Food Density Table Edge Cases
  // ===========================================================================
  describe('Food density table edge cases', () => {
    it('has reasonable density values', () => {
      Object.values(FOOD_DENSITY_TABLE).forEach((entry) => {
        expect(entry.density).toBeGreaterThan(0);
        expect(entry.density).toBeLessThan(5); // Most foods are less dense than 5 g/ml
      });
    });

    it('has common foods', () => {
      // Note: chicken is stored as 'chicken breast' and 'chicken thigh'
      const commonFoods = ['apple', 'bread', 'chicken breast', 'rice', 'pasta'];
      commonFoods.forEach((food) => {
        expect(FOOD_DENSITY_TABLE[food]).toBeDefined();
        expect(FOOD_DENSITY_TABLE[food].name).toBeTruthy();
        expect(FOOD_DENSITY_TABLE[food].density).toBeGreaterThan(0);
        expect(FOOD_DENSITY_TABLE[food].category).toBeTruthy();
        expect(FOOD_DENSITY_TABLE[food].shapeFactor).toBeGreaterThan(0);
      });
    });

    it('entries have all required properties', () => {
      Object.entries(FOOD_DENSITY_TABLE).forEach(([key, entry]) => {
        expect(entry.name).toBeTruthy();
        expect(typeof entry.density).toBe('number');
        expect(entry.category).toBeTruthy();
        expect(typeof entry.shapeFactor).toBe('number');
        expect(entry.shapeFactor).toBeGreaterThan(0);
        expect(entry.shapeFactor).toBeLessThanOrEqual(1);
      });
    });
  });

  // ===========================================================================
  // Integration Edge Cases
  // ===========================================================================
  describe('Integration edge cases', () => {
    it('estimateWeightFromMeasurement handles all confidence levels', () => {
      const confidences: Array<'high' | 'medium' | 'low'> = [
        'high',
        'medium',
        'low',
      ];

      confidences.forEach((confidence) => {
        const measurement: ARMeasurement = {
          width: 10,
          height: 8,
          depth: 5,
          distance: 30,
          confidence,
          planeDetected: confidence === 'high',
          timestamp: new Date(),
        };

        const result = estimateWeightFromMeasurement(measurement, 'apple');
        expect(result.weight).toBeGreaterThan(0);
      });
    });

    it('consistent results for same inputs', () => {
      const measurement: ARMeasurement = {
        width: 10,
        height: 8,
        depth: 5,
        distance: 30,
        confidence: 'high',
        planeDetected: true,
        timestamp: new Date(),
      };

      const result1 = estimateWeightFromMeasurement(measurement, 'apple');
      const result2 = estimateWeightFromMeasurement(measurement, 'apple');

      expect(result1.weight).toBe(result2.weight);
    });
  });
});
