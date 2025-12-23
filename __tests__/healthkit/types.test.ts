/**
 * HealthKit Types Tests
 */

import {
  HEALTHKIT_TO_METRIC_TYPE,
  METRIC_UNITS,
  HEALTHKIT_READ_PERMISSIONS,
  SYNC_TIMESTAMP_KEYS,
} from '@/lib/types/healthkit';

describe('HealthKit Types', () => {
  describe('HEALTHKIT_TO_METRIC_TYPE', () => {
    it('should map HKQuantityTypeIdentifierRestingHeartRate to RESTING_HEART_RATE', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['HKQuantityTypeIdentifierRestingHeartRate']).toBe(
        'RESTING_HEART_RATE'
      );
    });

    it('should map HKQuantityTypeIdentifierHeartRateVariabilitySDNN to HEART_RATE_VARIABILITY_SDNN', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['HKQuantityTypeIdentifierHeartRateVariabilitySDNN']).toBe(
        'HEART_RATE_VARIABILITY_SDNN'
      );
    });

    it('should map HKQuantityTypeIdentifierRespiratoryRate to RESPIRATORY_RATE', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['HKQuantityTypeIdentifierRespiratoryRate']).toBe(
        'RESPIRATORY_RATE'
      );
    });

    it('should map HKQuantityTypeIdentifierOxygenSaturation to OXYGEN_SATURATION', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['HKQuantityTypeIdentifierOxygenSaturation']).toBe(
        'OXYGEN_SATURATION'
      );
    });

    it('should map HKQuantityTypeIdentifierVO2Max to VO2_MAX', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['HKQuantityTypeIdentifierVO2Max']).toBe('VO2_MAX');
    });

    it('should map HKQuantityTypeIdentifierStepCount to STEPS', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['HKQuantityTypeIdentifierStepCount']).toBe('STEPS');
    });

    it('should map HKQuantityTypeIdentifierActiveEnergyBurned to ACTIVE_CALORIES', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['HKQuantityTypeIdentifierActiveEnergyBurned']).toBe(
        'ACTIVE_CALORIES'
      );
    });
  });

  describe('METRIC_UNITS', () => {
    it('should have correct unit for RESTING_HEART_RATE', () => {
      expect(METRIC_UNITS['RESTING_HEART_RATE']).toBe('bpm');
    });

    it('should have correct unit for HEART_RATE_VARIABILITY_SDNN', () => {
      expect(METRIC_UNITS['HEART_RATE_VARIABILITY_SDNN']).toBe('ms');
    });

    it('should have correct unit for OXYGEN_SATURATION', () => {
      expect(METRIC_UNITS['OXYGEN_SATURATION']).toBe('%');
    });

    it('should have correct unit for VO2_MAX', () => {
      expect(METRIC_UNITS['VO2_MAX']).toBe('mL/kg/min');
    });

    it('should have correct unit for SLEEP_DURATION', () => {
      expect(METRIC_UNITS['SLEEP_DURATION']).toBe('hours');
    });

    it('should have correct unit for STEPS', () => {
      expect(METRIC_UNITS['STEPS']).toBe('steps');
    });

    it('should have correct unit for ACTIVE_CALORIES', () => {
      expect(METRIC_UNITS['ACTIVE_CALORIES']).toBe('kcal');
    });
  });

  describe('HEALTHKIT_READ_PERMISSIONS', () => {
    it('should include HeartRate', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('HKQuantityTypeIdentifierHeartRate');
    });

    it('should include RestingHeartRate', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('HKQuantityTypeIdentifierRestingHeartRate');
    });

    it('should include HeartRateVariabilitySDNN', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('HKQuantityTypeIdentifierHeartRateVariabilitySDNN');
    });

    it('should include SleepAnalysis', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('HKCategoryTypeIdentifierSleepAnalysis');
    });

    it('should include StepCount', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('HKQuantityTypeIdentifierStepCount');
    });

    it('should include ActiveEnergyBurned', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('HKQuantityTypeIdentifierActiveEnergyBurned');
    });

    it('should include OxygenSaturation', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('HKQuantityTypeIdentifierOxygenSaturation');
    });

    it('should include Vo2Max', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('HKQuantityTypeIdentifierVO2Max');
    });
  });

  describe('SYNC_TIMESTAMP_KEYS', () => {
    it('should have CARDIOVASCULAR key', () => {
      expect(SYNC_TIMESTAMP_KEYS.CARDIOVASCULAR).toBe('healthkit_last_sync_cardiovascular');
    });

    it('should have RESPIRATORY key', () => {
      expect(SYNC_TIMESTAMP_KEYS.RESPIRATORY).toBe('healthkit_last_sync_respiratory');
    });

    it('should have SLEEP key', () => {
      expect(SYNC_TIMESTAMP_KEYS.SLEEP).toBe('healthkit_last_sync_sleep');
    });

    it('should have ACTIVITY key', () => {
      expect(SYNC_TIMESTAMP_KEYS.ACTIVITY).toBe('healthkit_last_sync_activity');
    });
  });
});
