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
    it('should map RestingHeartRate to RESTING_HEART_RATE', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['RestingHeartRate']).toBe('RESTING_HEART_RATE');
    });

    it('should map HeartRateVariabilitySDNN to HEART_RATE_VARIABILITY_SDNN', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['HeartRateVariabilitySDNN']).toBe(
        'HEART_RATE_VARIABILITY_SDNN'
      );
    });

    it('should map RespiratoryRate to RESPIRATORY_RATE', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['RespiratoryRate']).toBe('RESPIRATORY_RATE');
    });

    it('should map OxygenSaturation to OXYGEN_SATURATION', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['OxygenSaturation']).toBe('OXYGEN_SATURATION');
    });

    it('should map Vo2Max to VO2_MAX', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['Vo2Max']).toBe('VO2_MAX');
    });

    it('should map StepCount to STEPS', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['StepCount']).toBe('STEPS');
    });

    it('should map ActiveEnergyBurned to ACTIVE_CALORIES', () => {
      expect(HEALTHKIT_TO_METRIC_TYPE['ActiveEnergyBurned']).toBe('ACTIVE_CALORIES');
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
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('HeartRate');
    });

    it('should include RestingHeartRate', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('RestingHeartRate');
    });

    it('should include HeartRateVariabilitySDNN', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('HeartRateVariabilitySDNN');
    });

    it('should include SleepAnalysis', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('SleepAnalysis');
    });

    it('should include StepCount', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('StepCount');
    });

    it('should include ActiveEnergyBurned', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('ActiveEnergyBurned');
    });

    it('should include OxygenSaturation', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('OxygenSaturation');
    });

    it('should include Vo2Max', () => {
      expect(HEALTHKIT_READ_PERMISSIONS).toContain('Vo2Max');
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
