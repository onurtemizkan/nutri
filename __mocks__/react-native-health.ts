/**
 * Mock for react-native-health module
 * Used in jest tests when the native module isn't available
 */

const mockHealthKit = {
  initHealthKit: jest.fn().mockImplementation((permissions, callback) => {
    callback(null, true);
  }),
  isAvailable: jest.fn().mockResolvedValue(true),
  getAuthorizationStatus: jest.fn().mockResolvedValue({
    permissions: {
      read: [],
      write: [],
    },
  }),
  initStepCountObserver: jest.fn(),
  getStepCount: jest.fn().mockResolvedValue({ value: 0 }),
  getActiveEnergyBurned: jest.fn().mockResolvedValue({ value: 0 }),
  getBasalEnergyBurned: jest.fn().mockResolvedValue({ value: 0 }),
  getDailyDistanceWalkingRunningSamples: jest.fn().mockResolvedValue([]),
  getHeartRateSamples: jest.fn().mockResolvedValue([]),
  getRestingHeartRate: jest.fn().mockResolvedValue([]),
  getHeartRateVariabilitySamples: jest.fn().mockResolvedValue([]),
  getOxygenSaturationSamples: jest.fn().mockResolvedValue([]),
  getVo2MaxSamples: jest.fn().mockResolvedValue([]),
  getSleepSamples: jest.fn().mockResolvedValue([]),
  getRespiratoryRateSamples: jest.fn().mockResolvedValue([]),
  getFlightsClimbed: jest.fn().mockResolvedValue({ value: 0 }),
  Constants: {
    Permissions: {
      StepCount: 'HKQuantityTypeIdentifierStepCount',
      HeartRate: 'HKQuantityTypeIdentifierHeartRate',
      RestingHeartRate: 'HKQuantityTypeIdentifierRestingHeartRate',
      HeartRateVariabilitySDNN: 'HKQuantityTypeIdentifierHeartRateVariabilitySDNN',
      OxygenSaturation: 'HKQuantityTypeIdentifierOxygenSaturation',
      Vo2Max: 'HKQuantityTypeIdentifierVO2Max',
      SleepAnalysis: 'HKCategoryTypeIdentifierSleepAnalysis',
      RespiratoryRate: 'HKQuantityTypeIdentifierRespiratoryRate',
      ActiveEnergyBurned: 'HKQuantityTypeIdentifierActiveEnergyBurned',
      BasalEnergyBurned: 'HKQuantityTypeIdentifierBasalEnergyBurned',
      DistanceWalkingRunning: 'HKQuantityTypeIdentifierDistanceWalkingRunning',
      FlightsClimbed: 'HKQuantityTypeIdentifierFlightsClimbed',
    },
  },
};

export default mockHealthKit;
