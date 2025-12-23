/**
 * Mock for @kingstinct/react-native-healthkit module
 * Used in jest tests when the native module isn't available
 *
 * This mock provides a promise-based API matching @kingstinct/react-native-healthkit
 */

// Mock functions that can be controlled in tests
export const isHealthDataAvailable = jest.fn(() => true);

export const requestAuthorization = jest.fn(async () => true);

// Query functions used by the HealthKit services
export const queryQuantitySamples = jest.fn(async () => []);

export const queryCategorySamples = jest.fn(async () => []);

export const getQuantitySamples = jest.fn(async () => []);

export const getCategorySamples = jest.fn(async () => []);

export const getStatisticsCollection = jest.fn(async () => []);

// HealthKit data types - commonly used identifiers
export const HKQuantityTypeIdentifier = {
  heartRate: 'HKQuantityTypeIdentifierHeartRate',
  restingHeartRate: 'HKQuantityTypeIdentifierRestingHeartRate',
  heartRateVariabilitySDNN: 'HKQuantityTypeIdentifierHeartRateVariabilitySDNN',
  oxygenSaturation: 'HKQuantityTypeIdentifierOxygenSaturation',
  vo2Max: 'HKQuantityTypeIdentifierVO2Max',
  respiratoryRate: 'HKQuantityTypeIdentifierRespiratoryRate',
  stepCount: 'HKQuantityTypeIdentifierStepCount',
  activeEnergyBurned: 'HKQuantityTypeIdentifierActiveEnergyBurned',
  basalEnergyBurned: 'HKQuantityTypeIdentifierBasalEnergyBurned',
  distanceWalkingRunning: 'HKQuantityTypeIdentifierDistanceWalkingRunning',
  flightsClimbed: 'HKQuantityTypeIdentifierFlightsClimbed',
};

export const HKCategoryTypeIdentifier = {
  sleepAnalysis: 'HKCategoryTypeIdentifierSleepAnalysis',
};

// Export default as well for various import styles
export default {
  isHealthDataAvailable,
  requestAuthorization,
  queryQuantitySamples,
  queryCategorySamples,
  getQuantitySamples,
  getCategorySamples,
  getStatisticsCollection,
  HKQuantityTypeIdentifier,
  HKCategoryTypeIdentifier,
};
