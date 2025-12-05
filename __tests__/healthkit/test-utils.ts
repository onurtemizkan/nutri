/**
 * HealthKit Test Utilities
 * Mocks and helper functions for HealthKit tests
 */

import { HealthKitSample, SleepSample } from '@/lib/types/healthkit';

// Mock SecureStore
export const mockSecureStore = {
  getItemAsync: jest.fn(),
  setItemAsync: jest.fn(),
  deleteItemAsync: jest.fn(),
};

jest.mock('expo-secure-store', () => mockSecureStore);

// Mock react-native Platform
export const mockPlatform = {
  OS: 'ios' as 'ios' | 'android' | 'web',
  select: jest.fn((options: { ios?: unknown; android?: unknown }) => options.ios),
};

jest.mock('react-native', () => ({
  Platform: mockPlatform,
}));

// Error type that matches react-native-health behavior
type HealthKitError = string | { message?: string } | null;

// Mock HealthKit functions
export const mockHealthKit = {
  isAvailable: jest.fn((callback: (error: string | null, available: boolean) => void) =>
    callback(null, true)
  ),
  initHealthKit: jest.fn(
    (
      options: { permissions: { read: string[]; write: string[] } },
      callback: (error: HealthKitError) => void
    ) => callback(null)
  ),
  getRestingHeartRateSamples: jest.fn(),
  getHeartRateSamples: jest.fn(),
  getHeartRateVariabilitySamples: jest.fn(),
  getRespiratoryRateSamples: jest.fn(),
  getOxygenSaturationSamples: jest.fn(),
  getVo2MaxSamples: jest.fn(),
  getSleepSamples: jest.fn(),
  getStepCount: jest.fn(),
  getDailyStepCountSamples: jest.fn(),
  getActiveEnergyBurned: jest.fn(),
  getBasalEnergyBurned: jest.fn(),
  getDailyDistanceWalkingRunningSamples: jest.fn(),
};

jest.mock('react-native-health', () => ({
  default: mockHealthKit,
}));

// Sample data generators
export function createMockHeartRateSample(
  value: number = 65,
  date: Date = new Date()
): HealthKitSample {
  return {
    value,
    startDate: date.toISOString(),
    endDate: date.toISOString(),
    sourceName: 'Apple Watch',
    id: `hr_${Date.now()}_${Math.random()}`,
  };
}

export function createMockRestingHeartRateSample(
  value: number = 58,
  date: Date = new Date()
): HealthKitSample {
  return {
    value,
    startDate: date.toISOString(),
    endDate: date.toISOString(),
    sourceName: 'Apple Watch',
    id: `rhr_${Date.now()}_${Math.random()}`,
  };
}

export function createMockHRVSample(value: number = 45, date: Date = new Date()): HealthKitSample {
  return {
    value,
    startDate: date.toISOString(),
    endDate: date.toISOString(),
    sourceName: 'Apple Watch',
    id: `hrv_${Date.now()}_${Math.random()}`,
  };
}

export function createMockSleepSample(
  value: string = 'DEEP',
  startDate: Date = new Date(),
  durationHours: number = 1
): SleepSample {
  const endDate = new Date(startDate.getTime() + durationHours * 60 * 60 * 1000);
  return {
    value: value as 'DEEP' | 'REM' | 'CORE' | 'AWAKE' | 'INBED' | 'ASLEEP',
    startDate: startDate.toISOString(),
    endDate: endDate.toISOString(),
    sourceName: 'Apple Watch',
    id: `sleep_${Date.now()}_${Math.random()}`,
  };
}

export function createMockStepCountSample(
  value: number = 8500,
  date: Date = new Date()
): { value: number; startDate: string; endDate: string } {
  const startOfDay = new Date(date);
  startOfDay.setHours(0, 0, 0, 0);
  const endOfDay = new Date(date);
  endOfDay.setHours(23, 59, 59, 999);
  return {
    value,
    startDate: startOfDay.toISOString(),
    endDate: endOfDay.toISOString(),
  };
}

export function createMockCaloriesSample(
  value: number = 350,
  date: Date = new Date()
): HealthKitSample {
  return {
    value,
    startDate: date.toISOString(),
    endDate: date.toISOString(),
    sourceName: 'Apple Watch',
    id: `cal_${Date.now()}_${Math.random()}`,
  };
}

// Reset all mocks
export function resetMocks(): void {
  Object.values(mockSecureStore).forEach((fn) => {
    if (typeof fn === 'function' && 'mockReset' in fn) {
      (fn as jest.Mock).mockReset();
    }
  });
  Object.values(mockHealthKit).forEach((fn) => {
    if (typeof fn === 'function' && 'mockReset' in fn) {
      (fn as jest.Mock).mockReset();
    }
  });
}

// Setup default mock implementations
export function setupDefaultMocks(): void {
  mockHealthKit.isAvailable.mockImplementation(
    (callback: (error: string | null, available: boolean) => void) => callback(null, true)
  );
  mockHealthKit.initHealthKit.mockImplementation(
    (
      _options: { permissions: { read: string[]; write: string[] } },
      callback: (error: string | null) => void
    ) => callback(null)
  );

  mockSecureStore.getItemAsync.mockResolvedValue(null);
  mockSecureStore.setItemAsync.mockResolvedValue(undefined);
  mockSecureStore.deleteItemAsync.mockResolvedValue(undefined);
}
