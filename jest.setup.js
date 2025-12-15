/**
 * Jest Setup File
 * Global mocks and configurations for all tests
 */

// Use fake timers to prevent cleanup errors from timers firing after test teardown
// This prevents the "importing after Jest environment torn down" errors
jest.useFakeTimers();

// Run pending timers and clean up after each test
afterEach(() => {
  // Run any pending timers to completion
  jest.runOnlyPendingTimers();
  // Clear any remaining timers
  jest.clearAllTimers();
});

// Mock react-native with DeviceEventEmitter and enhanced Animated
jest.mock('react-native', () => {
  const RN = jest.requireActual('react-native');

  // Mock DeviceEventEmitter
  RN.DeviceEventEmitter = {
    emit: jest.fn(),
    addListener: jest.fn(() => ({ remove: jest.fn() })),
    removeListener: jest.fn(),
    removeAllListeners: jest.fn(),
    removeSubscription: jest.fn(),
    listeners: jest.fn(() => []),
  };

  // Enhanced Animated mocks for tests that use animations
  RN.Animated.timing = () => ({
    start: jest.fn((cb) => cb && cb()),
  });
  RN.Animated.spring = () => ({
    start: jest.fn((cb) => cb && cb()),
  });
  RN.Animated.loop = () => ({
    start: jest.fn(),
    stop: jest.fn(),
  });
  RN.Animated.sequence = () => ({
    start: jest.fn((cb) => cb && cb()),
  });
  RN.Animated.parallel = () => ({
    start: jest.fn((cb) => cb && cb()),
  });

  return RN;
});
