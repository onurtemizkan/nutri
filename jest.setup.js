/**
 * Jest Setup File
 * Global mocks and configurations for all tests
 */

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
