module.exports = {
  preset: 'jest-expo',
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  transformIgnorePatterns: [
    'node_modules/(?!((jest-)?react-native|@react-native(-community)?)|expo(nent)?|@expo(nent)?/.*|@expo-google-fonts/.*|react-navigation|@react-navigation/.*|@unimodules/.*|unimodules|sentry-expo|native-base|react-native-svg)',
  ],
  testPathIgnorePatterns: [
    '/node_modules/',
    '/server/',
    '/.expo/',
    '/dist/',
    'test-utils\\.ts$',
  ],
  moduleNameMapper: {
    // Mock native HealthKit modules that aren't installed in test environment
    '^react-native-health$': '<rootDir>/__mocks__/react-native-health.ts',
    '^@kingstinct/react-native-healthkit$':
      '<rootDir>/__mocks__/@kingstinct/react-native-healthkit.ts',
  },
  collectCoverageFrom: [
    'app/**/*.{ts,tsx}',
    'lib/**/*.{ts,tsx}',
    'components/**/*.{ts,tsx}',
    '!**/*.d.ts',
    '!**/node_modules/**',
    '!**/server/**',
  ],
};
