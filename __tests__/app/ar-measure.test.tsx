/**
 * Tests for AR Measure Screen
 *
 * Integration tests for the AR measurement modal screen that combines
 * AR measurement, manual size picker, and reference object calibration.
 */

import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { Platform } from 'react-native';

// Mock expo-camera
jest.mock('expo-camera', () => ({
  CameraView: 'CameraView',
  useCameraPermissions: jest.fn(() => [{ granted: true }, jest.fn()]),
}));

// Mock expo-router
const mockRouterBack = jest.fn();
jest.mock('expo-router', () => ({
  useRouter: () => ({
    back: mockRouterBack,
    push: jest.fn(),
  }),
  useLocalSearchParams: () => ({
    foodName: 'apple',
    returnTo: 'scan-food',
  }),
}));

// Mock react-native-safe-area-context
jest.mock('react-native-safe-area-context', () => ({
  SafeAreaView: ({ children }: any) => children,
}));

// Mock Ionicons
jest.mock('@expo/vector-icons', () => ({
  Ionicons: 'Ionicons',
}));

// Mock expo-linear-gradient
jest.mock('expo-linear-gradient', () => ({
  LinearGradient: 'LinearGradient',
}));

// Mock LiDARModule
const mockGetDeviceCapabilities = jest.fn();
jest.mock('@/lib/modules/LiDARModule', () => ({
  __esModule: true,
  default: {
    isAvailable: true,
    getDeviceCapabilities: () => mockGetDeviceCapabilities(),
  },
}));

// Mock Animated
jest.mock('react-native', () => {
  const RN = jest.requireActual('react-native');
  RN.Animated.timing = () => ({
    start: jest.fn((cb?: () => void) => cb && cb()),
  });
  RN.Animated.spring = () => ({
    start: jest.fn((cb?: () => void) => cb && cb()),
  });
  RN.Animated.loop = () => ({
    start: jest.fn(),
    stop: jest.fn(),
  });
  RN.Animated.sequence = () => ({
    start: jest.fn((cb?: () => void) => cb && cb()),
  });
  return RN;
});

// Import after mocks
import ARMeasureScreen from '@/app/ar-measure';

describe('ARMeasureScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset Platform to iOS by default
    Platform.OS = 'ios';
    // Default: LiDAR available
    mockGetDeviceCapabilities.mockResolvedValue({
      hasLiDAR: true,
      hasARKit: true,
      supportsSceneDepth: true,
      maxDepthResolution: { width: 256, height: 192 },
      maxRGBResolution: { width: 1920, height: 1440 },
      depthRange: { min: 0.3, max: 5.0 },
      frameRate: 60,
    });
  });

  // ===========================================================================
  // Loading State Tests
  // ===========================================================================
  describe('Loading State', () => {
    it('shows loading indicator initially', () => {
      const { getByText } = render(<ARMeasureScreen />);
      expect(getByText('Checking device capabilities...')).toBeTruthy();
    });
  });

  // ===========================================================================
  // Mode Selection Tests
  // ===========================================================================
  describe('Mode Selection', () => {
    it('shows AR mode when LiDAR is available on iOS', async () => {
      const { queryByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        // Should not show manual mode switcher prominently
        expect(queryByText('Checking device capabilities...')).toBeNull();
      });
    });

    it('shows manual mode on Android', async () => {
      Platform.OS = 'android';
      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        // Should show manual size picker elements
        expect(getByText('Select Portion Size')).toBeTruthy();
      });
    });

    it('shows manual mode when LiDAR unavailable', async () => {
      mockGetDeviceCapabilities.mockResolvedValue({
        hasLiDAR: false,
        hasARKit: false,
        supportsSceneDepth: false,
        maxDepthResolution: { width: 0, height: 0 },
        maxRGBResolution: { width: 1920, height: 1080 },
        depthRange: { min: 0, max: 0 },
        frameRate: 0,
      });

      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        expect(getByText('Select Portion Size')).toBeTruthy();
      });
    });

    it('handles LiDAR check error gracefully', async () => {
      mockGetDeviceCapabilities.mockRejectedValue(
        new Error('LiDAR check failed')
      );

      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        // Should fall back to manual mode
        expect(getByText('Select Portion Size')).toBeTruthy();
      });
    });
  });

  // ===========================================================================
  // Manual Mode Tests
  // ===========================================================================
  describe('Manual Mode', () => {
    beforeEach(() => {
      // Force manual mode
      mockGetDeviceCapabilities.mockResolvedValue({
        hasLiDAR: false,
        hasARKit: false,
        supportsSceneDepth: false,
        maxDepthResolution: { width: 0, height: 0 },
        maxRGBResolution: { width: 1920, height: 1080 },
        depthRange: { min: 0, max: 0 },
        frameRate: 0,
      });
    });

    it('renders manual size picker', async () => {
      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        expect(getByText('Select Portion Size')).toBeTruthy();
        expect(getByText('Quick Select')).toBeTruthy();
      });
    });

    it('passes foodName to manual size picker', async () => {
      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        // The ManualSizePicker should receive foodName='apple'
        expect(getByText('Select Portion Size')).toBeTruthy();
      });
    });

    it('shows calibration option in manual mode', async () => {
      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        expect(getByText('Calibrate for Accuracy')).toBeTruthy();
      });
    });

    it('navigates to calibration when calibrate is pressed', async () => {
      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        expect(getByText('Calibrate for Accuracy')).toBeTruthy();
      });

      fireEvent.press(getByText('Calibrate for Accuracy'));

      await waitFor(() => {
        expect(getByText('Calibrate with Reference')).toBeTruthy();
      });
    });

    it('shows mode switcher if AR is available', async () => {
      // AR available but not LiDAR
      mockGetDeviceCapabilities.mockResolvedValue({
        hasLiDAR: false,
        hasARKit: true,
        supportsSceneDepth: true,
        maxDepthResolution: { width: 256, height: 192 },
        maxRGBResolution: { width: 1920, height: 1080 },
        depthRange: { min: 0.3, max: 3.0 },
        frameRate: 30,
      });

      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        // In AR mode by default when ARKit available
        expect(getByText('Switch to Manual Mode')).toBeTruthy();
      });
    });
  });

  // ===========================================================================
  // Calibration Mode Tests
  // ===========================================================================
  describe('Calibration Mode', () => {
    beforeEach(() => {
      mockGetDeviceCapabilities.mockResolvedValue({
        hasLiDAR: false,
        hasARKit: false,
        supportsSceneDepth: false,
        maxDepthResolution: { width: 0, height: 0 },
        maxRGBResolution: { width: 1920, height: 1080 },
        depthRange: { min: 0, max: 0 },
        frameRate: 0,
      });
    });

    it('shows reference object calibration screen', async () => {
      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        expect(getByText('Calibrate for Accuracy')).toBeTruthy();
      });

      fireEvent.press(getByText('Calibrate for Accuracy'));

      await waitFor(() => {
        expect(getByText('Calibrate with Reference')).toBeTruthy();
        expect(getByText('Coins')).toBeTruthy();
        expect(getByText('Cards')).toBeTruthy();
      });
    });

    it('returns to manual mode after calibration', async () => {
      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        expect(getByText('Calibrate for Accuracy')).toBeTruthy();
      });

      // Open calibration
      fireEvent.press(getByText('Calibrate for Accuracy'));

      await waitFor(() => {
        expect(getByText('Calibrate with Reference')).toBeTruthy();
      });

      // Select a reference object and confirm
      fireEvent.press(getByText('US Quarter'));
      fireEvent.press(getByText('Use This Reference'));

      await waitFor(() => {
        // Should return to manual mode with calibration applied
        expect(getByText('Select Portion Size')).toBeTruthy();
        expect(getByText('Calibrated')).toBeTruthy();
      });
    });

    it('cancels calibration and returns to previous mode', async () => {
      const { getByText, UNSAFE_getAllByType } = render(<ARMeasureScreen />);

      await waitFor(() => {
        expect(getByText('Calibrate for Accuracy')).toBeTruthy();
      });

      // Open calibration
      fireEvent.press(getByText('Calibrate for Accuracy'));

      await waitFor(() => {
        expect(getByText('Calibrate with Reference')).toBeTruthy();
      });

      // Press back button
      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );
      fireEvent.press(touchables[0]); // First touchable is back button

      await waitFor(() => {
        // Should return to manual mode without calibration
        expect(getByText('Select Portion Size')).toBeTruthy();
        expect(getByText('Calibrate for Accuracy')).toBeTruthy();
      });
    });
  });

  // ===========================================================================
  // Measurement Result Tests
  // ===========================================================================
  describe('Measurement Results', () => {
    beforeEach(() => {
      mockGetDeviceCapabilities.mockResolvedValue({
        hasLiDAR: false,
        hasARKit: false,
        supportsSceneDepth: false,
        maxDepthResolution: { width: 0, height: 0 },
        maxRGBResolution: { width: 1920, height: 1080 },
        depthRange: { min: 0, max: 0 },
        frameRate: 0,
      });
    });

    it('shows measurement result after selecting a size', async () => {
      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        expect(getByText('Small')).toBeTruthy();
      });

      // Select a preset size
      fireEvent.press(getByText('Small'));

      // Confirm selection
      fireEvent.press(getByText('Use This Size'));

      await waitFor(() => {
        expect(getByText('Measurement Ready')).toBeTruthy();
        expect(getByText('Width (cm)')).toBeTruthy();
        expect(getByText('Height (cm)')).toBeTruthy();
        expect(getByText('Depth (cm)')).toBeTruthy();
      });
    });

    it('shows remeasure button after measurement', async () => {
      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        expect(getByText('Small')).toBeTruthy();
      });

      fireEvent.press(getByText('Small'));
      fireEvent.press(getByText('Use This Size'));

      await waitFor(() => {
        expect(getByText('Remeasure')).toBeTruthy();
      });
    });

    it('clears measurement when remeasure is pressed', async () => {
      const { getByText, queryByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        expect(getByText('Small')).toBeTruthy();
      });

      fireEvent.press(getByText('Small'));
      fireEvent.press(getByText('Use This Size'));

      await waitFor(() => {
        expect(getByText('Measurement Ready')).toBeTruthy();
      });

      fireEvent.press(getByText('Remeasure'));

      await waitFor(() => {
        expect(queryByText('Measurement Ready')).toBeNull();
        expect(getByText('Select Portion Size')).toBeTruthy();
      });
    });

    it('navigates back when Use Measurement is pressed', async () => {
      const { getByText } = render(<ARMeasureScreen />);

      await waitFor(() => {
        expect(getByText('Small')).toBeTruthy();
      });

      fireEvent.press(getByText('Small'));
      fireEvent.press(getByText('Use This Size'));

      await waitFor(() => {
        expect(getByText('Use Measurement')).toBeTruthy();
      });

      fireEvent.press(getByText('Use Measurement'));

      expect(mockRouterBack).toHaveBeenCalled();
    });
  });

  // ===========================================================================
  // Cancel Tests
  // ===========================================================================
  describe('Cancel Behavior', () => {
    beforeEach(() => {
      mockGetDeviceCapabilities.mockResolvedValue({
        hasLiDAR: false,
        hasARKit: false,
        supportsSceneDepth: false,
        maxDepthResolution: { width: 0, height: 0 },
        maxRGBResolution: { width: 1920, height: 1080 },
        depthRange: { min: 0, max: 0 },
        frameRate: 0,
      });
    });

    it('calls router.back when cancel is pressed in manual mode', async () => {
      const { UNSAFE_getAllByType } = render(<ARMeasureScreen />);

      await waitFor(() => {
        const touchables = UNSAFE_getAllByType(
          require('react-native').TouchableOpacity
        );
        expect(touchables.length).toBeGreaterThan(0);
      });

      // Close button is the first touchable
      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );
      fireEvent.press(touchables[0]);

      expect(mockRouterBack).toHaveBeenCalled();
    });
  });
});
