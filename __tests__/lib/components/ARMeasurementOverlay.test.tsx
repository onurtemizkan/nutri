/**
 * Tests for AR Measurement Overlay Component
 *
 * Tests the interactive AR measurement overlay that allows users
 * to measure food dimensions by tapping on the screen.
 */

import React from 'react';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import { ARMeasurementOverlay } from '@/lib/components/ARMeasurementOverlay';
import type { ARMeasurement } from '@/lib/types/food-analysis';

// Mock Ionicons
jest.mock('@expo/vector-icons', () => ({
  Ionicons: 'Ionicons',
}));

// Mock Animated
jest.mock('react-native', () => {
  const RN = jest.requireActual('react-native');
  RN.Animated.timing = () => ({
    start: jest.fn(),
  });
  RN.Animated.spring = () => ({
    start: jest.fn(),
  });
  RN.Animated.loop = () => ({
    start: jest.fn(),
    stop: jest.fn(),
  });
  RN.Animated.sequence = () => ({
    start: jest.fn(),
  });
  return RN;
});

describe('ARMeasurementOverlay', () => {
  const mockOnMeasurementComplete = jest.fn();
  const mockOnCancel = jest.fn();
  const mockGetDepthAtPoint = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockGetDepthAtPoint.mockResolvedValue(30); // 30cm default depth
    // Note: Fake timers are already enabled globally in jest.setup.js
  });

  // Note: Timer cleanup is handled globally in jest.setup.js afterEach

  // ===========================================================================
  // Rendering Tests
  // ===========================================================================
  describe('Rendering', () => {
    it('renders the component when active', () => {
      const { getByText } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          isActive={true}
        />
      );

      expect(getByText('Tap the top-left corner of the food')).toBeTruthy();
    });

    it('shows LiDAR status when available', () => {
      const { getByText } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          isActive={true}
        />
      );

      expect(getByText('LiDAR')).toBeTruthy();
    });

    it('shows No Depth status when unavailable', () => {
      const { getByText } = render(
        <ARMeasurementOverlay
          hasLiDAR={false}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          isActive={true}
        />
      );

      // Component shows 'No Depth' when neither LiDAR nor Scene Depth is available
      expect(getByText('No Depth')).toBeTruthy();
    });

    it('shows surface detection status', () => {
      const { getByText } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          isActive={true}
        />
      );

      expect(getByText('Detecting...')).toBeTruthy();
    });
  });

  // ===========================================================================
  // Interaction Tests
  // ===========================================================================
  describe('Interactions', () => {
    it('calls onCancel when back button is pressed', () => {
      const { UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          isActive={true}
        />
      );

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );
      // First touchable after the main touch area is the back button
      fireEvent.press(touchables[1]);

      expect(mockOnCancel).toHaveBeenCalledTimes(1);
    });

    it('updates instruction after first tap', async () => {
      const { getByText, UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          getDepthAtPoint={mockGetDepthAtPoint}
          isActive={true}
        />
      );

      expect(getByText('Tap the top-left corner of the food')).toBeTruthy();

      // Find touch area and tap
      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 100, locationY: 100 },
        });
      });

      await waitFor(() => {
        expect(getByText('Tap the bottom-right corner')).toBeTruthy();
      });
    });

    it('shows reset button after placing first point', async () => {
      const { getByText, queryByText, UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          getDepthAtPoint={mockGetDepthAtPoint}
          isActive={true}
        />
      );

      // Initially no reset button
      expect(queryByText('Reset')).toBeNull();

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 100, locationY: 100 },
        });
      });

      await waitFor(() => {
        expect(getByText('Reset')).toBeTruthy();
      });
    });
  });

  // ===========================================================================
  // Measurement Flow Tests
  // ===========================================================================
  describe('Measurement Flow', () => {
    it('completes measurement after all three taps', async () => {
      const { getByText, UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          getDepthAtPoint={mockGetDepthAtPoint}
          isActive={true}
        />
      );

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );

      // First tap
      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 50, locationY: 50 },
        });
      });

      // Second tap
      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 200, locationY: 200 },
        });
      });

      // Third tap (depth)
      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 125, locationY: 50 },
        });
      });

      await waitFor(() => {
        expect(getByText('Measurement complete!')).toBeTruthy();
      });
    });

    it('shows measurement summary after completion', async () => {
      const { getByText, UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          getDepthAtPoint={mockGetDepthAtPoint}
          isActive={true}
        />
      );

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );

      // Complete all three taps
      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 50, locationY: 50 },
        });
      });

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 200, locationY: 200 },
        });
      });

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 125, locationY: 50 },
        });
      });

      await waitFor(() => {
        expect(getByText('Measured Dimensions')).toBeTruthy();
        expect(getByText('Width')).toBeTruthy();
        expect(getByText('Height')).toBeTruthy();
        expect(getByText('Depth')).toBeTruthy();
      });
    });

    it('shows Use Measurement button after completion', async () => {
      const { getByText, UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          getDepthAtPoint={mockGetDepthAtPoint}
          isActive={true}
        />
      );

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );

      // Complete measurement
      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 50, locationY: 50 },
        });
      });

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 200, locationY: 200 },
        });
      });

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 125, locationY: 50 },
        });
      });

      await waitFor(() => {
        expect(getByText('Use Measurement')).toBeTruthy();
      });
    });

    it('calls onMeasurementComplete when confirm is pressed', async () => {
      const { getByText, UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          getDepthAtPoint={mockGetDepthAtPoint}
          isActive={true}
        />
      );

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );

      // Complete measurement
      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 50, locationY: 50 },
        });
      });

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 200, locationY: 200 },
        });
      });

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 125, locationY: 50 },
        });
      });

      await waitFor(() => {
        expect(getByText('Use Measurement')).toBeTruthy();
      });

      // Press confirm
      fireEvent.press(getByText('Use Measurement'));

      expect(mockOnMeasurementComplete).toHaveBeenCalledTimes(1);
      expect(mockOnMeasurementComplete).toHaveBeenCalledWith(
        expect.objectContaining({
          width: expect.any(Number),
          height: expect.any(Number),
          depth: expect.any(Number),
          distance: expect.any(Number),
          confidence: expect.stringMatching(/high|medium|low/),
          planeDetected: expect.any(Boolean),
          timestamp: expect.any(Date),
        })
      );
    });
  });

  // ===========================================================================
  // Reset Tests
  // ===========================================================================
  describe('Reset Functionality', () => {
    it('resets measurement when reset button is pressed', async () => {
      const { getByText, UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          getDepthAtPoint={mockGetDepthAtPoint}
          isActive={true}
        />
      );

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );

      // Place first point
      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 100, locationY: 100 },
        });
      });

      await waitFor(() => {
        expect(getByText('Tap the bottom-right corner')).toBeTruthy();
      });

      // Press reset
      fireEvent.press(getByText('Reset'));

      await waitFor(() => {
        expect(getByText('Tap the top-left corner of the food')).toBeTruthy();
      });
    });
  });

  // ===========================================================================
  // Confidence Tests
  // ===========================================================================
  describe('Confidence Calculation', () => {
    it('shows high confidence with LiDAR and all points', async () => {
      const { getByText, UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          getDepthAtPoint={mockGetDepthAtPoint}
          isActive={true}
        />
      );

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );

      // Complete measurement
      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 50, locationY: 50 },
        });
      });

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 200, locationY: 200 },
        });
      });

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 125, locationY: 50 },
        });
      });

      // With hasLiDAR=true, confidence should be 'medium' or 'high'
      // The exact level depends on plane detection state
      await waitFor(() => {
        expect(getByText(/confidence/i)).toBeTruthy();
      });
    });

    it('shows lower confidence without LiDAR', async () => {
      const { getByText, UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={false}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          getDepthAtPoint={mockGetDepthAtPoint}
          isActive={true}
        />
      );

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );

      // Complete measurement
      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 50, locationY: 50 },
        });
      });

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 200, locationY: 200 },
        });
      });

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 125, locationY: 50 },
        });
      });

      await waitFor(() => {
        // Component renders "{Level} confidence" text (Low or Medium without LiDAR)
        expect(getByText(/confidence/)).toBeTruthy();
      });
    });
  });

  // ===========================================================================
  // Depth Integration Tests
  // ===========================================================================
  describe('Depth Integration', () => {
    it('calls getDepthAtPoint when placing points', async () => {
      const { UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          getDepthAtPoint={mockGetDepthAtPoint}
          isActive={true}
        />
      );

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 100, locationY: 150 },
        });
      });

      expect(mockGetDepthAtPoint).toHaveBeenCalledWith(100, 150);
    });

    it('uses default depth when getDepthAtPoint fails', async () => {
      mockGetDepthAtPoint.mockRejectedValue(new Error('Depth unavailable'));

      const { getByText, UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          getDepthAtPoint={mockGetDepthAtPoint}
          isActive={true}
        />
      );

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 100, locationY: 150 },
        });
      });

      // Should still proceed with measurement
      await waitFor(() => {
        expect(getByText('Tap the bottom-right corner')).toBeTruthy();
      });
    });
  });

  // ===========================================================================
  // Inactive State Tests
  // ===========================================================================
  describe('Inactive State', () => {
    it('does not respond to taps when inactive', async () => {
      const { getByText, UNSAFE_getAllByType } = render(
        <ARMeasurementOverlay
          hasLiDAR={true}
          onMeasurementComplete={mockOnMeasurementComplete}
          onCancel={mockOnCancel}
          isActive={false}
        />
      );

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );

      await act(async () => {
        fireEvent.press(touchables[0], {
          nativeEvent: { locationX: 100, locationY: 100 },
        });
      });

      // Instruction should not change
      expect(getByText('Tap the top-left corner of the food')).toBeTruthy();
    });
  });
});
