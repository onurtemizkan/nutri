/**
 * Tests for Manual Size Picker Component
 *
 * This component provides fallback UI for devices without AR/LiDAR support,
 * allowing users to manually select portion sizes.
 */

import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { ManualSizePicker } from '@/lib/components/ManualSizePicker';
import {
  PRESET_SIZES,
  presetToMeasurement,
} from '@/lib/utils/portion-estimation';
import type { ARMeasurement } from '@/lib/types/food-analysis';

// Mock Ionicons
jest.mock('@expo/vector-icons', () => ({
  Ionicons: 'Ionicons',
}));

describe('ManualSizePicker', () => {
  const mockOnSelect = jest.fn();
  const mockOnCancel = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ===========================================================================
  // Rendering Tests
  // ===========================================================================
  describe('Rendering', () => {
    it('renders the component with title', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      expect(getByText('Select Portion Size')).toBeTruthy();
    });

    it('renders info banner', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      expect(
        getByText(/Select a size that best matches your food portion/)
      ).toBeTruthy();
    });

    it('renders Quick Select section title', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      expect(getByText('Quick Select')).toBeTruthy();
    });

    it('renders Custom Size section title', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      expect(getByText('Custom Size')).toBeTruthy();
    });

    it('renders all preset size options', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      PRESET_SIZES.forEach((preset) => {
        expect(getByText(preset.displayName)).toBeTruthy();
      });
    });

    it('renders custom dimension toggle', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      expect(getByText('Enter custom dimensions')).toBeTruthy();
    });

    it('renders Use This Size button', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      expect(getByText('Use This Size')).toBeTruthy();
    });
  });

  // ===========================================================================
  // Preset Selection Tests
  // ===========================================================================
  describe('Preset Selection', () => {
    it('selects a preset when tapped', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      const mediumPreset = getByText('Medium');
      fireEvent.press(mediumPreset);

      // After selection, summary should appear
      expect(getByText('Selected Size')).toBeTruthy();
    });

    it('shows summary with dimensions after preset selection', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      const smallPreset = getByText('Small');
      fireEvent.press(smallPreset);

      expect(getByText('Dimensions:')).toBeTruthy();
      expect(getByText('Volume:')).toBeTruthy();
      expect(getByText('Est. Weight:')).toBeTruthy();
    });

    it('shows initial selection if provided', () => {
      const { getByText } = render(
        <ManualSizePicker
          onSelect={mockOnSelect}
          onCancel={mockOnCancel}
          initialSize="large"
        />
      );

      // Summary should be visible with initial selection
      expect(getByText('Selected Size')).toBeTruthy();
    });

    it('displays reference object for presets', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      // Check that at least one reference object is displayed
      const mediumPreset = PRESET_SIZES.find((p) => p.name === 'medium');
      if (mediumPreset) {
        expect(getByText(mediumPreset.referenceObject)).toBeTruthy();
      }
    });
  });

  // ===========================================================================
  // Custom Dimension Mode Tests
  // ===========================================================================
  describe('Custom Dimension Mode', () => {
    it('enters custom mode when toggle is pressed', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      const customToggle = getByText('Enter custom dimensions');
      fireEvent.press(customToggle);

      // Sliders should appear
      expect(getByText('Width')).toBeTruthy();
      expect(getByText('Height')).toBeTruthy();
      expect(getByText('Depth')).toBeTruthy();
    });

    it('shows dimension values in cm', () => {
      const { getByText, getAllByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      const customToggle = getByText('Enter custom dimensions');
      fireEvent.press(customToggle);

      // Default values should be displayed (width and height both 8cm, depth 6cm)
      expect(getAllByText('8 cm').length).toBe(2); // Width and Height
      expect(getByText('6 cm')).toBeTruthy(); // Depth
    });

    it('clears preset selection when entering custom mode', () => {
      const { getByText, queryByText } = render(
        <ManualSizePicker
          onSelect={mockOnSelect}
          onCancel={mockOnCancel}
          initialSize="small"
        />
      );

      // Initial preset is selected
      expect(getByText('Selected Size')).toBeTruthy();

      // Enter custom mode
      const customToggle = getByText('Enter custom dimensions');
      fireEvent.press(customToggle);

      // Summary should still show (with custom dimensions)
      expect(getByText('Selected Size')).toBeTruthy();
    });
  });

  // ===========================================================================
  // Callback Tests
  // ===========================================================================
  describe('Callbacks', () => {
    it('calls onCancel when close button is pressed', () => {
      const { getByTestId, UNSAFE_getAllByType } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      // Find TouchableOpacity components - the first one after header should be close
      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );
      // First touchable is the close button
      fireEvent.press(touchables[0]);

      expect(mockOnCancel).toHaveBeenCalledTimes(1);
    });

    it('calls onSelect with measurement when confirm is pressed', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      // Select a preset first
      const mediumPreset = getByText('Medium');
      fireEvent.press(mediumPreset);

      // Press confirm
      const confirmButton = getByText('Use This Size');
      fireEvent.press(confirmButton);

      expect(mockOnSelect).toHaveBeenCalledTimes(1);
      expect(mockOnSelect).toHaveBeenCalledWith(
        expect.objectContaining({
          width: expect.any(Number),
          height: expect.any(Number),
          depth: expect.any(Number),
          confidence: 'low',
          planeDetected: false,
        })
      );
    });

    it('passes correct preset dimensions to onSelect', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      const smallPreset = PRESET_SIZES.find((p) => p.name === 'small')!;

      // Select small preset
      const smallButton = getByText('Small');
      fireEvent.press(smallButton);

      // Confirm
      const confirmButton = getByText('Use This Size');
      fireEvent.press(confirmButton);

      const calledMeasurement = mockOnSelect.mock.calls[0][0] as ARMeasurement;
      expect(calledMeasurement.width).toBe(smallPreset.width);
      expect(calledMeasurement.height).toBe(smallPreset.height);
      expect(calledMeasurement.depth).toBe(smallPreset.depth);
    });

    it('does not call onSelect when no selection is made', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      // Try to press confirm without selection
      const confirmButton = getByText('Use This Size');
      fireEvent.press(confirmButton);

      // Should not be called since button should be disabled
      expect(mockOnSelect).not.toHaveBeenCalled();
    });
  });

  // ===========================================================================
  // Weight Estimation Tests
  // ===========================================================================
  describe('Weight Estimation', () => {
    it('shows weight estimate for known food', () => {
      const { getByText } = render(
        <ManualSizePicker
          foodName="apple"
          onSelect={mockOnSelect}
          onCancel={mockOnCancel}
        />
      );

      // Select a preset
      const mediumPreset = getByText('Medium');
      fireEvent.press(mediumPreset);

      // Weight estimate should appear
      expect(getByText('Est. Weight:')).toBeTruthy();
    });

    it('shows weight estimate for unknown food', () => {
      const { getByText } = render(
        <ManualSizePicker
          foodName="mystery food"
          onSelect={mockOnSelect}
          onCancel={mockOnCancel}
        />
      );

      // Select a preset
      const mediumPreset = getByText('Medium');
      fireEvent.press(mediumPreset);

      // Weight estimate should still appear (using generic default)
      expect(getByText('Est. Weight:')).toBeTruthy();
    });

    it('shows confidence warning', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      // Select a preset
      const mediumPreset = getByText('Medium');
      fireEvent.press(mediumPreset);

      // Confidence warning should appear
      expect(
        getByText('Manual estimate - lower accuracy than AR measurement')
      ).toBeTruthy();
    });
  });

  // ===========================================================================
  // Button State Tests
  // ===========================================================================
  describe('Button States', () => {
    it('confirm button is disabled when no selection', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      const confirmButton = getByText('Use This Size');

      // Press should not trigger callback
      fireEvent.press(confirmButton);
      expect(mockOnSelect).not.toHaveBeenCalled();
    });

    it('confirm button is enabled after selection', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      // Select preset
      fireEvent.press(getByText('Medium'));

      // Now confirm should work
      fireEvent.press(getByText('Use This Size'));
      expect(mockOnSelect).toHaveBeenCalled();
    });
  });

  // ===========================================================================
  // Integration Tests
  // ===========================================================================
  describe('Integration', () => {
    it('switches between presets correctly', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      // Select small
      fireEvent.press(getByText('Small'));
      expect(getByText('Selected Size')).toBeTruthy();

      // Switch to large
      fireEvent.press(getByText('Large'));
      expect(getByText('Selected Size')).toBeTruthy();

      // Confirm and check it's the large preset
      fireEvent.press(getByText('Use This Size'));

      const largePreset = PRESET_SIZES.find((p) => p.name === 'large')!;
      const measurement = mockOnSelect.mock.calls[0][0] as ARMeasurement;
      expect(measurement.width).toBe(largePreset.width);
    });

    it('switches from preset to custom mode', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      // Select preset first
      fireEvent.press(getByText('Medium'));

      // Switch to custom
      fireEvent.press(getByText('Enter custom dimensions'));

      // Confirm sends custom dimensions (default: 8x8x6)
      fireEvent.press(getByText('Use This Size'));

      const measurement = mockOnSelect.mock.calls[0][0] as ARMeasurement;
      expect(measurement.width).toBe(8);
      expect(measurement.height).toBe(8);
      expect(measurement.depth).toBe(6);
    });

    it('switches from custom mode back to preset', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      // Enter custom mode
      fireEvent.press(getByText('Enter custom dimensions'));

      // Switch back to preset
      fireEvent.press(getByText('Small'));

      // Confirm sends preset dimensions
      fireEvent.press(getByText('Use This Size'));

      const smallPreset = PRESET_SIZES.find((p) => p.name === 'small')!;
      const measurement = mockOnSelect.mock.calls[0][0] as ARMeasurement;
      expect(measurement.width).toBe(smallPreset.width);
    });
  });

  // ===========================================================================
  // Measurement Output Tests
  // ===========================================================================
  describe('Measurement Output', () => {
    it('returns measurement with low confidence for presets', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      fireEvent.press(getByText('Medium'));
      fireEvent.press(getByText('Use This Size'));

      const measurement = mockOnSelect.mock.calls[0][0] as ARMeasurement;
      expect(measurement.confidence).toBe('low');
    });

    it('returns measurement with planeDetected false for presets', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      fireEvent.press(getByText('Medium'));
      fireEvent.press(getByText('Use This Size'));

      const measurement = mockOnSelect.mock.calls[0][0] as ARMeasurement;
      expect(measurement.planeDetected).toBe(false);
    });

    it('returns measurement with timestamp', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      fireEvent.press(getByText('Medium'));
      fireEvent.press(getByText('Use This Size'));

      const measurement = mockOnSelect.mock.calls[0][0] as ARMeasurement;
      expect(measurement.timestamp).toBeInstanceOf(Date);
    });

    it('returns measurement with default distance', () => {
      const { getByText } = render(
        <ManualSizePicker onSelect={mockOnSelect} onCancel={mockOnCancel} />
      );

      fireEvent.press(getByText('Medium'));
      fireEvent.press(getByText('Use This Size'));

      const measurement = mockOnSelect.mock.calls[0][0] as ARMeasurement;
      expect(measurement.distance).toBe(30);
    });
  });
});
