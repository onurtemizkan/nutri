/**
 * Tests for Reference Object Calibration Component
 *
 * Tests the calibration flow using known reference objects
 * to calculate scale factors for measurement.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import {
  ReferenceObjectCalibration,
  REFERENCE_OBJECTS,
  calculateCalibration,
  pixelsToCm,
  cmToPixels,
} from '@/lib/components/ReferenceObjectCalibration';
import type {
  ReferenceObject,
  CalibrationResult,
} from '@/lib/components/ReferenceObjectCalibration';

// Mock Ionicons
jest.mock('@expo/vector-icons', () => ({
  Ionicons: 'Ionicons',
}));

// Mock DeviceEventEmitter (needed since tests might import ar-measure indirectly)
jest.mock('react-native', () => {
  const RN = jest.requireActual('react-native');
  RN.DeviceEventEmitter = {
    emit: jest.fn(),
    addListener: jest.fn(() => ({ remove: jest.fn() })),
    removeListener: jest.fn(),
    removeAllListeners: jest.fn(),
  };
  return RN;
});

describe('ReferenceObjectCalibration', () => {
  const mockOnCalibrate = jest.fn();
  const mockOnCancel = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ===========================================================================
  // Reference Objects Data Tests
  // ===========================================================================
  describe('Reference Objects Data', () => {
    it('has a comprehensive list of reference objects', () => {
      expect(REFERENCE_OBJECTS.length).toBeGreaterThanOrEqual(8);
    });

    it('includes coins category', () => {
      const coins = REFERENCE_OBJECTS.filter((obj) => obj.category === 'coin');
      expect(coins.length).toBeGreaterThanOrEqual(3);
    });

    it('includes cards category', () => {
      const cards = REFERENCE_OBJECTS.filter((obj) => obj.category === 'card');
      expect(cards.length).toBeGreaterThanOrEqual(1);
    });

    it('includes devices category', () => {
      const devices = REFERENCE_OBJECTS.filter((obj) => obj.category === 'device');
      expect(devices.length).toBeGreaterThanOrEqual(1);
    });

    it('has US Quarter with correct dimensions', () => {
      const quarter = REFERENCE_OBJECTS.find((obj) => obj.id === 'us_quarter');
      expect(quarter).toBeDefined();
      expect(quarter?.widthCm).toBeCloseTo(2.426, 2);
      expect(quarter?.heightCm).toBeCloseTo(2.426, 2);
    });

    it('has Credit Card with correct ISO dimensions', () => {
      const card = REFERENCE_OBJECTS.find((obj) => obj.id === 'credit_card');
      expect(card).toBeDefined();
      expect(card?.widthCm).toBeCloseTo(8.56, 2);
      expect(card?.heightCm).toBeCloseTo(5.398, 2);
    });

    it('all objects have required properties', () => {
      REFERENCE_OBJECTS.forEach((obj) => {
        expect(obj.id).toBeTruthy();
        expect(obj.name).toBeTruthy();
        expect(obj.description).toBeTruthy();
        expect(obj.widthCm).toBeGreaterThan(0);
        expect(obj.heightCm).toBeGreaterThan(0);
        expect(obj.icon).toBeTruthy();
        expect(['coin', 'card', 'device', 'other']).toContain(obj.category);
      });
    });
  });

  // ===========================================================================
  // Calibration Calculation Tests
  // ===========================================================================
  describe('calculateCalibration', () => {
    const creditCard: ReferenceObject = {
      id: 'credit_card',
      name: 'Credit Card',
      description: 'Test card',
      widthCm: 8.56,
      heightCm: 5.398,
      icon: 'card',
      category: 'card',
    };

    it('calculates correct pixels per cm', () => {
      // If credit card (8.56cm) appears as 856 pixels wide
      const result = calculateCalibration(creditCard, 856, 539.8);

      expect(result.pixelsPerCmX).toBeCloseTo(100, 1);
      expect(result.pixelsPerCmY).toBeCloseTo(100, 1);
    });

    it('calculates correct scale factor as average', () => {
      const result = calculateCalibration(creditCard, 856, 539.8);

      expect(result.scaleFactor).toBeCloseTo(100, 1);
    });

    it('returns high confidence for matching aspect ratio', () => {
      // Perfect aspect ratio match
      const result = calculateCalibration(creditCard, 856, 539.8);

      expect(result.confidence).toBe('high');
    });

    it('returns medium confidence for slightly off aspect ratio', () => {
      // Off by ~15%
      const result = calculateCalibration(creditCard, 856, 620);

      expect(result.confidence).toBe('medium');
    });

    it('returns low confidence for very different aspect ratio', () => {
      // Square when should be rectangular
      const result = calculateCalibration(creditCard, 500, 500);

      expect(result.confidence).toBe('low');
    });

    it('includes timestamp in result', () => {
      const before = new Date();
      const result = calculateCalibration(creditCard, 856, 539.8);
      const after = new Date();

      expect(result.timestamp.getTime()).toBeGreaterThanOrEqual(before.getTime());
      expect(result.timestamp.getTime()).toBeLessThanOrEqual(after.getTime());
    });

    it('includes reference object in result', () => {
      const result = calculateCalibration(creditCard, 856, 539.8);

      expect(result.referenceObject).toBe(creditCard);
    });
  });

  // ===========================================================================
  // Conversion Utility Tests
  // ===========================================================================
  describe('pixelsToCm', () => {
    it('converts pixels to centimeters correctly', () => {
      const scaleFactor = 100; // 100 pixels per cm
      expect(pixelsToCm(200, scaleFactor)).toBe(2);
      expect(pixelsToCm(50, scaleFactor)).toBe(0.5);
      expect(pixelsToCm(100, scaleFactor)).toBe(1);
    });

    it('handles fractional results', () => {
      const scaleFactor = 75;
      expect(pixelsToCm(150, scaleFactor)).toBe(2);
      expect(pixelsToCm(100, scaleFactor)).toBeCloseTo(1.333, 2);
    });
  });

  describe('cmToPixels', () => {
    it('converts centimeters to pixels correctly', () => {
      const scaleFactor = 100;
      expect(cmToPixels(2, scaleFactor)).toBe(200);
      expect(cmToPixels(0.5, scaleFactor)).toBe(50);
      expect(cmToPixels(1, scaleFactor)).toBe(100);
    });

    it('handles fractional centimeters', () => {
      const scaleFactor = 75;
      expect(cmToPixels(2, scaleFactor)).toBe(150);
      expect(cmToPixels(1.5, scaleFactor)).toBe(112.5);
    });
  });

  // ===========================================================================
  // Component Rendering Tests
  // ===========================================================================
  describe('Rendering', () => {
    it('renders the component', () => {
      const { getByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
        />
      );

      expect(getByText('Calibrate with Reference')).toBeTruthy();
    });

    it('shows instructions', () => {
      const { getByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
        />
      );

      expect(
        getByText(/Place a reference object next to your food/)
      ).toBeTruthy();
    });

    it('shows category tabs', () => {
      const { getByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
        />
      );

      expect(getByText('Coins')).toBeTruthy();
      expect(getByText('Cards')).toBeTruthy();
      expect(getByText('Devices')).toBeTruthy();
      expect(getByText('Other')).toBeTruthy();
    });

    it('shows coin objects by default', () => {
      const { getByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
        />
      );

      expect(getByText('US Quarter')).toBeTruthy();
      expect(getByText('US Dime')).toBeTruthy();
    });

    it('shows confirm button', () => {
      const { getByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
        />
      );

      expect(getByText('Use This Reference')).toBeTruthy();
    });
  });

  // ===========================================================================
  // Interaction Tests
  // ===========================================================================
  describe('Interactions', () => {
    it('calls onCancel when back button is pressed', () => {
      const { UNSAFE_getAllByType } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
        />
      );

      const touchables = UNSAFE_getAllByType(
        require('react-native').TouchableOpacity
      );
      // First touchable is the back button
      fireEvent.press(touchables[0]);

      expect(mockOnCancel).toHaveBeenCalledTimes(1);
    });

    it('switches categories when tab is pressed', () => {
      const { getByText, queryByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
        />
      );

      // Initially showing coins
      expect(getByText('US Quarter')).toBeTruthy();
      expect(queryByText('Credit Card')).toBeNull();

      // Switch to cards
      fireEvent.press(getByText('Cards'));

      expect(getByText('Credit Card')).toBeTruthy();
      expect(queryByText('US Quarter')).toBeNull();
    });

    it('selects object when pressed', () => {
      const { getByText, getAllByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
        />
      );

      fireEvent.press(getByText('US Quarter'));

      // Should show preview with selected object dimensions
      expect(getByText('Selected Reference')).toBeTruthy();
      // US Quarter is circular so both width and height are 2.43 cm
      expect(getAllByText('2.43 cm').length).toBe(2);
    });

    it('calls onCalibrate when confirm button is pressed with selection', () => {
      const { getByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
          drawnWidth={242.6}
          drawnHeight={242.6}
        />
      );

      // Select an object
      fireEvent.press(getByText('US Quarter'));

      // Confirm
      fireEvent.press(getByText('Use This Reference'));

      expect(mockOnCalibrate).toHaveBeenCalledTimes(1);
      expect(mockOnCalibrate).toHaveBeenCalledWith(
        expect.objectContaining({
          referenceObject: expect.objectContaining({
            id: 'us_quarter',
          }),
          pixelsPerCmX: expect.any(Number),
          pixelsPerCmY: expect.any(Number),
          scaleFactor: expect.any(Number),
          confidence: expect.stringMatching(/high|medium|low/),
          timestamp: expect.any(Date),
        })
      );
    });

    it('does not call onCalibrate when no object is selected', () => {
      const { getByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
        />
      );

      fireEvent.press(getByText('Use This Reference'));

      expect(mockOnCalibrate).not.toHaveBeenCalled();
    });
  });

  // ===========================================================================
  // Initial Selection Tests
  // ===========================================================================
  describe('Initial Selection', () => {
    it('pre-selects object when initialObjectId is provided', () => {
      const { getByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
          initialObjectId="us_quarter"
        />
      );

      // Should show preview immediately
      expect(getByText('Selected Reference')).toBeTruthy();
    });

    it('handles invalid initialObjectId gracefully', () => {
      const { queryByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
          initialObjectId="invalid_id"
        />
      );

      // Should not show preview
      expect(queryByText('Selected Reference')).toBeNull();
    });
  });

  // ===========================================================================
  // Category Filtering Tests
  // ===========================================================================
  describe('Category Filtering', () => {
    it('shows only coins when Coins tab is active', () => {
      const { getByText, queryByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
        />
      );

      expect(getByText('US Quarter')).toBeTruthy();
      expect(getByText('US Dime')).toBeTruthy();
      expect(queryByText('Credit Card')).toBeNull();
    });

    it('shows only devices when Devices tab is active', () => {
      const { getByText, queryByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
        />
      );

      fireEvent.press(getByText('Devices'));

      expect(getByText('iPhone 15 Pro')).toBeTruthy();
      expect(queryByText('US Quarter')).toBeNull();
    });

    it('shows other items when Other tab is active', () => {
      const { getByText, queryByText } = render(
        <ReferenceObjectCalibration
          onCalibrate={mockOnCalibrate}
          onCancel={mockOnCancel}
        />
      );

      fireEvent.press(getByText('Other'));

      expect(getByText('US Dollar Bill')).toBeTruthy();
      expect(getByText('AA Battery')).toBeTruthy();
      expect(queryByText('Credit Card')).toBeNull();
    });
  });
});
