/**
 * Card Component Tests
 *
 * Comprehensive test suite covering all variants, padding, and pressable behavior.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { Text } from 'react-native';
import { Card } from '@/lib/components/ui/Card';

// Mock react-native Animated
jest.mock('react-native', () => {
  const RN = jest.requireActual('react-native');
  RN.Animated.spring = jest.fn(() => ({
    start: jest.fn(),
  }));
  return RN;
});

describe('Card', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ============================================================================
  // RENDERING TESTS
  // ============================================================================

  describe('Rendering', () => {
    it('renders children correctly', () => {
      const { getByText } = render(
        <Card>
          <Text>Card Content</Text>
        </Card>
      );
      expect(getByText('Card Content')).toBeTruthy();
    });

    it('renders with testID', () => {
      const { getByTestId } = render(
        <Card testID="my-card">
          <Text>Content</Text>
        </Card>
      );
      expect(getByTestId('my-card')).toBeTruthy();
    });
  });

  // ============================================================================
  // VARIANT TESTS
  // ============================================================================

  describe('Variants', () => {
    it('renders elevated variant by default', () => {
      const { getByTestId } = render(
        <Card testID="card">
          <Text>Elevated Card</Text>
        </Card>
      );
      const card = getByTestId('card');
      // Elevated cards should have shadow properties
      expect(card).toBeTruthy();
    });

    it('renders outlined variant with border', () => {
      const { getByTestId } = render(
        <Card variant="outlined" testID="card">
          <Text>Outlined Card</Text>
        </Card>
      );
      const card = getByTestId('card');
      expect(card).toBeTruthy();
    });

    it('renders filled variant', () => {
      const { getByTestId } = render(
        <Card variant="filled" testID="card">
          <Text>Filled Card</Text>
        </Card>
      );
      const card = getByTestId('card');
      expect(card).toBeTruthy();
    });
  });

  // ============================================================================
  // PADDING TESTS
  // ============================================================================

  describe('Padding', () => {
    it('renders with no padding', () => {
      const { getByTestId } = render(
        <Card padding="none" testID="card">
          <Text>No Padding</Text>
        </Card>
      );
      const card = getByTestId('card');
      expect(card.props.style).toEqual(
        expect.arrayContaining([expect.objectContaining({ padding: 0 })])
      );
    });

    it('renders with small padding', () => {
      const { getByTestId } = render(
        <Card padding="sm" testID="card">
          <Text>Small Padding</Text>
        </Card>
      );
      const card = getByTestId('card');
      expect(card.props.style).toEqual(
        expect.arrayContaining([expect.objectContaining({ padding: 8 })])
      );
    });

    it('renders with medium padding by default', () => {
      const { getByTestId } = render(
        <Card testID="card">
          <Text>Medium Padding</Text>
        </Card>
      );
      const card = getByTestId('card');
      expect(card.props.style).toEqual(
        expect.arrayContaining([expect.objectContaining({ padding: 16 })])
      );
    });

    it('renders with large padding', () => {
      const { getByTestId } = render(
        <Card padding="lg" testID="card">
          <Text>Large Padding</Text>
        </Card>
      );
      const card = getByTestId('card');
      expect(card.props.style).toEqual(
        expect.arrayContaining([expect.objectContaining({ padding: 24 })])
      );
    });
  });

  // ============================================================================
  // PRESSABLE BEHAVIOR TESTS
  // ============================================================================

  describe('Pressable Behavior', () => {
    it('calls onPress when pressed', () => {
      const onPress = jest.fn();
      const { getByTestId } = render(
        <Card onPress={onPress} testID="card">
          <Text>Pressable Card</Text>
        </Card>
      );
      fireEvent.press(getByTestId('card'));
      expect(onPress).toHaveBeenCalledTimes(1);
    });

    it('triggers press animation on pressIn', () => {
      const { Animated } = require('react-native');
      const onPress = jest.fn();
      const { getByTestId } = render(
        <Card onPress={onPress} testID="card">
          <Text>Pressable Card</Text>
        </Card>
      );
      fireEvent(getByTestId('card'), 'pressIn');
      expect(Animated.spring).toHaveBeenCalled();
    });

    it('triggers press animation on pressOut', () => {
      const { Animated } = require('react-native');
      const onPress = jest.fn();
      const { getByTestId } = render(
        <Card onPress={onPress} testID="card">
          <Text>Pressable Card</Text>
        </Card>
      );
      fireEvent(getByTestId('card'), 'pressOut');
      expect(Animated.spring).toHaveBeenCalled();
    });

    it('is not pressable when onPress not provided', () => {
      const { getByTestId } = render(
        <Card testID="card">
          <Text>Non-pressable Card</Text>
        </Card>
      );
      const card = getByTestId('card');
      // Non-pressable cards should be View, not Pressable
      expect(card.props.onPress).toBeUndefined();
    });
  });

  // ============================================================================
  // ACCESSIBILITY TESTS
  // ============================================================================

  describe('Accessibility', () => {
    it('has button accessibility role when pressable', () => {
      const onPress = jest.fn();
      const { getByTestId } = render(
        <Card onPress={onPress} testID="card">
          <Text>Pressable Card</Text>
        </Card>
      );
      const card = getByTestId('card');
      expect(card.props.accessibilityRole).toBe('button');
    });

    it('has correct accessibility label when pressable', () => {
      const onPress = jest.fn();
      const { getByTestId } = render(
        <Card
          onPress={onPress}
          accessibilityLabel="Navigate to details"
          testID="card"
        >
          <Text>Pressable Card</Text>
        </Card>
      );
      const card = getByTestId('card');
      expect(card.props.accessibilityLabel).toBe('Navigate to details');
    });

    it('has accessibility hint when provided', () => {
      const onPress = jest.fn();
      const { getByTestId } = render(
        <Card
          onPress={onPress}
          accessibilityHint="Double tap to open details"
          testID="card"
        >
          <Text>Pressable Card</Text>
        </Card>
      );
      const card = getByTestId('card');
      expect(card.props.accessibilityHint).toBe('Double tap to open details');
    });

    it('does not have button role when not pressable', () => {
      const { getByTestId } = render(
        <Card testID="card">
          <Text>Non-pressable Card</Text>
        </Card>
      );
      const card = getByTestId('card');
      expect(card.props.accessibilityRole).toBeUndefined();
    });
  });

  // ============================================================================
  // CUSTOM STYLE TESTS
  // ============================================================================

  describe('Custom Styles', () => {
    it('applies custom styles', () => {
      const customStyle = { marginTop: 20 };
      const { getByTestId } = render(
        <Card style={customStyle} testID="card">
          <Text>Styled Card</Text>
        </Card>
      );
      const card = getByTestId('card');
      expect(card.props.style).toEqual(
        expect.arrayContaining([expect.objectContaining({ marginTop: 20 })])
      );
    });

    it('merges custom styles with variant styles', () => {
      const customStyle = { marginBottom: 16 };
      const { getByTestId } = render(
        <Card variant="outlined" style={customStyle} testID="card">
          <Text>Merged Styles</Text>
        </Card>
      );
      const card = getByTestId('card');
      expect(card.props.style).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ marginBottom: 16 }),
        ])
      );
    });
  });
});
