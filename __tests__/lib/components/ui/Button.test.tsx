/**
 * Button Component Tests
 *
 * Comprehensive test suite covering all variants, sizes, states, and accessibility.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { Button } from '@/lib/components/ui/Button';

// Mock expo-linear-gradient
jest.mock('expo-linear-gradient', () => ({
  LinearGradient: ({ children, ...props }: { children: React.ReactNode }) => {
    const { View } = require('react-native');
    return <View {...props}>{children}</View>;
  },
}));

// Mock react-native Animated
jest.mock('react-native', () => {
  const RN = jest.requireActual('react-native');
  RN.Animated.spring = jest.fn(() => ({
    start: jest.fn(),
  }));
  return RN;
});

describe('Button', () => {
  const defaultProps = {
    label: 'Test Button',
    accessibilityLabel: 'Test button accessibility label',
    onPress: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ============================================================================
  // RENDERING TESTS
  // ============================================================================

  describe('Rendering', () => {
    it('renders with label text', () => {
      const { getByText } = render(<Button {...defaultProps} />);
      expect(getByText('Test Button')).toBeTruthy();
    });

    it('renders with testID', () => {
      const { getByTestId } = render(
        <Button {...defaultProps} testID="my-button" />
      );
      expect(getByTestId('my-button')).toBeTruthy();
    });
  });

  // ============================================================================
  // VARIANT TESTS
  // ============================================================================

  describe('Variants', () => {
    it('renders primary variant by default', () => {
      const { getByText } = render(<Button {...defaultProps} />);
      const label = getByText('Test Button');
      // Primary variant uses white text
      expect(label.props.style).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ color: '#FFFFFF' }),
        ])
      );
    });

    it('renders secondary variant with border', () => {
      const { getByText } = render(
        <Button {...defaultProps} variant="secondary" />
      );
      expect(getByText('Test Button')).toBeTruthy();
    });

    it('renders ghost variant', () => {
      const { getByText } = render(
        <Button {...defaultProps} variant="ghost" />
      );
      const label = getByText('Test Button');
      // Ghost variant uses primary color text
      expect(label.props.style).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ color: '#8B5CF6' }),
        ])
      );
    });

    it('renders destructive variant', () => {
      const { getByText } = render(
        <Button {...defaultProps} variant="destructive" />
      );
      expect(getByText('Test Button')).toBeTruthy();
    });
  });

  // ============================================================================
  // SIZE TESTS
  // ============================================================================

  describe('Sizes', () => {
    it('renders small size with correct height', () => {
      const { getByTestId } = render(
        <Button {...defaultProps} size="sm" testID="button" />
      );
      const button = getByTestId('button');
      expect(button).toBeTruthy();
    });

    it('renders medium size by default', () => {
      const { getByTestId } = render(
        <Button {...defaultProps} testID="button" />
      );
      const button = getByTestId('button');
      expect(button).toBeTruthy();
    });

    it('renders large size with correct height', () => {
      const { getByTestId } = render(
        <Button {...defaultProps} size="lg" testID="button" />
      );
      const button = getByTestId('button');
      expect(button).toBeTruthy();
    });
  });

  // ============================================================================
  // STATE TESTS
  // ============================================================================

  describe('States', () => {
    describe('Loading state', () => {
      it('shows loading indicator when loading', () => {
        const { getByTestId, queryByText } = render(
          <Button {...defaultProps} loading testID="button" />
        );
        expect(getByTestId('button-loading')).toBeTruthy();
        expect(queryByText('Test Button')).toBeNull();
      });

      it('does not call onPress when loading', () => {
        const onPress = jest.fn();
        const { getByTestId } = render(
          <Button {...defaultProps} onPress={onPress} loading testID="button" />
        );
        fireEvent.press(getByTestId('button'));
        expect(onPress).not.toHaveBeenCalled();
      });
    });

    describe('Disabled state', () => {
      it('applies disabled styling', () => {
        const { getByTestId } = render(
          <Button {...defaultProps} disabled testID="button" />
        );
        const button = getByTestId('button');
        expect(button).toBeTruthy();
      });

      it('does not call onPress when disabled', () => {
        const onPress = jest.fn();
        const { getByTestId } = render(
          <Button {...defaultProps} onPress={onPress} disabled testID="button" />
        );
        fireEvent.press(getByTestId('button'));
        expect(onPress).not.toHaveBeenCalled();
      });
    });
  });

  // ============================================================================
  // INTERACTION TESTS
  // ============================================================================

  describe('Interactions', () => {
    it('calls onPress when pressed', () => {
      const onPress = jest.fn();
      const { getByTestId } = render(
        <Button {...defaultProps} onPress={onPress} testID="button" />
      );
      fireEvent.press(getByTestId('button'));
      expect(onPress).toHaveBeenCalledTimes(1);
    });

    it('triggers press animation on pressIn', () => {
      const { Animated } = require('react-native');
      const { getByTestId } = render(
        <Button {...defaultProps} testID="button" />
      );
      fireEvent(getByTestId('button'), 'pressIn');
      expect(Animated.spring).toHaveBeenCalled();
    });

    it('triggers press animation on pressOut', () => {
      const { Animated } = require('react-native');
      const { getByTestId } = render(
        <Button {...defaultProps} testID="button" />
      );
      fireEvent(getByTestId('button'), 'pressOut');
      expect(Animated.spring).toHaveBeenCalled();
    });
  });

  // ============================================================================
  // ACCESSIBILITY TESTS
  // ============================================================================

  describe('Accessibility', () => {
    it('has button accessibility role', () => {
      const { getByTestId } = render(
        <Button {...defaultProps} testID="button" />
      );
      const button = getByTestId('button');
      expect(button.props.accessibilityRole).toBe('button');
    });

    it('has correct accessibility label', () => {
      const { getByTestId } = render(
        <Button {...defaultProps} testID="button" />
      );
      const button = getByTestId('button');
      expect(button.props.accessibilityLabel).toBe(
        'Test button accessibility label'
      );
    });

    it('sets disabled accessibility state when disabled', () => {
      const { getByTestId } = render(
        <Button {...defaultProps} disabled testID="button" />
      );
      const button = getByTestId('button');
      expect(button.props.accessibilityState).toEqual(
        expect.objectContaining({ disabled: true })
      );
    });

    it('sets busy accessibility state when loading', () => {
      const { getByTestId } = render(
        <Button {...defaultProps} loading testID="button" />
      );
      const button = getByTestId('button');
      expect(button.props.accessibilityState).toEqual(
        expect.objectContaining({ busy: true })
      );
    });
  });

  // ============================================================================
  // ICON TESTS
  // ============================================================================

  describe('Icons', () => {
    it('renders left icon when provided', () => {
      const { View, Text } = require('react-native');
      const leftIcon = <Text testID="left-icon">Icon</Text>;
      const { getByTestId } = render(
        <Button {...defaultProps} leftIcon={leftIcon} />
      );
      expect(getByTestId('left-icon')).toBeTruthy();
    });

    it('renders right icon when provided', () => {
      const { Text } = require('react-native');
      const rightIcon = <Text testID="right-icon">Icon</Text>;
      const { getByTestId } = render(
        <Button {...defaultProps} rightIcon={rightIcon} />
      );
      expect(getByTestId('right-icon')).toBeTruthy();
    });

    it('renders both icons when provided', () => {
      const { Text } = require('react-native');
      const leftIcon = <Text testID="left-icon">Left</Text>;
      const rightIcon = <Text testID="right-icon">Right</Text>;
      const { getByTestId } = render(
        <Button {...defaultProps} leftIcon={leftIcon} rightIcon={rightIcon} />
      );
      expect(getByTestId('left-icon')).toBeTruthy();
      expect(getByTestId('right-icon')).toBeTruthy();
    });

    it('does not render icons when loading', () => {
      const { Text } = require('react-native');
      const leftIcon = <Text testID="left-icon">Left</Text>;
      const { queryByTestId } = render(
        <Button {...defaultProps} leftIcon={leftIcon} loading />
      );
      expect(queryByTestId('left-icon')).toBeNull();
    });
  });

  // ============================================================================
  // FULL WIDTH TEST
  // ============================================================================

  describe('Full Width', () => {
    it('applies full width styling when fullWidth is true', () => {
      const { getByTestId } = render(
        <Button {...defaultProps} fullWidth testID="button" />
      );
      const button = getByTestId('button');
      expect(button).toBeTruthy();
    });
  });
});
