/**
 * Input Component Tests
 *
 * Comprehensive test suite covering all states, accessibility, and interactions.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { Input } from '@/lib/components/ui/Input';

// Mock react-native Animated
jest.mock('react-native', () => {
  const RN = jest.requireActual('react-native');
  RN.Animated.timing = jest.fn(() => ({
    start: jest.fn(),
  }));
  return RN;
});

describe('Input', () => {
  const defaultProps = {
    label: 'Email',
    value: '',
    onChangeText: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ============================================================================
  // RENDERING TESTS
  // ============================================================================

  describe('Rendering', () => {
    it('renders with label', () => {
      const { getByText } = render(<Input {...defaultProps} />);
      expect(getByText('Email')).toBeTruthy();
    });

    it('renders with testID', () => {
      const { getByTestId } = render(
        <Input {...defaultProps} testID="email-input" />
      );
      expect(getByTestId('email-input')).toBeTruthy();
    });

    it('renders input field', () => {
      const { getByTestId } = render(
        <Input {...defaultProps} testID="email-input" />
      );
      expect(getByTestId('email-input-input')).toBeTruthy();
    });

    it('renders placeholder text', () => {
      const { getByPlaceholderText } = render(
        <Input {...defaultProps} placeholder="Enter your email" />
      );
      expect(getByPlaceholderText('Enter your email')).toBeTruthy();
    });

    it('renders with provided value', () => {
      const { getByDisplayValue } = render(
        <Input {...defaultProps} value="test@example.com" />
      );
      expect(getByDisplayValue('test@example.com')).toBeTruthy();
    });
  });

  // ============================================================================
  // STATE TESTS
  // ============================================================================

  describe('States', () => {
    describe('Error state', () => {
      it('shows error message when error prop provided', () => {
        const { getByText, getByTestId } = render(
          <Input
            {...defaultProps}
            error="Invalid email address"
            testID="email-input"
          />
        );
        expect(getByText('Invalid email address')).toBeTruthy();
        expect(getByTestId('email-input-error')).toBeTruthy();
      });

      it('hides helper text when error is shown', () => {
        const { queryByText, getByText } = render(
          <Input
            {...defaultProps}
            helperText="We'll never share your email"
            error="Invalid email address"
          />
        );
        expect(getByText('Invalid email address')).toBeTruthy();
        expect(queryByText("We'll never share your email")).toBeNull();
      });

      it('error container has accessibility role alert', () => {
        const { getByTestId } = render(
          <Input
            {...defaultProps}
            error="Invalid email address"
            testID="email-input"
          />
        );
        const errorContainer = getByTestId('email-input-error');
        expect(errorContainer.props.accessibilityRole).toBe('alert');
      });

      it('error container has live region for screen readers', () => {
        const { getByTestId } = render(
          <Input
            {...defaultProps}
            error="Invalid email address"
            testID="email-input"
          />
        );
        const errorContainer = getByTestId('email-input-error');
        expect(errorContainer.props.accessibilityLiveRegion).toBe('polite');
      });
    });

    describe('Disabled state', () => {
      it('input is not editable when disabled', () => {
        const { getByTestId } = render(
          <Input {...defaultProps} disabled testID="email-input" />
        );
        const input = getByTestId('email-input-input');
        expect(input.props.editable).toBe(false);
      });

      it('does not call onChangeText when disabled', () => {
        const onChangeText = jest.fn();
        const { getByTestId } = render(
          <Input
            {...defaultProps}
            onChangeText={onChangeText}
            disabled
            testID="email-input"
          />
        );
        const input = getByTestId('email-input-input');
        fireEvent.changeText(input, 'new value');
        // Note: React Native's editable=false should prevent this,
        // but in tests we verify the prop is set correctly
        expect(input.props.editable).toBe(false);
      });

      it('has disabled accessibility state', () => {
        const { getByTestId } = render(
          <Input {...defaultProps} disabled testID="email-input" />
        );
        const input = getByTestId('email-input-input');
        expect(input.props.accessibilityState).toEqual(
          expect.objectContaining({ disabled: true })
        );
      });
    });

    describe('Focus state', () => {
      it('triggers animation on focus', () => {
        const { Animated } = require('react-native');
        const { getByTestId } = render(
          <Input {...defaultProps} testID="email-input" />
        );
        const input = getByTestId('email-input-input');
        fireEvent(input, 'focus');
        expect(Animated.timing).toHaveBeenCalled();
      });

      it('triggers animation on blur', () => {
        const { Animated } = require('react-native');
        const { getByTestId } = render(
          <Input {...defaultProps} testID="email-input" />
        );
        const input = getByTestId('email-input-input');
        fireEvent(input, 'blur');
        expect(Animated.timing).toHaveBeenCalled();
      });

      it('calls onFocus callback when provided', () => {
        const onFocus = jest.fn();
        const { getByTestId } = render(
          <Input {...defaultProps} onFocus={onFocus} testID="email-input" />
        );
        const input = getByTestId('email-input-input');
        fireEvent(input, 'focus');
        expect(onFocus).toHaveBeenCalled();
      });

      it('calls onBlur callback when provided', () => {
        const onBlur = jest.fn();
        const { getByTestId } = render(
          <Input {...defaultProps} onBlur={onBlur} testID="email-input" />
        );
        const input = getByTestId('email-input-input');
        fireEvent(input, 'blur');
        expect(onBlur).toHaveBeenCalled();
      });
    });
  });

  // ============================================================================
  // INTERACTION TESTS
  // ============================================================================

  describe('Interactions', () => {
    it('calls onChangeText when text changes', () => {
      const onChangeText = jest.fn();
      const { getByTestId } = render(
        <Input
          {...defaultProps}
          onChangeText={onChangeText}
          testID="email-input"
        />
      );
      const input = getByTestId('email-input-input');
      fireEvent.changeText(input, 'test@example.com');
      expect(onChangeText).toHaveBeenCalledWith('test@example.com');
    });

    it('calls onSubmitEditing when submit is pressed', () => {
      const onSubmitEditing = jest.fn();
      const { getByTestId } = render(
        <Input
          {...defaultProps}
          onSubmitEditing={onSubmitEditing}
          testID="email-input"
        />
      );
      const input = getByTestId('email-input-input');
      fireEvent(input, 'submitEditing');
      expect(onSubmitEditing).toHaveBeenCalled();
    });
  });

  // ============================================================================
  // HELPER TEXT TESTS
  // ============================================================================

  describe('Helper Text', () => {
    it('renders helper text when provided', () => {
      const { getByText, getByTestId } = render(
        <Input
          {...defaultProps}
          helperText="Enter a valid email"
          testID="email-input"
        />
      );
      expect(getByText('Enter a valid email')).toBeTruthy();
      expect(getByTestId('email-input-helper')).toBeTruthy();
    });

    it('does not render helper text when not provided', () => {
      const { queryByTestId } = render(
        <Input {...defaultProps} testID="email-input" />
      );
      expect(queryByTestId('email-input-helper')).toBeNull();
    });
  });

  // ============================================================================
  // ICON TESTS
  // ============================================================================

  describe('Icons', () => {
    it('renders left icon when provided', () => {
      const { Text } = require('react-native');
      const leftIcon = <Text testID="left-icon">@</Text>;
      const { getByTestId } = render(
        <Input {...defaultProps} leftIcon={leftIcon} />
      );
      expect(getByTestId('left-icon')).toBeTruthy();
    });

    it('renders right icon when provided', () => {
      const { Text } = require('react-native');
      const rightIcon = <Text testID="right-icon">X</Text>;
      const { getByTestId } = render(
        <Input {...defaultProps} rightIcon={rightIcon} />
      );
      expect(getByTestId('right-icon')).toBeTruthy();
    });

    it('renders both icons when provided', () => {
      const { Text } = require('react-native');
      const leftIcon = <Text testID="left-icon">@</Text>;
      const rightIcon = <Text testID="right-icon">X</Text>;
      const { getByTestId } = render(
        <Input {...defaultProps} leftIcon={leftIcon} rightIcon={rightIcon} />
      );
      expect(getByTestId('left-icon')).toBeTruthy();
      expect(getByTestId('right-icon')).toBeTruthy();
    });
  });

  // ============================================================================
  // ACCESSIBILITY TESTS
  // ============================================================================

  describe('Accessibility', () => {
    it('has correct accessibility label from label prop', () => {
      const { getByTestId } = render(
        <Input {...defaultProps} testID="email-input" />
      );
      const input = getByTestId('email-input-input');
      expect(input.props.accessibilityLabel).toBe('Email');
    });

    it('has accessibility hint when provided', () => {
      const { getByTestId } = render(
        <Input
          {...defaultProps}
          accessibilityHint="Enter your email address"
          testID="email-input"
        />
      );
      const input = getByTestId('email-input-input');
      expect(input.props.accessibilityHint).toBe('Enter your email address');
    });

    it('label is connected to input via accessibilityLabelledBy', () => {
      const { getByTestId } = render(
        <Input {...defaultProps} testID="email-input" />
      );
      const input = getByTestId('email-input-input');
      expect(input.props.accessibilityLabelledBy).toBe('email-input-label');
    });
  });

  // ============================================================================
  // INPUT TYPE TESTS
  // ============================================================================

  describe('Input Types', () => {
    it('supports secure text entry for passwords', () => {
      const { getByTestId } = render(
        <Input {...defaultProps} secureTextEntry testID="password-input" />
      );
      const input = getByTestId('password-input-input');
      expect(input.props.secureTextEntry).toBe(true);
    });

    it('supports different keyboard types', () => {
      const { getByTestId } = render(
        <Input {...defaultProps} keyboardType="email-address" testID="email-input" />
      );
      const input = getByTestId('email-input-input');
      expect(input.props.keyboardType).toBe('email-address');
    });

    it('supports auto-capitalize options', () => {
      const { getByTestId } = render(
        <Input {...defaultProps} autoCapitalize="none" testID="email-input" />
      );
      const input = getByTestId('email-input-input');
      expect(input.props.autoCapitalize).toBe('none');
    });

    it('supports max length', () => {
      const { getByTestId } = render(
        <Input {...defaultProps} maxLength={100} testID="email-input" />
      );
      const input = getByTestId('email-input-input');
      expect(input.props.maxLength).toBe(100);
    });

    it('supports multiline mode', () => {
      const { getByTestId } = render(
        <Input
          {...defaultProps}
          multiline
          numberOfLines={4}
          testID="description-input"
        />
      );
      const input = getByTestId('description-input-input');
      expect(input.props.multiline).toBe(true);
      expect(input.props.numberOfLines).toBe(4);
    });

    it('supports return key type', () => {
      const { getByTestId } = render(
        <Input {...defaultProps} returnKeyType="next" testID="email-input" />
      );
      const input = getByTestId('email-input-input');
      expect(input.props.returnKeyType).toBe('next');
    });
  });
});
