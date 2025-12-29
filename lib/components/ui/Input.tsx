/**
 * Shared Input Component
 *
 * A reusable TextInput component with built-in label, error state,
 * helper text, and comprehensive accessibility support.
 */

import React, { useState, useRef, useCallback } from 'react';
import {
  View,
  TextInput,
  Text,
  StyleSheet,
  Animated,
  KeyboardTypeOptions,
  NativeSyntheticEvent,
  TextInputFocusEventData,
  Platform,
  ViewStyle,
} from 'react-native';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';

// ============================================================================
// TYPES
// ============================================================================

export interface InputProps {
  /** Label text displayed above the input */
  label: string;
  /** Current input value */
  value: string;
  /** Callback when text changes */
  onChangeText: (text: string) => void;
  /** Placeholder text */
  placeholder?: string;
  /** Error message to display (triggers error state) */
  error?: string;
  /** Helper text displayed below input */
  helperText?: string;
  /** Whether to hide text (for passwords) */
  secureTextEntry?: boolean;
  /** Keyboard type for the input */
  keyboardType?: KeyboardTypeOptions;
  /** Auto-capitalization behavior */
  autoCapitalize?: 'none' | 'sentences' | 'words' | 'characters';
  /** Whether the input is disabled */
  disabled?: boolean;
  /** Icon to display on the left side */
  leftIcon?: React.ReactNode;
  /** Icon to display on the right side */
  rightIcon?: React.ReactNode;
  /** Maximum character length */
  maxLength?: number;
  /** Whether input supports multiple lines */
  multiline?: boolean;
  /** Number of lines for multiline input */
  numberOfLines?: number;
  /** Accessibility hint for additional context */
  accessibilityHint?: string;
  /** Optional test ID for testing */
  testID?: string;
  /** Callback when input is focused */
  onFocus?: (e: NativeSyntheticEvent<TextInputFocusEventData>) => void;
  /** Callback when input loses focus */
  onBlur?: (e: NativeSyntheticEvent<TextInputFocusEventData>) => void;
  /** Auto-focus on mount */
  autoFocus?: boolean;
  /** Return key type */
  returnKeyType?: 'done' | 'go' | 'next' | 'search' | 'send';
  /** Callback when submit button is pressed */
  onSubmitEditing?: () => void;
}

// ============================================================================
// COMPONENT
// ============================================================================

export function Input({
  label,
  value,
  onChangeText,
  placeholder,
  error,
  helperText,
  secureTextEntry = false,
  keyboardType = 'default',
  autoCapitalize = 'sentences',
  disabled = false,
  leftIcon,
  rightIcon,
  maxLength,
  multiline = false,
  numberOfLines = 1,
  accessibilityHint,
  testID,
  onFocus,
  onBlur,
  autoFocus = false,
  returnKeyType,
  onSubmitEditing,
}: InputProps) {
  const [isFocused, setIsFocused] = useState(false);
  const borderColorAnim = useRef(new Animated.Value(0)).current;

  // Handle focus state
  const handleFocus = useCallback(
    (e: NativeSyntheticEvent<TextInputFocusEventData>) => {
      setIsFocused(true);
      Animated.timing(borderColorAnim, {
        toValue: 1,
        duration: 150,
        useNativeDriver: false,
      }).start();
      onFocus?.(e);
    },
    [borderColorAnim, onFocus]
  );

  // Handle blur state
  const handleBlur = useCallback(
    (e: NativeSyntheticEvent<TextInputFocusEventData>) => {
      setIsFocused(false);
      Animated.timing(borderColorAnim, {
        toValue: 0,
        duration: 150,
        useNativeDriver: false,
      }).start();
      onBlur?.(e);
    },
    [borderColorAnim, onBlur]
  );

  // Determine border color based on state
  const getBorderColor = () => {
    if (error) {
      return colors.semantic.error;
    }
    if (isFocused) {
      return colors.primary.main;
    }
    return colors.border.secondary;
  };

  // Animated border color (for smooth transitions when not in error state)
  const animatedBorderColor = error
    ? colors.semantic.error
    : borderColorAnim.interpolate({
        inputRange: [0, 1],
        outputRange: [colors.border.secondary, colors.primary.main],
      });

  // Generate unique ID for accessibility
  const inputId = testID || `input-${label.toLowerCase().replace(/\s+/g, '-')}`;

  return (
    <View style={styles.container} testID={testID}>
      {/* Label */}
      <Text style={[styles.label, disabled && styles.labelDisabled]} nativeID={`${inputId}-label`}>
        {label}
      </Text>

      {/* Input Container */}
      <Animated.View
        style={[
          styles.inputContainer,
          {
            borderColor: animatedBorderColor,
          } as Animated.WithAnimatedValue<ViewStyle>,
          disabled && styles.inputContainerDisabled,
          multiline && { minHeight: 48 * numberOfLines },
        ]}
      >
        {/* Left Icon */}
        {leftIcon && <View style={styles.leftIconContainer}>{leftIcon}</View>}

        {/* Text Input */}
        <TextInput
          style={[
            styles.input,
            leftIcon && styles.inputWithLeftIcon,
            rightIcon && styles.inputWithRightIcon,
            multiline && styles.inputMultiline,
            disabled && styles.inputDisabled,
          ]}
          value={value}
          onChangeText={onChangeText}
          placeholder={placeholder}
          placeholderTextColor={colors.text.disabled}
          secureTextEntry={secureTextEntry}
          keyboardType={keyboardType}
          autoCapitalize={autoCapitalize}
          editable={!disabled}
          maxLength={maxLength}
          multiline={multiline}
          numberOfLines={multiline ? numberOfLines : undefined}
          onFocus={handleFocus}
          onBlur={handleBlur}
          autoFocus={autoFocus}
          returnKeyType={returnKeyType}
          onSubmitEditing={onSubmitEditing}
          // Accessibility
          accessibilityLabel={label}
          accessibilityHint={accessibilityHint}
          accessibilityState={{
            disabled: disabled,
          }}
          accessibilityLabelledBy={`${inputId}-label`}
          testID={`${inputId}-input`}
        />

        {/* Right Icon */}
        {rightIcon && <View style={styles.rightIconContainer}>{rightIcon}</View>}
      </Animated.View>

      {/* Error Message */}
      {error && (
        <View
          style={styles.errorContainer}
          accessibilityLiveRegion="polite"
          accessibilityRole="alert"
          testID={`${inputId}-error`}
        >
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {/* Helper Text (only shown when no error) */}
      {helperText && !error && (
        <Text style={styles.helperText} testID={`${inputId}-helper`}>
          {helperText}
        </Text>
      )}
    </View>
  );
}

// ============================================================================
// STYLES
// ============================================================================

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.md,
  },
  label: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
    marginBottom: spacing.xs,
  },
  labelDisabled: {
    opacity: 0.5,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 48,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    paddingHorizontal: spacing.sm,
  },
  inputContainerDisabled: {
    opacity: 0.5,
    backgroundColor: colors.background.secondary,
  },
  input: {
    flex: 1,
    height: '100%',
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    ...Platform.select({
      ios: {
        paddingVertical: spacing.sm,
      },
      android: {
        paddingVertical: spacing.xs,
      },
    }),
  },
  inputWithLeftIcon: {
    marginLeft: spacing.xs,
  },
  inputWithRightIcon: {
    marginRight: spacing.xs,
  },
  inputMultiline: {
    textAlignVertical: 'top',
    paddingTop: spacing.sm,
  },
  inputDisabled: {
    color: colors.text.disabled,
  },
  leftIconContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  rightIconContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorContainer: {
    marginTop: spacing.xs,
  },
  errorText: {
    fontSize: typography.fontSize.sm,
    color: colors.semantic.error,
  },
  helperText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
});

// ============================================================================
// EXPORTS
// ============================================================================

export default Input;
