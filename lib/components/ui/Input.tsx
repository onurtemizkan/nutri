import React, { useState, useRef, useImperativeHandle, forwardRef } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  Animated,
  Platform,
  type TextInputProps,
  type StyleProp,
  type ViewStyle,
  type TextStyle,
  type TextInput as TextInputType,
} from 'react-native';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';

export interface InputProps extends Omit<TextInputProps, 'style'> {
  /** Label text displayed above the input */
  label: string;
  /** Current input value */
  value: string;
  /** Callback when text changes */
  onChangeText: (text: string) => void;
  /** Error message to display (shows error state when provided) */
  error?: string;
  /** Helper text displayed below input (hidden when error is shown) */
  helperText?: string;
  /** Whether the input is disabled */
  disabled?: boolean;
  /** Icon component to render on the left side of input */
  leftIcon?: React.ReactNode;
  /** Icon component to render on the right side of input */
  rightIcon?: React.ReactNode;
  /** Custom container style */
  containerStyle?: StyleProp<ViewStyle>;
  /** Custom input style */
  inputStyle?: StyleProp<TextStyle>;
  /** Test ID for testing */
  testID?: string;
}

export interface InputRef {
  focus: () => void;
  blur: () => void;
  clear: () => void;
}

/**
 * Shared Input component with built-in label, error state, helper text, and accessibility support.
 *
 * Features:
 * - Floating label above input
 * - Error state with red border and error message
 * - Helper text below input
 * - Focus state with purple border
 * - Disabled state with reduced opacity
 * - Full accessibility support
 *
 * @example
 * ```tsx
 * <Input
 *   label="Email"
 *   value={email}
 *   onChangeText={setEmail}
 *   error={emailError}
 *   helperText="We'll never share your email"
 *   keyboardType="email-address"
 *   autoCapitalize="none"
 * />
 * ```
 */
export const Input = forwardRef<InputRef, InputProps>(
  (
    {
      label,
      value,
      onChangeText,
      error,
      helperText,
      disabled = false,
      leftIcon,
      rightIcon,
      containerStyle,
      inputStyle,
      testID,
      onFocus,
      onBlur,
      placeholder,
      ...rest
    },
    ref
  ) => {
    const [isFocused, setIsFocused] = useState(false);
    const inputRef = useRef<TextInputType>(null);
    const borderColorAnim = useRef(new Animated.Value(0)).current;

    useImperativeHandle(ref, () => ({
      focus: () => inputRef.current?.focus(),
      blur: () => inputRef.current?.blur(),
      clear: () => inputRef.current?.clear(),
    }));

    const handleFocus = (e: Parameters<NonNullable<TextInputProps['onFocus']>>[0]) => {
      setIsFocused(true);
      Animated.timing(borderColorAnim, {
        toValue: 1,
        duration: 150,
        useNativeDriver: false,
      }).start();
      onFocus?.(e);
    };

    const handleBlur = (e: Parameters<NonNullable<TextInputProps['onBlur']>>[0]) => {
      setIsFocused(false);
      Animated.timing(borderColorAnim, {
        toValue: 0,
        duration: 150,
        useNativeDriver: false,
      }).start();
      onBlur?.(e);
    };

    // Determine border color based on state
    const getBorderColor = () => {
      if (error) return colors.status.error;
      if (isFocused) return colors.primary.main;
      return colors.border.secondary;
    };

    const animatedBorderColor = borderColorAnim.interpolate({
      inputRange: [0, 1],
      outputRange: [
        error ? colors.status.error : colors.border.secondary,
        error ? colors.status.error : colors.primary.main,
      ],
    });

    return (
      <View style={[styles.container, containerStyle]} testID={testID}>
        {/* Label */}
        <Text
          style={[
            styles.label,
            error && styles.labelError,
            disabled && styles.labelDisabled,
          ]}
          accessibilityRole="text"
        >
          {label}
        </Text>

        {/* Input Container */}
        <Animated.View
          style={[
            styles.inputContainer,
            { borderColor: animatedBorderColor },
            error && styles.inputContainerError,
            disabled && styles.inputContainerDisabled,
          ]}
        >
          {leftIcon && <View style={styles.leftIconContainer}>{leftIcon}</View>}

          <TextInput
            ref={inputRef}
            style={[
              styles.input,
              leftIcon && styles.inputWithLeftIcon,
              rightIcon && styles.inputWithRightIcon,
              disabled && styles.inputDisabled,
              inputStyle,
            ]}
            value={value}
            onChangeText={onChangeText}
            onFocus={handleFocus}
            onBlur={handleBlur}
            editable={!disabled}
            placeholder={placeholder}
            placeholderTextColor={colors.text.disabled}
            selectionColor={colors.primary.main}
            accessibilityLabel={label}
            accessibilityHint={helperText}
            accessibilityState={{
              disabled,
            }}
            {...rest}
          />

          {rightIcon && <View style={styles.rightIconContainer}>{rightIcon}</View>}
        </Animated.View>

        {/* Error Message or Helper Text */}
        <View
          style={styles.messageContainer}
          accessibilityLiveRegion={error ? 'polite' : 'none'}
        >
          {error ? (
            <Text
              style={styles.errorText}
              accessibilityRole="alert"
              testID={testID ? `${testID}-error` : undefined}
            >
              {error}
            </Text>
          ) : helperText ? (
            <Text
              style={styles.helperText}
              testID={testID ? `${testID}-helper` : undefined}
            >
              {helperText}
            </Text>
          ) : null}
        </View>
      </View>
    );
  }
);

Input.displayName = 'Input';

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.md,
  },

  // Label
  label: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
    letterSpacing: 0.3,
  },
  labelError: {
    color: colors.status.error,
  },
  labelDisabled: {
    color: colors.text.disabled,
  },

  // Input Container
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    overflow: 'hidden',
  },
  inputContainerError: {
    borderColor: colors.status.error,
    backgroundColor: colors.special.errorLight,
  },
  inputContainerDisabled: {
    backgroundColor: colors.background.elevated,
    opacity: 0.6,
  },

  // Input
  input: {
    flex: 1,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    height: 52,
    textAlignVertical: 'center',
    ...Platform.select({
      web: {
        outlineStyle: 'none',
      },
    }),
  },
  inputWithLeftIcon: {
    paddingLeft: spacing.xs,
  },
  inputWithRightIcon: {
    paddingRight: spacing.xs,
  },
  inputDisabled: {
    color: colors.text.disabled,
  },

  // Icons
  leftIconContainer: {
    paddingLeft: spacing.md,
    justifyContent: 'center',
    alignItems: 'center',
  },
  rightIconContainer: {
    paddingRight: spacing.md,
    justifyContent: 'center',
    alignItems: 'center',
  },

  // Messages
  messageContainer: {
    minHeight: spacing.lg,
    marginTop: spacing.xs,
  },
  errorText: {
    fontSize: typography.fontSize.sm,
    color: colors.status.error,
    fontWeight: typography.fontWeight.medium,
  },
  helperText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
});

export default Input;
