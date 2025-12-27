/**
 * Shared Button Component
 *
 * A reusable button component that provides consistent styling, accessibility,
 * and interaction feedback across the entire application.
 */

import React, { useRef, useCallback } from 'react';
import {
  Pressable,
  Text,
  StyleSheet,
  ActivityIndicator,
  Animated,
  View,
  AccessibilityRole,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, gradients, spacing, borderRadius, typography } from '@/lib/theme/colors';

// ============================================================================
// TYPES
// ============================================================================

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'destructive';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps {
  /** Visual variant of the button */
  variant?: ButtonVariant;
  /** Size of the button */
  size?: ButtonSize;
  /** Button text label */
  label: string;
  /** Accessibility label for screen readers (required for a11y) */
  accessibilityLabel: string;
  /** Press handler */
  onPress: () => void;
  /** Whether the button is disabled */
  disabled?: boolean;
  /** Whether the button is in loading state */
  loading?: boolean;
  /** Icon to display on the left side of the label */
  leftIcon?: React.ReactNode;
  /** Icon to display on the right side of the label */
  rightIcon?: React.ReactNode;
  /** Whether the button should take full width of its container */
  fullWidth?: boolean;
  /** Optional test ID for testing */
  testID?: string;
}

// ============================================================================
// SIZE CONFIGURATIONS
// ============================================================================

const SIZE_CONFIG = {
  sm: {
    height: 36,
    paddingHorizontal: spacing.sm + spacing.xs, // 12
    fontSize: typography.fontSize.sm, // 14
    iconSize: 16,
  },
  md: {
    height: 44, // Apple HIG minimum tap target
    paddingHorizontal: spacing.md, // 16
    fontSize: typography.fontSize.md, // 16
    iconSize: 20,
  },
  lg: {
    height: 52,
    paddingHorizontal: spacing.lg, // 24
    fontSize: typography.fontSize.lg, // 18
    iconSize: 24,
  },
} as const;

// ============================================================================
// COMPONENT
// ============================================================================

export function Button({
  variant = 'primary',
  size = 'md',
  label,
  accessibilityLabel,
  onPress,
  disabled = false,
  loading = false,
  leftIcon,
  rightIcon,
  fullWidth = false,
  testID,
}: ButtonProps) {
  // Animation value for press feedback
  const scaleValue = useRef(new Animated.Value(1)).current;

  // Get size configuration
  const sizeConfig = SIZE_CONFIG[size];

  // Handle press in - animate scale down
  const handlePressIn = useCallback(() => {
    Animated.spring(scaleValue, {
      toValue: 0.97,
      useNativeDriver: true,
      speed: 50,
      bounciness: 4,
    }).start();
  }, [scaleValue]);

  // Handle press out - animate scale back
  const handlePressOut = useCallback(() => {
    Animated.spring(scaleValue, {
      toValue: 1,
      useNativeDriver: true,
      speed: 50,
      bounciness: 4,
    }).start();
  }, [scaleValue]);

  // Determine if button is interactive
  const isInteractive = !disabled && !loading;

  // Get variant-specific styles
  const getVariantStyles = () => {
    switch (variant) {
      case 'primary':
        return {
          backgroundColor: undefined, // Uses gradient
          textColor: colors.text.primary,
          borderWidth: 0,
          borderColor: undefined,
        };
      case 'secondary':
        return {
          backgroundColor: 'transparent',
          textColor: colors.text.primary,
          borderWidth: 1,
          borderColor: colors.border.primary,
        };
      case 'ghost':
        return {
          backgroundColor: 'transparent',
          textColor: colors.primary.main,
          borderWidth: 0,
          borderColor: undefined,
        };
      case 'destructive':
        return {
          backgroundColor: colors.semantic.error,
          textColor: colors.text.primary,
          borderWidth: 0,
          borderColor: undefined,
        };
      default:
        return {
          backgroundColor: undefined,
          textColor: colors.text.primary,
          borderWidth: 0,
          borderColor: undefined,
        };
    }
  };

  const variantStyles = getVariantStyles();

  // Render button content
  const renderContent = () => (
    <View style={styles.contentContainer}>
      {loading ? (
        <ActivityIndicator
          size="small"
          color={variantStyles.textColor}
          testID={testID ? `${testID}-loading` : undefined}
        />
      ) : (
        <>
          {leftIcon && <View style={styles.leftIconContainer}>{leftIcon}</View>}
          <Text
            style={[
              styles.label,
              {
                fontSize: sizeConfig.fontSize,
                color: variantStyles.textColor,
              },
              disabled && styles.labelDisabled,
            ]}
          >
            {label}
          </Text>
          {rightIcon && <View style={styles.rightIconContainer}>{rightIcon}</View>}
        </>
      )}
    </View>
  );

  // Container styles
  const containerStyles = [
    styles.container,
    {
      height: sizeConfig.height,
      paddingHorizontal: sizeConfig.paddingHorizontal,
      borderWidth: variantStyles.borderWidth,
      borderColor: variantStyles.borderColor,
      backgroundColor: variantStyles.backgroundColor,
    },
    fullWidth && styles.fullWidth,
    disabled && styles.disabled,
  ];

  // Animated container style
  const animatedStyle = {
    transform: [{ scale: scaleValue }],
  };

  // For primary variant, wrap with LinearGradient
  if (variant === 'primary') {
    return (
      <Animated.View style={animatedStyle}>
        <Pressable
          onPress={isInteractive ? onPress : undefined}
          onPressIn={isInteractive ? handlePressIn : undefined}
          onPressOut={isInteractive ? handlePressOut : undefined}
          disabled={!isInteractive}
          accessibilityRole={'button' as AccessibilityRole}
          accessibilityLabel={accessibilityLabel}
          accessibilityState={{
            disabled: disabled,
            busy: loading,
          }}
          testID={testID}
          style={[fullWidth && styles.fullWidth, disabled && styles.disabled]}
        >
          <LinearGradient
            colors={gradients.primary as unknown as string[]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={[
              styles.container,
              styles.gradientContainer,
              {
                height: sizeConfig.height,
                paddingHorizontal: sizeConfig.paddingHorizontal,
              },
            ]}
          >
            {renderContent()}
          </LinearGradient>
        </Pressable>
      </Animated.View>
    );
  }

  // For other variants
  return (
    <Animated.View style={animatedStyle}>
      <Pressable
        onPress={isInteractive ? onPress : undefined}
        onPressIn={isInteractive ? handlePressIn : undefined}
        onPressOut={isInteractive ? handlePressOut : undefined}
        disabled={!isInteractive}
        accessibilityRole={'button' as AccessibilityRole}
        accessibilityLabel={accessibilityLabel}
        accessibilityState={{
          disabled: disabled,
          busy: loading,
        }}
        testID={testID}
        style={containerStyles}
      >
        {renderContent()}
      </Pressable>
    </Animated.View>
  );
}

// ============================================================================
// STYLES
// ============================================================================

const styles = StyleSheet.create({
  container: {
    borderRadius: borderRadius.md,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  gradientContainer: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  contentContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  label: {
    ...typography.button,
    textAlign: 'center',
  },
  labelDisabled: {
    opacity: 0.6,
  },
  leftIconContainer: {
    marginRight: spacing.xs,
  },
  rightIconContainer: {
    marginLeft: spacing.xs,
  },
  fullWidth: {
    width: '100%',
  },
  disabled: {
    opacity: 0.5,
  },
});

// ============================================================================
// EXPORTS
// ============================================================================

export default Button;
