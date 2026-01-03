/**
 * Shared Card Component
 *
 * A reusable Card component with consistent styling, shadows,
 * and optional press handling for interactive cards.
 */

import React, { useRef, useCallback } from 'react';
import { View, Pressable, StyleSheet, ViewStyle, Animated, AccessibilityRole } from 'react-native';
import { colors, shadows, spacing, borderRadius } from '@/lib/theme/colors';

// ============================================================================
// TYPES
// ============================================================================

export type CardVariant = 'elevated' | 'outlined' | 'filled';
export type CardPadding = 'none' | 'sm' | 'md' | 'lg';

export interface CardProps {
  /** Visual variant of the card */
  variant?: CardVariant;
  /** Card content */
  children: React.ReactNode;
  /** Press handler - if provided, card becomes pressable */
  onPress?: () => void;
  /** Additional styles */
  style?: ViewStyle;
  /** Padding size */
  padding?: CardPadding;
  /** Accessibility label for pressable cards */
  accessibilityLabel?: string;
  /** Accessibility hint for additional context */
  accessibilityHint?: string;
  /** Optional test ID for testing */
  testID?: string;
}

// ============================================================================
// PADDING CONFIGURATIONS
// ============================================================================

const PADDING_CONFIG = {
  none: 0,
  sm: spacing.sm,
  md: spacing.md,
  lg: spacing.lg,
} as const;

// ============================================================================
// COMPONENT
// ============================================================================

export function Card({
  variant = 'elevated',
  children,
  onPress,
  style,
  padding = 'md',
  accessibilityLabel,
  accessibilityHint,
  testID,
}: CardProps) {
  // Animation value for press feedback
  const scaleValue = useRef(new Animated.Value(1)).current;

  // Handle press in - animate scale down
  const handlePressIn = useCallback(() => {
    Animated.spring(scaleValue, {
      toValue: 0.98,
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

  // Get variant-specific styles
  const getVariantStyles = (): ViewStyle => {
    switch (variant) {
      case 'elevated':
        return {
          backgroundColor: colors.surface.card,
          ...shadows.md,
        };
      case 'outlined':
        return {
          backgroundColor: colors.surface.card,
          borderWidth: 1,
          borderColor: colors.border.secondary,
        };
      case 'filled':
        return {
          backgroundColor: colors.background.secondary,
        };
      default:
        return {
          backgroundColor: colors.surface.card,
          ...shadows.md,
        };
    }
  };

  // Combined styles
  const cardStyles: ViewStyle[] = [
    styles.container,
    getVariantStyles(),
    { padding: PADDING_CONFIG[padding] },
    style,
  ];

  // Animated container style
  const animatedStyle = {
    transform: [{ scale: scaleValue }],
  };

  // If pressable, wrap with Pressable and Animated.View
  if (onPress) {
    return (
      <Animated.View style={animatedStyle}>
        <Pressable
          onPress={onPress}
          onPressIn={handlePressIn}
          onPressOut={handlePressOut}
          accessibilityRole={'button' as AccessibilityRole}
          accessibilityLabel={accessibilityLabel}
          accessibilityHint={accessibilityHint}
          testID={testID}
          style={cardStyles}
        >
          {children}
        </Pressable>
      </Animated.View>
    );
  }

  // Non-pressable card
  return (
    <View style={cardStyles} testID={testID}>
      {children}
    </View>
  );
}

// ============================================================================
// STYLES
// ============================================================================

const styles = StyleSheet.create({
  container: {
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
  },
});

// ============================================================================
// EXPORTS
// ============================================================================

export default Card;
