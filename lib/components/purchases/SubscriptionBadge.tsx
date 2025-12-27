/**
 * SubscriptionBadge Component
 *
 * Displays a "Pro" badge when user has an active subscription.
 * Can be used in headers, profile screens, and feature cards.
 */

import React from 'react';
import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';

interface SubscriptionBadgeProps {
  /** Size variant of the badge */
  size?: 'small' | 'medium' | 'large';
  /** Whether to show the star icon */
  showIcon?: boolean;
  /** Additional styles for the container */
  style?: ViewStyle;
}

export function SubscriptionBadge({
  size = 'medium',
  showIcon = true,
  style,
}: SubscriptionBadgeProps) {
  const sizeStyles = getSizeStyles(size);

  return (
    <LinearGradient
      colors={[colors.primary.main, colors.primary.gradient.end]}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 0 }}
      style={[styles.container, sizeStyles.container, style]}
    >
      {showIcon && (
        <Ionicons
          name="star"
          size={sizeStyles.iconSize}
          color={colors.text.primary}
          style={styles.icon}
        />
      )}
      <Text style={[styles.text, sizeStyles.text]}>PRO</Text>
    </LinearGradient>
  );
}

/**
 * Get size-specific styles
 */
function getSizeStyles(size: 'small' | 'medium' | 'large') {
  switch (size) {
    case 'small':
      return {
        container: {
          paddingHorizontal: spacing.sm,
          paddingVertical: 2,
          borderRadius: borderRadius.sm,
        },
        text: {
          fontSize: 10,
        },
        iconSize: 10,
      };
    case 'large':
      return {
        container: {
          paddingHorizontal: spacing.md,
          paddingVertical: spacing.xs,
          borderRadius: borderRadius.md,
        },
        text: {
          fontSize: 14,
        },
        iconSize: 14,
      };
    case 'medium':
    default:
      return {
        container: {
          paddingHorizontal: spacing.sm,
          paddingVertical: spacing.xs,
          borderRadius: borderRadius.sm,
        },
        text: {
          fontSize: 12,
        },
        iconSize: 12,
      };
  }
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
  },
  icon: {
    marginRight: 4,
  },
  text: {
    color: colors.text.primary,
    fontWeight: typography.fontWeight.bold,
    letterSpacing: 1,
  },
});

export default SubscriptionBadge;
