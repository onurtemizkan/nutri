/**
 * Shared EmptyState Component
 *
 * A reusable empty state component for screens/sections with no data.
 * Displays an icon, title, description, and optional call-to-action button.
 */

import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, gradients, spacing, borderRadius, typography } from '@/lib/theme/colors';

// ============================================================================
// TYPES
// ============================================================================

export interface EmptyStateProps {
  /** Ionicons icon name to display */
  icon: keyof typeof Ionicons.glyphMap;
  /** Main title text */
  title: string;
  /** Description text below title */
  description: string;
  /** Optional CTA button label */
  actionLabel?: string;
  /** Optional CTA button press handler */
  onAction?: () => void;
  /** Optional test ID for testing */
  testID?: string;
}

// ============================================================================
// COMPONENT
// ============================================================================

export function EmptyState({
  icon,
  title,
  description,
  actionLabel,
  onAction,
  testID,
}: EmptyStateProps) {
  return (
    <View style={styles.container} testID={testID}>
      {/* Icon */}
      <Ionicons
        name={icon}
        size={64}
        color={colors.text.disabled}
        testID={testID ? `${testID}-icon` : 'empty-state-icon'}
      />

      {/* Title */}
      <Text
        style={styles.title}
        accessibilityRole="header"
        testID={testID ? `${testID}-title` : 'empty-state-title'}
      >
        {title}
      </Text>

      {/* Description */}
      <Text
        style={styles.description}
        testID={testID ? `${testID}-description` : 'empty-state-description'}
      >
        {description}
      </Text>

      {/* CTA Button (only if actionLabel provided) */}
      {actionLabel && onAction && (
        <Pressable
          onPress={onAction}
          accessibilityRole="button"
          accessibilityLabel={actionLabel}
          testID={testID ? `${testID}-action` : 'empty-state-action'}
          style={styles.buttonWrapper}
        >
          <LinearGradient
            colors={gradients.primary as unknown as string[]}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.button}
          >
            <Text style={styles.buttonText}>{actionLabel}</Text>
          </LinearGradient>
        </Pressable>
      )}
    </View>
  );
}

// ============================================================================
// STYLES
// ============================================================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.xl,
  },
  title: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginTop: spacing.md,
    textAlign: 'center',
  },
  description: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
    textAlign: 'center',
    marginTop: spacing.sm,
    lineHeight: typography.lineHeight.normal * typography.fontSize.md,
  },
  buttonWrapper: {
    marginTop: spacing.lg,
  },
  button: {
    height: 44,
    paddingHorizontal: spacing.lg,
    borderRadius: borderRadius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
});

// ============================================================================
// EXPORTS
// ============================================================================

export default EmptyState;
