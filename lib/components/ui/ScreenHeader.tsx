/**
 * Shared ScreenHeader Component
 *
 * A consistent header component for all screens with back navigation,
 * title, and optional right actions. Handles safe area insets.
 */

import React from 'react';
import { View, Text, Pressable, StyleSheet, Platform } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, typography } from '@/lib/theme/colors';

// ============================================================================
// TYPES
// ============================================================================

export type TitleAlign = 'left' | 'center';

export interface ScreenHeaderProps {
  /** Screen title */
  title: string;
  /** Whether to show back button (default: true) */
  showBackButton?: boolean;
  /** Custom back press handler (defaults to router.back()) */
  onBackPress?: () => void;
  /** Right side action buttons/elements */
  rightActions?: React.ReactNode;
  /** Whether header background is transparent */
  transparent?: boolean;
  /** Title alignment (default: 'center') */
  titleAlign?: TitleAlign;
  /** Optional test ID for testing */
  testID?: string;
}

// ============================================================================
// CONSTANTS
// ============================================================================

const HEADER_HEIGHT = 56;
const BACK_BUTTON_SIZE = 44; // Apple HIG minimum tap target

// ============================================================================
// COMPONENT
// ============================================================================

export function ScreenHeader({
  title,
  showBackButton = true,
  onBackPress,
  rightActions,
  transparent = false,
  titleAlign = 'center',
  testID,
}: ScreenHeaderProps) {
  const insets = useSafeAreaInsets();
  const router = useRouter();

  // Handle back navigation
  const handleBackPress = () => {
    if (onBackPress) {
      onBackPress();
    } else {
      router.back();
    }
  };

  return (
    <View
      style={[styles.container, { paddingTop: insets.top }, transparent && styles.transparent]}
      testID={testID}
    >
      <View style={styles.header}>
        {/* Left Section - Back Button */}
        <View style={styles.leftSection}>
          {showBackButton && (
            <Pressable
              onPress={handleBackPress}
              style={styles.backButton}
              accessibilityRole="button"
              accessibilityLabel="Go back"
              accessibilityHint="Navigate to the previous screen"
              testID={testID ? `${testID}-back-button` : 'header-back-button'}
            >
              <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
            </Pressable>
          )}
        </View>

        {/* Center Section - Title */}
        <View style={[styles.titleContainer, titleAlign === 'left' && styles.titleContainerLeft]}>
          <Text
            style={[styles.title, titleAlign === 'left' && styles.titleLeft]}
            numberOfLines={1}
            accessibilityRole="header"
            testID={testID ? `${testID}-title` : 'header-title'}
          >
            {title}
          </Text>
        </View>

        {/* Right Section - Actions */}
        <View style={styles.rightSection}>{rightActions}</View>
      </View>
    </View>
  );
}

// ============================================================================
// STYLES
// ============================================================================

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.background.primary,
  },
  transparent: {
    backgroundColor: 'transparent',
  },
  header: {
    height: HEADER_HEIGHT,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.sm,
  },
  leftSection: {
    width: BACK_BUTTON_SIZE,
    alignItems: 'flex-start',
    justifyContent: 'center',
  },
  backButton: {
    width: BACK_BUTTON_SIZE,
    height: BACK_BUTTON_SIZE,
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: -spacing.xs, // Align icon visually with edge
  },
  titleContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  titleContainerLeft: {
    alignItems: 'flex-start',
    marginLeft: spacing.sm,
  },
  title: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    ...Platform.select({
      ios: {
        fontWeight: '600',
      },
      android: {
        fontWeight: 'bold',
      },
    }),
  },
  titleLeft: {
    textAlign: 'left',
  },
  rightSection: {
    minWidth: BACK_BUTTON_SIZE, // Match left section width for balance
    alignItems: 'flex-end',
    justifyContent: 'center',
    flexDirection: 'row',
  },
});

// ============================================================================
// EXPORTS
// ============================================================================

export default ScreenHeader;
