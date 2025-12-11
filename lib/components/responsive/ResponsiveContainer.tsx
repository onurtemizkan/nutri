/**
 * ResponsiveContainer Component
 *
 * A container component that applies responsive padding and optional max-width
 * constraints based on the current device category.
 *
 * @example
 * ```tsx
 * <ResponsiveContainer>
 *   <YourContent />
 * </ResponsiveContainer>
 *
 * // With custom max width for forms
 * <ResponsiveContainer maxWidth={480} centerOnTablet>
 *   <FormContent />
 * </ResponsiveContainer>
 * ```
 */

import React from 'react';
import { View, ViewStyle, StyleSheet, StyleProp } from 'react-native';
import { useResponsive } from '@/hooks/useResponsive';
import { createContainerPadding, createFormContainerStyle } from '@/lib/responsive/spacing';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';

export interface ResponsiveContainerProps {
  /** Child elements */
  children: React.ReactNode;
  /** Additional style to apply */
  style?: StyleProp<ViewStyle>;
  /** Custom max width (overrides default for device) */
  maxWidth?: number;
  /** Whether to center content on tablets */
  centerOnTablet?: boolean;
  /** Whether this is a form container (uses narrower max width) */
  isForm?: boolean;
  /** Whether to apply vertical padding */
  verticalPadding?: boolean;
  /** Test ID for testing */
  testID?: string;
}

/**
 * ResponsiveContainer
 *
 * Automatically applies responsive horizontal padding and optional max-width
 * constraints based on the device category.
 *
 * - On phones: Full width with appropriate padding
 * - On tablets: Constrained width, centered, with larger padding
 */
export function ResponsiveContainer({
  children,
  style,
  maxWidth,
  centerOnTablet = true,
  isForm = false,
  verticalPadding = false,
  testID,
}: ResponsiveContainerProps): React.JSX.Element {
  const { deviceCategory, width, getSpacing } = useResponsive();
  const spacing = getSpacing();

  // Get base container style
  const containerStyle: ViewStyle = isForm
    ? createFormContainerStyle(deviceCategory, width, maxWidth ?? FORM_MAX_WIDTH)
    : createContainerPadding(deviceCategory, width);

  // Apply custom max width if provided
  if (maxWidth && !isForm) {
    if (width > maxWidth) {
      containerStyle.maxWidth = maxWidth;
      if (centerOnTablet) {
        containerStyle.alignSelf = 'center';
        containerStyle.width = '100%';
      }
    }
  }

  // Add vertical padding if requested
  if (verticalPadding) {
    containerStyle.paddingVertical = spacing.vertical;
  }

  return (
    <View style={[styles.container, containerStyle, style]} testID={testID}>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

export default ResponsiveContainer;
