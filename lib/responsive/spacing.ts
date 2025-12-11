/**
 * Responsive Spacing System
 *
 * Provides responsive spacing values that adapt to different device categories.
 * Complements the static spacing in lib/theme/colors.ts with responsive scaling.
 *
 * @example
 * ```typescript
 * import { getResponsiveSpacing, createContainerPadding } from '@/lib/responsive/spacing';
 *
 * const padding = getResponsiveSpacing('md', 'tablet'); // Returns scaled spacing
 * const containerStyle = createContainerPadding('tablet', screenWidth);
 * ```
 */

import { ViewStyle } from 'react-native';
import type { DeviceCategory } from './types';
import {
  SPACING_MULTIPLIERS,
  CONTAINER_PADDING,
  MAX_CONTENT_WIDTH,
  FORM_MAX_WIDTH,
  MIN_TOUCH_TARGET,
} from './breakpoints';

/**
 * Base spacing scale (same as lib/theme/colors.ts spacing)
 * These are the reference values for iPhone 14 (medium device)
 */
export const BASE_SPACING = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  '2xl': 40,
  '3xl': 48,
  '4xl': 64,
} as const;

/**
 * Spacing keys
 */
export type SpacingKey = keyof typeof BASE_SPACING;

/**
 * Get responsive spacing based on device category
 *
 * @param size - The base spacing key
 * @param deviceCategory - Current device category
 * @returns The scaled spacing value
 */
export function getResponsiveSpacing(
  size: SpacingKey,
  deviceCategory: DeviceCategory
): number {
  const baseValue = BASE_SPACING[size];
  const multiplier = SPACING_MULTIPLIERS[deviceCategory];
  return Math.round(baseValue * multiplier);
}

/**
 * Get all spacing values for a device category
 *
 * @param deviceCategory - Current device category
 * @returns Object with all spacing values scaled for device
 */
export function getAllSpacing(
  deviceCategory: DeviceCategory
): Record<SpacingKey, number> {
  const multiplier = SPACING_MULTIPLIERS[deviceCategory];
  const result: Partial<Record<SpacingKey, number>> = {};

  for (const [key, value] of Object.entries(BASE_SPACING)) {
    result[key as SpacingKey] = Math.round(value * multiplier);
  }

  return result as Record<SpacingKey, number>;
}

/**
 * Get container horizontal padding for device category
 *
 * @param deviceCategory - Current device category
 * @returns Horizontal padding value
 */
export function getContainerPadding(deviceCategory: DeviceCategory): number {
  return CONTAINER_PADDING[deviceCategory];
}

/**
 * Create container padding style
 *
 * @param deviceCategory - Current device category
 * @param screenWidth - Current screen width
 * @returns ViewStyle with padding and optional maxWidth
 */
export function createContainerPadding(
  deviceCategory: DeviceCategory,
  screenWidth: number
): ViewStyle {
  const horizontalPadding = CONTAINER_PADDING[deviceCategory];
  const maxWidth = MAX_CONTENT_WIDTH[deviceCategory];

  const style: ViewStyle = {
    paddingHorizontal: horizontalPadding,
  };

  // Apply max width for tablets to prevent content from stretching too wide
  if (maxWidth !== null && screenWidth > maxWidth) {
    style.maxWidth = maxWidth;
    style.alignSelf = 'center';
    style.width = '100%';
  }

  return style;
}

/**
 * Create form container style with centered, constrained width
 *
 * @param deviceCategory - Current device category
 * @param screenWidth - Current screen width
 * @param maxFormWidth - Maximum form width (default FORM_MAX_WIDTH)
 * @returns ViewStyle for form container
 */
export function createFormContainerStyle(
  deviceCategory: DeviceCategory,
  screenWidth: number,
  maxFormWidth: number = FORM_MAX_WIDTH
): ViewStyle {
  const horizontalPadding = CONTAINER_PADDING[deviceCategory];

  const style: ViewStyle = {
    paddingHorizontal: horizontalPadding,
  };

  // Center forms on tablets for better readability
  if (deviceCategory === 'tablet') {
    style.maxWidth = maxFormWidth;
    style.alignSelf = 'center';
    style.width = '100%';
  }

  return style;
}

/**
 * Get responsive gap for flex layouts
 *
 * @param deviceCategory - Current device category
 * @param baseGap - Base gap value (default 12)
 * @returns Scaled gap value
 */
export function getResponsiveGap(
  deviceCategory: DeviceCategory,
  baseGap: number = 12
): number {
  const multiplier = SPACING_MULTIPLIERS[deviceCategory];
  return Math.round(baseGap * multiplier);
}

/**
 * Get responsive margin for components
 *
 * @param deviceCategory - Current device category
 * @param baseMargin - Base margin value (default 16)
 * @returns Scaled margin value
 */
export function getResponsiveMargin(
  deviceCategory: DeviceCategory,
  baseMargin: number = 16
): number {
  const multiplier = SPACING_MULTIPLIERS[deviceCategory];
  return Math.round(baseMargin * multiplier);
}

/**
 * Calculate safe area aware padding
 * Adds to existing padding to account for safe areas
 *
 * @param basePadding - Base padding value
 * @param safeAreaInset - Safe area inset to add
 * @returns Combined padding value
 */
export function getSafeAreaPadding(
  basePadding: number,
  safeAreaInset: number
): number {
  return basePadding + safeAreaInset;
}

/**
 * Pre-defined spacing presets for common use cases
 */
export const SPACING_PRESETS = {
  // Screen padding
  screenPadding: { horizontal: 'md' as SpacingKey, vertical: 'lg' as SpacingKey },

  // Card spacing
  cardPadding: { horizontal: 'md' as SpacingKey, vertical: 'md' as SpacingKey },
  cardMargin: { horizontal: 'sm' as SpacingKey, vertical: 'sm' as SpacingKey },

  // List spacing
  listItemPadding: { horizontal: 'md' as SpacingKey, vertical: 'sm' as SpacingKey },
  listGap: 'sm' as SpacingKey,

  // Form spacing
  formGap: 'lg' as SpacingKey,
  inputPadding: { horizontal: 'md' as SpacingKey, vertical: 'md' as SpacingKey },

  // Button spacing
  buttonPadding: { horizontal: 'lg' as SpacingKey, vertical: 'md' as SpacingKey },

  // Section spacing
  sectionMargin: '2xl' as SpacingKey,
  sectionGap: 'xl' as SpacingKey,
} as const;

/**
 * Get preset spacing values for device category
 *
 * @param preset - Spacing preset name
 * @param deviceCategory - Current device category
 * @returns Spacing value(s) for the preset
 */
export function getPresetSpacing(
  preset: keyof typeof SPACING_PRESETS,
  deviceCategory: DeviceCategory
): number | { horizontal: number; vertical: number } {
  const presetValue = SPACING_PRESETS[preset];

  if (typeof presetValue === 'string') {
    return getResponsiveSpacing(presetValue, deviceCategory);
  }

  return {
    horizontal: getResponsiveSpacing(presetValue.horizontal, deviceCategory),
    vertical: getResponsiveSpacing(presetValue.vertical, deviceCategory),
  };
}

/**
 * Minimum touch target size (44pt per Apple HIG)
 */
export { MIN_TOUCH_TARGET };

/**
 * Ensure a size meets minimum touch target requirements
 *
 * @param size - The proposed size
 * @returns Size that's at least MIN_TOUCH_TARGET
 */
export function ensureMinTouchTarget(size: number): number {
  return Math.max(size, MIN_TOUCH_TARGET);
}

/**
 * Get responsive input/button height
 *
 * @param deviceCategory - Current device category
 * @param baseHeight - Base height (default 52)
 * @returns Scaled height that meets min touch target
 */
export function getResponsiveControlHeight(
  deviceCategory: DeviceCategory,
  baseHeight: number = 52
): number {
  const multiplier = SPACING_MULTIPLIERS[deviceCategory];
  return ensureMinTouchTarget(Math.round(baseHeight * multiplier));
}
