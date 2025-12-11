/**
 * Responsive Typography System
 *
 * Provides responsive font sizes that adapt to different device categories.
 * Complements the static typography in lib/theme/colors.ts with responsive scaling.
 *
 * @example
 * ```typescript
 * import { getResponsiveFontSize, createResponsiveTextStyle } from '@/lib/responsive/typography';
 *
 * const fontSize = getResponsiveFontSize('lg', 'tablet'); // Returns scaled font size
 * const style = createResponsiveTextStyle('lg', 'bold', 'tablet');
 * ```
 */

import { PixelRatio, TextStyle } from 'react-native';
import type { DeviceCategory } from './types';
import { FONT_SCALE_MULTIPLIERS, MIN_TOUCH_TARGET } from './breakpoints';

/**
 * Base font sizes (same as lib/theme/colors.ts typography.fontSize)
 * These are the reference values for iPhone 14 (medium device)
 */
export const BASE_FONT_SIZES = {
  xs: 12,
  sm: 14,
  md: 16,
  lg: 18,
  xl: 20,
  '2xl': 24,
  '3xl': 30,
  '4xl': 36,
  '5xl': 48,
} as const;

/**
 * Font size keys
 */
export type FontSizeKey = keyof typeof BASE_FONT_SIZES;

/**
 * Font weight values
 */
export const FONT_WEIGHTS = {
  regular: '400' as const,
  medium: '500' as const,
  semibold: '600' as const,
  bold: '700' as const,
};

/**
 * Font weight keys
 */
export type FontWeightKey = keyof typeof FONT_WEIGHTS;

/**
 * Line height multipliers
 */
export const LINE_HEIGHTS = {
  tight: 1.2,
  normal: 1.5,
  relaxed: 1.75,
} as const;

/**
 * Line height keys
 */
export type LineHeightKey = keyof typeof LINE_HEIGHTS;

/**
 * Letter spacing values for different text sizes
 */
export const LETTER_SPACING = {
  tight: -0.5,
  normal: 0,
  wide: 0.3,
  wider: 0.5,
} as const;

/**
 * Get a responsive font size based on device category
 *
 * @param size - The base font size key
 * @param deviceCategory - Current device category
 * @param systemFontScale - System font scale (default 1)
 * @returns The scaled font size
 */
export function getResponsiveFontSize(
  size: FontSizeKey,
  deviceCategory: DeviceCategory,
  systemFontScale: number = 1
): number {
  const baseSize = BASE_FONT_SIZES[size];
  const categoryMultiplier = FONT_SCALE_MULTIPLIERS[deviceCategory];

  // Round to nearest pixel for crisp rendering
  return PixelRatio.roundToNearestPixel(
    baseSize * categoryMultiplier * systemFontScale
  );
}

/**
 * Get responsive line height based on font size
 *
 * @param fontSize - The font size in points
 * @param lineHeightKey - The line height multiplier key
 * @returns The line height in points
 */
export function getResponsiveLineHeight(
  fontSize: number,
  lineHeightKey: LineHeightKey = 'normal'
): number {
  const multiplier = LINE_HEIGHTS[lineHeightKey];
  return Math.round(fontSize * multiplier);
}

/**
 * Create a complete responsive text style object
 *
 * @param sizeKey - Font size key
 * @param weightKey - Font weight key
 * @param deviceCategory - Current device category
 * @param options - Additional options
 * @returns TextStyle object
 */
export function createResponsiveTextStyle(
  sizeKey: FontSizeKey,
  weightKey: FontWeightKey,
  deviceCategory: DeviceCategory,
  options: {
    lineHeight?: LineHeightKey;
    letterSpacing?: keyof typeof LETTER_SPACING;
    systemFontScale?: number;
  } = {}
): TextStyle {
  const {
    lineHeight = 'normal',
    letterSpacing = 'normal',
    systemFontScale = 1,
  } = options;

  const fontSize = getResponsiveFontSize(sizeKey, deviceCategory, systemFontScale);

  return {
    fontSize,
    fontWeight: FONT_WEIGHTS[weightKey],
    lineHeight: getResponsiveLineHeight(fontSize, lineHeight),
    letterSpacing: LETTER_SPACING[letterSpacing],
  };
}

/**
 * Pre-defined responsive text styles for common use cases
 */
export const TEXT_STYLES = {
  // Headings
  h1: { size: '5xl' as FontSizeKey, weight: 'bold' as FontWeightKey, lineHeight: 'tight' as LineHeightKey },
  h2: { size: '4xl' as FontSizeKey, weight: 'bold' as FontWeightKey, lineHeight: 'tight' as LineHeightKey },
  h3: { size: '3xl' as FontSizeKey, weight: 'semibold' as FontWeightKey, lineHeight: 'tight' as LineHeightKey },
  h4: { size: '2xl' as FontSizeKey, weight: 'semibold' as FontWeightKey, lineHeight: 'normal' as LineHeightKey },
  h5: { size: 'xl' as FontSizeKey, weight: 'semibold' as FontWeightKey, lineHeight: 'normal' as LineHeightKey },
  h6: { size: 'lg' as FontSizeKey, weight: 'semibold' as FontWeightKey, lineHeight: 'normal' as LineHeightKey },

  // Body text
  bodyLarge: { size: 'lg' as FontSizeKey, weight: 'regular' as FontWeightKey, lineHeight: 'relaxed' as LineHeightKey },
  body: { size: 'md' as FontSizeKey, weight: 'regular' as FontWeightKey, lineHeight: 'normal' as LineHeightKey },
  bodySmall: { size: 'sm' as FontSizeKey, weight: 'regular' as FontWeightKey, lineHeight: 'normal' as LineHeightKey },

  // Labels and captions
  label: { size: 'sm' as FontSizeKey, weight: 'semibold' as FontWeightKey, lineHeight: 'normal' as LineHeightKey },
  caption: { size: 'xs' as FontSizeKey, weight: 'regular' as FontWeightKey, lineHeight: 'normal' as LineHeightKey },

  // Buttons
  buttonLarge: { size: 'lg' as FontSizeKey, weight: 'semibold' as FontWeightKey, lineHeight: 'tight' as LineHeightKey },
  button: { size: 'md' as FontSizeKey, weight: 'semibold' as FontWeightKey, lineHeight: 'tight' as LineHeightKey },
  buttonSmall: { size: 'sm' as FontSizeKey, weight: 'semibold' as FontWeightKey, lineHeight: 'tight' as LineHeightKey },
} as const;

/**
 * Text style preset keys
 */
export type TextStylePreset = keyof typeof TEXT_STYLES;

/**
 * Get a pre-defined responsive text style
 *
 * @param preset - The text style preset name
 * @param deviceCategory - Current device category
 * @param systemFontScale - System font scale
 * @returns TextStyle object
 */
export function getTextStyle(
  preset: TextStylePreset,
  deviceCategory: DeviceCategory,
  systemFontScale: number = 1
): TextStyle {
  const config = TEXT_STYLES[preset];
  return createResponsiveTextStyle(config.size, config.weight, deviceCategory, {
    lineHeight: config.lineHeight,
    systemFontScale,
  });
}

/**
 * Calculate minimum font size for accessibility (12pt minimum recommended)
 */
export const MIN_ACCESSIBLE_FONT_SIZE = 12;

/**
 * Ensure font size meets minimum accessibility requirements
 *
 * @param fontSize - The calculated font size
 * @returns Font size that's at least MIN_ACCESSIBLE_FONT_SIZE
 */
export function ensureMinFontSize(fontSize: number): number {
  return Math.max(fontSize, MIN_ACCESSIBLE_FONT_SIZE);
}
