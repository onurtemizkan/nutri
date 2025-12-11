/**
 * Device Breakpoints and Constants
 *
 * Screen dimensions for all supported iPhone (2020+) and iPad devices.
 * All values are in logical pixels (points), not physical pixels.
 *
 * Reference: https://www.ios-resolution.com/
 */

import type {
  DeviceModel,
  DeviceCategory,
  ScreenDimensions,
  DeviceBreakpoint,
} from './types';

/**
 * Screen dimensions for each device model (in points)
 * Portrait orientation (width x height)
 */
export const DEVICE_DIMENSIONS: Record<DeviceModel, ScreenDimensions> = {
  // iPhones - Small category
  iPhoneSE: { width: 375, height: 667 }, // iPhone SE 3rd gen (2022) - 4.7"
  iPhoneMini: { width: 375, height: 812 }, // iPhone 12/13 Mini - 5.4"

  // iPhones - Medium category
  iPhoneMedium: { width: 390, height: 844 }, // iPhone 12/13/14 - 6.1"
  iPhonePro: { width: 393, height: 852 }, // iPhone 14/15/16 Pro - 6.1"

  // iPhones - Large category
  iPhoneMax: { width: 430, height: 932 }, // iPhone Pro Max/Plus - 6.7"

  // iPads
  iPadMini: { width: 744, height: 1133 }, // iPad Mini 6th gen - 8.3"
  iPad: { width: 820, height: 1180 }, // iPad 10th gen / iPad Air 11" - 10.9"
  iPadPro11: { width: 834, height: 1194 }, // iPad Pro 11" - 11"
  iPadAir13: { width: 1032, height: 1376 }, // iPad Air 13" - 13"
  iPadPro13: { width: 1024, height: 1366 }, // iPad Pro 13" - 13"
} as const;

/**
 * Device category mappings
 */
export const DEVICE_CATEGORIES: Record<DeviceCategory, DeviceModel[]> = {
  small: ['iPhoneSE', 'iPhoneMini'],
  medium: ['iPhoneMedium', 'iPhonePro'],
  large: ['iPhoneMax'],
  tablet: ['iPadMini', 'iPad', 'iPadPro11', 'iPadAir13', 'iPadPro13'],
} as const;

/**
 * Breakpoint thresholds for device category detection
 * Based on width in portrait orientation
 */
export const CATEGORY_BREAKPOINTS = {
  /** Minimum width for medium category (exclusive of small) */
  MEDIUM_MIN_WIDTH: 388,
  /** Minimum width for large category */
  LARGE_MIN_WIDTH: 420,
  /** Minimum width for tablet category */
  TABLET_MIN_WIDTH: 700,
} as const;

/**
 * Detailed breakpoints for each device model
 * Used for precise device detection
 */
export const DEVICE_BREAKPOINTS: DeviceBreakpoint[] = [
  // Small iPhones
  {
    minWidth: 0,
    maxWidth: 376,
    minHeight: 0,
    maxHeight: 700,
    category: 'small',
    name: 'iPhone SE',
  },
  {
    minWidth: 0,
    maxWidth: 376,
    minHeight: 700,
    maxHeight: 850,
    category: 'small',
    name: 'iPhone Mini',
  },

  // Medium iPhones
  {
    minWidth: 376,
    maxWidth: 392,
    minHeight: 800,
    maxHeight: 900,
    category: 'medium',
    name: 'iPhone Standard',
  },
  {
    minWidth: 392,
    maxWidth: 420,
    minHeight: 800,
    maxHeight: 900,
    category: 'medium',
    name: 'iPhone Pro',
  },

  // Large iPhones
  {
    minWidth: 420,
    maxWidth: 700,
    minHeight: 900,
    maxHeight: 1000,
    category: 'large',
    name: 'iPhone Pro Max / Plus',
  },

  // iPads
  {
    minWidth: 700,
    maxWidth: 800,
    minHeight: 1000,
    maxHeight: 1200,
    category: 'tablet',
    name: 'iPad Mini',
  },
  {
    minWidth: 800,
    maxWidth: 900,
    minHeight: 1100,
    maxHeight: 1250,
    category: 'tablet',
    name: 'iPad / iPad Air 11"',
  },
  {
    minWidth: 900,
    maxWidth: 1100,
    minHeight: 1200,
    maxHeight: 1500,
    category: 'tablet',
    name: 'iPad Pro / iPad Air 13"',
  },
] as const;

/**
 * Base design dimensions (iPhone 14 as reference)
 * Used for scaling calculations
 */
export const BASE_DESIGN = {
  /** Base width for scaling (iPhone 14) */
  WIDTH: 390,
  /** Base height for scaling (iPhone 14) */
  HEIGHT: 844,
  /** Base font scale */
  FONT_SCALE: 1,
} as const;

/**
 * Minimum touch target size (Apple HIG)
 * Ensures accessibility compliance
 */
export const MIN_TOUCH_TARGET = 44;

/**
 * Responsive spacing multipliers per device category
 */
export const SPACING_MULTIPLIERS: Record<DeviceCategory, number> = {
  small: 0.85,
  medium: 1,
  large: 1.1,
  tablet: 1.25,
} as const;

/**
 * Font scale multipliers per device category
 */
export const FONT_SCALE_MULTIPLIERS: Record<DeviceCategory, number> = {
  small: 0.92,
  medium: 1,
  large: 1.05,
  tablet: 1.1,
} as const;

/**
 * Maximum content widths for different device categories
 * Prevents content from stretching too wide on large screens
 */
export const MAX_CONTENT_WIDTH: Record<DeviceCategory, number | null> = {
  small: null, // No max width
  medium: null, // No max width
  large: null, // No max width
  tablet: 600, // Constrain content width on tablets
} as const;

/**
 * Container horizontal padding per device category
 */
export const CONTAINER_PADDING: Record<DeviceCategory, number> = {
  small: 16,
  medium: 20,
  large: 24,
  tablet: 32,
} as const;

/**
 * Form max width on tablets (centered forms look better)
 */
export const FORM_MAX_WIDTH = 480;

/**
 * Card min width for grid layouts
 */
export const CARD_MIN_WIDTH = 160;

/**
 * Grid columns per device category
 */
export const GRID_COLUMNS: Record<DeviceCategory, number> = {
  small: 1,
  medium: 1,
  large: 2,
  tablet: 2,
} as const;

/**
 * Landscape grid columns (tablet only)
 */
export const LANDSCAPE_GRID_COLUMNS = 3;
