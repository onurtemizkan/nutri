/**
 * Responsive Design Helper Functions
 *
 * Pure utility functions for device detection, scaling calculations,
 * and responsive value resolution.
 */

import { Platform, PixelRatio } from 'react-native';
import type {
  DeviceCategory,
  DeviceModel,
  PlatformType,
  ResponsiveValue,
  ResponsiveSpacing,
  ScreenDimensions,
} from './types';
import {
  DEVICE_DIMENSIONS,
  CATEGORY_BREAKPOINTS,
  BASE_DESIGN,
  SPACING_MULTIPLIERS,
  FONT_SCALE_MULTIPLIERS,
  MAX_CONTENT_WIDTH,
  CONTAINER_PADDING,
  MIN_TOUCH_TARGET,
  GRID_COLUMNS,
  LANDSCAPE_GRID_COLUMNS,
} from './breakpoints';

/**
 * Detect device category based on screen width
 *
 * @param width - Screen width in points
 * @returns The detected device category
 */
export function getDeviceCategory(width: number): DeviceCategory {
  if (width >= CATEGORY_BREAKPOINTS.TABLET_MIN_WIDTH) {
    return 'tablet';
  }
  if (width >= CATEGORY_BREAKPOINTS.LARGE_MIN_WIDTH) {
    return 'large';
  }
  if (width >= CATEGORY_BREAKPOINTS.MEDIUM_MIN_WIDTH) {
    return 'medium';
  }
  return 'small';
}

/**
 * Detect the closest matching device model based on screen dimensions
 *
 * @param width - Screen width in points
 * @param height - Screen height in points
 * @returns The detected device model
 */
export function getDeviceModel(width: number, height: number): DeviceModel {
  // Ensure we're working with portrait dimensions
  const portraitWidth = Math.min(width, height);
  const portraitHeight = Math.max(width, height);

  // Find the closest matching device
  let closestDevice: DeviceModel = 'iPhoneMedium';
  let closestDistance = Infinity;

  const deviceEntries = Object.entries(DEVICE_DIMENSIONS) as [
    DeviceModel,
    ScreenDimensions,
  ][];

  for (const [device, dimensions] of deviceEntries) {
    const distance =
      Math.abs(dimensions.width - portraitWidth) +
      Math.abs(dimensions.height - portraitHeight);

    if (distance < closestDistance) {
      closestDistance = distance;
      closestDevice = device;
    }
  }

  return closestDevice;
}

/**
 * Check if the device is a tablet
 *
 * @param width - Screen width in points
 * @returns True if the device is a tablet
 */
export function isTablet(width: number): boolean {
  return width >= CATEGORY_BREAKPOINTS.TABLET_MIN_WIDTH;
}

/**
 * Check if the device is in landscape orientation
 *
 * @param width - Screen width in points
 * @param height - Screen height in points
 * @returns True if in landscape orientation
 */
export function isLandscape(width: number, height: number): boolean {
  return width > height;
}

/**
 * Get the current platform type
 *
 * @returns The current platform
 */
export function getPlatform(): PlatformType {
  if (Platform.OS === 'ios') return 'ios';
  if (Platform.OS === 'android') return 'android';
  return 'web';
}

/**
 * Check if the current device is an iPad (iOS only)
 *
 * @returns True if the device is an iPad
 */
export function isIPad(): boolean {
  return Platform.OS === 'ios' && Platform.isPad === true;
}

/**
 * Scale a size value proportionally based on screen width
 *
 * @param size - The base size value
 * @param screenWidth - Current screen width
 * @returns The scaled size
 */
export function scale(size: number, screenWidth: number): number {
  const scaleFactor = screenWidth / BASE_DESIGN.WIDTH;
  return Math.round(size * scaleFactor);
}

/**
 * Scale a font size with optional system font scale respect
 *
 * @param size - The base font size
 * @param screenWidth - Current screen width
 * @param fontScale - System font scale (default 1)
 * @param respectSystemScale - Whether to apply system font scale
 * @returns The scaled font size
 */
export function scaleFont(
  size: number,
  screenWidth: number,
  fontScale: number = 1,
  respectSystemScale: boolean = true
): number {
  const category = getDeviceCategory(screenWidth);
  const categoryMultiplier = FONT_SCALE_MULTIPLIERS[category];
  const systemMultiplier = respectSystemScale ? fontScale : 1;

  // Use PixelRatio to round to nearest pixel
  return PixelRatio.roundToNearestPixel(
    size * categoryMultiplier * systemMultiplier
  );
}

/**
 * Get a responsive value based on device category
 *
 * @param values - Object with values for each device category
 * @param category - Current device category
 * @returns The appropriate value for the current device
 */
export function getResponsiveValue<T>(
  values: ResponsiveValue<T>,
  category: DeviceCategory
): T {
  // Try to get the specific value for this category
  const specificValue = values[category];
  if (specificValue !== undefined) {
    return specificValue;
  }

  // Fall back to default
  return values.default;
}

/**
 * Get responsive spacing values for current device
 *
 * @param category - Current device category
 * @returns Spacing values for the device
 */
export function getSpacing(category: DeviceCategory): ResponsiveSpacing {
  const multiplier = SPACING_MULTIPLIERS[category];
  const basePadding = CONTAINER_PADDING[category];

  return {
    horizontal: basePadding,
    vertical: Math.round(16 * multiplier),
    gap: Math.round(12 * multiplier),
  };
}

/**
 * Get maximum content width for current device
 *
 * @param category - Current device category
 * @param screenWidth - Current screen width
 * @returns Maximum content width or screen width if no max
 */
export function getMaxContentWidth(
  category: DeviceCategory,
  screenWidth: number
): number {
  const maxWidth = MAX_CONTENT_WIDTH[category];
  if (maxWidth === null) {
    return screenWidth;
  }
  return Math.min(maxWidth, screenWidth);
}

/**
 * Ensure a size meets minimum touch target requirements
 *
 * @param size - The proposed size
 * @returns The size, but at least MIN_TOUCH_TARGET
 */
export function ensureMinTouchTarget(size: number): number {
  return Math.max(size, MIN_TOUCH_TARGET);
}

/**
 * Get the number of grid columns for current device and orientation
 *
 * @param category - Current device category
 * @param landscape - Whether in landscape orientation
 * @returns Number of grid columns
 */
export function getGridColumns(
  category: DeviceCategory,
  landscape: boolean
): number {
  if (category === 'tablet' && landscape) {
    return LANDSCAPE_GRID_COLUMNS;
  }
  return GRID_COLUMNS[category];
}

/**
 * Calculate container style for responsive layout
 *
 * @param category - Current device category
 * @param screenWidth - Current screen width
 * @returns Style object for container
 */
export function getContainerStyle(
  category: DeviceCategory,
  screenWidth: number
): { paddingHorizontal: number; maxWidth?: number; alignSelf?: 'center' } {
  const padding = CONTAINER_PADDING[category];
  const maxWidth = MAX_CONTENT_WIDTH[category];

  if (maxWidth !== null && screenWidth > maxWidth) {
    return {
      paddingHorizontal: padding,
      maxWidth: maxWidth,
      alignSelf: 'center',
    };
  }

  return {
    paddingHorizontal: padding,
  };
}

/**
 * Calculate form container style (narrower max-width for forms)
 *
 * @param category - Current device category
 * @param screenWidth - Current screen width
 * @param maxFormWidth - Maximum form width (default 480)
 * @returns Style object for form container
 */
export function getFormContainerStyle(
  category: DeviceCategory,
  screenWidth: number,
  maxFormWidth: number = 480
): { paddingHorizontal: number; maxWidth?: number; alignSelf?: 'center' } {
  const padding = CONTAINER_PADDING[category];

  if (category === 'tablet') {
    return {
      paddingHorizontal: padding,
      maxWidth: maxFormWidth,
      alignSelf: 'center',
    };
  }

  return {
    paddingHorizontal: padding,
  };
}

/**
 * Get the appropriate input height based on device category
 *
 * @param category - Current device category
 * @returns Input height in points
 */
export function getInputHeight(category: DeviceCategory): number {
  const baseHeight = 52;
  const multiplier = SPACING_MULTIPLIERS[category];
  return ensureMinTouchTarget(Math.round(baseHeight * multiplier));
}

/**
 * Get the appropriate button height based on device category
 *
 * @param category - Current device category
 * @returns Button height in points
 */
export function getButtonHeight(category: DeviceCategory): number {
  const baseHeight = 52;
  const multiplier = SPACING_MULTIPLIERS[category];
  return ensureMinTouchTarget(Math.round(baseHeight * multiplier));
}

/**
 * Clamp a value between min and max
 *
 * @param value - The value to clamp
 * @param min - Minimum value
 * @param max - Maximum value
 * @returns The clamped value
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Linear interpolation between two values based on screen width
 *
 * @param smallValue - Value for smallest screen
 * @param largeValue - Value for largest screen
 * @param screenWidth - Current screen width
 * @param minWidth - Minimum reference width (default: iPhone SE)
 * @param maxWidth - Maximum reference width (default: iPad Pro 13")
 * @returns Interpolated value
 */
export function lerp(
  smallValue: number,
  largeValue: number,
  screenWidth: number,
  minWidth: number = 375,
  maxWidth: number = 1024
): number {
  const t = clamp((screenWidth - minWidth) / (maxWidth - minWidth), 0, 1);
  return smallValue + (largeValue - smallValue) * t;
}
