/**
 * Responsive Design Utilities
 *
 * Comprehensive responsive design system for React Native/Expo apps.
 * Supports all iPhone models from 2020+ and current iPad lineup.
 *
 * @example
 * ```typescript
 * import {
 *   getDeviceCategory,
 *   scale,
 *   DEVICE_DIMENSIONS,
 *   type DeviceCategory,
 * } from '@/lib/responsive';
 *
 * // Get current device category
 * const category = getDeviceCategory(screenWidth);
 *
 * // Scale a value proportionally
 * const scaledPadding = scale(16, screenWidth);
 * ```
 */

// Type exports
export type {
  DeviceModel,
  DeviceCategory,
  PlatformType,
  Orientation,
  ScreenDimensions,
  DeviceBreakpoint,
  ResponsiveValue,
  ResponsiveSpacing,
  SafeAreaInsets,
  ResponsiveContextData,
  OrientationConfig,
  ResponsiveStyleProps,
} from './types';

// Constants exports
export {
  DEVICE_DIMENSIONS,
  DEVICE_CATEGORIES,
  CATEGORY_BREAKPOINTS,
  DEVICE_BREAKPOINTS,
  BASE_DESIGN,
  MIN_TOUCH_TARGET,
  SPACING_MULTIPLIERS,
  FONT_SCALE_MULTIPLIERS,
  MAX_CONTENT_WIDTH,
  CONTAINER_PADDING,
  FORM_MAX_WIDTH,
  CARD_MIN_WIDTH,
  GRID_COLUMNS,
  LANDSCAPE_GRID_COLUMNS,
} from './breakpoints';

// Helper function exports
export {
  getDeviceCategory,
  getDeviceModel,
  isTablet,
  isLandscape,
  getPlatform,
  isIPad,
  scale,
  scaleFont,
  getResponsiveValue,
  getSpacing,
  getMaxContentWidth,
  ensureMinTouchTarget,
  getGridColumns,
  getContainerStyle,
  getFormContainerStyle,
  getInputHeight,
  getButtonHeight,
  clamp,
  lerp,
} from './helpers';

// Typography exports
export type { FontSizeKey, FontWeightKey, LineHeightKey, TextStylePreset } from './typography';
export {
  BASE_FONT_SIZES,
  FONT_WEIGHTS,
  LINE_HEIGHTS,
  LETTER_SPACING,
  TEXT_STYLES,
  MIN_ACCESSIBLE_FONT_SIZE,
  getResponsiveFontSize,
  getResponsiveLineHeight,
  createResponsiveTextStyle,
  getTextStyle,
  ensureMinFontSize,
} from './typography';

// Spacing exports
export type { SpacingKey } from './spacing';
export {
  BASE_SPACING,
  SPACING_PRESETS,
  getResponsiveSpacing,
  getAllSpacing,
  getContainerPadding,
  createContainerPadding,
  createFormContainerStyle as createFormPaddingStyle,
  getResponsiveGap,
  getResponsiveMargin,
  getSafeAreaPadding,
  getPresetSpacing,
  getResponsiveControlHeight,
} from './spacing';
