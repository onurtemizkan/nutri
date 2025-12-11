/**
 * Responsive Design Type Definitions
 *
 * TypeScript types for device detection, breakpoints, and responsive utilities.
 * Covers all iPhone models from 2020+ and current iPad lineup.
 */

/**
 * Specific device model identifiers
 * These represent the exact screen dimensions in logical pixels (points)
 */
export type DeviceModel =
  // iPhones - Small category
  | 'iPhoneSE' // iPhone SE 3rd gen (2022): 375x667
  | 'iPhoneMini' // iPhone 12/13 Mini: 375x812
  // iPhones - Medium category
  | 'iPhoneMedium' // iPhone 12/13/14: 390x844
  | 'iPhonePro' // iPhone 14/15/16 Pro: 393x852
  // iPhones - Large category
  | 'iPhoneMax' // iPhone Pro Max/Plus: 430x932
  // iPads
  | 'iPadMini' // iPad Mini 6th gen: 744x1133
  | 'iPad' // iPad 10th gen / iPad Air 11": 820x1180
  | 'iPadPro11' // iPad Pro 11": 834x1194
  | 'iPadAir13' // iPad Air 13": 1032x1376
  | 'iPadPro13'; // iPad Pro 13": 1024x1366

/**
 * Device categories for responsive design
 * - small: iPhone SE, Mini (compact screens)
 * - medium: Standard iPhone (most common)
 * - large: iPhone Pro Max, Plus (largest phones)
 * - tablet: All iPads
 */
export type DeviceCategory = 'small' | 'medium' | 'large' | 'tablet';

/**
 * Platform type for cross-platform logic
 */
export type PlatformType = 'ios' | 'android' | 'web';

/**
 * Screen orientation
 */
export type Orientation = 'portrait' | 'landscape';

/**
 * Screen dimensions in logical pixels (points)
 */
export interface ScreenDimensions {
  width: number;
  height: number;
}

/**
 * Device breakpoint definition
 */
export interface DeviceBreakpoint {
  /** Minimum width for this breakpoint */
  minWidth: number;
  /** Maximum width for this breakpoint (exclusive) */
  maxWidth: number;
  /** Minimum height for this breakpoint */
  minHeight: number;
  /** Maximum height for this breakpoint (exclusive) */
  maxHeight: number;
  /** Device category this breakpoint belongs to */
  category: DeviceCategory;
  /** Human-readable device name */
  name: string;
}

/**
 * Responsive values for different device categories
 * Allows specifying different values per device size
 */
export interface ResponsiveValue<T> {
  /** Value for small devices (iPhone SE, Mini) */
  small?: T;
  /** Value for medium devices (standard iPhone) */
  medium?: T;
  /** Value for large devices (iPhone Pro Max) */
  large?: T;
  /** Value for tablets (iPads) */
  tablet?: T;
  /** Default value when category not specified */
  default: T;
}

/**
 * Responsive spacing values
 */
export interface ResponsiveSpacing {
  /** Horizontal padding/margin */
  horizontal: number;
  /** Vertical padding/margin */
  vertical: number;
  /** Gap between elements */
  gap: number;
}

/**
 * Safe area insets structure
 */
export interface SafeAreaInsets {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/**
 * Complete responsive context data
 * Returned by the useResponsive hook
 */
export interface ResponsiveContextData {
  /** Current screen width in points */
  width: number;
  /** Current screen height in points */
  height: number;
  /** Current device category */
  deviceCategory: DeviceCategory;
  /** Detected device model (best match) */
  deviceModel: DeviceModel;
  /** Whether the device is a tablet */
  isTablet: boolean;
  /** Whether the device is in landscape orientation */
  isLandscape: boolean;
  /** Whether the device is in portrait orientation */
  isPortrait: boolean;
  /** Current platform */
  platform: PlatformType;
  /** Pixel ratio for high-DPI displays */
  pixelRatio: number;
  /** Font scale from system settings */
  fontScale: number;
  /** Scale function for proportional sizing */
  scale: (size: number) => number;
  /** Scale function for fonts (respects system font scale) */
  scaleFont: (size: number) => number;
  /** Get responsive value based on current device category */
  getResponsiveValue: <T>(values: ResponsiveValue<T>) => T;
  /** Get spacing values for current device */
  getSpacing: () => ResponsiveSpacing;
}

/**
 * Orientation lock configuration
 */
export interface OrientationConfig {
  /** Lock orientation on iPhone */
  iPhoneLock: Orientation | 'all';
  /** Lock orientation on iPad */
  iPadLock: Orientation | 'all';
}

/**
 * Responsive style props that can be passed to components
 */
export interface ResponsiveStyleProps {
  /** Maximum width constraint (useful for tablets) */
  maxWidth?: number;
  /** Whether to center content horizontally */
  centerHorizontally?: boolean;
  /** Responsive padding */
  padding?: ResponsiveValue<number>;
  /** Responsive margin */
  margin?: ResponsiveValue<number>;
}
