/**
 * useResponsive Hook
 *
 * Comprehensive responsive design hook that provides device detection,
 * scaling functions, and responsive utilities for React Native/Expo apps.
 *
 * Supports all iPhone models from 2020+ and current iPad lineup.
 *
 * @example
 * ```tsx
 * import { useResponsive } from '@/hooks/useResponsive';
 *
 * function MyComponent() {
 *   const { isTablet, deviceCategory, scale, getResponsiveValue } = useResponsive();
 *
 *   const padding = getResponsiveValue({
 *     small: 12,
 *     medium: 16,
 *     large: 20,
 *     tablet: 24,
 *     default: 16,
 *   });
 *
 *   return (
 *     <View style={{ padding, fontSize: scale(16) }}>
 *       {isTablet ? <TabletLayout /> : <PhoneLayout />}
 *     </View>
 *   );
 * }
 * ```
 */

import { useMemo } from 'react';
import { useWindowDimensions, Platform, PixelRatio } from 'react-native';
import type {
  DeviceCategory,
  DeviceModel,
  PlatformType,
  ResponsiveValue,
  ResponsiveSpacing,
  ResponsiveContextData,
} from '@/lib/responsive/types';
import {
  getDeviceCategory,
  getDeviceModel,
  isTablet as checkIsTablet,
  isLandscape as checkIsLandscape,
  getPlatform,
  scale as scaleValue,
  scaleFont as scaleFontValue,
  getResponsiveValue as resolveResponsiveValue,
  getSpacing as getSpacingForCategory,
} from '@/lib/responsive/helpers';

/**
 * useResponsive Hook
 *
 * Provides comprehensive responsive design utilities including:
 * - Device category detection (small, medium, large, tablet)
 * - Device model detection
 * - Orientation detection
 * - Scaling functions for sizes and fonts
 * - Responsive value resolution
 *
 * @returns ResponsiveContextData object with all responsive utilities
 */
export function useResponsive(): ResponsiveContextData {
  const { width, height, fontScale } = useWindowDimensions();

  // Memoize computed values to avoid recalculation on every render
  const responsiveData = useMemo((): ResponsiveContextData => {
    const deviceCategory = getDeviceCategory(width);
    const deviceModel = getDeviceModel(width, height);
    const isTabletDevice = checkIsTablet(width);
    const isLandscapeOrientation = checkIsLandscape(width, height);
    const platform = getPlatform();
    const pixelRatio = PixelRatio.get();

    // Create bound scaling functions
    const scale = (size: number): number => scaleValue(size, width);

    const scaleFont = (size: number): number =>
      scaleFontValue(size, width, fontScale, true);

    // Create bound responsive value getter
    const getResponsiveValue = <T>(values: ResponsiveValue<T>): T =>
      resolveResponsiveValue(values, deviceCategory);

    // Create bound spacing getter
    const getSpacing = (): ResponsiveSpacing => getSpacingForCategory(deviceCategory);

    return {
      width,
      height,
      deviceCategory,
      deviceModel,
      isTablet: isTabletDevice,
      isLandscape: isLandscapeOrientation,
      isPortrait: !isLandscapeOrientation,
      platform,
      pixelRatio,
      fontScale,
      scale,
      scaleFont,
      getResponsiveValue,
      getSpacing,
    };
  }, [width, height, fontScale]);

  return responsiveData;
}

/**
 * Type-safe hook for getting a single responsive value
 *
 * @example
 * ```tsx
 * const padding = useResponsiveValue({
 *   small: 12,
 *   medium: 16,
 *   tablet: 24,
 *   default: 16,
 * });
 * ```
 */
export function useResponsiveValue<T>(values: ResponsiveValue<T>): T {
  const { deviceCategory } = useResponsive();
  return resolveResponsiveValue(values, deviceCategory);
}

/**
 * Hook for checking if current device is a tablet
 *
 * @example
 * ```tsx
 * const isTablet = useIsTablet();
 * if (isTablet) {
 *   // Show tablet-specific UI
 * }
 * ```
 */
export function useIsTablet(): boolean {
  const { width } = useWindowDimensions();
  return checkIsTablet(width);
}

/**
 * Hook for checking current orientation
 *
 * @example
 * ```tsx
 * const isLandscape = useIsLandscape();
 * const columns = isLandscape ? 3 : 2;
 * ```
 */
export function useIsLandscape(): boolean {
  const { width, height } = useWindowDimensions();
  return checkIsLandscape(width, height);
}

/**
 * Hook for getting current device category
 *
 * @example
 * ```tsx
 * const category = useDeviceCategory();
 * // Returns: 'small' | 'medium' | 'large' | 'tablet'
 * ```
 */
export function useDeviceCategory(): DeviceCategory {
  const { width } = useWindowDimensions();
  return getDeviceCategory(width);
}

/**
 * Hook for getting responsive spacing values
 *
 * @example
 * ```tsx
 * const spacing = useResponsiveSpacing();
 * // { horizontal: 20, vertical: 16, gap: 12 }
 * ```
 */
export function useResponsiveSpacing(): ResponsiveSpacing {
  const category = useDeviceCategory();
  return getSpacingForCategory(category);
}

// Re-export types for convenience
export type {
  DeviceCategory,
  DeviceModel,
  PlatformType,
  ResponsiveValue,
  ResponsiveSpacing,
  ResponsiveContextData,
};
