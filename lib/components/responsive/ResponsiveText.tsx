/**
 * ResponsiveText Component
 *
 * A text component that automatically scales font size based on device category.
 * Uses predefined text style presets for consistent typography.
 *
 * @example
 * ```tsx
 * <ResponsiveText variant="h1">Welcome</ResponsiveText>
 * <ResponsiveText variant="body">Regular paragraph text</ResponsiveText>
 * <ResponsiveText variant="caption" color={colors.text.secondary}>
 *   Small caption text
 * </ResponsiveText>
 * ```
 */

import React from 'react';
import { Text, TextStyle, StyleSheet, StyleProp, TextProps } from 'react-native';
import { useResponsive } from '@/hooks/useResponsive';
import {
  TextStylePreset,
  getTextStyle,
  getResponsiveFontSize,
  FontSizeKey,
  FontWeightKey,
  FONT_WEIGHTS,
} from '@/lib/responsive/typography';

export interface ResponsiveTextProps extends Omit<TextProps, 'style'> {
  /** Child text content */
  children: React.ReactNode;
  /** Text style preset variant */
  variant?: TextStylePreset;
  /** Custom font size key (overrides variant) */
  size?: FontSizeKey;
  /** Custom font weight (overrides variant) */
  weight?: FontWeightKey;
  /** Text color */
  color?: string;
  /** Text alignment */
  align?: 'left' | 'center' | 'right';
  /** Additional style to apply */
  style?: StyleProp<TextStyle>;
}

/**
 * ResponsiveText
 *
 * Renders text with responsive font sizing based on device category.
 * Supports predefined style presets (h1-h6, body, label, button, etc.)
 * or custom size/weight combinations.
 */
export function ResponsiveText({
  children,
  variant = 'body',
  size,
  weight,
  color,
  align,
  style,
  ...textProps
}: ResponsiveTextProps): React.JSX.Element {
  const { deviceCategory, fontScale } = useResponsive();

  // Get base text style from preset
  const baseStyle = getTextStyle(variant, deviceCategory, fontScale);

  // Override with custom size if provided
  if (size) {
    baseStyle.fontSize = getResponsiveFontSize(size, deviceCategory, fontScale);
  }

  // Override with custom weight if provided
  if (weight) {
    baseStyle.fontWeight = FONT_WEIGHTS[weight];
  }

  // Build final style
  const textStyle: TextStyle = {
    ...baseStyle,
    ...(color && { color }),
    ...(align && { textAlign: align }),
  };

  return (
    <Text style={[textStyle, style]} {...textProps}>
      {children}
    </Text>
  );
}

/**
 * Heading Components
 *
 * Convenience components for common heading levels
 */
export function H1(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="h1" {...props} />;
}

export function H2(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="h2" {...props} />;
}

export function H3(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="h3" {...props} />;
}

export function H4(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="h4" {...props} />;
}

export function H5(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="h5" {...props} />;
}

export function H6(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="h6" {...props} />;
}

/**
 * Body Text Components
 */
export function BodyLarge(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="bodyLarge" {...props} />;
}

export function Body(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="body" {...props} />;
}

export function BodySmall(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="bodySmall" {...props} />;
}

/**
 * Label and Caption Components
 */
export function Label(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="label" {...props} />;
}

export function Caption(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="caption" {...props} />;
}

/**
 * Button Text Components
 */
export function ButtonText(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="button" {...props} />;
}

export function ButtonTextLarge(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="buttonLarge" {...props} />;
}

export function ButtonTextSmall(props: Omit<ResponsiveTextProps, 'variant'>): React.JSX.Element {
  return <ResponsiveText variant="buttonSmall" {...props} />;
}

export default ResponsiveText;
