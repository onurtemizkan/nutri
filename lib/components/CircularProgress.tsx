import React, { memo } from 'react';
import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import Svg, { Circle, Defs, LinearGradient as SvgLinearGradient, Stop } from 'react-native-svg';
import { colors, typography } from '@/lib/theme/colors';

interface CircularProgressProps {
  /** Progress percentage (0-100) */
  percentage: number;
  /** Size of the circle in pixels */
  size?: number;
  /** Stroke width of the progress ring */
  strokeWidth?: number;
  /** Primary color or gradient start color */
  color?: string;
  /** Secondary color for gradient (optional) */
  gradientEndColor?: string;
  /** Background ring color */
  backgroundColor?: string;
  /** Whether to show percentage text in the center */
  showPercentage?: boolean;
  /** Custom content to show in the center (overrides showPercentage) */
  children?: React.ReactNode;
  /** Custom style for the container */
  style?: ViewStyle;
  /** Unique ID for gradient (needed when multiple CircularProgress on same screen) */
  gradientId?: string;
}

/**
 * Reusable circular progress indicator component
 * Used for displaying goal progress with customizable appearance
 */
export const CircularProgress = memo(function CircularProgress({
  percentage,
  size = 88,
  strokeWidth = 8,
  color = colors.primary.main,
  gradientEndColor,
  backgroundColor = colors.background.elevated,
  showPercentage = false,
  children,
  style,
  gradientId = 'progressGradient',
}: CircularProgressProps) {
  // Clamp percentage between 0 and 100 for display
  const displayPercentage = Math.min(Math.max(percentage, 0), 100);

  // Calculate circle dimensions
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - displayPercentage / 100);
  const center = size / 2;

  // Determine if using gradient
  const useGradient = Boolean(gradientEndColor);

  return (
    <View style={[styles.container, { width: size, height: size }, style]}>
      <Svg width={size} height={size} style={styles.svg}>
        {useGradient && (
          <Defs>
            <SvgLinearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
              <Stop offset="0%" stopColor={color} />
              <Stop offset="100%" stopColor={gradientEndColor} />
            </SvgLinearGradient>
          </Defs>
        )}
        {/* Background circle */}
        <Circle
          cx={center}
          cy={center}
          r={radius}
          stroke={backgroundColor}
          strokeWidth={strokeWidth}
          fill="transparent"
        />
        {/* Progress circle */}
        <Circle
          cx={center}
          cy={center}
          r={radius}
          stroke={useGradient ? `url(#${gradientId})` : color}
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          rotation={-90}
          origin={`${center}, ${center}`}
        />
      </Svg>

      {/* Center content */}
      <View style={styles.centerContent}>
        {children ||
          (showPercentage && (
            <Text style={styles.percentageText}>{Math.round(displayPercentage)}%</Text>
          ))}
      </View>
    </View>
  );
});

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  svg: {
    position: 'absolute',
  },
  centerContent: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  percentageText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
});

export default CircularProgress;
