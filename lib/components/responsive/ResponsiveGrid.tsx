/**
 * ResponsiveGrid Component
 *
 * A responsive grid layout that adapts the number of columns based on
 * device category and orientation.
 *
 * @example
 * ```tsx
 * <ResponsiveGrid>
 *   {items.map(item => (
 *     <Card key={item.id}>{item.content}</Card>
 *   ))}
 * </ResponsiveGrid>
 *
 * // With custom columns
 * <ResponsiveGrid columns={{ small: 1, medium: 2, tablet: 3 }}>
 *   {cards}
 * </ResponsiveGrid>
 * ```
 */

import React from 'react';
import { View, ViewStyle, StyleSheet, StyleProp } from 'react-native';
import { useResponsive } from '@/hooks/useResponsive';
import type { ResponsiveValue } from '@/lib/responsive/types';
import { getResponsiveGap } from '@/lib/responsive/spacing';

export interface ResponsiveGridProps {
  /** Child elements (grid items) */
  children: React.ReactNode;
  /** Number of columns per device category */
  columns?: ResponsiveValue<number>;
  /** Gap between items */
  gap?: number;
  /** Additional style to apply */
  style?: StyleProp<ViewStyle>;
  /** Style to apply to each item wrapper */
  itemStyle?: StyleProp<ViewStyle>;
  /** Test ID for testing */
  testID?: string;
}

/** Default columns per device category */
const DEFAULT_COLUMNS: ResponsiveValue<number> = {
  small: 1,
  medium: 1,
  large: 2,
  tablet: 2,
  default: 1,
};

/**
 * ResponsiveGrid
 *
 * Creates a responsive grid layout that automatically adjusts columns
 * based on device size and orientation.
 *
 * - Phones (portrait): 1 column
 * - Large phones: 2 columns
 * - Tablets (portrait): 2 columns
 * - Tablets (landscape): 3 columns (when landscape variant provided)
 */
export function ResponsiveGrid({
  children,
  columns = DEFAULT_COLUMNS,
  gap,
  style,
  itemStyle,
  testID,
}: ResponsiveGridProps): React.JSX.Element {
  const { deviceCategory, isLandscape, getResponsiveValue } = useResponsive();

  // Get number of columns for current device
  let numColumns = getResponsiveValue(columns);

  // Increase columns in landscape mode for tablets
  if (deviceCategory === 'tablet' && isLandscape) {
    numColumns = Math.min(numColumns + 1, 4);
  }

  // Calculate responsive gap
  const gridGap = gap ?? getResponsiveGap(deviceCategory);

  // Calculate item width percentage
  const itemWidth = `${100 / numColumns}%` as const;

  // Convert children to array
  const childArray = React.Children.toArray(children);

  return (
    <View style={[styles.grid, { gap: gridGap }, style]} testID={testID}>
      {childArray.map((child, index) => (
        <View
          key={index}
          style={[
            styles.gridItem,
            {
              width: itemWidth,
              paddingLeft: index % numColumns !== 0 ? gridGap / 2 : 0,
              paddingRight: (index + 1) % numColumns !== 0 ? gridGap / 2 : 0,
            },
            itemStyle,
          ]}
        >
          {child}
        </View>
      ))}
    </View>
  );
}

/**
 * ResponsiveGridItem
 *
 * Optional wrapper for grid items that need consistent styling
 */
export interface ResponsiveGridItemProps {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
}

export function ResponsiveGridItem({
  children,
  style,
}: ResponsiveGridItemProps): React.JSX.Element {
  return <View style={[styles.gridItemContent, style]}>{children}</View>;
}

const styles = StyleSheet.create({
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  gridItem: {
    marginBottom: 0, // Gap handles spacing
  },
  gridItemContent: {
    flex: 1,
  },
});

export default ResponsiveGrid;
