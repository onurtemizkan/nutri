/**
 * Responsive Components
 *
 * Reusable responsive component primitives that adapt to different device sizes.
 *
 * @example
 * ```tsx
 * import {
 *   ResponsiveContainer,
 *   ResponsiveGrid,
 *   ResponsiveText,
 *   H1,
 *   Body,
 * } from '@/lib/components/responsive';
 *
 * function MyScreen() {
 *   return (
 *     <ResponsiveContainer>
 *       <H1>Welcome</H1>
 *       <Body>Content here</Body>
 *       <ResponsiveGrid columns={{ small: 1, tablet: 2 }}>
 *         {items.map(item => <Card key={item.id} />)}
 *       </ResponsiveGrid>
 *     </ResponsiveContainer>
 *   );
 * }
 * ```
 */

// Container exports
export {
  ResponsiveContainer,
  type ResponsiveContainerProps,
} from './ResponsiveContainer';

// Grid exports
export {
  ResponsiveGrid,
  ResponsiveGridItem,
  type ResponsiveGridProps,
  type ResponsiveGridItemProps,
} from './ResponsiveGrid';

// Text exports
export {
  ResponsiveText,
  H1,
  H2,
  H3,
  H4,
  H5,
  H6,
  BodyLarge,
  Body,
  BodySmall,
  Label,
  Caption,
  ButtonText,
  ButtonTextLarge,
  ButtonTextSmall,
  type ResponsiveTextProps,
} from './ResponsiveText';
