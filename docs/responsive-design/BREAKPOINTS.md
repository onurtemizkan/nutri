# Responsive Design Breakpoints

## Overview

The Nutri app uses a responsive design system that adapts layouts for iPhone and iPad devices across various screen sizes.

## Breakpoint System

### Device Categories

| Category | Width Range | Example Devices |
|----------|-------------|-----------------|
| Small | < 375pt | iPhone SE (older) |
| Medium | 375-413pt | iPhone SE 3rd, iPhone 12/13/14 |
| Large | 414-767pt | iPhone Plus/Max variants |
| Tablet | >= 768pt | All iPads |

### Key Constants

```typescript
// From lib/responsive/breakpoints.ts
export const BREAKPOINTS = {
  small: 375,
  medium: 414,
  large: 768,
  xlarge: 1024,
};

export const FORM_MAX_WIDTH = 600; // Maximum width for forms on tablets
```

## useResponsive Hook

The `useResponsive` hook provides responsive utilities:

```typescript
import { useResponsive } from '@/hooks/useResponsive';

function MyScreen() {
  const {
    isTablet,      // boolean: true if tablet-sized device
    getSpacing,    // function: returns responsive spacing values
    width,         // number: current screen width
    scale,         // function: scale values based on screen size
    deviceCategory // string: 'small' | 'medium' | 'large' | 'tablet'
  } = useResponsive();

  const responsiveSpacing = getSpacing();
  // responsiveSpacing.horizontal - horizontal padding (16-48pt based on device)
  // responsiveSpacing.vertical - vertical padding
}
```

## Responsive Patterns

### ScrollView with Responsive Padding

```tsx
<ScrollView
  style={styles.scrollView}
  showsVerticalScrollIndicator={false}
  contentContainerStyle={[
    styles.scrollContent,
    { paddingHorizontal: responsiveSpacing.horizontal },
    isTablet && styles.scrollContentTablet
  ]}
>
```

### Tablet-Specific Styles

```typescript
const styles = StyleSheet.create({
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: spacing.lg,
  },
  scrollContentTablet: {
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
  },
});
```

### Dynamic Chart Width

For components that require explicit width (like charts):

```typescript
const effectiveContentWidth = isTablet
  ? Math.min(screenWidth, FORM_MAX_WIDTH)
  : screenWidth;
const chartWidth = effectiveContentWidth - responsiveSpacing.horizontal * 2 - spacing.md * 2;
```

## Screen Orientation

- **iPhones**: Portrait only (locked via expo-screen-orientation)
- **iPads**: Both portrait and landscape supported

Configuration in `app/_layout.tsx`:
```typescript
if (Platform.OS === 'ios' && Platform.isPad) {
  await ScreenOrientation.unlockAsync();
} else {
  await ScreenOrientation.lockAsync(OrientationLock.PORTRAIT_UP);
}
```

## Files Updated for Responsive Design

### Auth Screens
- `app/auth/welcome.tsx`
- `app/auth/signin.tsx`
- `app/auth/signup.tsx`
- `app/auth/forgot-password.tsx`
- `app/auth/reset-password.tsx`

### Tab Screens
- `app/(tabs)/index.tsx` (Home)
- `app/(tabs)/profile.tsx`
- `app/(tabs)/health.tsx`

### Meal Screens
- `app/add-meal.tsx`
- `app/scan-food.tsx`
- `app/ar-scan-food.tsx`
- `app/ar-measure.tsx`

### Health Screens
- `app/health-settings.tsx`
- `app/health/[metricType].tsx`
- `app/health/add.tsx`

### System Screens
- `app/+not-found.tsx`
- `app/_layout.tsx`
