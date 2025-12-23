# Task ID: 12

**Title:** Implement Responsive UI Design for iPhone and iPad Devices

**Status:** done

**Dependencies:** None

**Priority:** high

**Description:** Comprehensive responsive design implementation covering all iPhone sizes (2020+) and iPads with both screen orientations. Lock iPhones to portrait mode while supporting iPad landscape/portrait. Test and optimize all 17 app screens across device categories.

**Details:**

## Technical Implementation Details

### Breakpoint System (in logical pixels)
```typescript
const breakpoints = {
  // iPhones
  iPhoneSE: { width: 375, height: 667 },      // iPhone SE 3rd gen
  iPhoneMini: { width: 375, height: 812 },    // iPhone 12/13 Mini
  iPhoneMedium: { width: 390, height: 844 },  // iPhone 12/13/14
  iPhonePro: { width: 393, height: 852 },     // iPhone 14/15/16 Pro
  iPhoneMax: { width: 430, height: 932 },     // iPhone Pro Max/Plus
  
  // iPads
  iPadMini: { width: 744, height: 1133 },     // iPad Mini
  iPad: { width: 820, height: 1180 },         // iPad/iPad Air 11"
  iPadPro11: { width: 834, height: 1194 },    // iPad Pro 11"
  iPadAir13: { width: 1032, height: 1376 },   // iPad Air 13"
  iPadPro13: { width: 1024, height: 1366 },   // iPad Pro 13"
};

const deviceCategories = {
  small: ['iPhoneSE', 'iPhoneMini'],
  medium: ['iPhoneMedium', 'iPhonePro'],
  large: ['iPhoneMax'],
  tablet: ['iPadMini', 'iPad', 'iPadPro11', 'iPadAir13', 'iPadPro13'],
};
```

### Responsive Hook Example
```typescript
// lib/hooks/useResponsive.ts
import { useWindowDimensions, Platform } from 'react-native';

export function useResponsive() {
  const { width, height } = useWindowDimensions();
  
  const isTablet = width >= 744;
  const isLandscape = width > height;
  const deviceCategory = getDeviceCategory(width);
  
  const scale = (size: number) => {
    const baseWidth = 390; // iPhone 14 as baseline
    return (width / baseWidth) * size;
  };
  
  return { width, height, isTablet, isLandscape, deviceCategory, scale };
}
```

### Orientation Lock (app.json)
```json
{
  "expo": {
    "orientation": "portrait",
    "ios": {
      "supportsTablet": true,
      "requireFullScreen": false,
      "userInterfaceStyle": "automatic"
    }
  }
}
```

### iPad-specific orientation unlock (runtime)
```typescript
// In root _layout.tsx
import * as ScreenOrientation from 'expo-screen-orientation';

useEffect(() => {
  async function configureOrientation() {
    if (Platform.OS === 'ios' && Platform.isPad) {
      await ScreenOrientation.unlockAsync();
    } else {
      await ScreenOrientation.lockAsync(
        ScreenOrientation.OrientationLock.PORTRAIT_UP
      );
    }
  }
  configureOrientation();
}, []);
```

### Simulator Testing Checklist
Each screen must be tested on these simulators:
- [ ] iPhone SE (3rd generation) - iOS 17+
- [ ] iPhone 13 Mini - iOS 17+
- [ ] iPhone 14 - iOS 17+
- [ ] iPhone 15 Pro - iOS 17+
- [ ] iPhone 15 Pro Max - iOS 17+
- [ ] iPad Mini (6th generation) - Portrait
- [ ] iPad Mini (6th generation) - Landscape
- [ ] iPad Pro 11-inch - Portrait
- [ ] iPad Pro 11-inch - Landscape
- [ ] iPad Pro 13-inch - Portrait
- [ ] iPad Pro 13-inch - Landscape

### Safe Area Considerations
- Use SafeAreaView consistently
- Handle Dynamic Island on iPhone 14 Pro+
- Handle home indicator on all Face ID devices
- Handle notch on older Face ID devices
- Handle status bar on iPhone SE

### Testing Commands
```bash
# List available simulators
xcrun simctl list devices available

# Boot specific simulator
xcrun simctl boot "iPhone SE (3rd generation)"
xcrun simctl boot "iPhone 15 Pro Max"
xcrun simctl boot "iPad Pro 13-inch (M4)"

# Run app on specific simulator
npx expo run:ios --device "iPhone SE (3rd generation)"
npx expo run:ios --device "iPad Pro 13-inch (M4)"
```

**Test Strategy:**

## Testing Strategy

### Unit Tests
- Test useResponsive hook returns correct device categories
- Test scale functions produce expected values
- Test breakpoint detection logic

### Visual Regression Testing
- Screenshot each screen on each device category
- Compare layouts visually
- Verify no text truncation or overflow
- Verify touch targets are accessible (44pt minimum)

### Manual Testing Checklist per Screen

#### For each of the 17 screens, verify:
1. **Layout Integrity**
   - No horizontal scrolling when not intended
   - Content fits within safe areas
   - Proper padding/margins on all edges

2. **Typography**
   - All text is readable
   - No text truncation (unless intentional with ellipsis)
   - Font sizes appropriate for device

3. **Interactive Elements**
   - Buttons are tappable (44pt minimum)
   - Form fields are usable
   - Scrolling works smoothly

4. **Orientation (iPad only)**
   - Smooth rotation transition
   - Layout adapts correctly
   - No content loss during rotation

### Device Matrix
| Screen | SE | Mini | Medium | Pro Max | iPad Mini P | iPad Mini L | iPad Pro P | iPad Pro L |
|--------|:--:|:----:|:------:|:-------:|:-----------:|:-----------:|:----------:|:----------:|
| welcome | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| signin | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| signup | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| forgot-password | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| reset-password | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| home (tabs/index) | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| profile | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| health | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| add-meal | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| scan-food | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| ar-scan-food | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| ar-measure | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| health-settings | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| health/[metricType] | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| health/add | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| not-found | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |
| layouts | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

### Acceptance Criteria
- All screens render correctly on all device categories
- No visual bugs, overflow, or truncation
- Forms are usable on all devices
- iPad supports both orientations seamlessly
- Performance remains smooth on older devices (iPhone SE)
- Safe areas properly respected on all devices

## Subtasks

### 12.1. Create Responsive Design Utility Library

**Status:** done  
**Dependencies:** None  

Build the foundational responsive design utilities including breakpoint definitions, device category constants, and helper functions for device detection. Create lib/responsive/breakpoints.ts with all iPhone (2020+) and iPad screen dimensions.

**Details:**

Create the following files:
- lib/responsive/breakpoints.ts - Device breakpoint constants
- lib/responsive/types.ts - TypeScript types for device categories
- lib/responsive/helpers.ts - Utility functions for device detection

Breakpoints to define:
- iPhoneSE: 375x667 pts
- iPhoneMini: 375x812 pts  
- iPhoneMedium: 390x844 pts
- iPhonePro: 393x852 pts
- iPhoneMax: 430x932 pts
- iPadMini: 744x1133 pts
- iPad: 820x1180 pts
- iPadPro11: 834x1194 pts
- iPadAir13: 1032x1376 pts
- iPadPro13: 1024x1366 pts

### 12.2. Configure Orientation Lock for iPhone/iPad

**Status:** done  
**Dependencies:** 12.1  

Set up orientation locking - portrait-only for iPhones, both orientations for iPads. Update app.json and implement runtime orientation control using expo-screen-orientation.

**Details:**

1. Update app.json with orientation: "portrait" and ios.supportsTablet: true
2. Install expo-screen-orientation if not present
3. Implement runtime detection in _layout.tsx:
   - Lock to PORTRAIT_UP for iPhone
   - Unlock for iPad (Platform.isPad)
4. Test orientation behavior on both device types

### 12.3. Implement useResponsive Hook

**Status:** done  
**Dependencies:** 12.1  

Create a comprehensive useResponsive hook that provides device category detection, scaling functions, and responsive utilities. This hook will be the primary interface for responsive design throughout the app.

**Details:**

Create lib/hooks/useResponsive.ts with:
- useWindowDimensions integration
- Device category detection (small/medium/large/tablet)
- isTablet boolean
- isLandscape boolean  
- scale() function for proportional sizing
- scaleFont() for typography
- getSpacing() for responsive margins/padding
- Platform-aware logic for iOS/Android differences

Export types and hook from lib/hooks/index.ts

### 12.4. Create Responsive Typography and Spacing System

**Status:** done  
**Dependencies:** 12.3  

Build a responsive typography scale and spacing system that adapts to different device sizes. Create design tokens for font sizes, line heights, and spacing values.

**Details:**

Create lib/responsive/typography.ts:
- Base font sizes for each device category
- Responsive font scale (xs, sm, base, lg, xl, 2xl, 3xl)
- Line height multipliers
- Letter spacing values

Create lib/responsive/spacing.ts:
- Spacing scale (xs: 4, sm: 8, md: 16, lg: 24, xl: 32, 2xl: 48)
- Responsive padding/margin helpers
- Safe area aware spacing

Ensure minimum touch targets of 44pt on all devices

### 12.5. Build Responsive Component Primitives

**Status:** done  
**Dependencies:** 12.3, 12.4  

Create reusable responsive component wrappers that handle common responsive patterns - containers, cards, grids, and form layouts that adapt to device size.

**Details:**

Create components in lib/components/responsive/:
- ResponsiveContainer.tsx - Max-width container with padding
- ResponsiveGrid.tsx - Adaptive grid (1-col phone, 2-col tablet)
- ResponsiveCard.tsx - Card with adaptive sizing
- ResponsiveForm.tsx - Form layout wrapper
- ResponsiveText.tsx - Text with automatic font scaling

Each component should:
- Use useResponsive hook
- Support iPad landscape/portrait layouts
- Handle safe areas properly
- Be fully typed with TypeScript

### 12.6. Update Auth Screens for Responsive Design

**Status:** done  
**Dependencies:** 12.5  

Adapt all 5 authentication screens (welcome, signin, signup, forgot-password, reset-password) to be responsive across all device categories.

**Details:**

Update these screens:
- app/auth/welcome.tsx
- app/auth/signin.tsx
- app/auth/signup.tsx
- app/auth/forgot-password.tsx
- app/auth/reset-password.tsx

For each screen:
1. Import and use useResponsive hook
2. Replace hardcoded dimensions with responsive values
3. Ensure forms have appropriate widths on tablets (max-width)
4. Adjust padding and margins for each device category
5. Verify text is readable on all sizes
6. Center content appropriately on larger screens
7. Handle keyboard avoiding behavior on all sizes

### 12.7. Update Main Tab Screens for Responsive Design

**Status:** done  
**Dependencies:** 12.5  

Adapt the 3 main tab screens (home/index, profile, health) and tab layout to be responsive across all device categories.

**Details:**

Update these screens:
- app/(tabs)/index.tsx - Home dashboard
- app/(tabs)/profile.tsx - User profile
- app/(tabs)/health.tsx - Health overview
- app/(tabs)/_layout.tsx - Tab navigation

Focus areas:
1. Dashboard cards should use grid on tablets
2. Profile layout may use side-by-side on landscape iPad
3. Health metrics should display in responsive grid
4. Tab bar should adapt sizing for tablets
5. Charts and graphs must scale appropriately
6. Lists should have appropriate row heights per device

### 12.8. Update Meal and Scanning Screens for Responsive Design

**Status:** done  
**Dependencies:** 12.5  

Adapt the 4 meal/scanning screens (add-meal, scan-food, ar-scan-food, ar-measure) to be responsive, with special attention to camera and AR views.

**Details:**

Update these screens:
- app/add-meal.tsx - Manual meal entry form
- app/scan-food.tsx - Camera food scanning
- app/ar-scan-food.tsx - AR food recognition
- app/ar-measure.tsx - AR portion measurement

Special considerations:
1. Camera views must fill appropriate area on all devices
2. AR overlays need to scale with screen size
3. Form inputs in add-meal need responsive widths
4. Scanning UI controls must have 44pt+ touch targets
5. Results display should use available space on tablets
6. Modal presentations should be appropriately sized

### 12.9. Update Health Screens for Responsive Design

**Status:** done  
**Dependencies:** 12.5  

Adapt the 3 health-related screens (health-settings, health/[metricType], health/add) and error screen (+not-found) to be responsive.

**Details:**

Update these screens:
- app/health-settings.tsx - Health settings
- app/health/[metricType].tsx - Metric detail view
- app/health/add.tsx - Add health metric
- app/+not-found.tsx - 404 error page
- app/_layout.tsx - Root layout

Focus areas:
1. Settings lists should have appropriate row heights
2. Metric charts must scale for different screen sizes
3. Add metric form should be responsive
4. Error page should center content on all devices
5. Root layout should handle safe areas consistently

### 12.10. iPhone Simulator Testing - All Categories

**Status:** done  
**Dependencies:** 12.6, 12.7, 12.8, 12.9  

Run comprehensive simulator testing on all iPhone device categories (SE, Mini, Medium, Pro Max) verifying all 17 screens render correctly in portrait mode.

**Details:**

Test on these simulators:
1. iPhone SE (3rd generation) - Small/Legacy (375x667)
2. iPhone 13 Mini - Mini category (375x812)
3. iPhone 14 - Medium category (390x844)
4. iPhone 15 Pro - Pro category (393x852)
5. iPhone 15 Pro Max - Max category (430x932)

For each device, verify ALL 17 screens:
- Auth: welcome, signin, signup, forgot-password, reset-password
- Tabs: home, profile, health
- Meals: add-meal, scan-food, ar-scan-food, ar-measure
- Health: health-settings, [metricType], add
- System: not-found, layouts

Checklist per screen:
☐ No horizontal overflow
☐ Text readable, no truncation
☐ Touch targets >= 44pt
☐ Safe areas respected
☐ Forms usable with keyboard

### 12.11. iPad Simulator Testing - Both Orientations

**Status:** done  
**Dependencies:** 12.6, 12.7, 12.8, 12.9  

Run comprehensive simulator testing on iPad devices (Mini, Air, Pro) in both portrait and landscape orientations, verifying all 17 screens adapt correctly.

**Details:**

Test on these simulators:
1. iPad Mini (6th generation) - Portrait & Landscape
2. iPad Air 11-inch - Portrait & Landscape
3. iPad Pro 11-inch - Portrait & Landscape
4. iPad Pro 13-inch - Portrait & Landscape

For each device AND orientation, verify ALL 17 screens:
- Auth: welcome, signin, signup, forgot-password, reset-password
- Tabs: home, profile, health
- Meals: add-meal, scan-food, ar-scan-food, ar-measure
- Health: health-settings, [metricType], add
- System: not-found, layouts

Checklist per screen:
☐ Layout adapts to orientation change
☐ No content loss on rotation
☐ Grids display correctly (multi-column where appropriate)
☐ Forms centered with max-width
☐ Charts scale appropriately
☐ Touch targets >= 44pt
☐ Smooth rotation animation

### 12.12. Create Testing Documentation and Verification Report

**Status:** done  
**Dependencies:** 12.10, 12.11  

Document all responsive design testing results, create a verification matrix with screenshots, and compile final report of any issues found and resolutions.

**Details:**

Create documentation in docs/responsive-design/:
1. TESTING-MATRIX.md - Device x Screen verification grid
2. BREAKPOINTS.md - Documentation of breakpoint system
3. SCREENSHOTS/ - Folder with screenshots from each device
4. ISSUES.md - Log of issues found and how resolved
5. USAGE-GUIDE.md - How to use responsive utilities

Final verification matrix should show:
| Screen | SE | Mini | Med | Max | iPad-P | iPad-L |
With ✓/✗ for each combination

Include simulator commands for future testing:
- How to boot each simulator
- How to run app on each device
- How to take screenshots
