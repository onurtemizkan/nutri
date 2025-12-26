# UI/UX Improvements PRD - Nutri Mobile App

## Overview

This PRD outlines comprehensive UI/UX improvements for the Nutri mobile application. The improvements focus on creating a consistent design system, improving accessibility, enhancing user feedback, and ensuring a polished user experience across all screens.

## Goals

1. Create a reusable component library for consistent UI patterns
2. Achieve WCAG 2.1 AA accessibility compliance
3. Improve form validation and user feedback mechanisms
4. Add proper empty states and loading experiences
5. Ensure visual consistency across all screens

## Technical Context

- **Framework**: React Native + Expo
- **Design System**: `lib/theme/colors.ts` (comprehensive dark theme with design tokens)
- **Responsive System**: `lib/responsive/` and `hooks/useResponsive.ts`
- **Existing Components**: `lib/components/` (CustomAlert, ErrorBoundary, SwipeableMealCard)

---

## Feature 1: Shared Button Component

### Description
Create a reusable Button component that provides consistent button styling, accessibility, and interaction feedback across the entire application.

### Current Problem
- Each screen implements buttons differently with inconsistent styling
- No shared pressed state visual feedback
- Missing accessibility props on most buttons
- Gradient FABs are implemented inline in multiple places

### Requirements

#### Variants
- `primary`: Gradient purple/pink background (uses existing gradient colors)
- `secondary`: Outlined with border, transparent background
- `ghost`: Transparent background, text only
- `destructive`: Red background for dangerous actions

#### Sizes
- `sm`: Height 36px, padding horizontal 12px, fontSize 14
- `md`: Height 44px (Apple HIG minimum), padding horizontal 16px, fontSize 16
- `lg`: Height 52px, padding horizontal 24px, fontSize 18
- All sizes should scale using `useResponsive` multipliers

#### States
- Default, Pressed (scale to 0.97 or opacity 0.8), Disabled (opacity 0.5), Loading (ActivityIndicator)

#### Accessibility
- `accessibilityRole="button"`
- Required `accessibilityLabel` prop
- `accessibilityState={{ disabled, busy }}` for loading/disabled states

#### Props Interface
```typescript
interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'ghost' | 'destructive';
  size?: 'sm' | 'md' | 'lg';
  label: string;
  accessibilityLabel: string;
  onPress: () => void;
  disabled?: boolean;
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  fullWidth?: boolean;
}
```

### File Location
`lib/components/ui/Button.tsx`

### Acceptance Criteria
- [ ] All variants render correctly with proper colors from design tokens
- [ ] Sizes are responsive using useResponsive hook
- [ ] Loading state shows ActivityIndicator and disables interaction
- [ ] Pressed state has visual feedback (animation)
- [ ] All accessibility props are properly set
- [ ] Component is exported and documented

---

## Feature 2: Shared Input Component

### Description
Create a reusable TextInput component with built-in label, error state, helper text, and accessibility support.

### Current Problem
- TextInput fields lack accessibility labels (critical accessibility gap)
- No visual error states on invalid inputs
- No helper text for password requirements or input hints
- Placeholder text is the only label (invisible to screen readers when focused)

### Requirements

#### Visual Elements
- Floating or static label above input
- Input field with consistent border styling
- Error message area below input (red text)
- Helper text area below input (gray text)
- Optional left/right icons

#### States
- Default: Gray border
- Focused: Purple border (primary color)
- Error: Red border with error message
- Disabled: Reduced opacity, non-editable

#### Accessibility
- `accessibilityLabel` connected to visible label
- `accessibilityHint` for additional context
- Error state announced via `accessibilityLiveRegion`

#### Props Interface
```typescript
interface InputProps {
  label: string;
  value: string;
  onChangeText: (text: string) => void;
  placeholder?: string;
  error?: string;
  helperText?: string;
  secureTextEntry?: boolean;
  keyboardType?: KeyboardTypeOptions;
  autoCapitalize?: 'none' | 'sentences' | 'words' | 'characters';
  disabled?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  maxLength?: number;
  multiline?: boolean;
  numberOfLines?: number;
}
```

### File Location
`lib/components/ui/Input.tsx`

### Acceptance Criteria
- [ ] Label renders above input and is connected for accessibility
- [ ] Error state shows red border and error message
- [ ] Helper text appears when provided
- [ ] Focus state has visual indication
- [ ] Screen readers can identify the input purpose
- [ ] Supports all standard TextInput props

---

## Feature 3: Shared Card Component

### Description
Create a reusable Card component with consistent styling, shadows, and optional press handling.

### Current Problem
- Each screen defines its own card styles with slightly different shadows, border radius, and padding
- Inconsistent elevation/shadow values across screens
- No shared pressable card pattern

### Requirements

#### Variants
- `elevated`: With shadow (default)
- `outlined`: Border only, no shadow
- `filled`: Solid background, no border

#### Props Interface
```typescript
interface CardProps {
  variant?: 'elevated' | 'outlined' | 'filled';
  children: React.ReactNode;
  onPress?: () => void;
  style?: ViewStyle;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  accessibilityLabel?: string;
}
```

### Styling
- Use `colors.background.card` for background
- Use `shadows.md` for elevated variant
- Use `borderRadius.lg` (16) for all variants
- Padding from spacing scale

### File Location
`lib/components/ui/Card.tsx`

### Acceptance Criteria
- [ ] All variants render correctly
- [ ] Pressable cards have visual feedback
- [ ] Shadows use design token values
- [ ] Padding options work correctly
- [ ] Accessibility props passed through

---

## Feature 4: Shared ScreenHeader Component

### Description
Create a consistent header component for all screens with back navigation, title, and optional right actions.

### Current Problem
- No consistent header pattern across screens
- Back buttons implemented differently (sometimes Ionicons chevron, sometimes missing)
- `headerShown: false` on all screens but no unified custom header

### Requirements

#### Elements
- Back button (chevron-left icon) - conditionally shown
- Title (centered or left-aligned)
- Optional right action buttons
- Safe area handling for notch devices

#### Props Interface
```typescript
interface ScreenHeaderProps {
  title: string;
  showBackButton?: boolean;
  onBackPress?: () => void; // defaults to router.back()
  rightActions?: React.ReactNode;
  transparent?: boolean;
}
```

### File Location
`lib/components/ui/ScreenHeader.tsx`

### Acceptance Criteria
- [ ] Back button navigates correctly
- [ ] Title displays properly
- [ ] Right actions render when provided
- [ ] Safe area insets handled correctly
- [ ] Transparent variant works for overlay screens

---

## Feature 5: Shared EmptyState Component

### Description
Create a reusable empty state component for screens/sections with no data.

### Current Problem
- Home screen has no empty state when no meals are logged
- Activity list has no empty state
- Supplements list has no empty state
- Only Health screen has a proper empty state implementation

### Requirements

#### Elements
- Icon (customizable)
- Title text
- Description text
- Optional CTA button

#### Props Interface
```typescript
interface EmptyStateProps {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  description: string;
  actionLabel?: string;
  onAction?: () => void;
}
```

### File Location
`lib/components/ui/EmptyState.tsx`

### Acceptance Criteria
- [ ] Icon renders with proper sizing and color
- [ ] Text is centered and readable
- [ ] CTA button uses shared Button component
- [ ] Responsive sizing for different devices

---

## Feature 6: Accessibility Improvements

### Description
Systematically add accessibility props to all interactive elements across the application.

### Current Problem
- Most TextInput fields lack accessibility labels (critical gap)
- Buttons missing `accessibilityRole`
- Cards and interactive elements missing labels
- Swipeable actions lack `accessibilityHint`

### Screens to Update

#### Auth Screens (`app/auth/`)
- `signin.tsx`: Add labels to email, password inputs
- `signup.tsx`: Add labels to all form fields
- `forgot-password.tsx`: Add label to email input
- `reset-password.tsx`: Add labels to password fields
- `welcome.tsx`: Add labels to buttons

#### Main Tab Screens (`app/(tabs)/`)
- `index.tsx` (Home): Add labels to meal cards, FAB button, date navigation
- `health.tsx`: Add labels to metric cards, add button
- `profile.tsx`: Add labels to all form fields, edit button, logout button

#### Feature Screens
- `add-meal.tsx`: Add labels to all nutrition inputs
- `edit-meal/[id].tsx`: Add labels to all editable fields
- `add-health-metric.tsx`: Add labels to metric type picker, value input
- `supplements.tsx`: Add labels to supplement list items

### Acceptance Criteria
- [ ] All TextInput fields have accessibilityLabel
- [ ] All buttons have accessibilityRole and accessibilityLabel
- [ ] All interactive cards have accessibilityLabel
- [ ] Swipe gestures have accessibilityHint explaining the action
- [ ] Form validation errors announced to screen readers

---

## Feature 7: Form Validation UX Improvements

### Description
Implement inline form validation with visual error states instead of using Alert dialogs.

### Current Problem
- Forms use Alert.alert() for validation errors
- No visual indication of which field has an error
- No real-time validation feedback
- No helper text for input requirements

### Screens to Update
- `signin.tsx`: Email format, password required
- `signup.tsx`: Email format, password strength, confirm password match
- `forgot-password.tsx`: Email format
- `reset-password.tsx`: Password strength, confirm match
- `add-meal.tsx`: Required fields, numeric validation
- `profile.tsx`: Email format, numeric fields validation

### Requirements
- Use new Input component with error prop
- Show error message below the specific field
- Validate on blur (first time) then on change (after first error)
- Keep submit button enabled but show all errors on submit attempt

### Acceptance Criteria
- [ ] Each form uses the shared Input component
- [ ] Validation errors appear below the relevant field
- [ ] Error fields have red border
- [ ] Errors clear when user fixes the input
- [ ] No more Alert dialogs for validation errors

---

## Feature 8: Empty States Implementation

### Description
Add proper empty states to all screens and lists that can have no data.

### Screens Needing Empty States

#### Home Screen (`app/(tabs)/index.tsx`)
- When no meals logged for selected date
- Message: "No meals logged yet"
- CTA: "Log your first meal"

#### Activity Screen (`app/activity/index.tsx`)
- When no activities exist
- Message: "No activities tracked yet"
- CTA: "Add an activity"

#### Supplements Screen (`app/supplements.tsx`)
- When no supplements added
- Message: "No supplements tracked"
- CTA: "Add a supplement"

### Requirements
- Use shared EmptyState component
- Icon should be contextually relevant
- CTA should navigate to add screen

### Acceptance Criteria
- [ ] Home screen shows empty state when no meals for date
- [ ] Activity screen shows empty state when no activities
- [ ] Supplements screen shows empty state when empty
- [ ] All CTAs navigate to correct add screens

---

## Feature 9: Loading Skeleton Components

### Description
Create skeleton loading components to improve perceived performance during data fetching.

### Current Problem
- Screens show ActivityIndicator during loading
- Content "pops in" suddenly when loaded
- No visual hint of what content will appear

### Components to Create

#### MealCardSkeleton
- Matches MealCard dimensions
- Animated shimmer effect
- Shows in Home screen during meal loading

#### MetricCardSkeleton
- Matches health metric card dimensions
- Used in Health screen during loading

#### ProfileSkeleton
- Matches profile layout
- Used during profile data loading

### File Location
`lib/components/ui/Skeleton.tsx` - Base skeleton with shimmer animation
`lib/components/ui/MealCardSkeleton.tsx`
`lib/components/ui/MetricCardSkeleton.tsx`

### Acceptance Criteria
- [ ] Base Skeleton component with shimmer animation
- [ ] MealCardSkeleton matches meal card layout
- [ ] MetricCardSkeleton matches metric card layout
- [ ] Skeletons shown during loading instead of spinner
- [ ] Smooth transition from skeleton to real content

---

## Feature 10: Visual Consistency Audit and Fixes

### Description
Audit and fix visual inconsistencies across the app to ensure strict adherence to design tokens.

### Issues to Fix

#### Spacing Inconsistencies
- Replace all hardcoded spacing values with design tokens
- Ensure consistent use of spacing scale (4, 8, 16, 24, 32, 40, 48, 64)
- Audit gap values in flex layouts

#### Border Radius Inconsistencies
- Standardize card border radius to `borderRadius.lg` (16)
- Standardize button border radius to `borderRadius.md` (12)
- Standardize input border radius to `borderRadius.sm` (8)

#### Shadow Inconsistencies
- Use `shadows.sm` for subtle elevation
- Use `shadows.md` for cards
- Use `shadows.lg` for modals/overlays
- Remove any hardcoded shadow values

#### Typography Inconsistencies
- Ensure all text uses typography presets
- Replace inline font sizes with responsive typography
- Consistent use of font weights

### Files to Audit
- All files in `app/(tabs)/`
- All files in `app/auth/`
- All files in `app/activity/`
- All files in `app/health/`
- `app/add-meal.tsx`
- `app/supplements.tsx`
- `app/profile.tsx`

### Acceptance Criteria
- [ ] No hardcoded spacing values (use spacing.xs, sm, md, lg, xl)
- [ ] No hardcoded border radius values (use borderRadius tokens)
- [ ] No hardcoded shadow values (use shadows tokens)
- [ ] Consistent typography across all screens
- [ ] All colors from design tokens

---

## Feature 11: Interactive Feedback Improvements

### Description
Add haptic feedback and improved visual feedback for user interactions.

### Requirements

#### Haptic Feedback
- Add haptic feedback on button presses (light impact)
- Add haptic feedback on successful actions (success notification)
- Add haptic feedback on errors (error notification)
- Add haptic feedback on swipe actions (medium impact)

#### Implementation
```typescript
import * as Haptics from 'expo-haptics';

// Light tap feedback
Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

// Success feedback
Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

// Error feedback
Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
```

#### Visual Press States
- All touchable elements should have visible press feedback
- Use Pressable with style function for dynamic press states
- Standard press opacity: 0.7-0.8

### Acceptance Criteria
- [ ] Haptic feedback on primary button presses
- [ ] Haptic feedback on meal card swipe actions
- [ ] Haptic feedback on successful form submissions
- [ ] All buttons have visible pressed state
- [ ] Haptic intensity appropriate for action type

---

## Feature 12: Toast/Notification System

### Description
Create a toast notification system for non-critical feedback messages.

### Current Problem
- All feedback uses Alert.alert() which is disruptive
- No way to show success messages without interrupting flow
- Error messages require dismissal before continuing

### Requirements

#### Toast Types
- `success`: Green background, checkmark icon
- `error`: Red background, X icon
- `warning`: Yellow background, warning icon
- `info`: Blue background, info icon

#### Behavior
- Appears at top of screen below safe area
- Auto-dismisses after 3 seconds (configurable)
- Can be manually dismissed by swipe or tap
- Queue system for multiple toasts

#### Props Interface
```typescript
interface ToastOptions {
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  duration?: number; // ms, default 3000
  action?: {
    label: string;
    onPress: () => void;
  };
}

// Usage via context
const { showToast } = useToast();
showToast({ type: 'success', message: 'Meal saved successfully!' });
```

### File Location
- `lib/components/ui/Toast.tsx` - Toast component
- `lib/context/ToastContext.tsx` - Toast provider and hook

### Acceptance Criteria
- [ ] Toast renders correctly for all types
- [ ] Auto-dismiss works with configurable duration
- [ ] Manual dismiss via swipe works
- [ ] Multiple toasts queue properly
- [ ] Accessible announcements for screen readers

---

## Implementation Priority

### Phase 1: Foundation (Week 1)
1. Shared Button component
2. Shared Input component
3. Shared Card component

### Phase 2: Screen Components (Week 2)
4. Shared ScreenHeader component
5. Shared EmptyState component
6. Loading Skeleton components

### Phase 3: Accessibility (Week 3)
7. Accessibility improvements across all screens
8. Form validation UX improvements

### Phase 4: Polish (Week 4)
9. Empty states implementation
10. Visual consistency audit and fixes
11. Interactive feedback improvements
12. Toast notification system

---

## Success Metrics

- All interactive elements have accessibility labels (100% coverage)
- No hardcoded design values in component styles
- Consistent component usage across all screens
- Form validation errors shown inline (0 Alert dialogs for validation)
- Loading states use skeletons instead of spinners
- All empty states have helpful messages and CTAs
