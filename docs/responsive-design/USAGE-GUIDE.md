# Responsive Design Usage Guide

## Quick Start

### 1. Import Required Dependencies

```typescript
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import { spacing } from '@/lib/theme/colors';
```

### 2. Use the Hook in Your Component

```typescript
export default function MyScreen() {
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  // Your component code
}
```

### 3. Apply Responsive Styles to ScrollView

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
  <View style={styles.content}>
    {/* Your content */}
  </View>
</ScrollView>
```

### 4. Add Required Styles

```typescript
const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: spacing.lg,
  },
  scrollContentTablet: {
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
  },
  content: {
    paddingVertical: spacing.lg,
  },
});
```

## Common Patterns

### Form Screens (SignIn, SignUp, Add Meal, etc.)

Forms should be centered on tablets with a maximum width:

```tsx
function FormScreen() {
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView
          contentContainerStyle={[
            styles.scrollContent,
            { paddingHorizontal: responsiveSpacing.horizontal },
            isTablet && styles.scrollContentTablet
          ]}
        >
          {/* Form fields */}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
```

### Charts and Graphs

For components requiring explicit width:

```typescript
const { isTablet, getSpacing, width: screenWidth } = useResponsive();
const responsiveSpacing = getSpacing();

// Calculate chart width based on container
const effectiveContentWidth = isTablet
  ? Math.min(screenWidth, FORM_MAX_WIDTH)
  : screenWidth;
const chartWidth = effectiveContentWidth - responsiveSpacing.horizontal * 2 - spacing.md * 2;

// Use in chart component
<LineChart
  data={chartData}
  width={chartWidth}
  height={200}
/>
```

### Simple Centered Content (404 Page, etc.)

```tsx
function NotFoundScreen() {
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  return (
    <View style={styles.container}>
      <View style={[
        styles.content,
        { paddingHorizontal: responsiveSpacing.horizontal },
        isTablet && styles.contentTablet
      ]}>
        <Text>Content here</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  content: {
    alignItems: 'center',
  },
  contentTablet: {
    maxWidth: FORM_MAX_WIDTH,
    width: '100%',
  },
});
```

## Hook API Reference

### useResponsive()

Returns an object with:

| Property | Type | Description |
|----------|------|-------------|
| `isTablet` | `boolean` | True if screen width >= 768pt |
| `getSpacing` | `() => object` | Returns responsive spacing values |
| `width` | `number` | Current screen width in points |
| `height` | `number` | Current screen height in points |
| `scale` | `(value: number) => number` | Scale a value based on screen size |
| `deviceCategory` | `string` | 'small' \| 'medium' \| 'large' \| 'tablet' |

### getSpacing() Return Value

```typescript
{
  horizontal: number,  // 16-48pt based on device
  vertical: number,    // Vertical spacing
}
```

## Testing Your Implementation

### Development Testing

1. Run the app: `npm start`
2. Press `i` for iOS simulator
3. Use `Cmd + D` to open developer menu
4. Test on different simulators from Xcode

### Device Category Testing

Test on at least one device from each category:
- **Small/Medium iPhone**: iPhone SE or iPhone 14
- **Large iPhone**: iPhone Pro Max variant
- **iPad**: Any iPad model

### Checklist

- [ ] No horizontal overflow on small screens
- [ ] Content centered with max-width on tablets
- [ ] Text readable at all sizes
- [ ] Touch targets >= 44pt
- [ ] Safe areas respected
- [ ] Keyboard doesn't obscure form fields
