# Custom Alert System

The app now uses a custom alert component that matches the dark theme instead of the native Alert dialogs.

## What Changed

- **Native Alerts**: System dialogs (light colored, can't be styled)
- **Custom Alerts**: Styled modal dialogs matching your dark theme with purple gradient buttons

## How to Use

### 1. Import the utility
```typescript
import { showAlert } from '@/lib/utils/alert';
```

### 2. Replace `Alert.alert` with `showAlert`

**Simple Alert:**
```typescript
// Before
Alert.alert('Success', 'Your changes have been saved!');

// After
showAlert('Success', 'Your changes have been saved!');
```

**Error Alert:**
```typescript
// Before
Alert.alert('Error', 'Please fill in all required fields');

// After
showAlert('Error', 'Please fill in all required fields');
```

**Confirmation Dialog:**
```typescript
// Before
Alert.alert('Logout', 'Are you sure you want to logout?', [
  { text: 'Cancel', style: 'cancel' },
  {
    text: 'Logout',
    style: 'destructive',
    onPress: async () => {
      await logout();
      router.replace('/auth/welcome');
    },
  },
]);

// After
showAlert('Logout', 'Are you sure you want to logout?', [
  { text: 'Cancel', style: 'cancel' },
  {
    text: 'Logout',
    style: 'destructive',
    onPress: async () => {
      await logout();
      router.replace('/auth/welcome');
    },
  },
]);
```

**Multiple Buttons:**
```typescript
showAlert(
  'Development Mode',
  `Reset token: ${response.resetToken}\n\nIn production, this would be sent via email.`,
  [
    {
      text: 'Copy Token',
      onPress: () => {
        console.log('Token:', response.resetToken);
      },
    },
    {
      text: 'Go to Reset',
      onPress: () => router.push('/auth/reset-password'),
    },
  ]
);
```

## Button Styles

- **`'default'`**: Purple gradient button (primary action)
- **`'cancel'`**: Outlined transparent button (secondary action)
- **`'destructive'`**: Red gradient button (dangerous action like delete/logout)

## Examples Updated

These files have been updated as examples:
- `app/(tabs)/profile.tsx` - Shows logout confirmation and success messages
- `app/add-meal.tsx` - Shows error and success messages

## Remaining Files to Update

You can replace `Alert.alert` with `showAlert` in these files:
- `app/auth/signin.tsx`
- `app/auth/signup.tsx`
- `app/auth/forgot-password.tsx`
- `app/auth/reset-password.tsx`

Just:
1. Import: `import { showAlert } from '@/lib/utils/alert';`
2. Remove: `import { Alert } from 'react-native';` (or remove Alert from the imports)
3. Replace: `Alert.alert(...)` with `showAlert(...)`

## Features

- **Dark themed** - Matches your app's color scheme
- **Animated** - Smooth fade and scale animations
- **Gradient buttons** - Purple/pink gradients for primary actions
- **Flexible** - Supports 1-3 buttons in various layouts
- **Backdrop dismiss** - Tap outside to close (also calls cancel button if present)
- **Type-safe** - Full TypeScript support
