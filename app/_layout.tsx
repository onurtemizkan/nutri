import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import Constants, { ExecutionEnvironment } from 'expo-constants';
import { useFonts } from 'expo-font';
import { Stack, useRouter, useSegments } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useState, useRef } from 'react';
import { Platform } from 'react-native';
import 'react-native-reanimated';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { PortalProvider, PortalHost } from '@gorhom/portal';

import { useColorScheme } from '@/hooks/useColorScheme';
import { AuthProvider, useAuth } from '@/lib/context/AuthContext';
import { NotificationProvider } from '@/lib/context/NotificationContext';
import { AlertProvider } from '@/lib/components/CustomAlert';
import { ErrorBoundary } from '@/lib/components/ErrorBoundary';
import { healthKitService } from '@/lib/services/healthkit';
import { onboardingApi } from '@/lib/onboarding/api';

// Check if running in Expo Go or a dev build without native modules
// Screen orientation native module is only available in production EAS builds
const isExpoGo = Constants.executionEnvironment === ExecutionEnvironment.StoreClient;

// Check if expo-screen-orientation native module is available
// This is a safer check that works for both Expo Go and dev builds without the native module
function isScreenOrientationAvailable(): boolean {
  try {
    // Try to access the native module registry
    const { NativeModulesProxy } = require('expo-modules-core');
    return NativeModulesProxy?.ExpoScreenOrientation != null;
  } catch {
    return false;
  }
}

// Prevent the splash screen from auto-hiding before asset loading is complete.
SplashScreen.preventAutoHideAsync();

function RootLayoutNav() {
  const { isAuthenticated, isLoading } = useAuth();
  const segments = useSegments();
  const router = useRouter();
  const [onboardingChecked, setOnboardingChecked] = useState(false);
  const [needsOnboarding, setNeedsOnboarding] = useState(false);

  // Track if we were in onboarding to detect when leaving (completion)
  const wasInOnboardingRef = useRef(false);
  const inOnboardingGroup = segments[0] === 'onboarding';

  // Check onboarding status when user is authenticated
  // Also re-check when leaving onboarding flow (to detect completion)
  useEffect(() => {
    // Detect when we just left onboarding
    const justLeftOnboarding = wasInOnboardingRef.current && !inOnboardingGroup;
    wasInOnboardingRef.current = inOnboardingGroup;

    async function checkOnboarding() {
      if (!isAuthenticated) {
        setOnboardingChecked(false);
        setNeedsOnboarding(false);
        return;
      }

      // If we just left onboarding, reset checked state to prevent
      // the navigation effect from redirecting before we re-verify
      if (justLeftOnboarding) {
        setOnboardingChecked(false);
      }

      try {
        const status = await onboardingApi.getStatus();
        // User needs onboarding if they haven't completed it yet
        setNeedsOnboarding(!status?.isComplete);
      } catch (error) {
        // If we can't get status, assume onboarding might be needed
        // 404 means onboarding hasn't started
        setNeedsOnboarding(true);
      } finally {
        setOnboardingChecked(true);
      }
    }

    // Run check if:
    // 1. Initial auth check (isAuthenticated && !isLoading && !onboardingChecked)
    // 2. We just left onboarding (need to verify completion)
    if (isAuthenticated && !isLoading) {
      if (!onboardingChecked || justLeftOnboarding) {
        checkOnboarding();
      }
    }
  }, [isAuthenticated, isLoading, inOnboardingGroup, onboardingChecked]);

  useEffect(() => {
    if (isLoading) return;

    const inAuthGroup = segments[0] === 'auth';
    // Note: inOnboardingGroup is already defined at component level
    // These routes are accessible without authentication
    const inPublicRoute = segments[0] === 'terms' || segments[0] === 'privacy';

    // Not authenticated - go to auth (unless on a public route)
    if (!isAuthenticated && !inAuthGroup && !inPublicRoute) {
      router.replace('/auth/welcome');
      return;
    }

    // Authenticated - wait for onboarding check before any navigation
    if (isAuthenticated && !onboardingChecked) {
      return; // Don't navigate until we know onboarding status
    }

    // Authenticated and onboarding checked - redirect based on status
    if (isAuthenticated && onboardingChecked) {
      if (inAuthGroup) {
        // Just logged in - go to appropriate screen
        if (needsOnboarding) {
          router.replace('/onboarding');
        } else {
          router.replace('/(tabs)');
        }
      } else if (needsOnboarding && !inOnboardingGroup) {
        // Needs onboarding but not in onboarding flow
        router.replace('/onboarding');
      } else if (!needsOnboarding && inOnboardingGroup) {
        // Completed onboarding but still in onboarding flow
        router.replace('/(tabs)');
      }
    }
  }, [
    isAuthenticated,
    segments,
    isLoading,
    router,
    onboardingChecked,
    needsOnboarding,
    inOnboardingGroup,
  ]);

  return (
    <Stack>
      <Stack.Screen name="auth/welcome" options={{ headerShown: false }} />
      <Stack.Screen name="auth/signin" options={{ headerShown: false }} />
      <Stack.Screen name="auth/signup" options={{ headerShown: false }} />
      <Stack.Screen name="auth/forgot-password" options={{ headerShown: false }} />
      <Stack.Screen name="auth/reset-password" options={{ headerShown: false }} />
      <Stack.Screen name="onboarding" options={{ headerShown: false }} />
      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
      <Stack.Screen
        name="add-meal"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
        }}
      />
      <Stack.Screen
        name="edit-meal/[id]"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="edit-health-metric/[id]"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="scan-food"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
        }}
      />
      <Stack.Screen
        name="scan-barcode"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
        }}
      />
      <Stack.Screen
        name="scan-supplement-barcode"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
        }}
      />
      <Stack.Screen
        name="health-settings"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="notification-settings"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="ar-measure"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
          presentation: 'modal',
        }}
      />
      <Stack.Screen
        name="health/[metricType]"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="health/add"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
        }}
      />
      <Stack.Screen
        name="ar-scan-food"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
          presentation: 'modal',
        }}
      />
      <Stack.Screen
        name="terms"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="privacy"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="supplements"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="food-search"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
        }}
      />
      <Stack.Screen
        name="notifications/priming"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
          presentation: 'modal',
        }}
      />
      <Stack.Screen
        name="notifications/preferences"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="water"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="water-goal"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="water-add"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
        }}
      />
      <Stack.Screen
        name="weight/index"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="weight/add"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
        }}
      />
      <Stack.Screen
        name="weight/edit/[id]"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="weight/goal"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
        }}
      />
      <Stack.Screen
        name="goals"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="simulate"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="paywall"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
          presentation: 'modal',
        }}
      />
      <Stack.Screen
        name="subscription"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="reports/weekly"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="reports/monthly"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="reports/export"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
          presentation: 'modal',
        }}
      />
      <Stack.Screen
        name="predictions/index"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="predictions/[metric]"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="insights"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen name="+not-found" options={{ headerShown: false }} />
    </Stack>
  );
}

export default function RootLayout() {
  const colorScheme = useColorScheme();
  const [loaded] = useFonts({
    SpaceMono: require('../assets/fonts/SpaceMono-Regular.ttf'),
  });

  // Configure orientation: Portrait-only for iPhones, both orientations for iPads
  // Note: expo-screen-orientation requires a production EAS build (not available in Expo Go or dev builds without native module)
  useEffect(() => {
    async function configureOrientation() {
      // Skip if native module isn't available (Expo Go, dev builds without native module)
      if (isExpoGo || !isScreenOrientationAvailable()) {
        console.log('ScreenOrientation: Skipping (native module not available)');
        return;
      }

      try {
        // Dynamic import to avoid bundling issues
        const ScreenOrientation = await import('expo-screen-orientation');

        if (Platform.OS === 'ios' && Platform.isPad) {
          // iPad: Allow all orientations (portrait and landscape)
          await ScreenOrientation.unlockAsync();
        } else {
          // iPhone/Android: Lock to portrait only
          await ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.PORTRAIT_UP);
        }
      } catch (error) {
        // Fallback: silently ignore if native module fails
        console.log('ScreenOrientation: Error configuring orientation:', error);
      }
    }
    configureOrientation();
  }, []);

  useEffect(() => {
    if (loaded) {
      SplashScreen.hideAsync();
    }
  }, [loaded]);

  // HealthKit test on startup (iOS only)
  useEffect(() => {
    async function testHealthKit() {
      if (Platform.OS !== 'ios') {
        console.log('🏥 HealthKit: Not available (not iOS)');
        return;
      }

      console.log('🏥 HealthKit: Testing availability...');

      try {
        const isAvailable = await healthKitService.isAvailable();
        console.log('🏥 HealthKit: Available =', isAvailable);

        if (isAvailable) {
          const status = await healthKitService.getStatus();
          console.log('🏥 HealthKit: Status =', JSON.stringify(status, null, 2));

          // Try requesting permissions
          console.log('🏥 HealthKit: Requesting permissions...');
          const permResult = await healthKitService.requestPermissions();
          console.log('🏥 HealthKit: Permission result =', JSON.stringify(permResult, null, 2));

          if (permResult.success) {
            console.log('🏥 HealthKit: ✅ Successfully connected!');
            // Try to get today's activity as a test
            try {
              const activity = await healthKitService.getTodayActivity();
              console.log('🏥 HealthKit: Today activity =', JSON.stringify(activity, null, 2));
            } catch (actError) {
              console.log('🏥 HealthKit: Activity fetch error (expected in simulator):', actError);
            }
          } else {
            console.log('🏥 HealthKit: ❌ Permission denied:', permResult.error);
          }
        } else {
          console.log('🏥 HealthKit: ❌ Not available on this device');
        }
      } catch (error) {
        console.log('🏥 HealthKit: Error during test:', error);
      }
    }

    testHealthKit();
  }, []);

  if (!loaded) {
    return null;
  }

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <ErrorBoundary
        onError={(error, errorInfo) => {
          // Log to console for development
          console.error('[App Error]', error.message);
          console.error('[App Error] Stack:', errorInfo.componentStack);
          // TODO: Send to error reporting service (Sentry, etc.)
        }}
      >
        <PortalProvider>
          <AlertProvider>
            <AuthProvider>
              <NotificationProvider>
                <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
                  <RootLayoutNav />
                  <StatusBar style="light" />
                </ThemeProvider>
              </NotificationProvider>
            </AuthProvider>
          </AlertProvider>
          <PortalHost name="alert-portal" />
        </PortalProvider>
      </ErrorBoundary>
    </GestureHandlerRootView>
  );
}
