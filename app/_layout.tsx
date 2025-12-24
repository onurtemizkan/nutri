import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import Constants, { ExecutionEnvironment } from 'expo-constants';
import { useFonts } from 'expo-font';
import { Stack, useRouter, useSegments } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useState } from 'react';
import { Platform } from 'react-native';
import 'react-native-reanimated';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { PortalProvider, PortalHost } from '@gorhom/portal';

import { useColorScheme } from '@/hooks/useColorScheme';
import { AuthProvider, useAuth } from '@/lib/context/AuthContext';
import { NotificationProvider } from '@/lib/context/NotificationContext';
import { AlertProvider } from '@/lib/components/CustomAlert';
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

  // Check onboarding status when user is authenticated
  useEffect(() => {
    async function checkOnboarding() {
      if (!isAuthenticated) {
        setOnboardingChecked(false);
        setNeedsOnboarding(false);
        return;
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

    if (isAuthenticated && !isLoading) {
      checkOnboarding();
    }
  }, [isAuthenticated, isLoading]);

  useEffect(() => {
    if (isLoading) return;

    const inAuthGroup = segments[0] === 'auth';
    const inOnboardingGroup = segments[0] === 'onboarding';

    // Not authenticated - go to auth
    if (!isAuthenticated && !inAuthGroup) {
      router.replace('/auth/welcome');
      return;
    }

    // Authenticated but in auth group - redirect based on onboarding status
    if (isAuthenticated && inAuthGroup) {
      if (!onboardingChecked) return; // Wait for onboarding check

      if (needsOnboarding) {
        router.replace('/onboarding');
      } else {
        router.replace('/(tabs)');
      }
      return;
    }

    // Authenticated - check onboarding status
    if (isAuthenticated && onboardingChecked) {
      if (needsOnboarding && !inOnboardingGroup) {
        router.replace('/onboarding');
      } else if (!needsOnboarding && inOnboardingGroup) {
        router.replace('/(tabs)');
      }
    }
  }, [isAuthenticated, segments, isLoading, router, onboardingChecked, needsOnboarding]);

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
          animation: 'slide_from_bottom'
        }}
      />
      <Stack.Screen
        name="edit-meal/[id]"
        options={{
          headerShown: false,
          animation: 'slide_from_right'
        }}
      />
      <Stack.Screen
        name="edit-health-metric/[id]"
        options={{
          headerShown: false,
          animation: 'slide_from_right'
        }}
      />
      <Stack.Screen
        name="scan-food"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom'
        }}
      />
      <Stack.Screen
        name="scan-barcode"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom'
        }}
      />
      <Stack.Screen
        name="scan-supplement-barcode"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom'
        }}
      />
      <Stack.Screen
        name="health-settings"
        options={{
          headerShown: false,
          animation: 'slide_from_right'
        }}
      />
      <Stack.Screen
        name="notification-settings"
        options={{
          headerShown: false,
          animation: 'slide_from_right'
        }}
      />
      <Stack.Screen
        name="ar-measure"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
          presentation: 'modal'
        }}
      />
      <Stack.Screen
        name="health/[metricType]"
        options={{
          headerShown: false,
          animation: 'slide_from_right'
        }}
      />
      <Stack.Screen
        name="health/add"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom'
        }}
      />
      <Stack.Screen
        name="ar-scan-food"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom',
          presentation: 'modal'
        }}
      />
      <Stack.Screen
        name="terms"
        options={{
          headerShown: false,
          animation: 'slide_from_right'
        }}
      />
      <Stack.Screen
        name="privacy"
        options={{
          headerShown: false,
          animation: 'slide_from_right'
        }}
      />
      <Stack.Screen
        name="supplements"
        options={{
          headerShown: false,
          animation: 'slide_from_right'
        }}
      />
      <Stack.Screen
        name="food-search"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom'
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
          await ScreenOrientation.lockAsync(
            ScreenOrientation.OrientationLock.PORTRAIT_UP
          );
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
    </GestureHandlerRootView>
  );
}
