import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import { useFonts } from 'expo-font';
import { Stack, useRouter, useSegments } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import { useEffect } from 'react';
import { Platform } from 'react-native';
import 'react-native-reanimated';
import { PortalProvider, PortalHost } from '@gorhom/portal';

import { useColorScheme } from '@/hooks/useColorScheme';
import { AuthProvider, useAuth } from '@/lib/context/AuthContext';
import { AlertProvider } from '@/lib/components/CustomAlert';
import { healthKitService } from '@/lib/services/healthkit';

// Prevent the splash screen from auto-hiding before asset loading is complete.
SplashScreen.preventAutoHideAsync();

function RootLayoutNav() {
  const { isAuthenticated, isLoading } = useAuth();
  const segments = useSegments();
  const router = useRouter();

  useEffect(() => {
    if (isLoading) return;

    const inAuthGroup = segments[0] === 'auth';

    if (!isAuthenticated && !inAuthGroup) {
      router.replace('/auth/welcome');
    } else if (isAuthenticated && inAuthGroup) {
      router.replace('/(tabs)');
    }
  }, [isAuthenticated, segments, isLoading, router]);

  return (
    <Stack>
      <Stack.Screen name="auth/welcome" options={{ headerShown: false }} />
      <Stack.Screen name="auth/signin" options={{ headerShown: false }} />
      <Stack.Screen name="auth/signup" options={{ headerShown: false }} />
      <Stack.Screen name="auth/forgot-password" options={{ headerShown: false }} />
      <Stack.Screen name="auth/reset-password" options={{ headerShown: false }} />
      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
      <Stack.Screen
        name="add-meal"
        options={{
          headerShown: false,
          animation: 'slide_from_bottom'
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
        name="health-settings"
        options={{
          headerShown: false,
          animation: 'slide_from_right'
        }}
      />
      <Stack.Screen name="+not-found" />
    </Stack>
  );
}

export default function RootLayout() {
  const colorScheme = useColorScheme();
  const [loaded] = useFonts({
    SpaceMono: require('../assets/fonts/SpaceMono-Regular.ttf'),
  });

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
    <PortalProvider>
      <AlertProvider>
        <AuthProvider>
          <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
            <RootLayoutNav />
            <StatusBar style="light" />
          </ThemeProvider>
        </AuthProvider>
      </AlertProvider>
      <PortalHost name="alert-portal" />
    </PortalProvider>
  );
}
