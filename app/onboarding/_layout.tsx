import { Stack } from 'expo-router';
import { OnboardingProvider } from '@/lib/context/OnboardingContext';

export default function OnboardingLayout() {
  return (
    <OnboardingProvider>
      <Stack
        screenOptions={{
          headerShown: false,
          animation: 'slide_from_right',
          gestureEnabled: false,
        }}
      >
        <Stack.Screen name="index" />
        <Stack.Screen name="profile" />
        <Stack.Screen name="goals" />
        <Stack.Screen name="permissions" />
        <Stack.Screen name="health-background" />
        <Stack.Screen name="lifestyle" />
        <Stack.Screen name="complete" />
      </Stack>
    </OnboardingProvider>
  );
}
