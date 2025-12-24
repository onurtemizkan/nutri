import { useEffect, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Alert, ActivityIndicator } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { colors, gradients, typography, spacing, borderRadius } from '@/lib/theme/colors';
import { useOnboarding } from '@/lib/context/OnboardingContext';
import { ONBOARDING_STEPS, getStepRoute } from '@/lib/onboarding/config';
import { getErrorMessage } from '@/lib/utils/errorHandling';

export default function OnboardingWelcome() {
  const router = useRouter();
  const { startOnboarding, isLoading, status } = useOnboarding();
  const [isRedirecting, setIsRedirecting] = useState(false);

  useEffect(() => {
    // If onboarding is already complete, redirect to main app
    if (status?.isComplete) {
      router.replace('/(tabs)');
      return;
    }

    // If onboarding is in progress (started but not complete), resume at current step
    if (status && !status.isComplete && status.currentStep >= 1) {
      setIsRedirecting(true);
      const route = getStepRoute(status.currentStep);
      router.replace(route as never);
    }
  }, [status, router]);

  const handleStart = async () => {
    try {
      await startOnboarding();
      router.push('/onboarding/profile');
    } catch (error) {
      Alert.alert(
        'Unable to Start',
        getErrorMessage(error, 'Failed to start onboarding. Please check your connection and try again.')
      );
    }
  };

  // Show loading while checking status or redirecting
  if (isLoading || isRedirecting) {
    return (
      <SafeAreaView style={styles.container}>
        <LinearGradient colors={gradients.dark} style={styles.gradient}>
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={colors.primary.main} />
          </View>
        </LinearGradient>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient colors={gradients.dark} style={styles.gradient}>
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Logo Section */}
          <View style={styles.logoSection}>
            <LinearGradient colors={gradients.primary} style={styles.logoContainer}>
              <Text style={styles.logoEmoji}>🥗</Text>
            </LinearGradient>
            <Text style={styles.title}>Welcome to Nutri</Text>
            <Text style={styles.subtitle}>
              Let's set up your profile to personalize your nutrition journey
            </Text>
          </View>

          {/* Features Preview */}
          <View style={styles.featuresContainer}>
            {ONBOARDING_STEPS.slice(0, 5).map((step) => (
              <View key={step.id} style={styles.featureItem}>
                <View style={styles.featureIcon}>
                  <Ionicons name={step.icon as keyof typeof Ionicons.glyphMap} size={24} color={colors.primary.main} />
                </View>
                <View style={styles.featureTextContainer}>
                  <Text style={styles.featureTitle}>{step.title}</Text>
                  <Text style={styles.featureDescription}>{step.description}</Text>
                </View>
              </View>
            ))}
          </View>
        </ScrollView>

        {/* Sticky Footer */}
        <View style={styles.footer}>
          <TouchableOpacity onPress={handleStart} disabled={isLoading} style={styles.startButton}>
            <LinearGradient colors={gradients.primary} style={styles.startButtonGradient}>
              <Text style={styles.startButtonText}>
                {isLoading ? 'Starting...' : "Let's Get Started"}
              </Text>
              <Ionicons name="arrow-forward" size={20} color={colors.text.primary} />
            </LinearGradient>
          </TouchableOpacity>

          <Text style={styles.timeEstimate}>Takes about 3-5 minutes</Text>
        </View>
      </LinearGradient>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  gradient: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.xl,
    paddingBottom: spacing.md,
  },
  logoSection: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  logoContainer: {
    width: 80,
    height: 80,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.md,
  },
  logoEmoji: {
    fontSize: 40,
  },
  title: {
    ...typography.h1,
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: spacing.xs,
  },
  subtitle: {
    ...typography.body,
    color: colors.text.secondary,
    textAlign: 'center',
    paddingHorizontal: spacing.lg,
  },
  featuresContainer: {
    marginTop: spacing.lg,
    marginBottom: spacing.xl,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
    paddingVertical: spacing.sm,
  },
  featureIcon: {
    width: 48,
    height: 48,
    borderRadius: borderRadius.md,
    backgroundColor: colors.surface.card,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  featureTextContainer: {
    flex: 1,
  },
  featureTitle: {
    ...typography.bodyBold,
    color: colors.text.primary,
    marginBottom: 2,
  },
  featureDescription: {
    ...typography.bodySmall,
    color: colors.text.secondary,
  },
  footer: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.xl,
  },
  startButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    marginBottom: spacing.md,
  },
  startButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
    gap: spacing.sm,
  },
  startButtonText: {
    ...typography.button,
    color: colors.text.primary,
  },
  timeEstimate: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
    textAlign: 'center',
  },
});
