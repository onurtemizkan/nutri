import { useEffect, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, Animated } from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useOnboarding } from '@/lib/context/OnboardingContext';
import { colors, gradients, typography, spacing, borderRadius } from '@/lib/theme/colors';
import { getErrorMessage } from '@/lib/utils/errorHandling';

export default function OnboardingComplete() {
  const router = useRouter();
  const { completeOnboarding, isLoading, status } = useOnboarding();
  const [fadeAnim] = useState(new Animated.Value(0));
  const [scaleAnim] = useState(new Animated.Value(0.8));

  useEffect(() => {
    // Entrance animation
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 8,
        tension: 40,
        useNativeDriver: true,
      }),
    ]).start();
  }, [fadeAnim, scaleAnim]);

  const handleComplete = async () => {
    try {
      await completeOnboarding();
      router.replace('/(tabs)');
    } catch (error) {
      Alert.alert('Error', getErrorMessage(error, 'Failed to complete onboarding'));
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient colors={gradients.dark} style={styles.gradient}>
        <View style={styles.content}>
          {/* Celebration Icon */}
          <Animated.View
            style={[
              styles.iconContainer,
              {
                opacity: fadeAnim,
                transform: [{ scale: scaleAnim }],
              },
            ]}
          >
            <LinearGradient colors={gradients.primary} style={styles.iconGradient}>
              <Ionicons name="checkmark-circle" size={64} color={colors.text.primary} />
            </LinearGradient>
          </Animated.View>

          {/* Title */}
          <Animated.View style={{ opacity: fadeAnim }}>
            <Text style={styles.title}>You're All Set!</Text>
            <Text style={styles.subtitle}>
              Your profile is ready. Let's start your nutrition journey.
            </Text>
          </Animated.View>

          {/* Summary Card */}
          <Animated.View style={[styles.summaryCard, { opacity: fadeAnim }]}>
            <Text style={styles.summaryTitle}>Setup Complete</Text>

            <View style={styles.summaryItem}>
              <Ionicons name="person-circle-outline" size={24} color={colors.primary.main} />
              <Text style={styles.summaryText}>Profile configured</Text>
              <Ionicons name="checkmark" size={20} color={colors.semantic.success} />
            </View>

            <View style={styles.summaryItem}>
              <Ionicons name="trophy-outline" size={24} color={colors.primary.main} />
              <Text style={styles.summaryText}>Goals personalized</Text>
              <Ionicons name="checkmark" size={20} color={colors.semantic.success} />
            </View>

            <View style={styles.summaryItem}>
              <Ionicons name="notifications-outline" size={24} color={colors.primary.main} />
              <Text style={styles.summaryText}>Permissions set</Text>
              <Ionicons name="checkmark" size={20} color={colors.semantic.success} />
            </View>

            {status && status.skippedSteps.length > 0 && (
              <View style={styles.skippedNote}>
                <Ionicons name="information-circle-outline" size={16} color={colors.text.tertiary} />
                <Text style={styles.skippedText}>
                  You skipped {status.skippedSteps.length} optional step{status.skippedSteps.length > 1 ? 's' : ''}.
                  You can complete them later in Settings.
                </Text>
              </View>
            )}
          </Animated.View>

          {/* What's Next */}
          <Animated.View style={[styles.nextStepsCard, { opacity: fadeAnim }]}>
            <Text style={styles.nextStepsTitle}>What's Next?</Text>

            <View style={styles.nextStep}>
              <View style={styles.nextStepNumber}>
                <Text style={styles.nextStepNumberText}>1</Text>
              </View>
              <Text style={styles.nextStepText}>Log your first meal to get started</Text>
            </View>

            <View style={styles.nextStep}>
              <View style={styles.nextStepNumber}>
                <Text style={styles.nextStepNumberText}>2</Text>
              </View>
              <Text style={styles.nextStepText}>Track your health metrics daily</Text>
            </View>

            <View style={styles.nextStep}>
              <View style={styles.nextStepNumber}>
                <Text style={styles.nextStepNumberText}>3</Text>
              </View>
              <Text style={styles.nextStepText}>Get AI-powered nutrition insights</Text>
            </View>
          </Animated.View>

          {/* Spacer */}
          <View style={styles.spacer} />

          {/* Action Button */}
          <Animated.View style={[styles.footer, { opacity: fadeAnim }]}>
            <TouchableOpacity
              style={styles.startButton}
              onPress={handleComplete}
              disabled={isLoading}
            >
              <LinearGradient colors={gradients.primary} style={styles.startButtonGradient}>
                <Text style={styles.startButtonText}>
                  {isLoading ? 'Getting Ready...' : 'Start Tracking'}
                </Text>
                <Ionicons name="arrow-forward" size={20} color={colors.text.primary} />
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View>
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
  content: {
    flex: 1,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.xl * 2,
    alignItems: 'center',
  },
  iconContainer: {
    marginBottom: spacing.xl,
  },
  iconGradient: {
    width: 120,
    height: 120,
    borderRadius: 60,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    ...typography.h1,
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: spacing.sm,
  },
  subtitle: {
    ...typography.body,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  summaryCard: {
    width: '100%',
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
  },
  summaryTitle: {
    ...typography.h3,
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  summaryItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.sm,
  },
  summaryText: {
    ...typography.body,
    color: colors.text.primary,
    flex: 1,
    marginLeft: spacing.md,
  },
  skippedNote: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.background.primary,
  },
  skippedText: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
    flex: 1,
    marginLeft: spacing.xs,
  },
  nextStepsCard: {
    width: '100%',
    backgroundColor: colors.surface.elevated,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
  },
  nextStepsTitle: {
    ...typography.bodyBold,
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  nextStep: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  nextStepNumber: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.primary.dark,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  nextStepNumberText: {
    ...typography.bodyBold,
    color: colors.primary.main,
    fontSize: 14,
  },
  nextStepText: {
    ...typography.body,
    color: colors.text.secondary,
    flex: 1,
  },
  spacer: {
    flex: 1,
  },
  footer: {
    width: '100%',
    paddingBottom: spacing.xl,
  },
  startButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
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
});
