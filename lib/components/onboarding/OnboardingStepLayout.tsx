import React, { RefObject } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import type { ScrollView as ScrollViewType } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { colors, gradients, typography, spacing, borderRadius } from '@/lib/theme/colors';

interface OnboardingStepLayoutProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  currentStep: number;
  totalSteps: number;
  onBack?: () => void;
  onNext?: () => void;
  onSkip?: () => void;
  nextLabel?: string;
  skipLabel?: string;
  isNextDisabled?: boolean;
  isLoading?: boolean;
  showSkip?: boolean;
  showBack?: boolean;
  scrollRef?: RefObject<ScrollViewType | null>;
}

export function OnboardingStepLayout({
  title,
  subtitle,
  children,
  currentStep,
  totalSteps,
  onBack,
  onNext,
  onSkip,
  nextLabel = 'Continue',
  skipLabel = 'Skip for now',
  isNextDisabled = false,
  isLoading = false,
  showSkip = false,
  showBack = true,
  scrollRef,
}: OnboardingStepLayoutProps) {
  const progress = ((currentStep - 1) / totalSteps) * 100;

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient colors={gradients.dark} style={styles.gradient}>
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={styles.keyboardView}
        >
          {/* Header with progress */}
          <View style={styles.header}>
            <View style={styles.headerTop}>
              {showBack && onBack ? (
                <TouchableOpacity onPress={onBack} style={styles.backButton}>
                  <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
                </TouchableOpacity>
              ) : (
                <View style={styles.backButton} />
              )}
              <Text style={styles.stepIndicator}>
                {currentStep} of {totalSteps}
              </Text>
              {showSkip && onSkip ? (
                <TouchableOpacity onPress={onSkip} style={styles.skipButton}>
                  <Text style={styles.skipText}>{skipLabel}</Text>
                </TouchableOpacity>
              ) : (
                <View style={styles.skipButton} />
              )}
            </View>

            {/* Progress bar */}
            <View style={styles.progressContainer}>
              <View style={styles.progressBackground}>
                <LinearGradient
                  colors={gradients.primary}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={[styles.progressFill, { width: `${progress}%` }]}
                />
              </View>
            </View>
          </View>

          {/* Content */}
          <ScrollView
            ref={scrollRef}
            style={styles.scrollView}
            contentContainerStyle={styles.scrollContent}
            showsVerticalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
          >
            <View style={styles.titleContainer}>
              <Text style={styles.title}>{title}</Text>
              {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
            </View>

            <View style={styles.content}>{children}</View>
          </ScrollView>

          {/* Footer with action button */}
          <View style={styles.footer}>
            {onNext && (
              <TouchableOpacity
                style={[styles.nextButton, isNextDisabled && styles.nextButtonDisabled]}
                onPress={onNext}
                disabled={isNextDisabled || isLoading}
              >
                <LinearGradient
                  colors={isNextDisabled ? [colors.surface.card, colors.surface.card] : gradients.primary}
                  style={styles.nextButtonGradient}
                >
                  {isLoading ? (
                    <ActivityIndicator color={colors.text.primary} />
                  ) : (
                    <Text
                      style={[styles.nextButtonText, isNextDisabled && styles.nextButtonTextDisabled]}
                    >
                      {nextLabel}
                    </Text>
                  )}
                </LinearGradient>
              </TouchableOpacity>
            )}
          </View>
        </KeyboardAvoidingView>
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
  keyboardView: {
    flex: 1,
  },
  header: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
  },
  headerTop: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  backButton: {
    flex: 1,
    height: 40,
    alignItems: 'flex-start',
    justifyContent: 'center',
  },
  stepIndicator: {
    ...typography.body,
    color: colors.text.secondary,
    textAlign: 'center',
  },
  skipButton: {
    flex: 1,
    alignItems: 'flex-end',
    justifyContent: 'center',
  },
  skipText: {
    ...typography.bodySmall,
    color: colors.primary.main,
  },
  progressContainer: {
    height: 4,
    width: '100%',
  },
  progressBackground: {
    height: '100%',
    backgroundColor: colors.surface.card,
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 2,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: spacing.lg,
  },
  titleContainer: {
    marginBottom: spacing.xl,
  },
  title: {
    ...typography.h1,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  subtitle: {
    ...typography.body,
    color: colors.text.secondary,
    lineHeight: 24,
  },
  content: {
    flex: 1,
  },
  footer: {
    padding: spacing.lg,
    paddingBottom: spacing.xl,
  },
  nextButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  nextButtonDisabled: {
    opacity: 0.6,
  },
  nextButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  nextButtonText: {
    ...typography.button,
    color: colors.text.primary,
  },
  nextButtonTextDisabled: {
    color: colors.text.secondary,
  },
});
