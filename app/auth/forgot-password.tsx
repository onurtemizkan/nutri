import { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  Alert,
  ActivityIndicator,
  ScrollView,
} from 'react-native';
import { Link, useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { authApi } from '@/lib/api/auth';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';

export default function ForgotPasswordScreen() {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [emailSent, setEmailSent] = useState(false);
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  const handleForgotPassword = async () => {
    if (!email) {
      Alert.alert('Error', 'Please enter your email address');
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      Alert.alert('Error', 'Please enter a valid email address');
      return;
    }

    setIsLoading(true);
    try {
      const response = await authApi.forgotPassword(email);
      setEmailSent(true);

      // In development, show the reset token
      if (response.resetToken) {
        Alert.alert(
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
      }
    } catch (error) {
      Alert.alert('Error', getErrorMessage(error, 'Failed to send reset email'));
    } finally {
      setIsLoading(false);
    }
  };

  if (emailSent) {
    return (
      <SafeAreaView style={styles.container} testID="forgot-password-success-screen">
        <View
          style={[
            styles.successContent,
            { paddingHorizontal: responsiveSpacing.horizontal },
            isTablet && styles.tabletContent,
          ]}
        >
          <View style={styles.successContainer}>
            {/* Success Icon */}
            <View style={styles.successIconContainer}>
              <LinearGradient colors={gradients.primary} style={styles.successIconGradient}>
                <Text style={styles.successEmoji}>✉️</Text>
              </LinearGradient>
            </View>

            {/* Success Message */}
            <Text style={styles.successTitle}>Check Your Email</Text>
            <Text style={styles.successMessage}>
              If an account exists for {email}, you will receive a password reset link shortly.
            </Text>

            {/* Actions */}
            <View style={styles.successActions}>
              <TouchableOpacity
                style={styles.button}
                onPress={() => router.push('/auth/signin')}
                activeOpacity={0.8}
                testID="forgot-password-back-to-signin-button"
                accessibilityRole="button"
                accessibilityLabel="Back to sign in"
                accessibilityHint="Navigate back to sign in screen"
              >
                <LinearGradient
                  colors={gradients.primary}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.buttonGradient}
                >
                  <Text style={styles.buttonText}>Back to Sign In</Text>
                </LinearGradient>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.secondaryButton}
                onPress={() => setEmailSent(false)}
                activeOpacity={0.8}
                testID="forgot-password-try-another-email-button"
                accessibilityRole="button"
                accessibilityLabel="Try another email"
                accessibilityHint="Go back to enter a different email address"
              >
                <Text style={styles.secondaryButtonText}>Try Another Email</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} testID="forgot-password-screen">
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={[
            styles.scrollContent,
            { paddingHorizontal: responsiveSpacing.horizontal },
            isTablet && styles.tabletContent,
          ]}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
          keyboardDismissMode="interactive"
        >
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.title}>Forgot Password?</Text>
            <Text style={styles.subtitle}>
              Enter your email address and we'll send you a link to reset your password.
            </Text>
          </View>

          {/* Form */}
          <View style={styles.form}>
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Email</Text>
              <View style={styles.inputWrapper}>
                <TextInput
                  style={styles.input}
                  placeholder="your@email.com"
                  placeholderTextColor={colors.text.disabled}
                  value={email}
                  onChangeText={setEmail}
                  autoCapitalize="none"
                  keyboardType="email-address"
                  editable={!isLoading}
                  autoFocus
                  autoComplete="email"
                  returnKeyType="done"
                  onSubmitEditing={handleForgotPassword}
                  testID="forgot-password-email-input"
                  accessibilityLabel="Email address"
                  accessibilityHint="Enter the email address associated with your account"
                />
              </View>
            </View>

            <TouchableOpacity
              style={[styles.button, isLoading && styles.buttonDisabled]}
              onPress={handleForgotPassword}
              disabled={isLoading}
              activeOpacity={0.8}
              testID="forgot-password-submit-button"
              accessibilityRole="button"
              accessibilityLabel={isLoading ? 'Sending reset link' : 'Send reset link'}
              accessibilityHint="Double tap to send password reset link to your email"
              accessibilityState={{ disabled: isLoading, busy: isLoading }}
            >
              <LinearGradient
                colors={
                  isLoading ? [colors.text.disabled, colors.text.disabled] : gradients.primary
                }
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.buttonGradient}
              >
                {isLoading ? (
                  <ActivityIndicator color={colors.text.primary} />
                ) : (
                  <Text style={styles.buttonText}>Send Reset Link</Text>
                )}
              </LinearGradient>
            </TouchableOpacity>

            <View style={styles.footer}>
              <Text style={styles.footerText}>Remember your password? </Text>
              <Link href="/auth/signin" asChild>
                <TouchableOpacity
                  disabled={isLoading}
                  testID="forgot-password-signin-link"
                  accessibilityRole="link"
                  accessibilityLabel="Sign in"
                  accessibilityHint="Navigate back to sign in screen"
                >
                  <Text style={styles.link}>Sign In</Text>
                </TouchableOpacity>
              </Link>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  keyboardView: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing['2xl'],
    paddingBottom: spacing.xl,
  },
  tabletContent: {
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
  },

  // Header
  header: {
    marginBottom: spacing['2xl'],
  },
  title: {
    fontSize: typography.fontSize['4xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
    lineHeight: typography.lineHeight.relaxed * typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
  },

  // Form
  form: {
    gap: spacing.lg,
  },

  // Input
  inputContainer: {
    gap: spacing.sm,
  },
  label: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    letterSpacing: 0.3,
  },
  inputWrapper: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    overflow: 'hidden',
  },
  input: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    height: 52,
    textAlignVertical: 'center',
  },

  // Button
  button: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    marginTop: spacing.sm,
    ...shadows.md,
  },
  buttonDisabled: {
    opacity: 0.7,
  },
  buttonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
    height: 52,
  },
  buttonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    letterSpacing: 0.5,
  },

  // Secondary Button
  secondaryButton: {
    backgroundColor: 'transparent',
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: colors.border.primary,
    height: 52,
  },
  secondaryButtonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    letterSpacing: 0.5,
  },

  // Footer
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: spacing.md,
  },
  footerText: {
    color: colors.text.tertiary,
    fontSize: typography.fontSize.sm,
  },
  link: {
    color: colors.primary.main,
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
  },

  // Success State
  successContent: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: spacing.lg,
  },
  successContainer: {
    alignItems: 'center',
  },
  successIconContainer: {
    marginBottom: spacing.xl,
  },
  successIconGradient: {
    width: 100,
    height: 100,
    borderRadius: borderRadius.xl,
    alignItems: 'center',
    justifyContent: 'center',
    ...shadows.glow,
  },
  successEmoji: {
    fontSize: 48,
  },
  successTitle: {
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.md,
    textAlign: 'center',
    letterSpacing: -0.5,
  },
  successMessage: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
    textAlign: 'center',
    lineHeight: typography.lineHeight.relaxed * typography.fontSize.md,
    marginBottom: spacing['2xl'],
    paddingHorizontal: spacing.md,
  },
  successActions: {
    width: '100%',
    gap: spacing.md,
  },
});
