import { useState, useEffect, useCallback, useRef } from 'react';
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
  type TextInput as TextInputType,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { authApi } from '@/lib/api/auth';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';

export default function ResetPasswordScreen() {
  const [token, setToken] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [tokenValid, setTokenValid] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const router = useRouter();
  const params = useLocalSearchParams();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  const confirmPasswordInputRef = useRef<TextInputType>(null);

  const verifyToken = useCallback(
    async (tokenToVerify: string) => {
      setIsVerifying(true);
      try {
        const response = await authApi.verifyResetToken(tokenToVerify);
        setTokenValid(true);
        setUserEmail(response.email);
      } catch {
        Alert.alert(
          'Invalid Token',
          'This password reset link is invalid or has expired. Please request a new one.',
          [
            {
              text: 'OK',
              onPress: () => router.replace('/auth/forgot-password'),
            },
          ]
        );
        setTokenValid(false);
      } finally {
        setIsVerifying(false);
      }
    },
    [router]
  );

  useEffect(() => {
    // Check if token is in URL params
    if (params.token && typeof params.token === 'string') {
      setToken(params.token);
      verifyToken(params.token);
    }
  }, [params.token, verifyToken]);

  const handleVerifyToken = async () => {
    if (!token) {
      Alert.alert('Error', 'Please enter your reset token');
      return;
    }

    verifyToken(token);
  };

  const handleResetPassword = async () => {
    if (!token) {
      Alert.alert('Error', 'Reset token is missing');
      return;
    }

    if (!newPassword || !confirmPassword) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    if (newPassword !== confirmPassword) {
      Alert.alert('Error', 'Passwords do not match');
      return;
    }

    if (newPassword.length < 6) {
      Alert.alert('Error', 'Password must be at least 6 characters');
      return;
    }

    setIsLoading(true);
    try {
      await authApi.resetPassword(token, newPassword);

      Alert.alert(
        'Success',
        'Your password has been reset successfully. You can now sign in with your new password.',
        [
          {
            text: 'Sign In',
            onPress: () => router.replace('/auth/signin'),
          },
        ]
      );
    } catch (error) {
      Alert.alert(
        'Reset Failed',
        getErrorMessage(error, 'Failed to reset password. The token may have expired.')
      );
    } finally {
      setIsLoading(false);
    }
  };

  if (isVerifying) {
    return (
      <SafeAreaView style={styles.container} testID="reset-password-verifying-screen">
        <View style={[styles.centerContainer, isTablet && styles.tabletContent]}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Verifying reset token...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!tokenValid && !params.token) {
    return (
      <SafeAreaView style={styles.container} testID="reset-password-token-screen">
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
            <View style={styles.header}>
              <Text style={styles.title}>Enter Reset Token</Text>
              <Text style={styles.subtitle}>
                Please enter the reset token you received via email.
              </Text>
            </View>

            <View style={styles.form}>
              <View style={styles.inputContainer}>
                <Text style={styles.label}>Reset Token</Text>
                <View style={styles.inputWrapper}>
                  <TextInput
                    style={[styles.input, styles.tokenInput]}
                    placeholder="Enter your reset token"
                    placeholderTextColor={colors.text.disabled}
                    value={token}
                    onChangeText={setToken}
                    autoCapitalize="none"
                    editable={!isLoading}
                    multiline
                    numberOfLines={3}
                    testID="reset-password-token-input"
                    accessibilityLabel="Reset token"
                    accessibilityHint="Enter the password reset token from your email"
                  />
                </View>
              </View>

              <TouchableOpacity
                style={[styles.button, isLoading && styles.buttonDisabled]}
                onPress={handleVerifyToken}
                disabled={isLoading}
                activeOpacity={0.8}
                testID="reset-password-verify-token-button"
                accessibilityRole="button"
                accessibilityLabel={isLoading ? 'Verifying token' : 'Verify token'}
                accessibilityHint="Double tap to verify your reset token"
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
                    <Text style={styles.buttonText}>Verify Token</Text>
                  )}
                </LinearGradient>
              </TouchableOpacity>

              <View style={styles.footer}>
                <Text style={styles.footerText}>Don't have a token? </Text>
                <TouchableOpacity
                  onPress={() => router.push('/auth/forgot-password')}
                  disabled={isLoading}
                  testID="reset-password-request-token-link"
                  accessibilityRole="link"
                  accessibilityLabel="Request a reset token"
                  accessibilityHint="Navigate to request a new password reset token"
                >
                  <Text style={styles.link}>Request One</Text>
                </TouchableOpacity>
              </View>
            </View>
          </ScrollView>
        </KeyboardAvoidingView>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} testID="reset-password-screen">
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
          <View style={styles.header}>
            <Text style={styles.title}>Reset Password</Text>
            {userEmail && (
              <Text style={styles.subtitle}>Create a new password for {userEmail}</Text>
            )}
          </View>

          <View style={styles.form}>
            <View style={styles.inputContainer}>
              <Text style={styles.label}>New Password</Text>
              <View style={styles.inputWrapper}>
                <TextInput
                  style={styles.input}
                  placeholder="At least 6 characters"
                  placeholderTextColor={colors.text.disabled}
                  value={newPassword}
                  onChangeText={setNewPassword}
                  secureTextEntry
                  editable={!isLoading}
                  autoComplete="password-new"
                  returnKeyType="next"
                  onSubmitEditing={() => confirmPasswordInputRef.current?.focus()}
                  blurOnSubmit={false}
                  testID="reset-password-new-password-input"
                  accessibilityLabel="New password"
                  accessibilityHint="Create a new password with at least 6 characters"
                />
              </View>
            </View>

            <View style={styles.inputContainer}>
              <Text style={styles.label}>Confirm New Password</Text>
              <View style={styles.inputWrapper}>
                <TextInput
                  ref={confirmPasswordInputRef}
                  style={styles.input}
                  placeholder="Re-enter your password"
                  placeholderTextColor={colors.text.disabled}
                  value={confirmPassword}
                  onChangeText={setConfirmPassword}
                  secureTextEntry
                  editable={!isLoading}
                  autoComplete="password-new"
                  returnKeyType="done"
                  onSubmitEditing={handleResetPassword}
                  testID="reset-password-confirm-password-input"
                  accessibilityLabel="Confirm new password"
                  accessibilityHint="Re-enter your new password to confirm"
                />
              </View>
            </View>

            <TouchableOpacity
              style={[styles.button, isLoading && styles.buttonDisabled]}
              onPress={handleResetPassword}
              disabled={isLoading}
              activeOpacity={0.8}
              testID="reset-password-submit-button"
              accessibilityRole="button"
              accessibilityLabel={isLoading ? 'Resetting password' : 'Reset password'}
              accessibilityHint="Double tap to reset your password"
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
                  <Text style={styles.buttonText}>Reset Password</Text>
                )}
              </LinearGradient>
            </TouchableOpacity>

            <View style={styles.footer}>
              <Text style={styles.footerText}>Remember your password? </Text>
              <TouchableOpacity
                onPress={() => router.push('/auth/signin')}
                disabled={isLoading}
                testID="reset-password-signin-link"
                accessibilityRole="link"
                accessibilityLabel="Sign in"
                accessibilityHint="Navigate back to sign in screen"
              >
                <Text style={styles.link}>Sign In</Text>
              </TouchableOpacity>
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
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
    fontWeight: typography.fontWeight.medium,
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
  tokenInput: {
    minHeight: 120,
    textAlignVertical: 'top',
    height: 'auto',
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
});
