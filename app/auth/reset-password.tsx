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
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { authApi } from '@/lib/api/auth';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import { Input, type InputRef } from '@/lib/components/ui/Input';

const MIN_PASSWORD_LENGTH = 6;

type PasswordFieldName = 'newPassword' | 'confirmPassword';

export default function ResetPasswordScreen() {
  const [token, setToken] = useState('');
  const [tokenTouched, setTokenTouched] = useState(false);
  const [tokenError, setTokenError] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [tokenValid, setTokenValid] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [touched, setTouched] = useState<Record<PasswordFieldName, boolean>>({
    newPassword: false,
    confirmPassword: false,
  });
  const [errors, setErrors] = useState<Record<PasswordFieldName, string>>({
    newPassword: '',
    confirmPassword: '',
  });

  const router = useRouter();
  const params = useLocalSearchParams();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  const confirmPasswordInputRef = useRef<InputRef>(null);

  // Token validation
  const validateToken = useCallback((value: string): string => {
    if (!value.trim()) {
      return 'Reset token is required';
    }
    return '';
  }, []);

  // Password validation
  const validateNewPassword = useCallback((value: string): string => {
    if (!value) {
      return 'Password is required';
    }
    if (value.length < MIN_PASSWORD_LENGTH) {
      return `Password must be at least ${MIN_PASSWORD_LENGTH} characters`;
    }
    if (!/[A-Za-z]/.test(value) || !/[0-9]/.test(value)) {
      return 'Password must contain both letters and numbers';
    }
    return '';
  }, []);

  const validateConfirmPassword = useCallback(
    (value: string, currentPassword: string): string => {
      if (!value) {
        return 'Please confirm your password';
      }
      if (value !== currentPassword) {
        return 'Passwords do not match';
      }
      return '';
    },
    []
  );

  const validatePasswordField = useCallback(
    (name: PasswordFieldName, value: string): string => {
      switch (name) {
        case 'newPassword':
          return validateNewPassword(value);
        case 'confirmPassword':
          return validateConfirmPassword(value, newPassword);
        default:
          return '';
      }
    },
    [validateNewPassword, validateConfirmPassword, newPassword]
  );

  // Token handlers
  const handleTokenBlur = useCallback(() => {
    setTokenTouched(true);
    const error = validateToken(token);
    setTokenError(error);
  }, [token, validateToken]);

  const handleTokenChange = useCallback(
    (value: string) => {
      setToken(value);
      if (tokenTouched) {
        const error = validateToken(value);
        setTokenError(error);
      }
    },
    [tokenTouched, validateToken]
  );

  // Password handlers
  const handlePasswordBlur = useCallback(
    (name: PasswordFieldName, value: string) => {
      setTouched((prev) => ({ ...prev, [name]: true }));
      const error = validatePasswordField(name, value);
      setErrors((prev) => ({ ...prev, [name]: error }));

      // If password changed and confirmPassword is touched, re-validate confirmPassword
      if (name === 'newPassword' && touched.confirmPassword) {
        const confirmError = validateConfirmPassword(confirmPassword, value);
        setErrors((prev) => ({ ...prev, confirmPassword: confirmError }));
      }
    },
    [validatePasswordField, touched.confirmPassword, confirmPassword, validateConfirmPassword]
  );

  const handlePasswordChange = useCallback(
    (name: PasswordFieldName, value: string, setter: (v: string) => void) => {
      setter(value);
      if (touched[name]) {
        const error = validatePasswordField(name, value);
        setErrors((prev) => ({ ...prev, [name]: error }));
      }

      // If password changes and confirmPassword has been touched, re-validate confirmPassword
      if (name === 'newPassword' && touched.confirmPassword) {
        const confirmError = validateConfirmPassword(confirmPassword, value);
        setErrors((prev) => ({ ...prev, confirmPassword: confirmError }));
      }
    },
    [touched, validatePasswordField, confirmPassword, validateConfirmPassword]
  );

  const verifyToken = useCallback(async (tokenToVerify: string) => {
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
  }, [router]);

  useEffect(() => {
    // Check if token is in URL params
    if (params.token && typeof params.token === 'string') {
      setToken(params.token);
      verifyToken(params.token);
    }
  }, [params.token, verifyToken]);

  const handleVerifyToken = async () => {
    // Validate on submit
    setTokenTouched(true);
    const error = validateToken(token);
    setTokenError(error);

    if (error) {
      return;
    }

    verifyToken(token);
  };

  const validateAllPasswordFields = useCallback((): boolean => {
    const newPasswordError = validatePasswordField('newPassword', newPassword);
    const confirmPasswordError = validateConfirmPassword(confirmPassword, newPassword);

    setErrors({
      newPassword: newPasswordError,
      confirmPassword: confirmPasswordError,
    });

    setTouched({
      newPassword: true,
      confirmPassword: true,
    });

    return !newPasswordError && !confirmPasswordError;
  }, [newPassword, confirmPassword, validatePasswordField, validateConfirmPassword]);

  const handleResetPassword = async () => {
    // Validate all fields on submit
    if (!validateAllPasswordFields()) {
      return;
    }

    if (!token) {
      Alert.alert('Error', 'Reset token is missing');
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
        <View style={[
          styles.centerContainer,
          isTablet && styles.tabletContent
        ]}>
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
              isTablet && styles.tabletContent
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
              {/* Token Input - custom multiline styling */}
              <View style={styles.inputContainer}>
                <Text style={[styles.label, tokenTouched && tokenError && styles.labelError]}>
                  Reset Token
                </Text>
                <View style={[
                  styles.inputWrapper,
                  tokenTouched && tokenError && styles.inputWrapperError,
                ]}>
                  <TextInput
                    style={[styles.input, styles.tokenInput]}
                    placeholder="Enter your reset token"
                    placeholderTextColor={colors.text.disabled}
                    value={token}
                    onChangeText={handleTokenChange}
                    onBlur={handleTokenBlur}
                    autoCapitalize="none"
                    editable={!isLoading}
                    multiline
                    numberOfLines={3}
                    testID="reset-password-token-input"
                    accessibilityLabel="Reset token"
                    accessibilityHint="Enter the reset token from your email"
                  />
                </View>
                <View style={styles.messageContainer}>
                  {tokenTouched && tokenError ? (
                    <Text style={styles.errorText} accessibilityRole="alert">
                      {tokenError}
                    </Text>
                  ) : null}
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
                  colors={isLoading ? [colors.text.disabled, colors.text.disabled] : gradients.primary}
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
                  accessibilityLabel="Request one"
                  accessibilityHint="Navigate to request a password reset token"
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
            isTablet && styles.tabletContent
          ]}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
          keyboardDismissMode="interactive"
        >
          <View style={styles.header}>
            <Text style={styles.title}>Reset Password</Text>
            {userEmail && (
              <Text style={styles.subtitle}>
                Create a new password for {userEmail}
              </Text>
            )}
          </View>

          <View style={styles.form}>
            {/* New Password Input */}
            <Input
              label="New Password"
              value={newPassword}
              onChangeText={(value) => handlePasswordChange('newPassword', value, setNewPassword)}
              onBlur={() => handlePasswordBlur('newPassword', newPassword)}
              placeholder="At least 6 characters"
              error={touched.newPassword ? errors.newPassword : undefined}
              helperText={!touched.newPassword || !errors.newPassword ? 'Use letters and numbers for security' : undefined}
              secureTextEntry
              autoComplete="password-new"
              returnKeyType="next"
              onSubmitEditing={() => confirmPasswordInputRef.current?.focus()}
              disabled={isLoading}
              testID="reset-password-new-password-input"
            />

            {/* Confirm Password Input */}
            <Input
              ref={confirmPasswordInputRef}
              label="Confirm New Password"
              value={confirmPassword}
              onChangeText={(value) => handlePasswordChange('confirmPassword', value, setConfirmPassword)}
              onBlur={() => handlePasswordBlur('confirmPassword', confirmPassword)}
              placeholder="Re-enter your password"
              error={touched.confirmPassword ? errors.confirmPassword : undefined}
              secureTextEntry
              autoComplete="password-new"
              returnKeyType="done"
              onSubmitEditing={handleResetPassword}
              disabled={isLoading}
              testID="reset-password-confirm-password-input"
            />

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
                colors={isLoading ? [colors.text.disabled, colors.text.disabled] : gradients.primary}
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
                accessibilityHint="Navigate to sign in to your account"
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
    gap: spacing.xs,
  },

  // Token input styles (custom for multiline)
  inputContainer: {
    marginBottom: spacing.md,
  },
  label: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
    letterSpacing: 0.3,
  },
  labelError: {
    color: colors.status.error,
  },
  inputWrapper: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    overflow: 'hidden',
  },
  inputWrapperError: {
    borderColor: colors.status.error,
    backgroundColor: colors.special.errorLight,
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
  messageContainer: {
    minHeight: spacing.lg,
    marginTop: spacing.xs,
  },
  errorText: {
    fontSize: typography.fontSize.sm,
    color: colors.status.error,
    fontWeight: typography.fontWeight.medium,
  },

  // Button
  button: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    marginTop: spacing.md,
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
    marginTop: spacing.lg,
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
