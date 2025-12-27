import { useState, useRef, useCallback } from 'react';
import {
  View,
  Text,
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
import { useAuth } from '@/lib/context/AuthContext';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import { Input, type InputRef } from '@/lib/components/ui/Input';

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MIN_PASSWORD_LENGTH = 6;

type FieldName = 'name' | 'email' | 'password' | 'confirmPassword';

export default function SignUpScreen() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [touched, setTouched] = useState<Record<FieldName, boolean>>({
    name: false,
    email: false,
    password: false,
    confirmPassword: false,
  });
  const [errors, setErrors] = useState<Record<FieldName, string>>({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
  });

  const { register } = useAuth();
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  const emailInputRef = useRef<InputRef>(null);
  const passwordInputRef = useRef<InputRef>(null);
  const confirmPasswordInputRef = useRef<InputRef>(null);

  // Validation functions
  const validateName = useCallback((value: string): string => {
    if (!value.trim()) {
      return 'Name is required';
    }
    return '';
  }, []);

  const validateEmail = useCallback((value: string): string => {
    if (!value.trim()) {
      return 'Email is required';
    }
    if (!EMAIL_REGEX.test(value)) {
      return 'Please enter a valid email address';
    }
    return '';
  }, []);

  const validatePassword = useCallback((value: string): string => {
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

  const validateField = useCallback(
    (name: FieldName, value: string): string => {
      switch (name) {
        case 'name':
          return validateName(value);
        case 'email':
          return validateEmail(value);
        case 'password':
          return validatePassword(value);
        case 'confirmPassword':
          return validateConfirmPassword(value, password);
        default:
          return '';
      }
    },
    [validateName, validateEmail, validatePassword, validateConfirmPassword, password]
  );

  const handleBlur = useCallback(
    (name: FieldName, value: string) => {
      setTouched((prev) => ({ ...prev, [name]: true }));
      const error = validateField(name, value);
      setErrors((prev) => ({ ...prev, [name]: error }));

      // If password changed and confirmPassword is touched, re-validate confirmPassword
      if (name === 'password' && touched.confirmPassword) {
        const confirmError = validateConfirmPassword(confirmPassword, value);
        setErrors((prev) => ({ ...prev, confirmPassword: confirmError }));
      }
    },
    [validateField, touched.confirmPassword, confirmPassword, validateConfirmPassword]
  );

  const handleChange = useCallback(
    (name: FieldName, value: string, setter: (v: string) => void) => {
      setter(value);
      // Only validate on change if the field has been touched
      if (touched[name]) {
        const error = validateField(name, value);
        setErrors((prev) => ({ ...prev, [name]: error }));
      }

      // If password changes and confirmPassword has been touched, re-validate confirmPassword
      if (name === 'password' && touched.confirmPassword) {
        const confirmError = validateConfirmPassword(confirmPassword, value);
        setErrors((prev) => ({ ...prev, confirmPassword: confirmError }));
      }
    },
    [touched, validateField, confirmPassword, validateConfirmPassword]
  );

  const validateAllFields = useCallback((): boolean => {
    const nameError = validateField('name', name);
    const emailError = validateField('email', email);
    const passwordError = validateField('password', password);
    const confirmPasswordError = validateConfirmPassword(confirmPassword, password);

    setErrors({
      name: nameError,
      email: emailError,
      password: passwordError,
      confirmPassword: confirmPasswordError,
    });

    setTouched({
      name: true,
      email: true,
      password: true,
      confirmPassword: true,
    });

    return !nameError && !emailError && !passwordError && !confirmPasswordError;
  }, [name, email, password, confirmPassword, validateField, validateConfirmPassword]);

  const handleSignUp = async () => {
    // Validate all fields on submit
    if (!validateAllFields()) {
      return;
    }

    setIsLoading(true);
    try {
      await register(email, password, name);
      router.replace('/(tabs)');
    } catch (error) {
      // Server-side errors still use Alert
      Alert.alert('Sign Up Failed', getErrorMessage(error, 'Failed to create account'));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container} testID="signup-screen">
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
            <Text style={styles.title}>Create Account</Text>
            <Text style={styles.subtitle}>Start tracking your nutrition journey</Text>
          </View>

          {/* Form */}
          <View style={styles.form}>
            {/* Name Input */}
            <Input
              label="Name"
              value={name}
              onChangeText={(value) => handleChange('name', value, setName)}
              onBlur={() => handleBlur('name', name)}
              placeholder="Your name"
              error={touched.name ? errors.name : undefined}
              autoComplete="name"
              returnKeyType="next"
              onSubmitEditing={() => emailInputRef.current?.focus()}
              disabled={isLoading}
              testID="signup-name-input"
            />

            {/* Email Input */}
            <Input
              ref={emailInputRef}
              label="Email"
              value={email}
              onChangeText={(value) => handleChange('email', value, setEmail)}
              onBlur={() => handleBlur('email', email)}
              placeholder="your@email.com"
              error={touched.email ? errors.email : undefined}
              keyboardType="email-address"
              autoCapitalize="none"
              autoComplete="email"
              returnKeyType="next"
              onSubmitEditing={() => passwordInputRef.current?.focus()}
              disabled={isLoading}
              testID="signup-email-input"
            />

            {/* Password Input */}
            <Input
              ref={passwordInputRef}
              label="Password"
              value={password}
              onChangeText={(value) => handleChange('password', value, setPassword)}
              onBlur={() => handleBlur('password', password)}
              placeholder="At least 6 characters"
              error={touched.password ? errors.password : undefined}
              helperText={!touched.password || !errors.password ? 'Use letters and numbers for stronger security' : undefined}
              secureTextEntry
              autoComplete="password-new"
              returnKeyType="next"
              onSubmitEditing={() => confirmPasswordInputRef.current?.focus()}
              disabled={isLoading}
              testID="signup-password-input"
            />

            {/* Confirm Password Input */}
            <Input
              ref={confirmPasswordInputRef}
              label="Confirm Password"
              value={confirmPassword}
              onChangeText={(value) => handleChange('confirmPassword', value, setConfirmPassword)}
              onBlur={() => handleBlur('confirmPassword', confirmPassword)}
              placeholder="Re-enter password"
              error={touched.confirmPassword ? errors.confirmPassword : undefined}
              secureTextEntry
              autoComplete="password-new"
              returnKeyType="done"
              onSubmitEditing={handleSignUp}
              disabled={isLoading}
              testID="signup-confirm-password-input"
            />

            {/* Sign Up Button */}
            <TouchableOpacity
              style={[styles.button, isLoading && styles.buttonDisabled]}
              onPress={handleSignUp}
              disabled={isLoading}
              activeOpacity={0.8}
              testID="signup-submit-button"
              accessibilityRole="button"
              accessibilityLabel={isLoading ? 'Creating account' : 'Create account'}
              accessibilityHint="Double tap to create your account"
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
                  <Text style={styles.buttonText}>Create Account</Text>
                )}
              </LinearGradient>
            </TouchableOpacity>

            {/* Sign In Link */}
            <View style={styles.footer}>
              <Text style={styles.footerText}>Already have an account? </Text>
              <Link href="/auth/signin" asChild>
                <TouchableOpacity
                  disabled={isLoading}
                  testID="signup-signin-link"
                  accessibilityRole="link"
                  accessibilityLabel="Sign in"
                  accessibilityHint="Navigate to sign in to your existing account"
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
    marginBottom: spacing.xl,
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
    fontWeight: typography.fontWeight.medium,
  },

  // Form
  form: {
    gap: spacing.xs,
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
