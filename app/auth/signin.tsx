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

type FieldName = 'email' | 'password';

export default function SignInScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [touched, setTouched] = useState<Record<FieldName, boolean>>({
    email: false,
    password: false,
  });
  const [errors, setErrors] = useState<Record<FieldName, string>>({
    email: '',
    password: '',
  });

  const { login } = useAuth();
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  const passwordInputRef = useRef<InputRef>(null);

  // Validation functions
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
    return '';
  }, []);

  const validateField = useCallback(
    (name: FieldName, value: string): string => {
      switch (name) {
        case 'email':
          return validateEmail(value);
        case 'password':
          return validatePassword(value);
        default:
          return '';
      }
    },
    [validateEmail, validatePassword]
  );

  const handleBlur = useCallback(
    (name: FieldName, value: string) => {
      setTouched((prev) => ({ ...prev, [name]: true }));
      const error = validateField(name, value);
      setErrors((prev) => ({ ...prev, [name]: error }));
    },
    [validateField]
  );

  const handleChange = useCallback(
    (name: FieldName, value: string, setter: (v: string) => void) => {
      setter(value);
      // Only validate on change if the field has been touched
      if (touched[name]) {
        const error = validateField(name, value);
        setErrors((prev) => ({ ...prev, [name]: error }));
      }
    },
    [touched, validateField]
  );

  const validateAllFields = useCallback((): boolean => {
    const emailError = validateField('email', email);
    const passwordError = validateField('password', password);

    setErrors({
      email: emailError,
      password: passwordError,
    });

    setTouched({
      email: true,
      password: true,
    });

    return !emailError && !passwordError;
  }, [email, password, validateField]);

  const handleSignIn = async () => {
    // Validate all fields on submit
    if (!validateAllFields()) {
      return;
    }

    setIsLoading(true);
    try {
      await login(email, password);
      router.replace('/(tabs)');
    } catch (error) {
      // Server-side errors still use Alert (not validation errors)
      Alert.alert('Sign In Failed', getErrorMessage(error, 'Invalid email or password'));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container} testID="signin-screen">
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
            <Text style={styles.title}>Welcome Back</Text>
            <Text style={styles.subtitle}>Sign in to your account</Text>
          </View>

          {/* Form */}
          <View style={styles.form}>
            {/* Email Input */}
            <Input
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
              testID="signin-email-input"
            />

            {/* Password Row with Forgot Password Link */}
            <View style={styles.passwordSection}>
              <View style={styles.passwordHeader}>
                <Text style={[styles.label, touched.password && errors.password && styles.labelError]}>
                  Password
                </Text>
                <Link href="/auth/forgot-password" asChild>
                  <TouchableOpacity
                    disabled={isLoading}
                    testID="signin-forgot-password-link"
                    accessibilityRole="link"
                    accessibilityLabel="Forgot password"
                    accessibilityHint="Navigate to reset your password"
                  >
                    <Text style={styles.forgotPasswordLink}>Forgot?</Text>
                  </TouchableOpacity>
                </Link>
              </View>
              <Input
                ref={passwordInputRef}
                label=""
                value={password}
                onChangeText={(value) => handleChange('password', value, setPassword)}
                onBlur={() => handleBlur('password', password)}
                placeholder="Enter your password"
                error={touched.password ? errors.password : undefined}
                secureTextEntry
                autoComplete="password"
                returnKeyType="done"
                onSubmitEditing={handleSignIn}
                disabled={isLoading}
                testID="signin-password-input"
                containerStyle={styles.passwordInputContainer}
              />
            </View>

            {/* Sign In Button */}
            <TouchableOpacity
              style={[styles.button, isLoading && styles.buttonDisabled]}
              onPress={handleSignIn}
              disabled={isLoading}
              activeOpacity={0.8}
              testID="signin-submit-button"
              accessibilityRole="button"
              accessibilityLabel={isLoading ? 'Signing in' : 'Sign in'}
              accessibilityHint="Double tap to sign in to your account"
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
                  <Text style={styles.buttonText}>Sign In</Text>
                )}
              </LinearGradient>
            </TouchableOpacity>

            {/* Sign Up Link */}
            <View style={styles.footer}>
              <Text style={styles.footerText}>Don't have an account? </Text>
              <Link href="/auth/signup" asChild>
                <TouchableOpacity
                  disabled={isLoading}
                  testID="signin-signup-link"
                  accessibilityRole="link"
                  accessibilityLabel="Sign up"
                  accessibilityHint="Navigate to create a new account"
                >
                  <Text style={styles.link}>Sign Up</Text>
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
    fontWeight: typography.fontWeight.medium,
  },

  // Form
  form: {
    gap: spacing.sm,
  },

  // Password section (custom layout with forgot link)
  passwordSection: {
    marginBottom: spacing.sm,
  },
  passwordHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  label: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    letterSpacing: 0.3,
  },
  labelError: {
    color: colors.status.error,
  },
  forgotPasswordLink: {
    fontSize: typography.fontSize.sm,
    color: colors.primary.main,
    fontWeight: typography.fontWeight.semibold,
  },
  passwordInputContainer: {
    marginBottom: 0,
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
