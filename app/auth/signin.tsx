import { useState, useRef } from 'react';
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
import { Link, useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '@/lib/context/AuthContext';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';

export default function SignInScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  const passwordInputRef = useRef<TextInputType>(null);

  const handleSignIn = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      Alert.alert('Error', 'Please enter a valid email address');
      return;
    }

    setIsLoading(true);
    try {
      await login(email, password);
      router.replace('/(tabs)');
    } catch (error) {
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
                  autoComplete="email"
                  returnKeyType="next"
                  onSubmitEditing={() => passwordInputRef.current?.focus()}
                  blurOnSubmit={false}
                  testID="signin-email-input"
                  accessibilityLabel="Email address"
                  accessibilityHint="Enter your email to sign in"
                />
              </View>
            </View>

            {/* Password Input */}
            <View style={styles.inputContainer}>
              <View style={styles.labelRow}>
                <Text style={styles.label}>Password</Text>
                <Link href="/auth/forgot-password" asChild>
                  <TouchableOpacity
                    disabled={isLoading}
                    testID="signin-forgot-password-link"
                    accessibilityRole="link"
                    accessibilityLabel="Forgot password"
                    accessibilityHint="Navigate to password recovery"
                  >
                    <Text style={styles.forgotPasswordLink}>Forgot?</Text>
                  </TouchableOpacity>
                </Link>
              </View>
              <View style={styles.inputWrapper}>
                <TextInput
                  ref={passwordInputRef}
                  style={styles.input}
                  placeholder="Enter your password"
                  placeholderTextColor={colors.text.disabled}
                  value={password}
                  onChangeText={setPassword}
                  secureTextEntry
                  editable={!isLoading}
                  autoComplete="password"
                  returnKeyType="done"
                  onSubmitEditing={handleSignIn}
                  testID="signin-password-input"
                  accessibilityLabel="Password"
                  accessibilityHint="Enter your password to sign in"
                />
              </View>
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
    gap: spacing.lg,
  },

  // Input
  inputContainer: {
    gap: spacing.sm,
  },
  labelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  label: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    letterSpacing: 0.3,
  },
  forgotPasswordLink: {
    fontSize: typography.fontSize.sm,
    color: colors.primary.main,
    fontWeight: typography.fontWeight.semibold,
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
