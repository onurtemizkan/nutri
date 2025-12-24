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

export default function SignUpScreen() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { register } = useAuth();
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  const emailInputRef = useRef<TextInputType>(null);
  const passwordInputRef = useRef<TextInputType>(null);
  const confirmPasswordInputRef = useRef<TextInputType>(null);

  const handleSignUp = async () => {
    if (!name || !email || !password || !confirmPassword) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      Alert.alert('Error', 'Please enter a valid email address');
      return;
    }

    if (password !== confirmPassword) {
      Alert.alert('Error', 'Passwords do not match');
      return;
    }

    if (password.length < 6) {
      Alert.alert('Error', 'Password must be at least 6 characters');
      return;
    }

    if (!/[A-Za-z]/.test(password) || !/[0-9]/.test(password)) {
      Alert.alert(
        'Weak Password',
        'Password should contain both letters and numbers for better security.'
      );
      return;
    }

    setIsLoading(true);
    try {
      await register(email, password, name);
      router.replace('/(tabs)');
    } catch (error) {
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
            isTablet && styles.tabletContent
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
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Name</Text>
              <View style={styles.inputWrapper}>
                <TextInput
                  style={styles.input}
                  placeholder="Your name"
                  placeholderTextColor={colors.text.disabled}
                  value={name}
                  onChangeText={setName}
                  editable={!isLoading}
                  autoComplete="name"
                  returnKeyType="next"
                  onSubmitEditing={() => emailInputRef.current?.focus()}
                  blurOnSubmit={false}
                  testID="signup-name-input"
                />
              </View>
            </View>

            {/* Email Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Email</Text>
              <View style={styles.inputWrapper}>
                <TextInput
                  ref={emailInputRef}
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
                  testID="signup-email-input"
                />
              </View>
            </View>

            {/* Password Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Password</Text>
              <View style={styles.inputWrapper}>
                <TextInput
                  ref={passwordInputRef}
                  style={styles.input}
                  placeholder="At least 6 characters"
                  placeholderTextColor={colors.text.disabled}
                  value={password}
                  onChangeText={setPassword}
                  secureTextEntry
                  editable={!isLoading}
                  autoComplete="password-new"
                  returnKeyType="next"
                  onSubmitEditing={() => confirmPasswordInputRef.current?.focus()}
                  blurOnSubmit={false}
                  testID="signup-password-input"
                />
              </View>
              <Text style={styles.hint}>Use letters and numbers for stronger security</Text>
            </View>

            {/* Confirm Password Input */}
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Confirm Password</Text>
              <View style={styles.inputWrapper}>
                <TextInput
                  ref={confirmPasswordInputRef}
                  style={styles.input}
                  placeholder="Re-enter password"
                  placeholderTextColor={colors.text.disabled}
                  value={confirmPassword}
                  onChangeText={setConfirmPassword}
                  secureTextEntry
                  editable={!isLoading}
                  autoComplete="password-new"
                  returnKeyType="done"
                  onSubmitEditing={handleSignUp}
                  testID="signup-confirm-password-input"
                />
              </View>
            </View>

            {/* Sign Up Button */}
            <TouchableOpacity
              style={[styles.button, isLoading && styles.buttonDisabled]}
              onPress={handleSignUp}
              disabled={isLoading}
              activeOpacity={0.8}
              testID="signup-submit-button"
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
                <TouchableOpacity disabled={isLoading} testID="signup-signin-link">
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
  hint: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    marginTop: -spacing.xs,
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
