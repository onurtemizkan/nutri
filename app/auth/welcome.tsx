import { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Platform, Alert, ScrollView } from 'react-native';
import { Link, useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as AppleAuthentication from 'expo-apple-authentication';
import { useAuth } from '@/lib/context/AuthContext';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';

export default function WelcomeScreen() {
  const [isLoading, setIsLoading] = useState(false);
  const { appleSignIn } = useAuth();
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  const handleAppleSignIn = async () => {
    setIsLoading(true);
    try {
      const credential = await AppleAuthentication.signInAsync({
        requestedScopes: [
          AppleAuthentication.AppleAuthenticationScope.FULL_NAME,
          AppleAuthentication.AppleAuthenticationScope.EMAIL,
        ],
      });

      await appleSignIn({
        identityToken: credential.identityToken || '',
        authorizationCode: credential.authorizationCode || '',
        user: {
          email: credential.email || undefined,
          name: credential.fullName
            ? {
                firstName: credential.fullName.givenName || undefined,
                lastName: credential.fullName.familyName || undefined,
              }
            : undefined,
        },
      });

      router.replace('/(tabs)');
    } catch (error: unknown) {
      if (
        typeof error === 'object' &&
        error !== null &&
        'code' in error &&
        error.code === 'ERR_REQUEST_CANCELED'
      ) {
        return;
      }
      Alert.alert('Apple Sign In Failed', getErrorMessage(error, 'Could not sign in with Apple'));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container} testID="welcome-screen">
      <LinearGradient colors={gradients.dark} style={styles.gradient}>
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={[
            styles.content,
            { paddingHorizontal: responsiveSpacing.horizontal },
            isTablet && styles.tabletContent
          ]}
          showsVerticalScrollIndicator={false}
          bounces={false}
          testID="welcome-content"
        >
          {/* Header Section */}
          <View style={styles.header}>
            {/* Logo/Icon with gradient background */}
            <View style={styles.logoContainer}>
              <LinearGradient colors={gradients.primary} style={styles.logoGradient}>
                <Text style={styles.logoEmoji}>🥗</Text>
              </LinearGradient>
            </View>

            {/* Title */}
            <Text style={styles.title}>Nutri</Text>
            <Text style={styles.subtitle}>Track, Analyze, Optimize</Text>

            {/* Feature highlights */}
            <View style={styles.features}>
              <View style={styles.featureItem}>
                <Text style={styles.featureDot}>•</Text>
                <Text style={styles.featureText}>AI-powered insights</Text>
              </View>
              <View style={styles.featureItem}>
                <Text style={styles.featureDot}>•</Text>
                <Text style={styles.featureText}>Health metric correlation</Text>
              </View>
              <View style={styles.featureItem}>
                <Text style={styles.featureDot}>•</Text>
                <Text style={styles.featureText}>Personalized nutrition</Text>
              </View>
            </View>
          </View>

          {/* Action Buttons */}
          <View style={styles.buttonContainer}>
            {/* Apple Sign In Button */}
            {Platform.OS === 'ios' && (
              <>
                <View pointerEvents={isLoading ? 'none' : 'auto'} style={{ opacity: isLoading ? 0.6 : 1 }}>
                  <AppleAuthentication.AppleAuthenticationButton
                    buttonType={AppleAuthentication.AppleAuthenticationButtonType.SIGN_IN}
                    buttonStyle={AppleAuthentication.AppleAuthenticationButtonStyle.WHITE}
                    cornerRadius={borderRadius.md}
                    style={styles.appleButton}
                    onPress={handleAppleSignIn}
                  />
                </View>

                <View style={styles.divider}>
                  <View style={styles.dividerLine} />
                  <Text style={styles.dividerText}>or continue with email</Text>
                  <View style={styles.dividerLine} />
                </View>
              </>
            )}

            {/* Sign In Button */}
            <Link href="/auth/signin" asChild>
              <TouchableOpacity style={styles.primaryButton} disabled={isLoading} testID="welcome-signin-button">
                <LinearGradient
                  colors={gradients.primary}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.primaryButtonGradient}
                >
                  <Text style={styles.primaryButtonText}>Sign In</Text>
                </LinearGradient>
              </TouchableOpacity>
            </Link>

            {/* Create Account Button */}
            <Link href="/auth/signup" asChild>
              <TouchableOpacity style={styles.secondaryButton} disabled={isLoading} testID="welcome-signup-button">
                <Text style={styles.secondaryButtonText}>Create Account</Text>
              </TouchableOpacity>
            </Link>

            {/* Footer text */}
            <Text style={styles.footerText}>
              By continuing, you agree to our{' '}
              <Text
                style={styles.footerLink}
                onPress={() => router.push('/terms')}
                accessibilityRole="link"
                accessibilityLabel="View Terms and Conditions"
              >
                Terms & Privacy Policy
              </Text>
            </Text>
          </View>
        </ScrollView>
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
  scrollView: {
    flex: 1,
  },
  content: {
    flexGrow: 1,
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.xl,
    paddingBottom: spacing.lg,
    minHeight: '100%',
  },
  tabletContent: {
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
  },

  // Header
  header: {
    alignItems: 'center',
    marginTop: spacing.lg,
  },
  logoContainer: {
    marginBottom: spacing.lg,
  },
  logoGradient: {
    width: 100,
    height: 100,
    borderRadius: borderRadius.xl,
    alignItems: 'center',
    justifyContent: 'center',
    ...shadows.glow,
  },
  logoEmoji: {
    fontSize: 48,
  },
  title: {
    fontSize: typography.fontSize['5xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
    letterSpacing: -1,
  },
  subtitle: {
    fontSize: typography.fontSize.lg,
    color: colors.text.secondary,
    marginBottom: spacing.xl,
    fontWeight: typography.fontWeight.medium,
  },

  // Features
  features: {
    marginTop: spacing.md,
    gap: spacing.sm,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  featureDot: {
    fontSize: typography.fontSize.lg,
    color: colors.primary.main,
  },
  featureText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    fontWeight: typography.fontWeight.medium,
  },

  // Buttons
  buttonContainer: {
    gap: spacing.md,
  },
  appleButton: {
    width: '100%',
    height: 52,
  },

  // Primary Button (Gradient)
  primaryButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    ...shadows.md,
  },
  primaryButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
    height: 52,
  },
  primaryButtonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    letterSpacing: 0.5,
  },

  // Secondary Button (Outlined)
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
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    letterSpacing: 0.5,
  },

  // Divider
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: spacing.xs,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: colors.border.secondary,
  },
  dividerText: {
    marginHorizontal: spacing.md,
    color: colors.text.tertiary,
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
  },

  // Footer
  footerText: {
    textAlign: 'center',
    color: colors.text.disabled,
    fontSize: typography.fontSize.xs,
    marginTop: spacing.sm,
    lineHeight: typography.lineHeight.relaxed * typography.fontSize.xs,
  },
  footerLink: {
    color: colors.primary.main,
    textDecorationLine: 'underline',
  },
});
