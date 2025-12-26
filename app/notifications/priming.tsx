/**
 * Notification Pre-Permission Priming Screen
 *
 * This screen explains the benefits of notifications BEFORE asking for permission.
 * Pre-permission priming improves opt-in rates by setting expectations and showing value.
 *
 * Best practices followed:
 * - Explain WHAT notifications will be sent
 * - Show HOW they will help the user
 * - Make it easy to skip (no pressure)
 * - Use visuals and friendly language
 */

import { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  Dimensions,
  Platform,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { colors, gradients, typography, spacing, borderRadius, shadows } from '@/lib/theme/colors';
import { useNotifications } from '@/lib/context/NotificationContext';

const { width } = Dimensions.get('window');

// Notification benefits to display
const NOTIFICATION_BENEFITS = [
  {
    icon: 'restaurant-outline' as const,
    title: 'Meal Reminders',
    description: 'Never forget to log your meals with timely reminders',
    color: colors.primary.main,
  },
  {
    icon: 'analytics-outline' as const,
    title: 'Health Insights',
    description: 'Get personalized nutrition-health correlations',
    color: '#9C27B0',
  },
  {
    icon: 'flame-outline' as const,
    title: 'Streak Motivation',
    description: "Stay on track and don't break your logging streak",
    color: '#FF5722',
  },
  {
    icon: 'bar-chart-outline' as const,
    title: 'Weekly Progress',
    description: 'Receive summaries of your nutrition journey',
    color: '#2196F3',
  },
];

interface BenefitCardProps {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  description: string;
  color: string;
  delay: number;
}

function BenefitCard({ icon, title, description, color, delay }: BenefitCardProps) {
  const animatedValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(animatedValue, {
      toValue: 1,
      duration: 400,
      delay,
      useNativeDriver: true,
    }).start();
  }, [animatedValue, delay]);

  return (
    <Animated.View
      style={[
        styles.benefitCard,
        {
          opacity: animatedValue,
          transform: [
            {
              translateY: animatedValue.interpolate({
                inputRange: [0, 1],
                outputRange: [20, 0],
              }),
            },
          ],
        },
      ]}
    >
      <View style={[styles.benefitIconContainer, { backgroundColor: `${color}20` }]}>
        <Ionicons name={icon} size={24} color={color} />
      </View>
      <View style={styles.benefitContent}>
        <Text style={styles.benefitTitle}>{title}</Text>
        <Text style={styles.benefitDescription}>{description}</Text>
      </View>
    </Animated.View>
  );
}

export default function NotificationPrimingScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ returnTo?: string }>();
  const { requestPermission, permissionStatus } = useNotifications();
  const [isRequesting, setIsRequesting] = useState(false);
  const [isCheckingPermission, setIsCheckingPermission] = useState(true);

  // Animation values
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.9)).current;

  // Check permission status before showing content
  useEffect(() => {
    if (permissionStatus !== 'loading') {
      setIsCheckingPermission(false);
    }
  }, [permissionStatus]);

  useEffect(() => {
    if (!isCheckingPermission) {
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 400,
          useNativeDriver: true,
        }),
        Animated.spring(scaleAnim, {
          toValue: 1,
          friction: 8,
          tension: 40,
          useNativeDriver: true,
        }),
      ]).start();
    }
  }, [fadeAnim, scaleAnim, isCheckingPermission]);

  const handleEnableNotifications = async () => {
    setIsRequesting(true);
    try {
      const granted = await requestPermission();
      if (granted) {
        // Navigate to the return destination or back
        if (params.returnTo) {
          router.replace(params.returnTo as `/${string}`);
        } else {
          router.back();
        }
      } else {
        // Permission denied - still allow them to continue
        if (params.returnTo) {
          router.replace(params.returnTo as `/${string}`);
        } else {
          router.back();
        }
      }
    } catch (error) {
      console.error('Error requesting notification permission:', error);
    } finally {
      setIsRequesting(false);
    }
  };

  const handleSkip = () => {
    if (params.returnTo) {
      router.replace(params.returnTo as `/${string}`);
    } else {
      router.back();
    }
  };

  // If permission is already granted, redirect
  useEffect(() => {
    if (!isCheckingPermission && permissionStatus === 'granted') {
      if (params.returnTo) {
        router.replace(params.returnTo as `/${string}`);
      } else {
        router.back();
      }
    }
  }, [permissionStatus, params.returnTo, router, isCheckingPermission]);

  // Show loading state while checking permission to prevent flash
  if (isCheckingPermission) {
    return (
      <SafeAreaView style={styles.container} testID="notification-priming-screen">
        <LinearGradient colors={gradients.dark} style={styles.gradient}>
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={colors.primary.main} />
          </View>
        </LinearGradient>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} testID="notification-priming-screen">
      <LinearGradient colors={gradients.dark} style={styles.gradient}>
        {/* Skip Button */}
        <TouchableOpacity
          style={styles.skipButton}
          onPress={handleSkip}
          accessibilityLabel="Skip notifications setup"
          testID="notification-priming-skip-button"
        >
          <Text style={styles.skipButtonText}>Maybe Later</Text>
        </TouchableOpacity>

        {/* Main Content */}
        <Animated.View
          style={[
            styles.content,
            {
              opacity: fadeAnim,
              transform: [{ scale: scaleAnim }],
            },
          ]}
        >
          {/* Header with illustration */}
          <View style={styles.header}>
            <View style={styles.iconCircle}>
              <LinearGradient colors={gradients.primary} style={styles.iconGradient}>
                <Ionicons name="notifications" size={48} color={colors.text.primary} />
              </LinearGradient>
            </View>

            <Text style={styles.title}>Stay on Track</Text>
            <Text style={styles.subtitle}>
              Get helpful reminders and insights to reach your nutrition goals faster
            </Text>
          </View>

          {/* Benefits List */}
          <View style={styles.benefitsList}>
            {NOTIFICATION_BENEFITS.map((benefit, index) => (
              <BenefitCard
                key={benefit.title}
                icon={benefit.icon}
                title={benefit.title}
                description={benefit.description}
                color={benefit.color}
                delay={index * 100}
              />
            ))}
          </View>

          {/* Control note */}
          <View style={styles.controlNote}>
            <Ionicons name="settings-outline" size={16} color={colors.text.tertiary} />
            <Text style={styles.controlNoteText}>
              You can customize or disable notifications anytime in Settings
            </Text>
          </View>
        </Animated.View>

        {/* Action Buttons */}
        <View style={styles.footer}>
          <TouchableOpacity
            style={styles.enableButton}
            onPress={handleEnableNotifications}
            disabled={isRequesting}
            accessibilityLabel="Enable notifications"
            testID="notification-priming-enable-button"
          >
            <LinearGradient colors={gradients.primary} style={styles.enableButtonGradient}>
              <Ionicons
                name="notifications"
                size={20}
                color={colors.text.primary}
                style={styles.enableButtonIcon}
              />
              <Text style={styles.enableButtonText}>
                {isRequesting ? 'Enabling...' : 'Enable Notifications'}
              </Text>
            </LinearGradient>
          </TouchableOpacity>

          <Text style={styles.privacyNote}>
            {Platform.OS === 'ios'
              ? 'We respect your privacy. Notifications are processed securely.'
              : 'We respect your privacy. You can manage notification channels in Android settings.'}
          </Text>
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  skipButton: {
    alignSelf: 'flex-end',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  skipButtonText: {
    ...typography.body,
    color: colors.text.tertiary,
  },
  content: {
    flex: 1,
    paddingHorizontal: spacing.lg,
  },
  header: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  iconCircle: {
    marginBottom: spacing.lg,
  },
  iconGradient: {
    width: 96,
    height: 96,
    borderRadius: 48,
    alignItems: 'center',
    justifyContent: 'center',
    ...shadows.lg,
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
    maxWidth: 300,
  },
  benefitsList: {
    marginBottom: spacing.lg,
  },
  benefitCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.sm,
  },
  benefitIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  benefitContent: {
    flex: 1,
  },
  benefitTitle: {
    ...typography.bodyBold,
    color: colors.text.primary,
    marginBottom: 2,
  },
  benefitDescription: {
    ...typography.bodySmall,
    color: colors.text.secondary,
  },
  controlNote: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.sm,
  },
  controlNoteText: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
    marginLeft: spacing.xs,
  },
  footer: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xl,
  },
  enableButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    marginBottom: spacing.md,
  },
  enableButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
  },
  enableButtonIcon: {
    marginRight: spacing.sm,
  },
  enableButtonText: {
    ...typography.button,
    color: colors.text.primary,
  },
  privacyNote: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
    textAlign: 'center',
  },
});
