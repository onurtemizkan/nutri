/**
 * NotificationPrimingModal - Pre-permission priming screen for notifications
 *
 * Shows a friendly explanation of notification benefits before requesting
 * system permission. Displayed after the user logs their first meal.
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Modal,
  Animated,
  Easing,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { LinearGradient } from 'expo-linear-gradient';

import { colors, spacing, borderRadius, typography, gradients } from '../theme/colors';
import { useNotifications } from '../context/NotificationContext';

const PRIMING_SHOWN_KEY = '@nutri:notification_priming_shown';
const PRIMING_SHOWN_DATE_KEY = '@nutri:notification_priming_date';

interface NotificationPrimingModalProps {
  visible: boolean;
  onClose: () => void;
  onEnableNotifications?: () => void;
}

interface BenefitItemProps {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  description: string;
  delay: number;
}

function BenefitItem({ icon, title, description, delay }: BenefitItemProps) {
  const fadeAnim = React.useRef(new Animated.Value(0)).current;
  const slideAnim = React.useRef(new Animated.Value(20)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 400,
        delay,
        useNativeDriver: true,
        easing: Easing.out(Easing.ease),
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 400,
        delay,
        useNativeDriver: true,
        easing: Easing.out(Easing.ease),
      }),
    ]).start();
  }, [fadeAnim, slideAnim, delay]);

  return (
    <Animated.View
      style={[
        styles.benefitItem,
        {
          opacity: fadeAnim,
          transform: [{ translateY: slideAnim }],
        },
      ]}
    >
      <View style={styles.benefitIconContainer}>
        <Ionicons name={icon} size={24} color={colors.primary.main} />
      </View>
      <View style={styles.benefitContent}>
        <Text style={styles.benefitTitle}>{title}</Text>
        <Text style={styles.benefitDescription}>{description}</Text>
      </View>
    </Animated.View>
  );
}

export function NotificationPrimingModal({
  visible,
  onClose,
  onEnableNotifications,
}: NotificationPrimingModalProps) {
  const { requestPermission, permissionStatus } = useNotifications();
  const [isEnabling, setIsEnabling] = useState(false);

  // Animation values
  const scaleAnim = React.useRef(new Animated.Value(0.9)).current;
  const fadeAnim = React.useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (visible) {
      Animated.parallel([
        Animated.spring(scaleAnim, {
          toValue: 1,
          friction: 8,
          tension: 65,
          useNativeDriver: true,
        }),
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        }),
      ]).start();
    } else {
      scaleAnim.setValue(0.9);
      fadeAnim.setValue(0);
    }
  }, [visible, scaleAnim, fadeAnim]);

  const handleEnableNotifications = useCallback(async () => {
    setIsEnabling(true);
    try {
      const granted = await requestPermission();

      // Mark priming as shown
      await AsyncStorage.setItem(PRIMING_SHOWN_KEY, 'true');
      await AsyncStorage.setItem(PRIMING_SHOWN_DATE_KEY, new Date().toISOString());

      if (granted) {
        onEnableNotifications?.();
      }
      onClose();
    } catch (error) {
      console.error('Error enabling notifications:', error);
    } finally {
      setIsEnabling(false);
    }
  }, [requestPermission, onEnableNotifications, onClose]);

  const handleMaybeLater = useCallback(async () => {
    // Mark priming as shown even if declined
    await AsyncStorage.setItem(PRIMING_SHOWN_KEY, 'true');
    await AsyncStorage.setItem(PRIMING_SHOWN_DATE_KEY, new Date().toISOString());
    onClose();
  }, [onClose]);

  // Don't show if permission already granted
  if (permissionStatus === 'granted') {
    return null;
  }

  const benefits = [
    {
      icon: 'restaurant-outline' as const,
      title: 'Meal Reminders',
      description: 'Never forget to log your meals with gentle reminders at your preferred times.',
    },
    {
      icon: 'trophy-outline' as const,
      title: 'Goal Progress',
      description: 'Celebrate your wins and stay motivated with progress updates.',
    },
    {
      icon: 'heart-outline' as const,
      title: 'Health Insights',
      description: 'Get personalized insights about how your nutrition affects your health.',
    },
    {
      icon: 'flame-outline' as const,
      title: 'Streak Alerts',
      description: "Don't break your streak! We'll remind you to log before it's too late.",
    },
  ];

  return (
    <Modal
      visible={visible}
      transparent
      animationType="none"
      onRequestClose={handleMaybeLater}
    >
      <Animated.View style={[styles.overlay, { opacity: fadeAnim }]}>
        <Animated.View
          style={[
            styles.container,
            {
              transform: [{ scale: scaleAnim }],
            },
          ]}
        >
          {/* Header with icon */}
          <View style={styles.header}>
            <LinearGradient
              colors={gradients.primary}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={styles.iconGradient}
            >
              <Ionicons name="notifications" size={40} color={colors.text.primary} />
            </LinearGradient>
            <Text style={styles.title}>Stay on Track</Text>
            <Text style={styles.subtitle}>
              Enable notifications to help you build healthy habits and reach your goals.
            </Text>
          </View>

          {/* Benefits list */}
          <View style={styles.benefitsContainer}>
            {benefits.map((benefit, index) => (
              <BenefitItem
                key={benefit.title}
                icon={benefit.icon}
                title={benefit.title}
                description={benefit.description}
                delay={200 + index * 100}
              />
            ))}
          </View>

          {/* Privacy note */}
          <View style={styles.privacyNote}>
            <Ionicons
              name="shield-checkmark-outline"
              size={16}
              color={colors.text.tertiary}
            />
            <Text style={styles.privacyText}>
              You can customize notification types anytime in settings.
            </Text>
          </View>

          {/* Action buttons */}
          <View style={styles.actions}>
            <TouchableOpacity
              style={styles.enableButton}
              onPress={handleEnableNotifications}
              disabled={isEnabling}
              accessibilityLabel="Enable notifications"
              accessibilityRole="button"
            >
              <LinearGradient
                colors={gradients.primary}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.enableButtonGradient}
              >
                <Ionicons
                  name={isEnabling ? 'hourglass-outline' : 'notifications'}
                  size={20}
                  color={colors.text.primary}
                />
                <Text style={styles.enableButtonText}>
                  {isEnabling ? 'Enabling...' : 'Enable Notifications'}
                </Text>
              </LinearGradient>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.laterButton}
              onPress={handleMaybeLater}
              disabled={isEnabling}
              accessibilityLabel="Maybe later"
              accessibilityRole="button"
            >
              <Text style={styles.laterButtonText}>Maybe Later</Text>
            </TouchableOpacity>
          </View>
        </Animated.View>
      </Animated.View>
    </Modal>
  );
}

/**
 * Hook to check if notification priming should be shown
 */
export function useNotificationPriming() {
  const [shouldShowPriming, setShouldShowPriming] = useState(false);
  const { permissionStatus } = useNotifications();

  const checkPrimingStatus = useCallback(async () => {
    // Don't show if permission already granted or denied
    if (permissionStatus === 'granted' || permissionStatus === 'denied') {
      setShouldShowPriming(false);
      return false;
    }

    // Check if already shown
    const shown = await AsyncStorage.getItem(PRIMING_SHOWN_KEY);
    if (shown === 'true') {
      // Check if it's been more than 7 days - could show again
      const shownDate = await AsyncStorage.getItem(PRIMING_SHOWN_DATE_KEY);
      if (shownDate) {
        const daysSinceShown = (Date.now() - new Date(shownDate).getTime()) / (1000 * 60 * 60 * 24);
        // Don't re-show within 7 days
        if (daysSinceShown < 7) {
          setShouldShowPriming(false);
          return false;
        }
      } else {
        setShouldShowPriming(false);
        return false;
      }
    }

    return true;
  }, [permissionStatus]);

  /**
   * Trigger the priming modal after first meal logged
   */
  const triggerPriming = useCallback(async () => {
    const canShow = await checkPrimingStatus();
    if (canShow) {
      setShouldShowPriming(true);
    }
  }, [checkPrimingStatus]);

  /**
   * Dismiss the priming modal
   */
  const dismissPriming = useCallback(() => {
    setShouldShowPriming(false);
  }, []);

  /**
   * Reset priming state (for testing)
   */
  const resetPriming = useCallback(async () => {
    await AsyncStorage.removeItem(PRIMING_SHOWN_KEY);
    await AsyncStorage.removeItem(PRIMING_SHOWN_DATE_KEY);
  }, []);

  return {
    shouldShowPriming,
    triggerPriming,
    dismissPriming,
    resetPriming,
  };
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: colors.overlay.medium,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.lg,
  },
  container: {
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.xl,
    padding: spacing.lg,
    maxWidth: 400,
    width: '100%',
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.3,
        shadowRadius: 20,
      },
      android: {
        elevation: 10,
      },
    }),
  },

  // Header
  header: {
    alignItems: 'center',
    marginBottom: spacing.lg,
  },
  iconGradient: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  title: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
    textAlign: 'center',
    lineHeight: typography.fontSize.md * typography.lineHeight.normal,
  },

  // Benefits
  benefitsContainer: {
    marginBottom: spacing.lg,
  },
  benefitItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: spacing.md,
  },
  benefitIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  benefitContent: {
    flex: 1,
  },
  benefitTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: 2,
  },
  benefitDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    lineHeight: typography.fontSize.sm * typography.lineHeight.normal,
  },

  // Privacy note
  privacyNote: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.lg,
    gap: spacing.sm,
  },
  privacyText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    flex: 1,
  },

  // Actions
  actions: {
    gap: spacing.sm,
  },
  enableButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  enableButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
    gap: spacing.sm,
  },
  enableButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  laterButton: {
    alignItems: 'center',
    paddingVertical: spacing.md,
  },
  laterButtonText: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
});
