import { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, Switch, Platform } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { OnboardingStepLayout } from '@/lib/components/onboarding';
import { useOnboarding } from '@/lib/context/OnboardingContext';
import {
  NOTIFICATION_TYPE_OPTIONS,
  HEALTHKIT_SCOPE_OPTIONS,
  TOTAL_ONBOARDING_STEPS,
} from '@/lib/onboarding/config';
import { OnboardingStep3Data, NotificationType, HealthKitScope } from '@/lib/onboarding/types';
import { colors, typography, spacing, borderRadius } from '@/lib/theme/colors';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { notificationService, healthKitService } from '@/lib/services';

export default function OnboardingPermissions() {
  const router = useRouter();
  const { saveStep, isLoading, getDraftForStep, updateDraft } = useOnboarding();

  // Form state
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [notificationTypes, setNotificationTypes] = useState<NotificationType[]>([]);
  const [healthKitEnabled, setHealthKitEnabled] = useState(false);
  const [healthKitScopes, setHealthKitScopes] = useState<HealthKitScope[]>([]);

  // Load draft data on mount
  useEffect(() => {
    const draft = getDraftForStep<OnboardingStep3Data>(3);
    if (draft) {
      setNotificationsEnabled(draft.notificationsEnabled ?? false);
      setNotificationTypes(draft.notificationTypes || []);
      setHealthKitEnabled(draft.healthKitEnabled ?? false);
      setHealthKitScopes(draft.healthKitScopes || []);
    }
  }, [getDraftForStep]);

  // Save draft when form changes
  useEffect(() => {
    updateDraft(3, {
      notificationsEnabled,
      notificationTypes,
      healthKitEnabled,
      healthKitScopes,
      healthConnectEnabled: false,
      healthConnectScopes: [],
    });
  }, [notificationsEnabled, notificationTypes, healthKitEnabled, healthKitScopes, updateDraft]);

  const toggleNotificationType = (type: NotificationType) => {
    setNotificationTypes((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    );
  };

  const toggleHealthKitScope = (scope: HealthKitScope) => {
    setHealthKitScopes((prev) =>
      prev.includes(scope) ? prev.filter((s) => s !== scope) : [...prev, scope]
    );
  };

  const handleNotificationToggle = async (value: boolean) => {
    if (value) {
      try {
        const result = await notificationService.requestPermissions();
        if (result.granted) {
          setNotificationsEnabled(true);
          setNotificationTypes(['meal_reminders', 'insights', 'goals', 'weekly_summary']);
        } else {
          setNotificationsEnabled(false);
          if (!result.canAskAgain) {
            Alert.alert(
              'Notifications Disabled',
              'To enable notifications, please go to your device settings and allow notifications for Nutri.',
              [{ text: 'OK' }]
            );
          }
        }
      } catch (error) {
        console.error('Error requesting notification permissions:', error);
        setNotificationsEnabled(false);
      }
    } else {
      setNotificationsEnabled(false);
      setNotificationTypes([]);
    }
  };

  const handleHealthKitToggle = async (value: boolean) => {
    if (value) {
      try {
        const isAvailable = await healthKitService.isAvailable();
        if (!isAvailable) {
          Alert.alert(
            'HealthKit Not Available',
            'Apple Health is not available on this device.',
            [{ text: 'OK' }]
          );
          return;
        }

        const result = await healthKitService.requestPermissions([
          'heartRate',
          'restingHeartRate',
          'hrv',
          'steps',
          'activeEnergy',
          'sleep',
          'weight',
        ]);

        if (result.success) {
          setHealthKitEnabled(true);
          setHealthKitScopes(['heartRate', 'steps', 'sleep', 'activeEnergy', 'weight']);
        } else {
          setHealthKitEnabled(false);
          Alert.alert(
            'HealthKit Permission',
            result.error || 'Unable to get HealthKit permissions. You can enable this later in Settings.',
            [{ text: 'OK' }]
          );
        }
      } catch (error) {
        console.error('Error requesting HealthKit permissions:', error);
        setHealthKitEnabled(false);
      }
    } else {
      setHealthKitEnabled(false);
      setHealthKitScopes([]);
    }
  };

  const handleNext = async () => {
    const data: OnboardingStep3Data = {
      notificationsEnabled,
      notificationTypes,
      healthKitEnabled,
      healthKitScopes,
      healthConnectEnabled: false,
      healthConnectScopes: [],
    };

    try {
      await saveStep(3, data);
      router.push('/onboarding/health-background');
    } catch (error) {
      Alert.alert('Error', getErrorMessage(error, 'Failed to save permissions'));
    }
  };

  const handleBack = () => {
    router.back();
  };

  return (
    <OnboardingStepLayout
      title="Permissions"
      subtitle="Enable features to get the most out of Nutri"
      currentStep={3}
      totalSteps={TOTAL_ONBOARDING_STEPS}
      onBack={handleBack}
      onNext={handleNext}
      isLoading={isLoading}
      showBack={true}
    >
      {/* Push Notifications */}
      <View style={styles.section}>
        <View style={styles.permissionHeader}>
          <View style={styles.permissionIcon}>
            <Ionicons name="notifications-outline" size={24} color={colors.primary.main} />
          </View>
          <View style={styles.permissionInfo}>
            <Text style={styles.permissionTitle}>Push Notifications</Text>
            <Text style={styles.permissionDescription}>
              Get reminders to log meals and receive health insights
            </Text>
          </View>
          <Switch
            value={notificationsEnabled}
            onValueChange={handleNotificationToggle}
            trackColor={{ false: colors.surface.card, true: colors.primary.main }}
            thumbColor={colors.text.primary}
          />
        </View>

        {notificationsEnabled && (
          <View style={styles.permissionOptions}>
            {NOTIFICATION_TYPE_OPTIONS.map((option) => (
              <TouchableOpacity
                key={option.value}
                style={styles.optionRow}
                onPress={() => toggleNotificationType(option.value)}
              >
                <Ionicons
                  name={notificationTypes.includes(option.value) ? 'checkbox' : 'square-outline'}
                  size={24}
                  color={
                    notificationTypes.includes(option.value)
                      ? colors.primary.main
                      : colors.text.secondary
                  }
                />
                <View style={styles.optionTextContainer}>
                  <Text style={styles.optionLabel}>{option.label}</Text>
                  <Text style={styles.optionDescription}>{option.description}</Text>
                </View>
              </TouchableOpacity>
            ))}
          </View>
        )}
      </View>

      {/* HealthKit (iOS only) */}
      {Platform.OS === 'ios' && (
        <View style={styles.section}>
          <View style={styles.permissionHeader}>
            <View style={styles.permissionIcon}>
              <Ionicons name="heart-outline" size={24} color={colors.semantic.error} />
            </View>
            <View style={styles.permissionInfo}>
              <Text style={styles.permissionTitle}>Apple Health</Text>
              <Text style={styles.permissionDescription}>
                Sync health metrics for personalized insights
              </Text>
            </View>
            <Switch
              value={healthKitEnabled}
              onValueChange={handleHealthKitToggle}
              trackColor={{ false: colors.surface.card, true: colors.primary.main }}
              thumbColor={colors.text.primary}
            />
          </View>

          {healthKitEnabled && (
            <View style={styles.permissionOptions}>
              {HEALTHKIT_SCOPE_OPTIONS.map((option) => (
                <TouchableOpacity
                  key={option.value}
                  style={styles.optionRow}
                  onPress={() => toggleHealthKitScope(option.value)}
                >
                  <Ionicons
                    name={healthKitScopes.includes(option.value) ? 'checkbox' : 'square-outline'}
                    size={24}
                    color={
                      healthKitScopes.includes(option.value)
                        ? colors.primary.main
                        : colors.text.secondary
                    }
                  />
                  <View style={styles.optionTextContainer}>
                    <Text style={styles.optionLabel}>{option.label}</Text>
                    <Text style={styles.optionDescription}>{option.description}</Text>
                  </View>
                </TouchableOpacity>
              ))}
            </View>
          )}
        </View>
      )}

      {/* Privacy note */}
      <View style={styles.privacyNote}>
        <Ionicons name="shield-checkmark-outline" size={20} color={colors.text.tertiary} />
        <Text style={styles.privacyText}>
          Your data is encrypted and never shared with third parties. You can change these
          permissions anytime in Settings.
        </Text>
      </View>
    </OnboardingStepLayout>
  );
}

const styles = StyleSheet.create({
  section: {
    marginBottom: spacing.xl,
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
  },
  permissionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  permissionIcon: {
    width: 48,
    height: 48,
    borderRadius: borderRadius.md,
    backgroundColor: colors.background.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  permissionInfo: {
    flex: 1,
  },
  permissionTitle: {
    ...typography.bodyBold,
    color: colors.text.primary,
  },
  permissionDescription: {
    ...typography.bodySmall,
    color: colors.text.secondary,
    marginTop: 2,
  },
  permissionOptions: {
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.background.primary,
  },
  optionRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: spacing.sm,
  },
  optionTextContainer: {
    flex: 1,
    marginLeft: spacing.sm,
  },
  optionLabel: {
    ...typography.body,
    color: colors.text.primary,
  },
  optionDescription: {
    ...typography.bodySmall,
    color: colors.text.secondary,
    marginTop: 2,
  },
  privacyNote: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    padding: spacing.md,
    backgroundColor: colors.surface.elevated,
    borderRadius: borderRadius.md,
  },
  privacyText: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
    flex: 1,
    marginLeft: spacing.sm,
  },
});
