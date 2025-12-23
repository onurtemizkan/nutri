/**
 * Notification Settings Screen
 * Manage push notification preferences and schedule
 */

import { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Switch,
  Platform,
  Linking,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import DateTimePicker from '@react-native-community/datetimepicker';
import { colors, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import { showAlert } from '@/lib/utils/alert';
import { useNotifications } from '@/lib/context/NotificationContext';
import { notificationsApi } from '@/lib/api/notifications';
import { NotificationCategory, NotificationPreferences } from '@/lib/types';

// Notification category config
const CATEGORIES: {
  id: NotificationCategory;
  title: string;
  description: string;
  icon: keyof typeof Ionicons.glyphMap;
}[] = [
  {
    id: 'MEAL_REMINDER',
    title: 'Meal Reminders',
    description: 'Get reminded to log your meals',
    icon: 'restaurant-outline',
  },
  {
    id: 'GOAL_PROGRESS',
    title: 'Goal Progress',
    description: 'Daily and weekly progress updates',
    icon: 'trophy-outline',
  },
  {
    id: 'HEALTH_INSIGHT',
    title: 'Health Insights',
    description: 'ML-powered nutrition-health correlations',
    icon: 'analytics-outline',
  },
  {
    id: 'SUPPLEMENT_REMINDER',
    title: 'Supplement Reminders',
    description: 'Remember to take your supplements',
    icon: 'medical-outline',
  },
  {
    id: 'STREAK_ALERT',
    title: 'Streak Alerts',
    description: "Don't lose your logging streak",
    icon: 'flame-outline',
  },
  {
    id: 'WEEKLY_SUMMARY',
    title: 'Weekly Summary',
    description: 'Your week in review every Sunday',
    icon: 'bar-chart-outline',
  },
  {
    id: 'MARKETING',
    title: 'Tips & Updates',
    description: 'Product tips and feature updates',
    icon: 'bulb-outline',
  },
];

// Default meal reminder times
const DEFAULT_MEAL_TIMES = {
  breakfast: '08:00',
  lunch: '12:00',
  dinner: '18:00',
  snack: '15:00',
};

// Parse time string to Date
function parseTimeToDate(time: string): Date {
  const [hours, minutes] = time.split(':').map(Number);
  const date = new Date();
  date.setHours(hours, minutes, 0, 0);
  return date;
}

// Format Date to time string
function formatDateToTime(date: Date): string {
  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  return `${hours}:${minutes}`;
}

// Format time for display
function formatTimeDisplay(time: string): string {
  const [hours, minutes] = time.split(':').map(Number);
  const period = hours >= 12 ? 'PM' : 'AM';
  const displayHours = hours % 12 || 12;
  return `${displayHours}:${minutes.toString().padStart(2, '0')} ${period}`;
}

// Category toggle component
function CategoryToggle({
  category,
  enabled,
  onToggle,
}: {
  category: (typeof CATEGORIES)[number];
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
}) {
  return (
    <View style={styles.categoryItem}>
      <View style={styles.categoryLeft}>
        <View style={styles.categoryIcon}>
          <Ionicons name={category.icon} size={20} color={colors.primary.main} />
        </View>
        <View style={styles.categoryInfo}>
          <Text style={styles.categoryTitle}>{category.title}</Text>
          <Text style={styles.categoryDescription}>{category.description}</Text>
        </View>
      </View>
      <Switch
        value={enabled}
        onValueChange={onToggle}
        trackColor={{ false: colors.background.elevated, true: colors.primary.main }}
        thumbColor={colors.text.primary}
        accessibilityLabel={`${enabled ? 'Disable' : 'Enable'} ${category.title}`}
      />
    </View>
  );
}

// Time picker component
function MealTimePicker({
  mealType,
  label,
  time,
  onTimeChange,
}: {
  mealType: string;
  label: string;
  time: string;
  onTimeChange: (time: string) => void;
}) {
  const [showPicker, setShowPicker] = useState(false);

  const handleChange = (_event: unknown, selectedDate?: Date) => {
    if (Platform.OS === 'android') {
      setShowPicker(false);
    }
    if (selectedDate) {
      onTimeChange(formatDateToTime(selectedDate));
    }
  };

  return (
    <View style={styles.timePickerRow}>
      <Text style={styles.timePickerLabel}>{label}</Text>
      <TouchableOpacity
        style={styles.timePickerButton}
        onPress={() => setShowPicker(true)}
        accessibilityLabel={`Set ${label} reminder time`}
      >
        <Text style={styles.timePickerValue}>{formatTimeDisplay(time)}</Text>
        <Ionicons name="time-outline" size={18} color={colors.text.tertiary} />
      </TouchableOpacity>
      {showPicker && (
        <DateTimePicker
          value={parseTimeToDate(time)}
          mode="time"
          is24Hour={false}
          display={Platform.OS === 'ios' ? 'spinner' : 'default'}
          onChange={handleChange}
          minuteInterval={5}
        />
      )}
    </View>
  );
}

export default function NotificationSettingsScreen() {
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();
  const { permissionStatus, requestPermission, isRegistered } = useNotifications();

  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [preferences, setPreferences] = useState<NotificationPreferences | null>(null);
  const [hasChanges, setHasChanges] = useState(false);

  // Load preferences from API
  const loadPreferences = useCallback(async () => {
    setIsLoading(true);
    try {
      const prefs = await notificationsApi.getPreferences();
      setPreferences(prefs);
    } catch (error) {
      console.error('Error loading preferences:', error);
      showAlert('Error', 'Failed to load notification preferences');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPreferences();
  }, [loadPreferences]);

  // Save preferences to API
  const savePreferences = async () => {
    if (!preferences || !hasChanges) return;

    setIsSaving(true);
    try {
      await notificationsApi.updatePreferences({
        enabled: preferences.enabled,
        enabledCategories: preferences.enabledCategories,
        quietHoursEnabled: preferences.quietHoursEnabled,
        quietHoursStart: preferences.quietHoursStart,
        quietHoursEnd: preferences.quietHoursEnd,
        mealReminderTimes: preferences.mealReminderTimes,
      });
      setHasChanges(false);
      showAlert('Saved', 'Notification preferences updated');
    } catch (error) {
      console.error('Error saving preferences:', error);
      showAlert('Error', 'Failed to save notification preferences');
    } finally {
      setIsSaving(false);
    }
  };

  // Handle master toggle
  const handleMasterToggle = (enabled: boolean) => {
    if (!preferences) return;
    setPreferences({ ...preferences, enabled });
    setHasChanges(true);
  };

  // Handle category toggle
  const handleCategoryToggle = (category: NotificationCategory, enabled: boolean) => {
    if (!preferences) return;

    const enabledCategories = enabled
      ? [...preferences.enabledCategories, category]
      : preferences.enabledCategories.filter((c) => c !== category);

    setPreferences({ ...preferences, enabledCategories });
    setHasChanges(true);
  };

  // Handle quiet hours toggle
  const handleQuietHoursToggle = (enabled: boolean) => {
    if (!preferences) return;
    setPreferences({
      ...preferences,
      quietHoursEnabled: enabled,
      quietHoursStart: enabled ? preferences.quietHoursStart || '22:00' : preferences.quietHoursStart,
      quietHoursEnd: enabled ? preferences.quietHoursEnd || '08:00' : preferences.quietHoursEnd,
    });
    setHasChanges(true);
  };

  // Handle quiet hours time change
  const handleQuietHoursTimeChange = (field: 'start' | 'end', time: string) => {
    if (!preferences) return;
    setPreferences({
      ...preferences,
      [field === 'start' ? 'quietHoursStart' : 'quietHoursEnd']: time,
    });
    setHasChanges(true);
  };

  // Handle meal reminder time change
  const handleMealTimeChange = (mealType: string, time: string) => {
    if (!preferences) return;
    setPreferences({
      ...preferences,
      mealReminderTimes: {
        ...preferences.mealReminderTimes,
        [mealType]: time,
      },
    });
    setHasChanges(true);
  };

  // Handle enable notifications
  const handleEnableNotifications = async () => {
    const granted = await requestPermission();
    if (!granted) {
      showAlert(
        'Permission Required',
        'Please enable notifications in Settings to receive reminders.',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Open Settings',
            onPress: () => Linking.openSettings(),
          },
        ]
      );
    }
  };

  // Determine if notifications are disabled at system level
  const isSystemDisabled = permissionStatus === 'denied';
  const isMealReminderEnabled = preferences?.enabledCategories.includes('MEAL_REMINDER');

  return (
    <SafeAreaView style={styles.container} testID="notification-settings-screen">
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => router.back()}
          accessibilityLabel="Go back"
          testID="notification-settings-back-button"
        >
          <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Notifications</Text>
        <TouchableOpacity
          style={[styles.saveButton, (!hasChanges || isSaving) && styles.saveButtonDisabled]}
          onPress={savePreferences}
          disabled={!hasChanges || isSaving}
          accessibilityLabel="Save preferences"
          testID="notification-settings-save-button"
        >
          {isSaving ? (
            <ActivityIndicator size="small" color={colors.primary.main} />
          ) : (
            <Text style={[styles.saveButtonText, !hasChanges && styles.saveButtonTextDisabled]}>
              Save
            </Text>
          )}
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={[
          styles.scrollContent,
          { paddingHorizontal: responsiveSpacing.horizontal },
          isTablet && styles.scrollContentTablet,
        ]}
      >
        <View style={styles.content}>
          {isLoading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={colors.primary.main} />
              <Text style={styles.loadingText}>Loading preferences...</Text>
            </View>
          ) : (
            <>
              {/* Permission Status Banner */}
              {isSystemDisabled && (
                <View style={styles.permissionBanner}>
                  <Ionicons name="notifications-off" size={24} color={colors.status.error} />
                  <View style={styles.permissionBannerContent}>
                    <Text style={styles.permissionBannerTitle}>Notifications Disabled</Text>
                    <Text style={styles.permissionBannerText}>
                      Enable notifications in Settings to receive reminders.
                    </Text>
                  </View>
                  <TouchableOpacity
                    style={styles.permissionBannerButton}
                    onPress={() => Linking.openSettings()}
                    accessibilityLabel="Open Settings"
                  >
                    <Text style={styles.permissionBannerButtonText}>Settings</Text>
                  </TouchableOpacity>
                </View>
              )}

              {permissionStatus === 'undetermined' && (
                <TouchableOpacity
                  style={styles.enableButton}
                  onPress={handleEnableNotifications}
                  accessibilityLabel="Enable notifications"
                >
                  <Ionicons name="notifications" size={24} color={colors.text.primary} />
                  <Text style={styles.enableButtonText}>Enable Notifications</Text>
                </TouchableOpacity>
              )}

              {/* Master Toggle */}
              {preferences && (
                <View style={styles.section}>
                  <View style={styles.masterToggle}>
                    <View style={styles.masterToggleLeft}>
                      <Ionicons name="notifications" size={24} color={colors.primary.main} />
                      <View style={styles.masterToggleInfo}>
                        <Text style={styles.masterToggleTitle}>All Notifications</Text>
                        <Text style={styles.masterToggleSubtitle}>
                          {preferences.enabled ? 'Notifications are on' : 'Notifications are off'}
                        </Text>
                      </View>
                    </View>
                    <Switch
                      value={preferences.enabled}
                      onValueChange={handleMasterToggle}
                      trackColor={{ false: colors.background.elevated, true: colors.primary.main }}
                      thumbColor={colors.text.primary}
                      disabled={isSystemDisabled}
                      accessibilityLabel={`${preferences.enabled ? 'Disable' : 'Enable'} all notifications`}
                    />
                  </View>
                </View>
              )}

              {/* Category Toggles */}
              {preferences && preferences.enabled && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Notification Types</Text>
                  {CATEGORIES.map((category) => (
                    <CategoryToggle
                      key={category.id}
                      category={category}
                      enabled={preferences.enabledCategories.includes(category.id)}
                      onToggle={(enabled) => handleCategoryToggle(category.id, enabled)}
                    />
                  ))}
                </View>
              )}

              {/* Meal Reminder Times */}
              {preferences && preferences.enabled && isMealReminderEnabled && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Meal Reminder Times</Text>
                  <Text style={styles.sectionDescription}>
                    Set when you'd like to be reminded to log each meal
                  </Text>
                  <View style={styles.timePickerContainer}>
                    <MealTimePicker
                      mealType="breakfast"
                      label="Breakfast"
                      time={preferences.mealReminderTimes?.breakfast || DEFAULT_MEAL_TIMES.breakfast}
                      onTimeChange={(time) => handleMealTimeChange('breakfast', time)}
                    />
                    <MealTimePicker
                      mealType="lunch"
                      label="Lunch"
                      time={preferences.mealReminderTimes?.lunch || DEFAULT_MEAL_TIMES.lunch}
                      onTimeChange={(time) => handleMealTimeChange('lunch', time)}
                    />
                    <MealTimePicker
                      mealType="dinner"
                      label="Dinner"
                      time={preferences.mealReminderTimes?.dinner || DEFAULT_MEAL_TIMES.dinner}
                      onTimeChange={(time) => handleMealTimeChange('dinner', time)}
                    />
                    <MealTimePicker
                      mealType="snack"
                      label="Snack"
                      time={preferences.mealReminderTimes?.snack || DEFAULT_MEAL_TIMES.snack}
                      onTimeChange={(time) => handleMealTimeChange('snack', time)}
                    />
                  </View>
                </View>
              )}

              {/* Quiet Hours */}
              {preferences && preferences.enabled && (
                <View style={styles.section}>
                  <View style={styles.quietHoursHeader}>
                    <View style={styles.quietHoursLeft}>
                      <Ionicons name="moon" size={20} color={colors.primary.main} />
                      <View style={styles.quietHoursInfo}>
                        <Text style={styles.quietHoursTitle}>Quiet Hours</Text>
                        <Text style={styles.quietHoursSubtitle}>
                          No notifications during these hours
                        </Text>
                      </View>
                    </View>
                    <Switch
                      value={preferences.quietHoursEnabled}
                      onValueChange={handleQuietHoursToggle}
                      trackColor={{ false: colors.background.elevated, true: colors.primary.main }}
                      thumbColor={colors.text.primary}
                      accessibilityLabel={`${preferences.quietHoursEnabled ? 'Disable' : 'Enable'} quiet hours`}
                    />
                  </View>

                  {preferences.quietHoursEnabled && (
                    <View style={styles.quietHoursTimeContainer}>
                      <View style={styles.quietHoursTimeRow}>
                        <Text style={styles.quietHoursTimeLabel}>Start</Text>
                        <TouchableOpacity
                          style={styles.quietHoursTimeButton}
                          onPress={() => {
                            // For now, use a simple implementation
                            // A more complete implementation would show a time picker
                          }}
                        >
                          <Text style={styles.quietHoursTimeValue}>
                            {formatTimeDisplay(preferences.quietHoursStart || '22:00')}
                          </Text>
                        </TouchableOpacity>
                      </View>
                      <View style={styles.quietHoursTimeRow}>
                        <Text style={styles.quietHoursTimeLabel}>End</Text>
                        <TouchableOpacity style={styles.quietHoursTimeButton}>
                          <Text style={styles.quietHoursTimeValue}>
                            {formatTimeDisplay(preferences.quietHoursEnd || '08:00')}
                          </Text>
                        </TouchableOpacity>
                      </View>
                    </View>
                  )}
                </View>
              )}

              {/* Device Status */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Device Status</Text>
                <View style={styles.deviceStatusRow}>
                  <Text style={styles.deviceStatusLabel}>Permission Status</Text>
                  <View style={styles.deviceStatusValue}>
                    <View
                      style={[
                        styles.statusDot,
                        permissionStatus === 'granted'
                          ? styles.statusDotSuccess
                          : permissionStatus === 'denied'
                            ? styles.statusDotError
                            : styles.statusDotWarning,
                      ]}
                    />
                    <Text style={styles.deviceStatusText}>
                      {permissionStatus === 'granted'
                        ? 'Allowed'
                        : permissionStatus === 'denied'
                          ? 'Denied'
                          : 'Not Requested'}
                    </Text>
                  </View>
                </View>
                <View style={styles.deviceStatusRow}>
                  <Text style={styles.deviceStatusLabel}>Device Registered</Text>
                  <View style={styles.deviceStatusValue}>
                    <View
                      style={[
                        styles.statusDot,
                        isRegistered ? styles.statusDotSuccess : styles.statusDotWarning,
                      ]}
                    />
                    <Text style={styles.deviceStatusText}>{isRegistered ? 'Yes' : 'No'}</Text>
                  </View>
                </View>
              </View>
            </>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  backButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  saveButton: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  saveButtonDisabled: {
    opacity: 0.5,
  },
  saveButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },
  saveButtonTextDisabled: {
    color: colors.text.tertiary,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: spacing.lg,
  },
  scrollContentTablet: {
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
  },
  content: {
    paddingVertical: spacing.lg,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: spacing['4xl'],
  },
  loadingText: {
    marginTop: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },

  // Permission Banner
  permissionBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.special.errorLight,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginBottom: spacing.lg,
    gap: spacing.md,
  },
  permissionBannerContent: {
    flex: 1,
  },
  permissionBannerTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.status.error,
    marginBottom: 2,
  },
  permissionBannerText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },
  permissionBannerButton: {
    backgroundColor: colors.status.error,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.sm,
  },
  permissionBannerButtonText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Enable Button
  enableButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary.main,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    marginBottom: spacing.lg,
    gap: spacing.sm,
    ...shadows.sm,
  },
  enableButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Section
  section: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  sectionTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    marginBottom: spacing.md,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  sectionDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.md,
  },

  // Master Toggle
  masterToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  masterToggleLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    gap: spacing.md,
  },
  masterToggleInfo: {
    flex: 1,
  },
  masterToggleTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: 2,
  },
  masterToggleSubtitle: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },

  // Category Toggle
  categoryItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  categoryLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  categoryIcon: {
    width: 36,
    height: 36,
    borderRadius: borderRadius.sm,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  categoryInfo: {
    flex: 1,
  },
  categoryTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
    marginBottom: 2,
  },
  categoryDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },

  // Time Picker
  timePickerContainer: {
    marginTop: spacing.sm,
  },
  timePickerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  timePickerLabel: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
  },
  timePickerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: colors.background.elevated,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.sm,
  },
  timePickerValue: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    fontWeight: typography.fontWeight.medium,
  },

  // Quiet Hours
  quietHoursHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  quietHoursLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    gap: spacing.md,
  },
  quietHoursInfo: {
    flex: 1,
  },
  quietHoursTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: 2,
  },
  quietHoursSubtitle: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  quietHoursTimeContainer: {
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  quietHoursTimeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
  },
  quietHoursTimeLabel: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
  },
  quietHoursTimeButton: {
    backgroundColor: colors.background.elevated,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.sm,
  },
  quietHoursTimeValue: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    fontWeight: typography.fontWeight.medium,
  },

  // Device Status
  deviceStatusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
  },
  deviceStatusLabel: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
  },
  deviceStatusValue: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  deviceStatusText: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  statusDotSuccess: {
    backgroundColor: colors.status.success,
  },
  statusDotError: {
    backgroundColor: colors.status.error,
  },
  statusDotWarning: {
    backgroundColor: colors.status.warning,
  },
});
