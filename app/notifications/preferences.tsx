/**
 * Notification Preferences Screen
 *
 * Allows users to configure their notification preferences including:
 * - Enable/disable specific notification categories
 * - Set quiet hours
 * - Configure meal reminder times
 *
 * This is a more focused version that can be navigated to from the priming screen
 * or accessed directly from profile settings.
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
  Modal,
} from 'react-native';
import { isAxiosError } from 'axios';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import DateTimePicker from '@react-native-community/datetimepicker';
import { colors, typography, spacing, borderRadius, shadows } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import { showAlert } from '@/lib/utils/alert';
import { useNotifications } from '@/lib/context/NotificationContext';
import { notificationsApi } from '@/lib/api/notifications';
import { NotificationCategory, NotificationPreferences } from '@/lib/types';

// Notification category configuration with icons and descriptions
const CATEGORIES: {
  id: NotificationCategory;
  title: string;
  description: string;
  icon: keyof typeof Ionicons.glyphMap;
  color: string;
}[] = [
  {
    id: 'MEAL_REMINDER',
    title: 'Meal Reminders',
    description: 'Get reminded to log your meals',
    icon: 'restaurant-outline',
    color: colors.primary.main,
  },
  {
    id: 'GOAL_PROGRESS',
    title: 'Goal Progress',
    description: 'Daily and weekly progress updates',
    icon: 'trophy-outline',
    color: '#FF9800',
  },
  {
    id: 'HEALTH_INSIGHT',
    title: 'Health Insights',
    description: 'ML-powered nutrition-health correlations',
    icon: 'analytics-outline',
    color: '#9C27B0',
  },
  {
    id: 'SUPPLEMENT_REMINDER',
    title: 'Supplement Reminders',
    description: 'Remember to take your supplements',
    icon: 'medical-outline',
    color: '#2196F3',
  },
  {
    id: 'STREAK_ALERT',
    title: 'Streak Alerts',
    description: "Don't lose your logging streak",
    icon: 'flame-outline',
    color: '#FF5722',
  },
  {
    id: 'WEEKLY_SUMMARY',
    title: 'Weekly Summary',
    description: 'Your week in review every Sunday',
    icon: 'bar-chart-outline',
    color: '#607D8B',
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

interface TimePickerButtonProps {
  label: string;
  time: string;
  onTimeChange: (time: string) => void;
}

function TimePickerButton({ label, time, onTimeChange }: TimePickerButtonProps) {
  const [showPicker, setShowPicker] = useState(false);
  const [tempTime, setTempTime] = useState(time);

  const handleChange = (_event: unknown, selectedDate?: Date) => {
    if (Platform.OS === 'android') {
      setShowPicker(false);
      if (selectedDate) {
        onTimeChange(formatDateToTime(selectedDate));
      }
    } else if (selectedDate) {
      // iOS: just update temp value, wait for Done button
      setTempTime(formatDateToTime(selectedDate));
    }
  };

  const handleDone = () => {
    onTimeChange(tempTime);
    setShowPicker(false);
  };

  const handleCancel = () => {
    setTempTime(time); // Reset to original
    setShowPicker(false);
  };

  const handleOpen = () => {
    setTempTime(time); // Sync temp with current
    setShowPicker(true);
  };

  return (
    <View style={styles.timePickerRow}>
      <Text style={styles.timePickerLabel}>{label}</Text>
      <TouchableOpacity
        style={styles.timePickerButton}
        onPress={handleOpen}
        accessibilityLabel={`Set ${label} reminder time`}
      >
        <Text style={styles.timePickerValue}>{formatTimeDisplay(time)}</Text>
        <Ionicons name="time-outline" size={18} color={colors.text.tertiary} />
      </TouchableOpacity>

      {/* Android: inline picker */}
      {showPicker && Platform.OS === 'android' && (
        <DateTimePicker
          value={parseTimeToDate(time)}
          mode="time"
          is24Hour={false}
          display="default"
          onChange={handleChange}
          minuteInterval={5}
        />
      )}

      {/* iOS: modal with Done/Cancel */}
      {Platform.OS === 'ios' && (
        <Modal visible={showPicker} transparent animationType="slide">
          <View style={styles.pickerModalOverlay}>
            <View style={styles.pickerModalContent}>
              <View style={styles.pickerModalHeader}>
                <TouchableOpacity onPress={handleCancel} accessibilityLabel="Cancel">
                  <Text style={styles.pickerCancelButton}>Cancel</Text>
                </TouchableOpacity>
                <Text style={styles.pickerModalTitle}>{label}</Text>
                <TouchableOpacity onPress={handleDone} accessibilityLabel="Done">
                  <Text style={styles.pickerDoneButton}>Done</Text>
                </TouchableOpacity>
              </View>
              <DateTimePicker
                value={parseTimeToDate(tempTime)}
                mode="time"
                is24Hour={false}
                display="spinner"
                onChange={handleChange}
                minuteInterval={5}
                style={styles.iosPicker}
              />
            </View>
          </View>
        </Modal>
      )}
    </View>
  );
}

export default function NotificationPreferencesScreen() {
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();
  const { permissionStatus, requestPermission, isRegistered, isProvisional, requestFullPermission } =
    useNotifications();

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
      // Check if it's a 404 (no preferences yet) vs other errors
      if (isAxiosError(error) && error.response?.status === 404) {
        // User has no preferences yet - initialize with defaults
        setPreferences({
          enabled: true,
          enabledCategories: ['MEAL_REMINDER', 'GOAL_PROGRESS', 'HEALTH_INSIGHT', 'STREAK_ALERT'],
          quietHoursEnabled: false,
          mealReminderTimes: DEFAULT_MEAL_TIMES,
        });
      } else {
        // Network error or other failure - show error to user
        console.error('Error loading preferences:', error);
        showAlert('Error', 'Failed to load notification preferences. Please check your connection and try again.');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (permissionStatus === 'granted') {
      loadPreferences();
    } else {
      setIsLoading(false);
    }
  }, [loadPreferences, permissionStatus]);

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
      showAlert('Saved', 'Your notification preferences have been updated.');
    } catch (error) {
      console.error('Error saving preferences:', error);
      showAlert('Error', 'Failed to save notification preferences. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  // Toggle notification category
  const handleCategoryToggle = (category: NotificationCategory, enabled: boolean) => {
    if (!preferences) return;

    const enabledCategories = enabled
      ? [...preferences.enabledCategories, category]
      : preferences.enabledCategories.filter((c) => c !== category);

    setPreferences({ ...preferences, enabledCategories });
    setHasChanges(true);
  };

  // Handle meal time change
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

  // Handle enabling notifications
  const handleEnableNotifications = async () => {
    const granted = await requestPermission();
    if (granted) {
      loadPreferences();
    }
  };

  // Handle upgrading from provisional
  const handleUpgradeFromProvisional = async () => {
    const granted = await requestFullPermission();
    if (granted) {
      showAlert('Success', 'Notifications are now fully enabled!');
    }
  };

  const isSystemDisabled = permissionStatus === 'denied';
  const isMealReminderEnabled = preferences?.enabledCategories.includes('MEAL_REMINDER');

  return (
    <SafeAreaView style={styles.container} testID="notification-preferences-screen">
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => router.back()}
          accessibilityLabel="Go back"
          testID="notification-preferences-back-button"
        >
          <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Notification Preferences</Text>
        {hasChanges && (
          <TouchableOpacity
            style={[styles.saveButton, isSaving && styles.saveButtonDisabled]}
            onPress={savePreferences}
            disabled={isSaving}
            accessibilityLabel="Save preferences"
            testID="notification-preferences-save-button"
          >
            {isSaving ? (
              <ActivityIndicator size="small" color={colors.primary.main} />
            ) : (
              <Text style={styles.saveButtonText}>Save</Text>
            )}
          </TouchableOpacity>
        )}
        {!hasChanges && <View style={styles.saveButtonPlaceholder} />}
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[
          styles.scrollContent,
          { paddingHorizontal: responsiveSpacing.horizontal },
          isTablet && styles.scrollContentTablet,
        ]}
        showsVerticalScrollIndicator={false}
      >
        {isLoading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={colors.primary.main} />
            <Text style={styles.loadingText}>Loading preferences...</Text>
          </View>
        ) : permissionStatus !== 'granted' ? (
          // Permission not granted - show enable CTA
          <View style={styles.permissionSection}>
            {isSystemDisabled ? (
              <>
                <View style={styles.permissionIcon}>
                  <Ionicons name="notifications-off" size={48} color={colors.status.warning} />
                </View>
                <Text style={styles.permissionTitle}>Notifications Disabled</Text>
                <Text style={styles.permissionDescription}>
                  To receive reminders and insights, please enable notifications in your device
                  settings.
                </Text>
                <TouchableOpacity
                  style={styles.settingsButton}
                  onPress={() => Linking.openSettings()}
                  accessibilityLabel="Open device settings"
                >
                  <Ionicons name="settings-outline" size={20} color={colors.text.primary} />
                  <Text style={styles.settingsButtonText}>Open Settings</Text>
                </TouchableOpacity>
              </>
            ) : (
              <>
                <View style={styles.permissionIcon}>
                  <Ionicons name="notifications-outline" size={48} color={colors.primary.main} />
                </View>
                <Text style={styles.permissionTitle}>Enable Notifications</Text>
                <Text style={styles.permissionDescription}>
                  Get helpful reminders to log meals and receive personalized health insights.
                </Text>
                <TouchableOpacity
                  style={styles.enableButton}
                  onPress={handleEnableNotifications}
                  accessibilityLabel="Enable notifications"
                >
                  <Ionicons name="notifications" size={20} color={colors.text.primary} />
                  <Text style={styles.enableButtonText}>Enable Notifications</Text>
                </TouchableOpacity>
              </>
            )}
          </View>
        ) : (
          // Permission granted - show preferences
          <View style={styles.content}>
            {/* iOS Provisional Warning */}
            {Platform.OS === 'ios' && isProvisional && (
              <TouchableOpacity style={styles.provisionalBanner} onPress={handleUpgradeFromProvisional}>
                <Ionicons name="information-circle" size={24} color={colors.status.info} />
                <View style={styles.provisionalContent}>
                  <Text style={styles.provisionalTitle}>Quiet Notifications</Text>
                  <Text style={styles.provisionalText}>
                    Notifications go to your Notification Center. Tap to enable alerts.
                  </Text>
                </View>
                <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
              </TouchableOpacity>
            )}

            {/* Categories Section */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Notification Types</Text>
              <Text style={styles.sectionDescription}>
                Choose which notifications you'd like to receive
              </Text>

              {CATEGORIES.map((category) => (
                <View key={category.id} style={styles.categoryItem}>
                  <View style={styles.categoryLeft}>
                    <View style={[styles.categoryIcon, { backgroundColor: `${category.color}20` }]}>
                      <Ionicons name={category.icon} size={20} color={category.color} />
                    </View>
                    <View style={styles.categoryInfo}>
                      <Text style={styles.categoryTitle}>{category.title}</Text>
                      <Text style={styles.categoryDescription}>{category.description}</Text>
                    </View>
                  </View>
                  <Switch
                    value={preferences?.enabledCategories.includes(category.id) ?? false}
                    onValueChange={(enabled) => handleCategoryToggle(category.id, enabled)}
                    trackColor={{ false: colors.background.elevated, true: colors.primary.main }}
                    thumbColor={colors.text.primary}
                    accessibilityLabel={`${preferences?.enabledCategories.includes(category.id) ? 'Disable' : 'Enable'} ${category.title}`}
                  />
                </View>
              ))}
            </View>

            {/* Meal Reminder Times */}
            {preferences && isMealReminderEnabled && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Meal Reminder Times</Text>
                <Text style={styles.sectionDescription}>
                  Set when you'd like to be reminded to log each meal
                </Text>

                <View style={styles.timePickersContainer}>
                  <TimePickerButton
                    label="Breakfast"
                    time={preferences.mealReminderTimes?.breakfast || DEFAULT_MEAL_TIMES.breakfast}
                    onTimeChange={(time) => handleMealTimeChange('breakfast', time)}
                  />
                  <TimePickerButton
                    label="Lunch"
                    time={preferences.mealReminderTimes?.lunch || DEFAULT_MEAL_TIMES.lunch}
                    onTimeChange={(time) => handleMealTimeChange('lunch', time)}
                  />
                  <TimePickerButton
                    label="Dinner"
                    time={preferences.mealReminderTimes?.dinner || DEFAULT_MEAL_TIMES.dinner}
                    onTimeChange={(time) => handleMealTimeChange('dinner', time)}
                  />
                  <TimePickerButton
                    label="Snack"
                    time={preferences.mealReminderTimes?.snack || DEFAULT_MEAL_TIMES.snack}
                    onTimeChange={(time) => handleMealTimeChange('snack', time)}
                  />
                </View>
              </View>
            )}

            {/* Quiet Hours */}
            {preferences && (
              <View style={styles.section}>
                <View style={styles.quietHoursHeader}>
                  <View style={styles.quietHoursLeft}>
                    <View style={[styles.categoryIcon, { backgroundColor: `${colors.text.tertiary}20` }]}>
                      <Ionicons name="moon" size={20} color={colors.text.tertiary} />
                    </View>
                    <View style={styles.quietHoursInfo}>
                      <Text style={styles.categoryTitle}>Quiet Hours</Text>
                      <Text style={styles.categoryDescription}>
                        No notifications during these hours
                      </Text>
                    </View>
                  </View>
                  <Switch
                    value={preferences.quietHoursEnabled ?? false}
                    onValueChange={handleQuietHoursToggle}
                    trackColor={{ false: colors.background.elevated, true: colors.primary.main }}
                    thumbColor={colors.text.primary}
                    accessibilityLabel={`${preferences.quietHoursEnabled ? 'Disable' : 'Enable'} quiet hours`}
                  />
                </View>

                {preferences.quietHoursEnabled && (
                  <View style={styles.quietHoursTimeContainer}>
                    <TimePickerButton
                      label="Start"
                      time={preferences.quietHoursStart || '22:00'}
                      onTimeChange={(time) => {
                        setPreferences({ ...preferences, quietHoursStart: time });
                        setHasChanges(true);
                      }}
                    />
                    <TimePickerButton
                      label="End"
                      time={preferences.quietHoursEnd || '08:00'}
                      onTimeChange={(time) => {
                        setPreferences({ ...preferences, quietHoursEnd: time });
                        setHasChanges(true);
                      }}
                    />
                  </View>
                )}
              </View>
            )}

            {/* Device Status */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Status</Text>
              <View style={styles.statusRow}>
                <Text style={styles.statusLabel}>Device Registered</Text>
                <View style={styles.statusValue}>
                  <View
                    style={[
                      styles.statusDot,
                      isRegistered ? styles.statusDotSuccess : styles.statusDotWarning,
                    ]}
                  />
                  <Text style={styles.statusText}>{isRegistered ? 'Yes' : 'No'}</Text>
                </View>
              </View>
              {Platform.OS === 'ios' && (
                <View style={styles.statusRow}>
                  <Text style={styles.statusLabel}>Authorization Type</Text>
                  <Text style={styles.statusText}>
                    {isProvisional ? 'Provisional (Quiet)' : 'Full'}
                  </Text>
                </View>
              )}
            </View>
          </View>
        )}
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
    ...typography.h3,
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
    ...typography.bodyBold,
    color: colors.primary.main,
  },
  saveButtonPlaceholder: {
    width: 60,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingVertical: spacing.lg,
  },
  scrollContentTablet: {
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: spacing['4xl'],
  },
  loadingText: {
    marginTop: spacing.md,
    ...typography.body,
    color: colors.text.tertiary,
  },
  content: {},

  // Permission Section
  permissionSection: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: spacing.xl,
  },
  permissionIcon: {
    marginBottom: spacing.lg,
  },
  permissionTitle: {
    ...typography.h2,
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: spacing.sm,
  },
  permissionDescription: {
    ...typography.body,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  enableButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary.main,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    borderRadius: borderRadius.md,
    gap: spacing.sm,
    ...shadows.sm,
  },
  enableButtonText: {
    ...typography.button,
    color: colors.text.primary,
  },
  settingsButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.surface.card,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    borderRadius: borderRadius.md,
    gap: spacing.sm,
  },
  settingsButtonText: {
    ...typography.button,
    color: colors.text.primary,
  },

  // Provisional Banner
  provisionalBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.special.infoLight,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginBottom: spacing.lg,
  },
  provisionalContent: {
    flex: 1,
    marginLeft: spacing.md,
  },
  provisionalTitle: {
    ...typography.bodyBold,
    color: colors.status.info,
  },
  provisionalText: {
    ...typography.bodySmall,
    color: colors.text.secondary,
  },

  // Section
  section: {
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
  },
  sectionTitle: {
    ...typography.bodySmall,
    fontWeight: '600',
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  sectionDescription: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
    marginBottom: spacing.md,
  },

  // Category Item
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
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  categoryInfo: {
    flex: 1,
  },
  categoryTitle: {
    ...typography.bodyBold,
    color: colors.text.primary,
  },
  categoryDescription: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
  },

  // Time Pickers
  timePickersContainer: {
    marginTop: spacing.sm,
  },
  timePickerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  timePickerLabel: {
    ...typography.body,
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
    ...typography.bodyBold,
    color: colors.text.primary,
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
  },
  quietHoursInfo: {
    flex: 1,
  },
  quietHoursTimeContainer: {
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },

  // Status
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
  },
  statusLabel: {
    ...typography.body,
    color: colors.text.secondary,
  },
  statusValue: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  statusText: {
    ...typography.body,
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
  statusDotWarning: {
    backgroundColor: colors.status.warning,
  },

  // iOS Time Picker Modal
  pickerModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  pickerModalContent: {
    backgroundColor: colors.surface.card,
    borderTopLeftRadius: borderRadius.lg,
    borderTopRightRadius: borderRadius.lg,
    paddingBottom: spacing.xl,
  },
  pickerModalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  pickerModalTitle: {
    ...typography.bodyBold,
    color: colors.text.primary,
  },
  pickerCancelButton: {
    ...typography.body,
    color: colors.text.tertiary,
    paddingHorizontal: spacing.sm,
  },
  pickerDoneButton: {
    ...typography.bodyBold,
    color: colors.primary.main,
    paddingHorizontal: spacing.sm,
  },
  iosPicker: {
    height: 200,
  },
});
