/**
 * Add/Edit Activity Screen
 *
 * Form for manually entering or editing activities with
 * type picker, intensity selector, duration input, and optional metrics.
 */

import { useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import DateTimePicker from '@react-native-community/datetimepicker';
import { activitiesApi } from '@/lib/api/activities';
import {
  ActivityType,
  ActivityIntensity,
  CreateActivityInput,
  ACTIVITY_TYPES,
  ACTIVITY_TYPE_CONFIG,
  INTENSITY_LEVELS,
  INTENSITY_CONFIG,
  getActivitiesByCategory,
  CATEGORY_DISPLAY_NAMES,
  estimateCalories,
  ActivityCategory,
} from '@/lib/types/activities';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { getErrorMessage } from '@/lib/utils/errorHandling';

export default function AddActivityScreen() {
  const router = useRouter();
  const { editId } = useLocalSearchParams<{ editId?: string }>();
  const isEditing = !!editId;
  const { getResponsiveValue } = useResponsive();

  // Form state
  const [activityType, setActivityType] = useState<ActivityType>('RUNNING');
  const [intensity, setIntensity] = useState<ActivityIntensity>('MODERATE');
  const [startDate, setStartDate] = useState(new Date());
  const [durationHours, setDurationHours] = useState('0');
  const [durationMinutes, setDurationMinutes] = useState('30');
  const [caloriesBurned, setCaloriesBurned] = useState('');
  const [averageHeartRate, setAverageHeartRate] = useState('');
  const [maxHeartRate, setMaxHeartRate] = useState('');
  const [distance, setDistance] = useState('');
  const [steps, setSteps] = useState('');
  const [notes, setNotes] = useState('');

  // UI state
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingActivity, setIsLoadingActivity] = useState(isEditing);
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [showTimePicker, setShowTimePicker] = useState(false);
  const [showTypePicker, setShowTypePicker] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const contentPadding = getResponsiveValue({
    small: spacing.md,
    medium: spacing.lg,
    large: spacing.lg,
    tablet: spacing.xl,
    default: spacing.lg,
  });

  // Load existing activity if editing
  useEffect(() => {
    if (editId) {
      loadActivity();
    }
  }, [editId]);

  const loadActivity = async () => {
    if (!editId) return;
    try {
      const activity = await activitiesApi.getById(editId);
      setActivityType(activity.activityType);
      setIntensity(activity.intensity);
      setStartDate(new Date(activity.startedAt));
      const hours = Math.floor(activity.duration / 60);
      const mins = activity.duration % 60;
      setDurationHours(hours.toString());
      setDurationMinutes(mins.toString());
      if (activity.caloriesBurned) setCaloriesBurned(activity.caloriesBurned.toString());
      if (activity.averageHeartRate) setAverageHeartRate(activity.averageHeartRate.toString());
      if (activity.maxHeartRate) setMaxHeartRate(activity.maxHeartRate.toString());
      if (activity.distance) setDistance((activity.distance / 1000).toFixed(2)); // Convert to km
      if (activity.steps) setSteps(activity.steps.toString());
      if (activity.notes) setNotes(activity.notes);
    } catch (err) {
      Alert.alert('Error', 'Failed to load activity');
      router.back();
    } finally {
      setIsLoadingActivity(false);
    }
  };

  // Auto-estimate calories when type, intensity, or duration changes
  useEffect(() => {
    if (!caloriesBurned || caloriesBurned === '') {
      const totalMinutes = parseInt(durationHours || '0') * 60 + parseInt(durationMinutes || '0');
      if (totalMinutes > 0) {
        const estimated = estimateCalories(activityType, intensity, totalMinutes);
        setCaloriesBurned(estimated.toString());
      }
    }
  }, [activityType, intensity, durationHours, durationMinutes]);

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};
    const totalMinutes = parseInt(durationHours || '0') * 60 + parseInt(durationMinutes || '0');

    if (totalMinutes < 1) {
      newErrors.duration = 'Duration must be at least 1 minute';
    }

    if (averageHeartRate && (parseInt(averageHeartRate) < 30 || parseInt(averageHeartRate) > 250)) {
      newErrors.averageHeartRate = 'Heart rate must be between 30 and 250 bpm';
    }

    if (maxHeartRate && (parseInt(maxHeartRate) < 30 || parseInt(maxHeartRate) > 250)) {
      newErrors.maxHeartRate = 'Heart rate must be between 30 and 250 bpm';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSave = async () => {
    if (!validateForm()) return;

    setIsLoading(true);
    try {
      const totalMinutes = parseInt(durationHours || '0') * 60 + parseInt(durationMinutes || '0');
      const endDate = new Date(startDate.getTime() + totalMinutes * 60 * 1000);

      const data: CreateActivityInput = {
        activityType,
        intensity,
        startedAt: startDate.toISOString(),
        endedAt: endDate.toISOString(),
        duration: totalMinutes,
        source: 'manual',
        ...(caloriesBurned && { caloriesBurned: parseFloat(caloriesBurned) }),
        ...(averageHeartRate && { averageHeartRate: parseInt(averageHeartRate) }),
        ...(maxHeartRate && { maxHeartRate: parseInt(maxHeartRate) }),
        ...(distance && { distance: parseFloat(distance) * 1000 }), // Convert km to meters
        ...(steps && { steps: parseInt(steps) }),
        ...(notes && { notes }),
      };

      if (isEditing && editId) {
        await activitiesApi.update(editId, data);
        Alert.alert('Success', 'Activity updated!', [{ text: 'OK', onPress: () => router.back() }]);
      } else {
        await activitiesApi.create(data);
        Alert.alert('Success', 'Activity added!', [{ text: 'OK', onPress: () => router.back() }]);
      }
    } catch (err) {
      Alert.alert('Error', getErrorMessage(err, 'Failed to save activity'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    router.back();
  };

  const typeConfig = ACTIVITY_TYPE_CONFIG[activityType];
  const activitiesByCategory = getActivitiesByCategory();

  if (isLoadingActivity) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top']} testID="add-activity-screen">
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        {/* Header */}
        <View style={[styles.header, { paddingHorizontal: contentPadding }]}>
          <TouchableOpacity
            onPress={handleCancel}
            style={styles.headerButton}
            accessibilityLabel="Cancel"
            testID="activity-cancel-button"
          >
            <Text style={styles.cancelText}>Cancel</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>{isEditing ? 'Edit Activity' : 'Add Activity'}</Text>
          <TouchableOpacity
            onPress={handleSave}
            style={styles.headerButton}
            disabled={isLoading}
            accessibilityLabel="Save"
            testID="activity-save-button"
          >
            {isLoading ? (
              <ActivityIndicator size="small" color={colors.primary.main} />
            ) : (
              <Text style={styles.saveText}>Save</Text>
            )}
          </TouchableOpacity>
        </View>

        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={[styles.scrollContent, { paddingHorizontal: contentPadding }]}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Activity Type Selector */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Activity Type</Text>
            <TouchableOpacity
              style={styles.typeSelector}
              onPress={() => setShowTypePicker(!showTypePicker)}
              testID="activity-type-selector"
            >
              <View style={[styles.typeIconSmall, { backgroundColor: typeConfig.color + '20' }]}>
                <Ionicons
                  name={typeConfig.icon as keyof typeof Ionicons.glyphMap}
                  size={24}
                  color={typeConfig.color}
                />
              </View>
              <Text style={styles.typeSelectorText}>{typeConfig.displayName}</Text>
              <Ionicons
                name={showTypePicker ? 'chevron-up' : 'chevron-down'}
                size={20}
                color={colors.text.secondary}
              />
            </TouchableOpacity>

            {showTypePicker && (
              <View style={styles.typePickerContainer}>
                {(Object.keys(activitiesByCategory) as ActivityCategory[])
                  .filter((cat) => cat !== 'all')
                  .map((category) => (
                    <View key={category} style={styles.typeCategory}>
                      <Text style={styles.typeCategoryTitle}>
                        {CATEGORY_DISPLAY_NAMES[category]}
                      </Text>
                      <View style={styles.typeGrid}>
                        {activitiesByCategory[category].map((type) => {
                          const config = ACTIVITY_TYPE_CONFIG[type];
                          const isSelected = activityType === type;
                          return (
                            <TouchableOpacity
                              key={type}
                              style={[
                                styles.typeOption,
                                isSelected && { borderColor: config.color, backgroundColor: config.color + '10' },
                              ]}
                              onPress={() => {
                                setActivityType(type);
                                setShowTypePicker(false);
                              }}
                              testID={`activity-type-${type}`}
                            >
                              <Ionicons
                                name={config.icon as keyof typeof Ionicons.glyphMap}
                                size={20}
                                color={isSelected ? config.color : colors.text.secondary}
                              />
                              <Text
                                style={[
                                  styles.typeOptionText,
                                  isSelected && { color: config.color },
                                ]}
                              >
                                {config.shortName}
                              </Text>
                            </TouchableOpacity>
                          );
                        })}
                      </View>
                    </View>
                  ))}
              </View>
            )}
          </View>

          {/* Intensity Selector */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Intensity</Text>
            <View style={styles.intensityGrid}>
              {INTENSITY_LEVELS.map((level) => {
                const config = INTENSITY_CONFIG[level];
                const isSelected = intensity === level;
                return (
                  <TouchableOpacity
                    key={level}
                    style={[
                      styles.intensityOption,
                      isSelected && { borderColor: config.color, backgroundColor: config.color + '20' },
                    ]}
                    onPress={() => setIntensity(level)}
                    testID={`activity-intensity-${level}`}
                  >
                    <Ionicons
                      name={config.icon as keyof typeof Ionicons.glyphMap}
                      size={20}
                      color={isSelected ? config.color : colors.text.secondary}
                    />
                    <Text
                      style={[
                        styles.intensityOptionText,
                        isSelected && { color: config.color, fontWeight: typography.fontWeight.semibold as '600' },
                      ]}
                    >
                      {config.displayName}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
          </View>

          {/* Date and Time */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>When</Text>
            <View style={styles.dateTimeRow}>
              <TouchableOpacity
                style={styles.dateTimeButton}
                onPress={() => setShowDatePicker(true)}
                testID="activity-date-picker"
              >
                <Ionicons name="calendar-outline" size={20} color={colors.text.secondary} />
                <Text style={styles.dateTimeText}>
                  {startDate.toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                  })}
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.dateTimeButton}
                onPress={() => setShowTimePicker(true)}
                testID="activity-time-picker"
              >
                <Ionicons name="time-outline" size={20} color={colors.text.secondary} />
                <Text style={styles.dateTimeText}>
                  {startDate.toLocaleTimeString('en-US', {
                    hour: 'numeric',
                    minute: '2-digit',
                  })}
                </Text>
              </TouchableOpacity>
            </View>
            {showDatePicker && (
              <DateTimePicker
                value={startDate}
                mode="date"
                display="spinner"
                onChange={(_, date) => {
                  setShowDatePicker(false);
                  if (date) setStartDate(date);
                }}
              />
            )}
            {showTimePicker && (
              <DateTimePicker
                value={startDate}
                mode="time"
                display="spinner"
                onChange={(_, date) => {
                  setShowTimePicker(false);
                  if (date) setStartDate(date);
                }}
              />
            )}
          </View>

          {/* Duration */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Duration</Text>
            <View style={styles.durationRow}>
              <View style={styles.durationInput}>
                <TextInput
                  style={styles.durationTextInput}
                  value={durationHours}
                  onChangeText={setDurationHours}
                  keyboardType="number-pad"
                  maxLength={2}
                  placeholder="0"
                  placeholderTextColor={colors.text.disabled}
                  testID="activity-duration-hours"
                />
                <Text style={styles.durationLabel}>hours</Text>
              </View>
              <Text style={styles.durationSeparator}>:</Text>
              <View style={styles.durationInput}>
                <TextInput
                  style={styles.durationTextInput}
                  value={durationMinutes}
                  onChangeText={setDurationMinutes}
                  keyboardType="number-pad"
                  maxLength={2}
                  placeholder="30"
                  placeholderTextColor={colors.text.disabled}
                  testID="activity-duration-minutes"
                />
                <Text style={styles.durationLabel}>min</Text>
              </View>
            </View>
            {errors.duration && <Text style={styles.errorText}>{errors.duration}</Text>}
          </View>

          {/* Calories */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Calories Burned</Text>
            <View style={styles.inputContainer}>
              <Ionicons name="flame-outline" size={20} color={colors.text.secondary} />
              <TextInput
                style={styles.textInput}
                value={caloriesBurned}
                onChangeText={setCaloriesBurned}
                keyboardType="number-pad"
                placeholder="Auto-estimated"
                placeholderTextColor={colors.text.disabled}
                testID="activity-calories-input"
              />
              <Text style={styles.inputUnit}>cal</Text>
            </View>
          </View>

          {/* Heart Rate (Optional) */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Heart Rate (Optional)</Text>
            <View style={styles.twoColumnRow}>
              <View style={styles.halfInputContainer}>
                <TextInput
                  style={styles.textInput}
                  value={averageHeartRate}
                  onChangeText={setAverageHeartRate}
                  keyboardType="number-pad"
                  placeholder="Avg"
                  placeholderTextColor={colors.text.disabled}
                  testID="activity-avg-hr-input"
                />
                <Text style={styles.inputUnit}>bpm</Text>
              </View>
              <View style={styles.halfInputContainer}>
                <TextInput
                  style={styles.textInput}
                  value={maxHeartRate}
                  onChangeText={setMaxHeartRate}
                  keyboardType="number-pad"
                  placeholder="Max"
                  placeholderTextColor={colors.text.disabled}
                  testID="activity-max-hr-input"
                />
                <Text style={styles.inputUnit}>bpm</Text>
              </View>
            </View>
            {errors.averageHeartRate && <Text style={styles.errorText}>{errors.averageHeartRate}</Text>}
            {errors.maxHeartRate && <Text style={styles.errorText}>{errors.maxHeartRate}</Text>}
          </View>

          {/* Distance (if applicable) */}
          {typeConfig.hasDistance && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Distance (Optional)</Text>
              <View style={styles.inputContainer}>
                <Ionicons name="navigate-outline" size={20} color={colors.text.secondary} />
                <TextInput
                  style={styles.textInput}
                  value={distance}
                  onChangeText={setDistance}
                  keyboardType="decimal-pad"
                  placeholder="0.00"
                  placeholderTextColor={colors.text.disabled}
                  testID="activity-distance-input"
                />
                <Text style={styles.inputUnit}>km</Text>
              </View>
            </View>
          )}

          {/* Steps (if applicable) */}
          {typeConfig.hasSteps && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Steps (Optional)</Text>
              <View style={styles.inputContainer}>
                <Ionicons name="footsteps-outline" size={20} color={colors.text.secondary} />
                <TextInput
                  style={styles.textInput}
                  value={steps}
                  onChangeText={setSteps}
                  keyboardType="number-pad"
                  placeholder="0"
                  placeholderTextColor={colors.text.disabled}
                  testID="activity-steps-input"
                />
                <Text style={styles.inputUnit}>steps</Text>
              </View>
            </View>
          )}

          {/* Notes */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Notes (Optional)</Text>
            <TextInput
              style={styles.notesInput}
              value={notes}
              onChangeText={setNotes}
              placeholder="Add any notes about your workout..."
              placeholderTextColor={colors.text.disabled}
              multiline
              numberOfLines={3}
              textAlignVertical="top"
              testID="activity-notes-input"
            />
          </View>

          {/* Source Badge */}
          <View style={styles.sourceBadge}>
            <Ionicons name="create-outline" size={16} color={colors.text.tertiary} />
            <Text style={styles.sourceText}>Manual Entry</Text>
          </View>

          {/* Bottom padding */}
          <View style={{ height: spacing.xl }} />
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
  },
  headerButton: {
    minWidth: 60,
  },
  headerTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
  },
  cancelText: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
  },
  saveText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.primary.main,
    textAlign: 'right',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingTop: spacing.lg,
    paddingBottom: spacing.xl,
  },
  section: {
    marginBottom: spacing.lg,
  },
  sectionTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.tertiary,
    marginBottom: spacing.sm,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  typeSelector: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  typeIconSmall: {
    width: 40,
    height: 40,
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  typeSelectorText: {
    flex: 1,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium as '500',
    color: colors.text.primary,
  },
  typePickerContainer: {
    marginTop: spacing.md,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  typeCategory: {
    marginBottom: spacing.md,
  },
  typeCategoryTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.secondary,
    marginBottom: spacing.sm,
  },
  typeGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  typeOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    gap: spacing.xs,
  },
  typeOptionText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },
  intensityGrid: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  intensityOption: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: spacing.md,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  intensityOptionText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginTop: spacing.xs,
  },
  dateTimeRow: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  dateTimeButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    gap: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  dateTimeText: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
  },
  durationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
  },
  durationInput: {
    alignItems: 'center',
  },
  durationTextInput: {
    width: 80,
    height: 56,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold as '700',
    color: colors.text.primary,
    textAlign: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  durationLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  durationSeparator: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold as '700',
    color: colors.text.secondary,
    marginTop: -spacing.lg,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    gap: spacing.sm,
  },
  textInput: {
    flex: 1,
    height: 48,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
  },
  inputUnit: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  twoColumnRow: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  halfInputContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  notesInput: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    minHeight: 100,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  errorText: {
    fontSize: typography.fontSize.sm,
    color: colors.status.error,
    marginTop: spacing.xs,
  },
  sourceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.special.highlight + '30',
    borderRadius: borderRadius.full,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    alignSelf: 'center',
    gap: spacing.xs,
  },
  sourceText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
});
