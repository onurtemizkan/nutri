import { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { userSupplementsApi } from '@/lib/api/supplements';
import {
  UserSupplement,
  ScheduleType,
  DayOfWeek,
  SCHEDULE_TYPES,
  DAYS_OF_WEEK,
  COMMON_UNITS,
  UpdateUserSupplementInput,
} from '@/lib/types/supplements';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';

export default function EditSupplementScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();

  const [userSupplement, setUserSupplement] = useState<UserSupplement | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // Form fields
  const [dosage, setDosage] = useState('');
  const [unit, setUnit] = useState('mg');
  const [scheduleType, setScheduleType] = useState<ScheduleType>('DAILY');
  const [scheduleTimes, setScheduleTimes] = useState<string[]>(['08:00']);
  const [weeklySchedule, setWeeklySchedule] = useState<Record<string, string[]>>({});
  const [intervalDays, setIntervalDays] = useState('2');
  const [notes, setNotes] = useState('');
  const [isActive, setIsActive] = useState(true);

  useEffect(() => {
    if (id) {
      loadSupplement();
    }
  }, [id]);

  const loadSupplement = async () => {
    try {
      const data = await userSupplementsApi.getById(id as string);
      setUserSupplement(data);

      // Populate form
      setDosage(data.dosage);
      setUnit(data.unit);
      setScheduleType(data.scheduleType);
      setIsActive(data.isActive);
      setNotes(data.notes || '');

      if (data.scheduleTimes && Array.isArray(data.scheduleTimes)) {
        setScheduleTimes(data.scheduleTimes as string[]);
      }
      if (data.weeklySchedule) {
        setWeeklySchedule(data.weeklySchedule as Record<string, string[]>);
      }
      if (data.intervalDays) {
        setIntervalDays(data.intervalDays.toString());
      }
    } catch (error) {
      console.error('Failed to load supplement:', error);
      showAlert('Error', getErrorMessage(error, 'Failed to load supplement'), [
        { text: 'OK', onPress: () => router.back() },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const addTime = () => {
    setScheduleTimes([...scheduleTimes, '12:00']);
  };

  const removeTime = (index: number) => {
    if (scheduleTimes.length > 1) {
      setScheduleTimes(scheduleTimes.filter((_, i) => i !== index));
    }
  };

  const updateTime = (index: number, time: string) => {
    const newTimes = [...scheduleTimes];
    newTimes[index] = time;
    setScheduleTimes(newTimes);
  };

  const toggleDay = (day: DayOfWeek) => {
    const newSchedule = { ...weeklySchedule };
    if (newSchedule[day]) {
      delete newSchedule[day];
    } else {
      newSchedule[day] = ['08:00'];
    }
    setWeeklySchedule(newSchedule);
  };

  const updateDayTimes = (day: string, times: string[]) => {
    setWeeklySchedule({ ...weeklySchedule, [day]: times });
  };

  const handleSave = async () => {
    if (!dosage) {
      showAlert('Error', 'Please enter a dosage');
      return;
    }
    if (scheduleType === 'WEEKLY' && Object.keys(weeklySchedule).length === 0) {
      showAlert('Error', 'Please select at least one day for weekly schedule');
      return;
    }

    setIsSaving(true);
    try {
      const input: UpdateUserSupplementInput = {
        dosage,
        unit,
        scheduleType,
        isActive,
        notes: notes || null,
      };

      // Add schedule-specific fields
      if (scheduleType === 'DAILY_MULTIPLE') {
        input.scheduleTimes = scheduleTimes;
      } else if (scheduleType === 'WEEKLY') {
        input.weeklySchedule = weeklySchedule;
      } else if (scheduleType === 'INTERVAL') {
        input.intervalDays = parseInt(intervalDays, 10);
      }

      await userSupplementsApi.update(id as string, input);
      showAlert('Success', 'Supplement schedule updated!', [
        { text: 'OK', onPress: () => router.back() },
      ]);
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to update supplement'));
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeactivate = () => {
    showAlert(
      'Deactivate Supplement',
      'This will remove the supplement from your daily schedule. You can reactivate it later.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Deactivate',
          style: 'destructive',
          onPress: async () => {
            setIsDeleting(true);
            try {
              await userSupplementsApi.deactivate(id as string);
              showAlert('Success', 'Supplement deactivated', [
                { text: 'OK', onPress: () => router.back() },
              ]);
            } catch (error) {
              showAlert('Error', getErrorMessage(error, 'Failed to deactivate supplement'));
            } finally {
              setIsDeleting(false);
            }
          },
        },
      ]
    );
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
        </View>
      </SafeAreaView>
    );
  }

  if (!userSupplement) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <Text style={styles.errorText}>Supplement not found</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()}>
            <Ionicons name="arrow-back" size={24} color={colors.text.secondary} />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Edit Supplement</Text>
          <TouchableOpacity onPress={handleSave} disabled={isSaving}>
            {isSaving ? (
              <ActivityIndicator color={colors.primary.main} />
            ) : (
              <Text style={styles.saveButton}>Save</Text>
            )}
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
          <View style={styles.content}>
            {/* Supplement Info */}
            <View style={styles.supplementInfo}>
              <Text style={styles.supplementName}>{userSupplement.supplement.name}</Text>
              <View style={styles.statusBadge}>
                <View
                  style={[
                    styles.statusDot,
                    { backgroundColor: isActive ? colors.status.success : colors.status.warning },
                  ]}
                />
                <Text style={styles.statusText}>{isActive ? 'Active' : 'Inactive'}</Text>
              </View>
            </View>

            {/* Dosage */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Dosage</Text>
              <View style={styles.row}>
                <View style={styles.flex2}>
                  <Text style={styles.label}>Amount *</Text>
                  <View style={styles.inputWrapper}>
                    <TextInput
                      style={styles.input}
                      placeholder="e.g., 500"
                      placeholderTextColor={colors.text.disabled}
                      value={dosage}
                      onChangeText={setDosage}
                      keyboardType="numeric"
                      editable={!isSaving}
                    />
                  </View>
                </View>
                <View style={styles.flex1}>
                  <Text style={styles.label}>Unit</Text>
                  <View style={styles.unitSelector}>
                    <ScrollView
                      horizontal
                      showsHorizontalScrollIndicator={false}
                      contentContainerStyle={styles.unitContainer}
                    >
                      {COMMON_UNITS.slice(0, 6).map((u) => (
                        <TouchableOpacity
                          key={u}
                          style={[styles.unitButton, unit === u && styles.unitButtonActive]}
                          onPress={() => setUnit(u)}
                        >
                          <Text
                            style={[
                              styles.unitButtonText,
                              unit === u && styles.unitButtonTextActive,
                            ]}
                          >
                            {u}
                          </Text>
                        </TouchableOpacity>
                      ))}
                    </ScrollView>
                  </View>
                </View>
              </View>
            </View>

            {/* Schedule Type */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Schedule</Text>
              <View style={styles.scheduleTypeContainer}>
                {SCHEDULE_TYPES.filter((t) => t.value !== 'ONE_TIME').map((type) => (
                  <TouchableOpacity
                    key={type.value}
                    style={styles.scheduleTypeButton}
                    onPress={() => setScheduleType(type.value)}
                    activeOpacity={0.8}
                  >
                    {scheduleType === type.value ? (
                      <LinearGradient
                        colors={gradients.primary}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 0 }}
                        style={styles.scheduleTypeButtonActive}
                      >
                        <Text style={styles.scheduleTypeTextActive}>{type.label}</Text>
                        <Text style={styles.scheduleTypeDescActive}>{type.description}</Text>
                      </LinearGradient>
                    ) : (
                      <View style={styles.scheduleTypeButtonInactive}>
                        <Text style={styles.scheduleTypeText}>{type.label}</Text>
                        <Text style={styles.scheduleTypeDesc}>{type.description}</Text>
                      </View>
                    )}
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            {/* Schedule Details */}
            {scheduleType === 'DAILY_MULTIPLE' && (
              <View style={styles.section}>
                <View style={styles.sectionHeader}>
                  <Text style={styles.sectionTitle}>Times</Text>
                  <TouchableOpacity onPress={addTime}>
                    <Ionicons name="add-circle" size={24} color={colors.primary.main} />
                  </TouchableOpacity>
                </View>
                {scheduleTimes.map((time, index) => (
                  <View key={index} style={styles.timeRow}>
                    <View style={styles.timeInputWrapper}>
                      <TextInput
                        style={styles.timeInput}
                        value={time}
                        onChangeText={(t) => updateTime(index, t)}
                        placeholder="HH:MM"
                        placeholderTextColor={colors.text.disabled}
                      />
                    </View>
                    {scheduleTimes.length > 1 && (
                      <TouchableOpacity onPress={() => removeTime(index)}>
                        <Ionicons name="close-circle" size={24} color={colors.status.error} />
                      </TouchableOpacity>
                    )}
                  </View>
                ))}
              </View>
            )}

            {scheduleType === 'WEEKLY' && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Days</Text>
                <View style={styles.daysContainer}>
                  {DAYS_OF_WEEK.map((day) => (
                    <TouchableOpacity
                      key={day.value}
                      style={[
                        styles.dayButton,
                        weeklySchedule[day.value] && styles.dayButtonActive,
                      ]}
                      onPress={() => toggleDay(day.value)}
                    >
                      <Text
                        style={[
                          styles.dayButtonText,
                          weeklySchedule[day.value] && styles.dayButtonTextActive,
                        ]}
                      >
                        {day.short}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
                {Object.entries(weeklySchedule).map(([day, times]) => (
                  <View key={day} style={styles.weeklyDayTimes}>
                    <Text style={styles.weeklyDayLabel}>
                      {DAYS_OF_WEEK.find((d) => d.value === day)?.label}
                    </Text>
                    <View style={styles.timeInputWrapper}>
                      <TextInput
                        style={styles.timeInput}
                        value={times[0]}
                        onChangeText={(t) => updateDayTimes(day, [t])}
                        placeholder="HH:MM"
                        placeholderTextColor={colors.text.disabled}
                      />
                    </View>
                  </View>
                ))}
              </View>
            )}

            {scheduleType === 'INTERVAL' && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Interval</Text>
                <View style={styles.intervalRow}>
                  <Text style={styles.intervalText}>Every</Text>
                  <View style={styles.intervalInputWrapper}>
                    <TextInput
                      style={styles.intervalInput}
                      value={intervalDays}
                      onChangeText={setIntervalDays}
                      keyboardType="numeric"
                      placeholder="2"
                      placeholderTextColor={colors.text.disabled}
                    />
                  </View>
                  <Text style={styles.intervalText}>days</Text>
                </View>
              </View>
            )}

            {/* Notes */}
            <View style={styles.section}>
              <Text style={styles.label}>Notes</Text>
              <View style={styles.inputWrapper}>
                <TextInput
                  style={[styles.input, styles.notesInput]}
                  placeholder="Add any notes about this supplement..."
                  placeholderTextColor={colors.text.disabled}
                  value={notes}
                  onChangeText={setNotes}
                  multiline
                  numberOfLines={3}
                  textAlignVertical="top"
                  editable={!isSaving}
                />
              </View>
            </View>

            {/* Active Toggle */}
            <View style={styles.section}>
              <TouchableOpacity
                style={styles.toggleRow}
                onPress={() => setIsActive(!isActive)}
              >
                <View>
                  <Text style={styles.toggleLabel}>Active</Text>
                  <Text style={styles.toggleDescription}>
                    Show in daily schedule
                  </Text>
                </View>
                <View
                  style={[
                    styles.toggle,
                    isActive && styles.toggleActive,
                  ]}
                >
                  <View
                    style={[
                      styles.toggleKnob,
                      isActive && styles.toggleKnobActive,
                    ]}
                  />
                </View>
              </TouchableOpacity>
            </View>

            {/* Deactivate Button */}
            <TouchableOpacity
              style={styles.deactivateButton}
              onPress={handleDeactivate}
              disabled={isDeleting}
            >
              {isDeleting ? (
                <ActivityIndicator color={colors.status.error} />
              ) : (
                <>
                  <Ionicons name="trash-outline" size={20} color={colors.status.error} />
                  <Text style={styles.deactivateButtonText}>Deactivate Supplement</Text>
                </>
              )}
            </TouchableOpacity>
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorText: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
    backgroundColor: colors.background.secondary,
  },
  headerTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  saveButton: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: spacing['3xl'],
  },

  // Supplement Info
  supplementInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xl,
    padding: spacing.md,
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  supplementName: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    flex: 1,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: spacing.xs,
  },
  statusText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
  },

  // Sections
  section: {
    marginBottom: spacing.xl,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  sectionTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.md,
    letterSpacing: -0.3,
  },
  label: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
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
    height: 48,
  },
  notesInput: {
    height: 80,
    textAlignVertical: 'top',
    paddingTop: spacing.md,
  },
  row: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  flex1: {
    flex: 1,
  },
  flex2: {
    flex: 2,
  },

  // Unit Selector
  unitSelector: {
    height: 48,
  },
  unitContainer: {
    flexDirection: 'row',
    gap: spacing.xs,
    alignItems: 'center',
  },
  unitButton: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
    backgroundColor: colors.background.tertiary,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  unitButtonActive: {
    backgroundColor: colors.primary.main,
    borderColor: colors.primary.main,
  },
  unitButtonText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  unitButtonTextActive: {
    color: colors.text.primary,
    fontWeight: typography.fontWeight.semibold,
  },

  // Schedule Type
  scheduleTypeContainer: {
    gap: spacing.sm,
  },
  scheduleTypeButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  scheduleTypeButtonActive: {
    padding: spacing.md,
  },
  scheduleTypeButtonInactive: {
    padding: spacing.md,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  scheduleTypeText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
  },
  scheduleTypeTextActive: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  scheduleTypeDesc: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  scheduleTypeDescActive: {
    fontSize: typography.fontSize.xs,
    color: 'rgba(255, 255, 255, 0.8)',
    marginTop: 2,
  },

  // Time inputs
  timeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.sm,
  },
  timeInputWrapper: {
    flex: 1,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  timeInput: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    textAlign: 'center',
  },

  // Days selector
  daysContainer: {
    flexDirection: 'row',
    gap: spacing.xs,
    marginBottom: spacing.md,
  },
  dayButton: {
    flex: 1,
    paddingVertical: spacing.sm,
    alignItems: 'center',
    borderRadius: borderRadius.sm,
    backgroundColor: colors.background.tertiary,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  dayButtonActive: {
    backgroundColor: colors.primary.main,
    borderColor: colors.primary.main,
  },
  dayButtonText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
  },
  dayButtonTextActive: {
    color: colors.text.primary,
  },
  weeklyDayTimes: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    marginBottom: spacing.sm,
  },
  weeklyDayLabel: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
    width: 80,
  },

  // Interval
  intervalRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  intervalText: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
  },
  intervalInputWrapper: {
    width: 80,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  intervalInput: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    textAlign: 'center',
  },

  // Toggle
  toggleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.md,
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  toggleLabel: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  toggleDescription: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  toggle: {
    width: 50,
    height: 30,
    borderRadius: 15,
    backgroundColor: colors.background.tertiary,
    padding: 2,
    justifyContent: 'center',
  },
  toggleActive: {
    backgroundColor: colors.primary.main,
  },
  toggleKnob: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: colors.text.primary,
  },
  toggleKnobActive: {
    alignSelf: 'flex-end',
  },

  // Deactivate Button
  deactivateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.md,
    marginTop: spacing.lg,
    borderWidth: 1,
    borderColor: colors.status.error,
    borderRadius: borderRadius.md,
    gap: spacing.sm,
  },
  deactivateButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.status.error,
  },
});
