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
  Modal,
  FlatList,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { supplementsApi, userSupplementsApi } from '@/lib/api/supplements';
import {
  Supplement,
  ScheduleType,
  DayOfWeek,
  SUPPLEMENT_CATEGORIES,
  SCHEDULE_TYPES,
  DAYS_OF_WEEK,
  COMMON_UNITS,
  CreateUserSupplementInput,
} from '@/lib/types/supplements';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';

export default function AddSupplementScreen() {
  // Supplement selection
  const [supplements, setSupplements] = useState<Supplement[]>([]);
  const [selectedSupplement, setSelectedSupplement] = useState<Supplement | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSupplementPicker, setShowSupplementPicker] = useState(false);
  const [loadingSupplements, setLoadingSupplements] = useState(true);

  // Form fields
  const [dosage, setDosage] = useState('');
  const [unit, setUnit] = useState('mg');
  const [scheduleType, setScheduleType] = useState<ScheduleType>('DAILY');
  const [scheduleTimes, setScheduleTimes] = useState<string[]>(['08:00']);
  const [weeklySchedule, setWeeklySchedule] = useState<Record<string, string[]>>({});
  const [intervalDays, setIntervalDays] = useState('2');
  const [notes, setNotes] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  useEffect(() => {
    loadSupplements();
  }, []);

  const loadSupplements = async () => {
    try {
      const data = await supplementsApi.getSupplements();
      setSupplements(data);
    } catch (error) {
      console.error('Failed to load supplements:', error);
      showAlert('Error', getErrorMessage(error, 'Failed to load supplements'));
    } finally {
      setLoadingSupplements(false);
    }
  };

  const filteredSupplements = supplements.filter(
    (s) =>
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleSelectSupplement = (supplement: Supplement) => {
    setSelectedSupplement(supplement);
    if (supplement.defaultDosage) setDosage(supplement.defaultDosage);
    if (supplement.defaultUnit) setUnit(supplement.defaultUnit);
    setShowSupplementPicker(false);
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
    if (!selectedSupplement) {
      showAlert('Error', 'Please select a supplement');
      return;
    }
    if (!dosage) {
      showAlert('Error', 'Please enter a dosage');
      return;
    }
    if (scheduleType === 'WEEKLY' && Object.keys(weeklySchedule).length === 0) {
      showAlert('Error', 'Please select at least one day for weekly schedule');
      return;
    }

    setIsLoading(true);
    try {
      const input: CreateUserSupplementInput = {
        supplementId: selectedSupplement.id,
        dosage,
        unit,
        scheduleType,
        startDate: new Date().toISOString(),
        notes: notes || undefined,
      };

      // Add schedule-specific fields
      if (scheduleType === 'DAILY_MULTIPLE') {
        input.scheduleTimes = scheduleTimes;
      } else if (scheduleType === 'WEEKLY') {
        input.weeklySchedule = weeklySchedule;
      } else if (scheduleType === 'INTERVAL') {
        input.intervalDays = parseInt(intervalDays, 10);
      }

      await userSupplementsApi.create(input);
      showAlert('Success', 'Supplement schedule created!', [
        { text: 'OK', onPress: () => router.back() },
      ]);
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to create supplement schedule'));
    } finally {
      setIsLoading(false);
    }
  };

  const getCategoryLabel = (category: string) => {
    return SUPPLEMENT_CATEGORIES.find((c) => c.value === category)?.label || category;
  };

  const renderSupplementItem = ({ item }: { item: Supplement }) => (
    <TouchableOpacity
      style={styles.supplementItem}
      onPress={() => handleSelectSupplement(item)}
    >
      <View style={styles.supplementItemInfo}>
        <Text style={styles.supplementItemName}>{item.name}</Text>
        <Text style={styles.supplementItemCategory}>
          {getCategoryLabel(item.category)}
        </Text>
      </View>
      {item.defaultDosage && (
        <Text style={styles.supplementItemDosage}>
          {item.defaultDosage} {item.defaultUnit}
        </Text>
      )}
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()}>
            <Text style={styles.cancelButton}>Cancel</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Add Supplement</Text>
          <TouchableOpacity onPress={handleSave} disabled={isLoading}>
            {isLoading ? (
              <ActivityIndicator color={colors.primary.main} />
            ) : (
              <Text style={styles.saveButton}>Save</Text>
            )}
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
          <View style={styles.content}>
            {/* Supplement Selection */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Supplement</Text>
              <TouchableOpacity
                style={styles.pickerButton}
                onPress={() => setShowSupplementPicker(true)}
              >
                {selectedSupplement ? (
                  <View style={styles.selectedSupplement}>
                    <Text style={styles.selectedSupplementName}>
                      {selectedSupplement.name}
                    </Text>
                    <Text style={styles.selectedSupplementCategory}>
                      {getCategoryLabel(selectedSupplement.category)}
                    </Text>
                  </View>
                ) : (
                  <Text style={styles.pickerPlaceholder}>Select a supplement</Text>
                )}
                <Ionicons name="chevron-down" size={20} color={colors.text.tertiary} />
              </TouchableOpacity>
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
                      editable={!isLoading}
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
                  editable={!isLoading}
                />
              </View>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>

      {/* Supplement Picker Modal */}
      <Modal visible={showSupplementPicker} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Select Supplement</Text>
              <TouchableOpacity onPress={() => setShowSupplementPicker(false)}>
                <Ionicons name="close" size={24} color={colors.text.secondary} />
              </TouchableOpacity>
            </View>
            <View style={styles.searchContainer}>
              <Ionicons name="search" size={20} color={colors.text.tertiary} />
              <TextInput
                style={styles.searchInput}
                placeholder="Search supplements..."
                placeholderTextColor={colors.text.disabled}
                value={searchQuery}
                onChangeText={setSearchQuery}
              />
            </View>
            {loadingSupplements ? (
              <ActivityIndicator size="large" color={colors.primary.main} style={styles.loader} />
            ) : (
              <FlatList
                data={filteredSupplements}
                keyExtractor={(item) => item.id}
                renderItem={renderSupplementItem}
                contentContainerStyle={styles.supplementList}
                ItemSeparatorComponent={() => <View style={styles.separator} />}
                ListEmptyComponent={
                  <View style={styles.emptyList}>
                    <Text style={styles.emptyListText}>No supplements found</Text>
                  </View>
                }
              />
            )}
          </View>
        </View>
      </Modal>
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
  cancelButton: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
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

  // Picker
  pickerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    padding: spacing.md,
    minHeight: 56,
  },
  pickerPlaceholder: {
    fontSize: typography.fontSize.md,
    color: colors.text.disabled,
  },
  selectedSupplement: {
    flex: 1,
  },
  selectedSupplementName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  selectedSupplementCategory: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
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

  // Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: colors.background.primary,
    borderTopLeftRadius: borderRadius.xl,
    borderTopRightRadius: borderRadius.xl,
    maxHeight: '80%',
    paddingBottom: spacing.xl,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  modalTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    margin: spacing.md,
    paddingHorizontal: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  searchInput: {
    flex: 1,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.sm,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
  },
  supplementList: {
    paddingHorizontal: spacing.md,
  },
  supplementItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.md,
  },
  supplementItemInfo: {
    flex: 1,
  },
  supplementItemName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
  },
  supplementItemCategory: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  supplementItemDosage: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },
  separator: {
    height: 1,
    backgroundColor: colors.border.secondary,
  },
  emptyList: {
    padding: spacing.xl,
    alignItems: 'center',
  },
  emptyListText: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
  loader: {
    padding: spacing.xl,
  },
});
