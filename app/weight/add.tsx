import { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import DateTimePicker from '@react-native-community/datetimepicker';
import { weightApi } from '@/lib/api/weight';
import { WeightUnit, kgToLb, lbToKg } from '@/lib/types/weight';
import { colors, gradients, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';
import { getErrorMessage } from '@/lib/utils/errorHandling';

export default function AddWeightScreen() {
  const router = useRouter();
  const [weightInput, setWeightInput] = useState('');
  const [unit, setUnit] = useState<WeightUnit>('kg');
  const [recordedAt, setRecordedAt] = useState(new Date());
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [showTimePicker, setShowTimePicker] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSave = async () => {
    const weightValue = parseFloat(weightInput);

    if (isNaN(weightValue) || weightValue <= 0) {
      showAlert('Invalid Weight', 'Please enter a valid weight value');
      return;
    }

    // Convert to kg if needed
    const weightInKg = unit === 'lb' ? lbToKg(weightValue) : weightValue;

    // Validate weight range
    if (weightInKg < 20 || weightInKg > 500) {
      showAlert('Invalid Weight', 'Weight must be between 20 kg and 500 kg');
      return;
    }

    setIsSubmitting(true);

    try {
      await weightApi.create({
        weight: weightInKg,
        recordedAt: recordedAt.toISOString(),
      });
      router.back();
    } catch (error) {
      showAlert('Error', getErrorMessage(error, 'Failed to save weight record'));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDateChange = (event: unknown, selectedDate?: Date) => {
    setShowDatePicker(false);
    if (selectedDate) {
      // Preserve time, update date
      const newDate = new Date(recordedAt);
      newDate.setFullYear(selectedDate.getFullYear());
      newDate.setMonth(selectedDate.getMonth());
      newDate.setDate(selectedDate.getDate());
      setRecordedAt(newDate);
    }
  };

  const handleTimeChange = (event: unknown, selectedTime?: Date) => {
    setShowTimePicker(false);
    if (selectedTime) {
      // Preserve date, update time
      const newDate = new Date(recordedAt);
      newDate.setHours(selectedTime.getHours());
      newDate.setMinutes(selectedTime.getMinutes());
      setRecordedAt(newDate);
    }
  };

  const formatDate = (date: Date) => {
    const today = new Date();
    const isToday =
      date.getDate() === today.getDate() &&
      date.getMonth() === today.getMonth() &&
      date.getFullYear() === today.getFullYear();

    if (isToday) return 'Today';

    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const isYesterday =
      date.getDate() === yesterday.getDate() &&
      date.getMonth() === yesterday.getMonth() &&
      date.getFullYear() === yesterday.getFullYear();

    if (isYesterday) return 'Yesterday';

    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: date.getFullYear() !== today.getFullYear() ? 'numeric' : undefined,
    });
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Header */}
          <View style={styles.header}>
            <TouchableOpacity
              onPress={() => router.back()}
              style={styles.closeButton}
              accessibilityLabel="Close"
            >
              <Ionicons name="close" size={24} color={colors.text.primary} />
            </TouchableOpacity>
            <Text style={styles.title}>Add Weight</Text>
            <View style={styles.placeholder} />
          </View>

          {/* Weight Input Section */}
          <View style={styles.inputSection}>
            <View style={styles.weightInputContainer}>
              <TextInput
                style={styles.weightInput}
                value={weightInput}
                onChangeText={setWeightInput}
                placeholder="0.0"
                placeholderTextColor={colors.text.disabled}
                keyboardType="decimal-pad"
                autoFocus
                maxLength={6}
              />
              <View style={styles.unitToggle}>
                <TouchableOpacity
                  style={[styles.unitButton, unit === 'kg' && styles.unitButtonActive]}
                  onPress={() => setUnit('kg')}
                >
                  <Text style={[styles.unitText, unit === 'kg' && styles.unitTextActive]}>kg</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.unitButton, unit === 'lb' && styles.unitButtonActive]}
                  onPress={() => setUnit('lb')}
                >
                  <Text style={[styles.unitText, unit === 'lb' && styles.unitTextActive]}>lb</Text>
                </TouchableOpacity>
              </View>
            </View>

            {/* Quick Adjust Buttons */}
            <View style={styles.quickAdjustContainer}>
              {[-1, -0.5, 0.5, 1].map((amount) => (
                <TouchableOpacity
                  key={amount}
                  style={styles.quickAdjustButton}
                  onPress={() => {
                    const current = parseFloat(weightInput) || 0;
                    const adjusted = Math.max(0, current + amount);
                    setWeightInput(adjusted.toFixed(1));
                  }}
                >
                  <Text style={styles.quickAdjustText}>
                    {amount > 0 ? '+' : ''}
                    {amount}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Date/Time Selection */}
          <View style={styles.dateTimeSection}>
            <Text style={styles.sectionLabel}>When</Text>

            <TouchableOpacity style={styles.dateTimeButton} onPress={() => setShowDatePicker(true)}>
              <View style={styles.dateTimeIconContainer}>
                <Ionicons name="calendar-outline" size={20} color={colors.primary.main} />
              </View>
              <View style={styles.dateTimeInfo}>
                <Text style={styles.dateTimeLabel}>Date</Text>
                <Text style={styles.dateTimeValue}>{formatDate(recordedAt)}</Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color={colors.text.disabled} />
            </TouchableOpacity>

            <TouchableOpacity style={styles.dateTimeButton} onPress={() => setShowTimePicker(true)}>
              <View style={styles.dateTimeIconContainer}>
                <Ionicons name="time-outline" size={20} color={colors.primary.main} />
              </View>
              <View style={styles.dateTimeInfo}>
                <Text style={styles.dateTimeLabel}>Time</Text>
                <Text style={styles.dateTimeValue}>{formatTime(recordedAt)}</Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color={colors.text.disabled} />
            </TouchableOpacity>
          </View>

          {/* Date/Time Pickers */}
          {showDatePicker && (
            <DateTimePicker
              value={recordedAt}
              mode="date"
              display={Platform.OS === 'ios' ? 'spinner' : 'default'}
              onChange={handleDateChange}
              maximumDate={new Date()}
              themeVariant="dark"
            />
          )}

          {showTimePicker && (
            <DateTimePicker
              value={recordedAt}
              mode="time"
              display={Platform.OS === 'ios' ? 'spinner' : 'default'}
              onChange={handleTimeChange}
              themeVariant="dark"
            />
          )}
        </ScrollView>

        {/* Save Button */}
        <View style={styles.footer}>
          <TouchableOpacity
            style={styles.saveButton}
            onPress={handleSave}
            disabled={isSubmitting || !weightInput}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={
                isSubmitting || !weightInput
                  ? [colors.border.secondary, colors.border.secondary]
                  : gradients.primary
              }
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.saveButtonGradient}
            >
              {isSubmitting ? (
                <ActivityIndicator color={colors.text.primary} />
              ) : (
                <Text style={styles.saveButtonText}>Save Weight</Text>
              )}
            </LinearGradient>
          </TouchableOpacity>
        </View>
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
    padding: spacing.lg,
  },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.xl,
  },
  closeButton: {
    padding: spacing.xs,
  },
  title: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  placeholder: {
    width: 32,
  },

  // Weight Input
  inputSection: {
    marginBottom: spacing.xl,
  },
  weightInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.lg,
  },
  weightInput: {
    fontSize: 64,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    textAlign: 'center',
    minWidth: 150,
  },
  unitToggle: {
    flexDirection: 'row',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.xs,
    marginLeft: spacing.md,
  },
  unitButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: borderRadius.sm,
  },
  unitButtonActive: {
    backgroundColor: colors.primary.main,
  },
  unitText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
  },
  unitTextActive: {
    color: colors.text.primary,
  },

  // Quick Adjust
  quickAdjustContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.md,
  },
  quickAdjustButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  quickAdjustText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
  },

  // Date/Time Section
  dateTimeSection: {
    marginBottom: spacing.xl,
  },
  sectionLabel: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: spacing.md,
  },
  dateTimeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  dateTimeIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  dateTimeInfo: {
    flex: 1,
  },
  dateTimeLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  dateTimeValue: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Footer
  footer: {
    padding: spacing.lg,
    paddingBottom: spacing.xl,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  saveButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  saveButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 50,
  },
  saveButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
});
