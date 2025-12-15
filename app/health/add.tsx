import { useState } from 'react';
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
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import DateTimePicker from '@react-native-community/datetimepicker';
import { healthMetricsApi } from '@/lib/api/health-metrics';
import {
  CreateHealthMetricInput,
  HealthMetricType,
  METRIC_CONFIG,
  HEALTH_METRIC_TYPES,
  getMetricsByCategory,
  CATEGORY_DISPLAY_NAMES,
  validateMetricValue,
  MetricCategory,
} from '@/lib/types/health-metrics';
import { colors, gradients, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';

export default function AddHealthMetricScreen() {
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  const [metricType, setMetricType] = useState<HealthMetricType>('RESTING_HEART_RATE');
  const [value, setValue] = useState('');
  const [recordedAt, setRecordedAt] = useState(new Date());
  const [isLoading, setIsLoading] = useState(false);
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [showTimePicker, setShowTimePicker] = useState(false);
  const [showMetricPicker, setShowMetricPicker] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const config = METRIC_CONFIG[metricType];
  const metricsByCategory = getMetricsByCategory();

  const handleValueChange = (text: string) => {
    // Allow only numbers and decimal point
    const sanitized = text.replace(/[^0-9.]/g, '');
    setValue(sanitized);
    setError(null);
  };

  const validateForm = (): boolean => {
    if (!value || value.trim() === '') {
      setError('Value is required');
      return false;
    }

    const numValue = parseFloat(value);
    if (isNaN(numValue)) {
      setError('Please enter a valid number');
      return false;
    }

    const validation = validateMetricValue(metricType, numValue);
    if (!validation.valid) {
      setError(validation.error || 'Invalid value');
      return false;
    }

    return true;
  };

  const handleSave = async () => {
    if (!validateForm()) {
      return;
    }

    setIsLoading(true);
    try {
      const data: CreateHealthMetricInput = {
        metricType,
        value: parseFloat(value),
        unit: config.unit,
        recordedAt: recordedAt.toISOString(),
        source: 'manual',
      };

      await healthMetricsApi.create(data);

      showAlert('Success', 'Health metric added successfully!', [
        {
          text: 'OK',
          onPress: () => router.back(),
        },
      ]);
    } catch (err) {
      showAlert('Error', getErrorMessage(err, 'Failed to add health metric'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleDateChange = (_event: unknown, selectedDate?: Date) => {
    setShowDatePicker(false);
    if (selectedDate) {
      // Keep the time from recordedAt, just change the date
      const newDate = new Date(recordedAt);
      newDate.setFullYear(selectedDate.getFullYear());
      newDate.setMonth(selectedDate.getMonth());
      newDate.setDate(selectedDate.getDate());
      setRecordedAt(newDate);
    }
  };

  const handleTimeChange = (_event: unknown, selectedTime?: Date) => {
    setShowTimePicker(false);
    if (selectedTime) {
      // Keep the date from recordedAt, just change the time
      const newDate = new Date(recordedAt);
      newDate.setHours(selectedTime.getHours());
      newDate.setMinutes(selectedTime.getMinutes());
      setRecordedAt(newDate);
    }
  };

  const formatDate = (date: Date): string => {
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const renderMetricPickerModal = () => (
    <Modal
      visible={showMetricPicker}
      animationType="slide"
      transparent={true}
      onRequestClose={() => setShowMetricPicker(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Select Metric Type</Text>
            <TouchableOpacity
              onPress={() => setShowMetricPicker(false)}
              style={styles.modalCloseButton}
            >
              <Ionicons name="close" size={24} color={colors.text.primary} />
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.modalScrollView} showsVerticalScrollIndicator={false}>
            {(Object.keys(metricsByCategory) as MetricCategory[]).map((category) => {
              const metrics = metricsByCategory[category];
              if (metrics.length === 0) return null;

              return (
                <View key={category} style={styles.categorySection}>
                  <Text style={styles.categoryTitle}>
                    {CATEGORY_DISPLAY_NAMES[category]}
                  </Text>
                  {metrics.map((type) => {
                    const typeConfig = METRIC_CONFIG[type];
                    const isSelected = type === metricType;

                    return (
                      <TouchableOpacity
                        key={type}
                        style={[
                          styles.metricOption,
                          isSelected && styles.metricOptionSelected,
                        ]}
                        onPress={() => {
                          setMetricType(type);
                          setValue('');
                          setError(null);
                          setShowMetricPicker(false);
                        }}
                        activeOpacity={0.7}
                      >
                        <Ionicons
                          name={typeConfig.icon as keyof typeof Ionicons.glyphMap}
                          size={20}
                          color={isSelected ? colors.primary.main : colors.text.tertiary}
                        />
                        <View style={styles.metricOptionText}>
                          <Text
                            style={[
                              styles.metricOptionName,
                              isSelected && styles.metricOptionNameSelected,
                            ]}
                          >
                            {typeConfig.displayName}
                          </Text>
                          <Text style={styles.metricOptionUnit}>{typeConfig.unit}</Text>
                        </View>
                        {isSelected && (
                          <Ionicons
                            name="checkmark-circle"
                            size={20}
                            color={colors.primary.main}
                          />
                        )}
                      </TouchableOpacity>
                    );
                  })}
                </View>
              );
            })}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );

  return (
    <SafeAreaView style={styles.container} testID="add-health-metric-screen">
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} disabled={isLoading} testID="add-health-metric-cancel-button">
            <Text style={styles.cancelButton}>Cancel</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Add Health Metric</Text>
          <TouchableOpacity onPress={handleSave} disabled={isLoading} testID="add-health-metric-save-button">
            {isLoading ? (
              <ActivityIndicator color={colors.primary.main} />
            ) : (
              <Text style={styles.saveButton}>Save</Text>
            )}
          </TouchableOpacity>
        </View>

        <ScrollView
          style={styles.scrollView}
          showsVerticalScrollIndicator={false}
          contentContainerStyle={[
            styles.scrollContent,
            { paddingHorizontal: responsiveSpacing.horizontal },
            isTablet && styles.scrollContentTablet
          ]}
        >
          <View style={styles.content}>
            {/* Metric Type Selector */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Metric Type</Text>
              <TouchableOpacity
                style={styles.pickerButton}
                onPress={() => setShowMetricPicker(true)}
                disabled={isLoading}
                activeOpacity={0.8}
              >
                <Ionicons
                  name={config.icon as keyof typeof Ionicons.glyphMap}
                  size={20}
                  color={colors.primary.main}
                />
                <View style={styles.pickerButtonText}>
                  <Text style={styles.pickerButtonTitle}>{config.displayName}</Text>
                  <Text style={styles.pickerButtonSubtitle}>Unit: {config.unit}</Text>
                </View>
                <Ionicons name="chevron-down" size={20} color={colors.text.tertiary} />
              </TouchableOpacity>
            </View>

            {/* Value Input */}
            <View style={styles.section}>
              <Text style={styles.label}>
                Value ({config.unit}) *
              </Text>
              <View style={[styles.inputWrapper, error && styles.inputWrapperError]}>
                <TextInput
                  style={styles.input}
                  placeholder={`Enter ${config.shortName.toLowerCase()}`}
                  placeholderTextColor={colors.text.disabled}
                  value={value}
                  onChangeText={handleValueChange}
                  keyboardType="decimal-pad"
                  editable={!isLoading}
                />
              </View>
              {error && <Text style={styles.errorText}>{error}</Text>}
              {config.minValue !== undefined && config.maxValue !== undefined && (
                <Text style={styles.hintText}>
                  Valid range: {config.minValue} - {config.maxValue} {config.unit}
                </Text>
              )}
            </View>

            {/* Date Picker */}
            <View style={styles.section}>
              <Text style={styles.label}>Date</Text>
              <TouchableOpacity
                style={styles.dateButton}
                onPress={() => setShowDatePicker(true)}
                disabled={isLoading}
                activeOpacity={0.8}
              >
                <Ionicons name="calendar-outline" size={20} color={colors.primary.main} />
                <Text style={styles.dateButtonText}>{formatDate(recordedAt)}</Text>
                <Ionicons name="chevron-forward" size={16} color={colors.text.tertiary} />
              </TouchableOpacity>
            </View>

            {/* Time Picker */}
            <View style={styles.section}>
              <Text style={styles.label}>Time</Text>
              <TouchableOpacity
                style={styles.dateButton}
                onPress={() => setShowTimePicker(true)}
                disabled={isLoading}
                activeOpacity={0.8}
              >
                <Ionicons name="time-outline" size={20} color={colors.primary.main} />
                <Text style={styles.dateButtonText}>{formatTime(recordedAt)}</Text>
                <Ionicons name="chevron-forward" size={16} color={colors.text.tertiary} />
              </TouchableOpacity>
            </View>

            {/* Source Badge */}
            <View style={styles.section}>
              <Text style={styles.label}>Data Source</Text>
              <View style={styles.sourceBadge}>
                <Ionicons name="create-outline" size={16} color={colors.status.success} />
                <Text style={styles.sourceBadgeText}>Manual Entry</Text>
              </View>
            </View>

            {/* Description */}
            {config.description && (
              <View style={styles.descriptionCard}>
                <Ionicons name="information-circle-outline" size={20} color={colors.primary.main} />
                <Text style={styles.descriptionText}>{config.description}</Text>
              </View>
            )}
          </View>
        </ScrollView>

        {/* Date Picker Modal */}
        {showDatePicker && (
          <DateTimePicker
            value={recordedAt}
            mode="date"
            display={Platform.OS === 'ios' ? 'spinner' : 'default'}
            onChange={handleDateChange}
            maximumDate={new Date()}
          />
        )}

        {/* Time Picker Modal */}
        {showTimePicker && (
          <DateTimePicker
            value={recordedAt}
            mode="time"
            display={Platform.OS === 'ios' ? 'spinner' : 'default'}
            onChange={handleTimeChange}
          />
        )}

        {/* Metric Picker Modal */}
        {renderMetricPickerModal()}
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

  // Header
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

  // Content
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
    paddingBottom: spacing['3xl'],
  },
  section: {
    marginBottom: spacing.xl,
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

  // Picker Button
  pickerButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  pickerButtonText: {
    flex: 1,
    marginLeft: spacing.md,
  },
  pickerButtonTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  pickerButtonSubtitle: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },

  // Input
  inputWrapper: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    overflow: 'hidden',
  },
  inputWrapperError: {
    borderColor: colors.status.error,
  },
  input: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    height: 48,
  },
  errorText: {
    fontSize: typography.fontSize.xs,
    color: colors.status.error,
    marginTop: spacing.xs,
  },
  hintText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    marginTop: spacing.xs,
  },

  // Date Button
  dateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  dateButtonText: {
    flex: 1,
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    marginLeft: spacing.md,
  },

  // Source Badge
  sourceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.special.highlight,
    padding: spacing.md,
    borderRadius: borderRadius.sm,
    borderWidth: 1,
    borderColor: colors.status.success,
  },
  sourceBadgeText: {
    fontSize: typography.fontSize.sm,
    color: colors.status.success,
    marginLeft: spacing.sm,
    fontWeight: typography.fontWeight.medium,
  },

  // Description
  descriptionCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: colors.special.highlight,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.focus,
  },
  descriptionText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginLeft: spacing.sm,
    lineHeight: 20,
  },

  // Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: colors.overlay.medium,
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: colors.background.secondary,
    borderTopLeftRadius: borderRadius.xl,
    borderTopRightRadius: borderRadius.xl,
    maxHeight: '80%',
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
  modalCloseButton: {
    padding: spacing.xs,
  },
  modalScrollView: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing['3xl'],
  },

  // Category Section
  categorySection: {
    marginTop: spacing.lg,
    marginBottom: spacing.md,
  },
  categoryTitle: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: spacing.sm,
  },

  // Metric Option
  metricOption: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
    borderRadius: borderRadius.md,
    backgroundColor: colors.background.tertiary,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  metricOptionSelected: {
    borderColor: colors.primary.main,
    backgroundColor: colors.special.highlight,
  },
  metricOptionText: {
    flex: 1,
    marginLeft: spacing.md,
  },
  metricOptionName: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  metricOptionNameSelected: {
    fontWeight: typography.fontWeight.semibold,
  },
  metricOptionUnit: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
});
