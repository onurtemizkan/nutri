import { useState, useEffect, useCallback } from 'react';
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
import { Ionicons } from '@expo/vector-icons';
import DateTimePicker from '@react-native-community/datetimepicker';
import { healthMetricsApi } from '@/lib/api/health-metrics';
import {
  HealthMetric,
  UpdateHealthMetricInput,
  METRIC_CONFIG,
  SOURCE_CONFIG,
  validateMetricValue,
} from '@/lib/types/health-metrics';
import { getErrorMessage } from '@/lib/utils/errorHandling';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { showAlert } from '@/lib/utils/alert';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';

export default function EditHealthMetricScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  const [metric, setMetric] = useState<HealthMetric | null>(null);
  const [value, setValue] = useState('');
  const [recordedAt, setRecordedAt] = useState(new Date());
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [showTimePicker, setShowTimePicker] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadMetric = useCallback(async () => {
    if (!id) {
      showAlert('Error', 'Invalid metric ID');
      router.back();
      return;
    }

    try {
      const data = await healthMetricsApi.getById(id);
      setMetric(data);
      setValue(data.value.toString());
      setRecordedAt(new Date(data.recordedAt));
    } catch (err) {
      showAlert('Error', getErrorMessage(err, 'Failed to load health metric'));
      router.back();
    } finally {
      setIsLoading(false);
    }
  }, [id, router]);

  useEffect(() => {
    loadMetric();
  }, [loadMetric]);

  const config = metric ? METRIC_CONFIG[metric.metricType] : null;
  const sourceConfig = metric ? SOURCE_CONFIG[metric.source] : null;

  const handleValueChange = (text: string) => {
    const sanitized = text.replace(/[^0-9.]/g, '');
    setValue(sanitized);
    setError(null);
  };

  const validateForm = (): boolean => {
    if (!value || value.trim() === '' || !metric) {
      setError('Value is required');
      return false;
    }

    const numValue = parseFloat(value);
    if (isNaN(numValue)) {
      setError('Please enter a valid number');
      return false;
    }

    const validation = validateMetricValue(metric.metricType, numValue);
    if (!validation.valid) {
      setError(validation.error || 'Invalid value');
      return false;
    }

    return true;
  };

  const handleSave = async () => {
    if (!validateForm() || !metric || !config) {
      return;
    }

    setIsSaving(true);
    try {
      const data: UpdateHealthMetricInput = {
        value: parseFloat(value),
        unit: config.unit,
        recordedAt: recordedAt.toISOString(),
      };

      await healthMetricsApi.update(metric.id, data);
      router.back();
    } catch (err) {
      showAlert('Error', getErrorMessage(err, 'Failed to update health metric'));
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = () => {
    if (!metric || !config) return;

    showAlert(
      'Delete Metric',
      `Are you sure you want to delete this ${config.displayName} reading?`,
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await healthMetricsApi.delete(metric.id);
              router.back();
            } catch (err) {
              showAlert('Error', getErrorMessage(err, 'Failed to delete health metric'));
            }
          },
        },
      ]
    );
  };

  const handleDateChange = (_event: unknown, selectedDate?: Date) => {
    setShowDatePicker(false);
    if (selectedDate) {
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

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Loading metric...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (!metric || !config || !sourceConfig) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle-outline" size={64} color={colors.status.error} />
          <Text style={styles.errorTitle}>Metric not found</Text>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Text style={styles.backButtonText}>Go Back</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} testID="edit-health-metric-screen">
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => router.back()}
            disabled={isSaving}
            style={styles.headerBackButton}
            accessibilityLabel="Go back"
            accessibilityRole="button"
            testID="edit-health-metric-back-button"
          >
            <Ionicons name="chevron-back" size={24} color={colors.primary.main} />
            <Text style={styles.cancelButton}>Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Edit Metric</Text>
          <TouchableOpacity onPress={handleSave} disabled={isSaving} testID="edit-health-metric-save-button">
            {isSaving ? (
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
            {/* Metric Type Display (read-only) */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Metric Type</Text>
              <View style={styles.metricTypeDisplay}>
                <Ionicons
                  name={config.icon as keyof typeof Ionicons.glyphMap}
                  size={20}
                  color={colors.primary.main}
                />
                <View style={styles.metricTypeText}>
                  <Text style={styles.metricTypeName}>{config.displayName}</Text>
                  <Text style={styles.metricTypeUnit}>Unit: {config.unit}</Text>
                </View>
              </View>
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
                  editable={!isSaving}
                  testID="edit-health-metric-value-input"
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
                disabled={isSaving}
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
                disabled={isSaving}
                activeOpacity={0.8}
              >
                <Ionicons name="time-outline" size={20} color={colors.primary.main} />
                <Text style={styles.dateButtonText}>{formatTime(recordedAt)}</Text>
                <Ionicons name="chevron-forward" size={16} color={colors.text.tertiary} />
              </TouchableOpacity>
            </View>

            {/* Source Badge (read-only) */}
            <View style={styles.section}>
              <Text style={styles.label}>Data Source</Text>
              <View style={styles.sourceBadge}>
                <Ionicons
                  name={sourceConfig.icon as keyof typeof Ionicons.glyphMap}
                  size={16}
                  color={colors.text.tertiary}
                />
                <Text style={styles.sourceBadgeText}>{sourceConfig.displayName}</Text>
              </View>
            </View>

            {/* Delete Button */}
            <TouchableOpacity
              style={styles.deleteButton}
              onPress={handleDelete}
              disabled={isSaving}
              activeOpacity={0.8}
            >
              <Ionicons name="trash-outline" size={20} color={colors.status.error} />
              <Text style={styles.deleteButtonText}>Delete This Metric</Text>
            </TouchableOpacity>
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
  loadingText: {
    marginTop: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
  },
  errorTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginTop: spacing.lg,
    marginBottom: spacing.lg,
  },
  backButton: {
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.md,
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.md,
  },
  backButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
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
  headerBackButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: -spacing.xs,
  },
  cancelButton: {
    fontSize: typography.fontSize.md,
    color: colors.primary.main,
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

  // Metric Type Display
  metricTypeDisplay: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  metricTypeText: {
    flex: 1,
    marginLeft: spacing.md,
  },
  metricTypeName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  metricTypeUnit: {
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
    backgroundColor: colors.background.tertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  sourceBadgeText: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
    marginLeft: spacing.sm,
  },

  // Delete Button
  deleteButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background.tertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.status.error,
    marginTop: spacing.xl,
  },
  deleteButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.status.error,
    marginLeft: spacing.sm,
  },
});
