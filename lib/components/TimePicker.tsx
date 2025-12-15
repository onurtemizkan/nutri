import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Platform,
  Modal,
} from 'react-native';
import DateTimePicker, { DateTimePickerEvent } from '@react-native-community/datetimepicker';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { formatMealTime } from '@/lib/utils/formatters';

interface TimePickerProps {
  value: Date;
  onChange: (date: Date) => void;
  label?: string;
  disabled?: boolean;
  testID?: string;
}

export function TimePicker({
  value,
  onChange,
  label,
  disabled = false,
  testID,
}: TimePickerProps) {
  const [showPicker, setShowPicker] = useState(false);
  // For iOS modal, we need a temp value to confirm/cancel
  const [tempDate, setTempDate] = useState(value);

  const handlePress = () => {
    if (disabled) return;
    setTempDate(value);
    setShowPicker(true);
  };

  const handleChange = (event: DateTimePickerEvent, selectedDate?: Date) => {
    if (Platform.OS === 'android') {
      // Android closes automatically
      setShowPicker(false);
      if (event.type === 'set' && selectedDate) {
        onChange(selectedDate);
      }
    } else {
      // iOS - update temp value
      if (selectedDate) {
        setTempDate(selectedDate);
      }
    }
  };

  const handleIOSConfirm = () => {
    onChange(tempDate);
    setShowPicker(false);
  };

  const handleIOSCancel = () => {
    setShowPicker(false);
  };

  return (
    <View style={styles.container}>
      {label && <Text style={styles.label}>{label}</Text>}

      <TouchableOpacity
        style={[styles.inputWrapper, disabled && styles.inputWrapperDisabled]}
        onPress={handlePress}
        disabled={disabled}
        activeOpacity={0.7}
        accessibilityLabel={`Select time, current value ${formatMealTime(value)}`}
        accessibilityRole="button"
        testID={testID}
      >
        <View style={styles.inputContent}>
          <Ionicons
            name="time-outline"
            size={20}
            color={disabled ? colors.text.disabled : colors.text.tertiary}
          />
          <Text style={[styles.timeText, disabled && styles.timeTextDisabled]}>
            {formatMealTime(value)}
          </Text>
        </View>
        <Ionicons
          name="chevron-down"
          size={16}
          color={disabled ? colors.text.disabled : colors.text.tertiary}
        />
      </TouchableOpacity>

      {/* Android: Native picker shows as dialog */}
      {Platform.OS === 'android' && showPicker && (
        <DateTimePicker
          value={value}
          mode="time"
          display="default"
          onChange={handleChange}
          testID={testID ? `${testID}-picker` : undefined}
        />
      )}

      {/* iOS: Show in modal for better UX */}
      {Platform.OS === 'ios' && (
        <Modal
          visible={showPicker}
          transparent
          animationType="slide"
          onRequestClose={handleIOSCancel}
        >
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <View style={styles.modalHeader}>
                <TouchableOpacity
                  onPress={handleIOSCancel}
                  style={styles.modalButton}
                  accessibilityLabel="Cancel"
                  accessibilityRole="button"
                >
                  <Text style={styles.modalButtonTextCancel}>Cancel</Text>
                </TouchableOpacity>
                <Text style={styles.modalTitle}>Select Time</Text>
                <TouchableOpacity
                  onPress={handleIOSConfirm}
                  style={styles.modalButton}
                  accessibilityLabel="Confirm"
                  accessibilityRole="button"
                >
                  <Text style={styles.modalButtonTextConfirm}>Done</Text>
                </TouchableOpacity>
              </View>
              <DateTimePicker
                value={tempDate}
                mode="time"
                display="spinner"
                onChange={handleChange}
                style={styles.iosPicker}
                textColor={colors.text.primary}
                testID={testID ? `${testID}-picker` : undefined}
              />
            </View>
          </View>
        </Modal>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.md,
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
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: 48,
  },
  inputWrapperDisabled: {
    opacity: 0.6,
  },
  inputContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  timeText: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    fontWeight: typography.fontWeight.medium,
  },
  timeTextDisabled: {
    color: colors.text.disabled,
  },

  // iOS Modal Styles
  modalOverlay: {
    flex: 1,
    backgroundColor: colors.overlay.light,
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: colors.background.secondary,
    borderTopLeftRadius: borderRadius.xl,
    borderTopRightRadius: borderRadius.xl,
    paddingBottom: spacing['2xl'],
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  modalTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  modalButton: {
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.sm,
  },
  modalButtonTextCancel: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
  modalButtonTextConfirm: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },
  iosPicker: {
    height: 200,
    backgroundColor: colors.background.secondary,
  },
});
