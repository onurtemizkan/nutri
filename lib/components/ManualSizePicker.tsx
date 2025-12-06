/**
 * Manual Size Picker Component
 *
 * Fallback UI for devices without AR/LiDAR support.
 * Allows users to manually select portion size using presets or custom sliders.
 */

import React, { useState, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Animated,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import {
  colors,
  spacing,
  borderRadius,
  typography,
  shadows,
} from '@/lib/theme/colors';
import {
  PRESET_SIZES,
  presetToMeasurement,
  estimateWeight,
  formatDimensions,
  formatWeight,
  formatVolume,
  calculateVolume,
  type PresetSize,
} from '@/lib/utils/portion-estimation';
import type { ARMeasurement } from '@/lib/types/food-analysis';
import type { CalibrationResult } from './ReferenceObjectCalibration';

export interface ManualSizePickerProps {
  /** Food name (optional, used for weight estimation) */
  foodName?: string;
  /** Initial selection (optional) */
  initialSize?: string;
  /** Called when user confirms selection */
  onSelect: (measurement: ARMeasurement) => void;
  /** Called when user cancels */
  onCancel: () => void;
  /** Called when user wants to calibrate (optional) */
  onRequestCalibration?: () => void;
  /** Current calibration result (optional) */
  calibration?: CalibrationResult | null;
}

interface CustomDimensions {
  width: number;
  height: number;
  depth: number;
}

const MIN_DIMENSION = 1;
const MAX_DIMENSION = 30;

/**
 * Manual Size Picker Component
 */
export function ManualSizePicker({
  foodName = '',
  initialSize,
  onSelect,
  onCancel,
  onRequestCalibration,
  calibration,
}: ManualSizePickerProps): React.ReactElement {
  const [selectedPreset, setSelectedPreset] = useState<string | null>(
    initialSize || null
  );
  const [isCustomMode, setIsCustomMode] = useState(false);
  const [customDimensions, setCustomDimensions] = useState<CustomDimensions>({
    width: 8,
    height: 8,
    depth: 6,
  });

  // Calculate current measurement based on selection
  // Calibration improves confidence level
  const currentMeasurement = useMemo((): ARMeasurement | null => {
    // Determine confidence based on calibration state
    let confidence: 'high' | 'medium' | 'low' = 'low';
    if (calibration) {
      // Use calibration confidence level
      confidence = calibration.confidence;
    }

    if (isCustomMode) {
      return {
        width: customDimensions.width,
        height: customDimensions.height,
        depth: customDimensions.depth,
        distance: 30,
        confidence,
        planeDetected: false,
        timestamp: new Date(),
      };
    }

    if (selectedPreset) {
      const preset = PRESET_SIZES.find((p) => p.name === selectedPreset);
      if (preset) {
        const measurement = presetToMeasurement(preset);
        // Override confidence with calibration-based confidence
        return {
          ...measurement,
          confidence,
        };
      }
    }

    return null;
  }, [selectedPreset, isCustomMode, customDimensions, calibration]);

  // Calculate estimated weight
  const weightEstimate = useMemo(() => {
    if (!currentMeasurement) return null;

    return estimateWeight(
      currentMeasurement.width,
      currentMeasurement.height,
      currentMeasurement.depth,
      foodName
    );
  }, [currentMeasurement, foodName]);

  // Calculate volume
  const volume = useMemo(() => {
    if (!currentMeasurement) return 0;
    return calculateVolume(
      currentMeasurement.width,
      currentMeasurement.height,
      currentMeasurement.depth
    );
  }, [currentMeasurement]);

  const handlePresetSelect = useCallback((presetName: string) => {
    setSelectedPreset(presetName);
    setIsCustomMode(false);
  }, []);

  const handleCustomModeToggle = useCallback(() => {
    setIsCustomMode(true);
    setSelectedPreset(null);
  }, []);

  const handleDimensionChange = useCallback(
    (dimension: keyof CustomDimensions, value: number) => {
      const clampedValue = Math.max(MIN_DIMENSION, Math.min(MAX_DIMENSION, value));
      setCustomDimensions((prev) => ({
        ...prev,
        [dimension]: clampedValue,
      }));
    },
    []
  );

  const handleConfirm = useCallback(() => {
    if (currentMeasurement) {
      onSelect(currentMeasurement);
    }
  }, [currentMeasurement, onSelect]);

  const canConfirm = currentMeasurement !== null;

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={onCancel} style={styles.closeButton}>
          <Ionicons name="close" size={24} color={colors.text.secondary} />
        </TouchableOpacity>
        <Text style={styles.title}>Select Portion Size</Text>
        <View style={styles.placeholder} />
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Info Banner */}
        <View style={styles.infoBanner}>
          <Ionicons
            name="information-circle-outline"
            size={20}
            color={colors.secondary.main}
          />
          <Text style={styles.infoText}>
            Select a size that best matches your food portion. For more accurate
            measurements, use AR on a compatible device.
          </Text>
        </View>

        {/* Calibration Card */}
        {onRequestCalibration && (
          <TouchableOpacity
            style={[
              styles.calibrationCard,
              calibration && styles.calibrationCardActive,
            ]}
            onPress={onRequestCalibration}
            activeOpacity={0.7}
          >
            <View style={styles.calibrationIconContainer}>
              <Ionicons
                name={calibration ? 'checkmark-circle' : 'scan-outline'}
                size={24}
                color={calibration ? colors.status.success : colors.primary.main}
              />
            </View>
            <View style={styles.calibrationContent}>
              <Text style={styles.calibrationTitle}>
                {calibration ? 'Calibrated' : 'Calibrate for Accuracy'}
              </Text>
              <Text style={styles.calibrationSubtitle}>
                {calibration
                  ? `Using ${calibration.referenceObject.name} (${calibration.confidence} confidence)`
                  : 'Place a reference object for better size estimation'}
              </Text>
            </View>
            <Ionicons
              name="chevron-forward"
              size={20}
              color={colors.text.tertiary}
            />
          </TouchableOpacity>
        )}

        {/* Preset Size Cards */}
        <Text style={styles.sectionTitle}>Quick Select</Text>
        <View style={styles.presetsContainer}>
          {PRESET_SIZES.map((preset) => (
            <PresetCard
              key={preset.name}
              preset={preset}
              isSelected={selectedPreset === preset.name && !isCustomMode}
              onSelect={() => handlePresetSelect(preset.name)}
              foodName={foodName}
            />
          ))}
        </View>

        {/* Custom Size Option */}
        <Text style={styles.sectionTitle}>Custom Size</Text>
        <TouchableOpacity
          style={[
            styles.customToggle,
            isCustomMode && styles.customToggleActive,
          ]}
          onPress={handleCustomModeToggle}
          activeOpacity={0.7}
        >
          <Ionicons
            name="resize-outline"
            size={24}
            color={isCustomMode ? colors.primary.main : colors.text.tertiary}
          />
          <Text
            style={[
              styles.customToggleText,
              isCustomMode && styles.customToggleTextActive,
            ]}
          >
            Enter custom dimensions
          </Text>
          {isCustomMode && (
            <Ionicons name="checkmark-circle" size={20} color={colors.primary.main} />
          )}
        </TouchableOpacity>

        {/* Custom Sliders */}
        {isCustomMode && (
          <View style={styles.slidersContainer}>
            <DimensionSlider
              label="Width"
              value={customDimensions.width}
              onChange={(v) => handleDimensionChange('width', v)}
              min={MIN_DIMENSION}
              max={MAX_DIMENSION}
            />
            <DimensionSlider
              label="Height"
              value={customDimensions.height}
              onChange={(v) => handleDimensionChange('height', v)}
              min={MIN_DIMENSION}
              max={MAX_DIMENSION}
            />
            <DimensionSlider
              label="Depth"
              value={customDimensions.depth}
              onChange={(v) => handleDimensionChange('depth', v)}
              min={MIN_DIMENSION}
              max={MAX_DIMENSION}
            />
          </View>
        )}

        {/* Current Selection Summary */}
        {currentMeasurement && (
          <View style={styles.summaryContainer}>
            <Text style={styles.summaryTitle}>Selected Size</Text>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Dimensions:</Text>
              <Text style={styles.summaryValue}>
                {formatDimensions(
                  currentMeasurement.width,
                  currentMeasurement.height,
                  currentMeasurement.depth
                )}
              </Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Volume:</Text>
              <Text style={styles.summaryValue}>{formatVolume(volume)}</Text>
            </View>
            {weightEstimate && (
              <View style={styles.summaryRow}>
                <Text style={styles.summaryLabel}>Est. Weight:</Text>
                <Text style={styles.summaryValue}>
                  {formatWeight(weightEstimate.weight)}
                </Text>
              </View>
            )}
            <View style={styles.confidenceRow}>
              <Ionicons
                name={calibration ? 'checkmark-circle-outline' : 'alert-circle-outline'}
                size={14}
                color={calibration ? colors.status.success : colors.status.warning}
              />
              <Text
                style={[
                  styles.confidenceText,
                  calibration && styles.confidenceTextCalibrated,
                ]}
              >
                {calibration
                  ? `Calibrated with ${calibration.referenceObject.name}`
                  : 'Manual estimate - lower accuracy than AR measurement'}
              </Text>
            </View>
          </View>
        )}
      </ScrollView>

      {/* Footer with Confirm Button */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={[styles.confirmButton, !canConfirm && styles.confirmButtonDisabled]}
          onPress={handleConfirm}
          disabled={!canConfirm}
          activeOpacity={0.8}
        >
          <Text
            style={[
              styles.confirmButtonText,
              !canConfirm && styles.confirmButtonTextDisabled,
            ]}
          >
            Use This Size
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// =============================================================================
// Sub-components
// =============================================================================

interface PresetCardProps {
  preset: PresetSize;
  isSelected: boolean;
  onSelect: () => void;
  foodName: string;
}

function PresetCard({
  preset,
  isSelected,
  onSelect,
  foodName,
}: PresetCardProps): React.ReactElement {
  const volume = calculateVolume(preset.width, preset.height, preset.depth);
  const weightEstimate = estimateWeight(
    preset.width,
    preset.height,
    preset.depth,
    foodName
  );

  return (
    <TouchableOpacity
      style={[styles.presetCard, isSelected && styles.presetCardSelected]}
      onPress={onSelect}
      activeOpacity={0.7}
    >
      <View style={styles.presetHeader}>
        <Text
          style={[
            styles.presetName,
            isSelected && styles.presetNameSelected,
          ]}
        >
          {preset.displayName}
        </Text>
        {isSelected && (
          <Ionicons
            name="checkmark-circle"
            size={20}
            color={colors.primary.main}
          />
        )}
      </View>

      <Text style={styles.presetReference}>{preset.referenceObject}</Text>

      <View style={styles.presetDetails}>
        <View style={styles.presetDetailItem}>
          <Ionicons name="cube-outline" size={14} color={colors.text.tertiary} />
          <Text style={styles.presetDetailText}>{formatVolume(volume)}</Text>
        </View>
        <View style={styles.presetDetailItem}>
          <Ionicons name="scale-outline" size={14} color={colors.text.tertiary} />
          <Text style={styles.presetDetailText}>
            ~{formatWeight(weightEstimate.weight)}
          </Text>
        </View>
      </View>

      <Text style={styles.presetDescription}>{preset.description}</Text>
    </TouchableOpacity>
  );
}

interface DimensionSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
}

function DimensionSlider({
  label,
  value,
  onChange,
  min,
  max,
}: DimensionSliderProps): React.ReactElement {
  const handleDecrement = () => {
    if (value > min) {
      onChange(value - 1);
    }
  };

  const handleIncrement = () => {
    if (value < max) {
      onChange(value + 1);
    }
  };

  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <View style={styles.sliderContainer}>
      <View style={styles.sliderHeader}>
        <Text style={styles.sliderLabel}>{label}</Text>
        <Text style={styles.sliderValue}>{value} cm</Text>
      </View>

      <View style={styles.sliderControls}>
        <TouchableOpacity
          style={[styles.sliderButton, value <= min && styles.sliderButtonDisabled]}
          onPress={handleDecrement}
          disabled={value <= min}
        >
          <Ionicons
            name="remove"
            size={20}
            color={value <= min ? colors.text.disabled : colors.text.primary}
          />
        </TouchableOpacity>

        <View style={styles.sliderTrack}>
          <View
            style={[styles.sliderFill, { width: `${percentage}%` }]}
          />
          <View
            style={[
              styles.sliderThumb,
              { left: `${percentage}%` },
            ]}
          />
        </View>

        <TouchableOpacity
          style={[styles.sliderButton, value >= max && styles.sliderButtonDisabled]}
          onPress={handleIncrement}
          disabled={value >= max}
        >
          <Ionicons
            name="add"
            size={20}
            color={value >= max ? colors.text.disabled : colors.text.primary}
          />
        </TouchableOpacity>
      </View>
    </View>
  );
}

// =============================================================================
// Styles
// =============================================================================

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
    paddingTop: spacing.lg,
    paddingBottom: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  closeButton: {
    padding: spacing.xs,
  },
  title: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  placeholder: {
    width: 32,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: spacing.md,
    paddingBottom: spacing.xl,
  },
  infoBanner: {
    flexDirection: 'row',
    backgroundColor: colors.special.highlight,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginBottom: spacing.lg,
    gap: spacing.sm,
  },
  infoText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    lineHeight: 20,
  },
  calibrationCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    gap: spacing.md,
  },
  calibrationCardActive: {
    borderColor: colors.status.success,
    backgroundColor: colors.special.highlight,
  },
  calibrationIconContainer: {
    width: 44,
    height: 44,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.md,
  },
  calibrationContent: {
    flex: 1,
  },
  calibrationTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  calibrationSubtitle: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    lineHeight: 16,
  },
  sectionTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.md,
    marginTop: spacing.sm,
  },
  presetsContainer: {
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  presetCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    borderWidth: 2,
    borderColor: 'transparent',
    ...shadows.sm,
  },
  presetCardSelected: {
    borderColor: colors.primary.main,
    backgroundColor: colors.special.highlight,
  },
  presetHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  presetName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  presetNameSelected: {
    color: colors.primary.main,
  },
  presetReference: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.sm,
  },
  presetDetails: {
    flexDirection: 'row',
    gap: spacing.md,
    marginBottom: spacing.xs,
  },
  presetDetailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  presetDetailText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  presetDescription: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
  },
  customToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    gap: spacing.sm,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  customToggleActive: {
    borderColor: colors.primary.main,
    backgroundColor: colors.special.highlight,
  },
  customToggleText: {
    flex: 1,
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
  customToggleTextActive: {
    color: colors.text.primary,
  },
  slidersContainer: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginTop: spacing.md,
    gap: spacing.md,
  },
  sliderContainer: {
    gap: spacing.sm,
  },
  sliderHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  sliderLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },
  sliderValue: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },
  sliderControls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  sliderButton: {
    width: 36,
    height: 36,
    borderRadius: borderRadius.sm,
    backgroundColor: colors.background.elevated,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sliderButtonDisabled: {
    opacity: 0.5,
  },
  sliderTrack: {
    flex: 1,
    height: 8,
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.full,
    position: 'relative',
    overflow: 'visible',
  },
  sliderFill: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.full,
  },
  sliderThumb: {
    position: 'absolute',
    top: -6,
    width: 20,
    height: 20,
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.full,
    marginLeft: -10,
    borderWidth: 3,
    borderColor: colors.background.primary,
    ...shadows.md,
  },
  summaryContainer: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginTop: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  summaryTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  summaryLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  summaryValue: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  confidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginTop: spacing.sm,
    paddingTop: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  confidenceText: {
    fontSize: typography.fontSize.xs,
    color: colors.status.warning,
    flex: 1,
  },
  confidenceTextCalibrated: {
    color: colors.status.success,
  },
  footer: {
    padding: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
    backgroundColor: colors.background.secondary,
  },
  confirmButton: {
    backgroundColor: colors.primary.main,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    alignItems: 'center',
    ...shadows.md,
  },
  confirmButtonDisabled: {
    backgroundColor: colors.background.elevated,
  },
  confirmButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  confirmButtonTextDisabled: {
    color: colors.text.disabled,
  },
});

export default ManualSizePicker;
