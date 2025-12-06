/**
 * Reference Object Calibration Component
 *
 * Allows users to calibrate measurements by placing a known reference object
 * (coin, credit card, etc.) in the frame. This improves measurement accuracy
 * especially when AR/LiDAR is unavailable.
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import {
  colors,
  spacing,
  borderRadius,
  typography,
} from '@/lib/theme/colors';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

/**
 * Reference object with known real-world dimensions
 */
export interface ReferenceObject {
  id: string;
  name: string;
  description: string;
  /** Width in centimeters */
  widthCm: number;
  /** Height in centimeters */
  heightCm: number;
  icon: keyof typeof Ionicons.glyphMap;
  category: 'coin' | 'card' | 'device' | 'other';
}

/**
 * Calibration result with scale factor
 */
export interface CalibrationResult {
  referenceObject: ReferenceObject;
  /** Pixels per centimeter (horizontal) */
  pixelsPerCmX: number;
  /** Pixels per centimeter (vertical) */
  pixelsPerCmY: number;
  /** Average scale factor */
  scaleFactor: number;
  /** Timestamp of calibration */
  timestamp: Date;
  /** Confidence based on aspect ratio match */
  confidence: 'high' | 'medium' | 'low';
}

/**
 * Common reference objects with known dimensions
 */
export const REFERENCE_OBJECTS: ReferenceObject[] = [
  // Coins
  {
    id: 'us_quarter',
    name: 'US Quarter',
    description: '24.26mm diameter',
    widthCm: 2.426,
    heightCm: 2.426,
    icon: 'ellipse',
    category: 'coin',
  },
  {
    id: 'us_dime',
    name: 'US Dime',
    description: '17.91mm diameter',
    widthCm: 1.791,
    heightCm: 1.791,
    icon: 'ellipse-outline',
    category: 'coin',
  },
  {
    id: 'us_nickel',
    name: 'US Nickel',
    description: '21.21mm diameter',
    widthCm: 2.121,
    heightCm: 2.121,
    icon: 'ellipse',
    category: 'coin',
  },
  {
    id: 'euro_1',
    name: 'Euro 1',
    description: '23.25mm diameter',
    widthCm: 2.325,
    heightCm: 2.325,
    icon: 'ellipse',
    category: 'coin',
  },
  // Cards
  {
    id: 'credit_card',
    name: 'Credit Card',
    description: 'Standard ISO/IEC 7810 ID-1',
    widthCm: 8.56,
    heightCm: 5.398,
    icon: 'card',
    category: 'card',
  },
  {
    id: 'drivers_license',
    name: "Driver's License",
    description: 'Standard US format',
    widthCm: 8.56,
    heightCm: 5.398,
    icon: 'id-card',
    category: 'card',
  },
  // Devices
  {
    id: 'iphone_15_pro',
    name: 'iPhone 15 Pro',
    description: '146.6 x 70.6mm',
    widthCm: 7.06,
    heightCm: 14.66,
    icon: 'phone-portrait',
    category: 'device',
  },
  {
    id: 'iphone_15',
    name: 'iPhone 15',
    description: '147.6 x 71.6mm',
    widthCm: 7.16,
    heightCm: 14.76,
    icon: 'phone-portrait',
    category: 'device',
  },
  // Other
  {
    id: 'us_dollar_bill',
    name: 'US Dollar Bill',
    description: '155.81 x 66.42mm',
    widthCm: 15.581,
    heightCm: 6.642,
    icon: 'cash',
    category: 'other',
  },
  {
    id: 'aa_battery',
    name: 'AA Battery',
    description: '50.5 x 14.5mm',
    widthCm: 1.45,
    heightCm: 5.05,
    icon: 'battery-half',
    category: 'other',
  },
];

export interface ReferenceObjectCalibrationProps {
  /** Called when user completes calibration */
  onCalibrate: (result: CalibrationResult) => void;
  /** Called when user cancels */
  onCancel: () => void;
  /** Width of the reference object box drawn by user (in pixels) */
  drawnWidth?: number;
  /** Height of the reference object box drawn by user (in pixels) */
  drawnHeight?: number;
  /** Pre-selected reference object ID */
  initialObjectId?: string;
}

/**
 * Categories for grouping reference objects
 */
const CATEGORIES = [
  { id: 'coin', label: 'Coins', icon: 'ellipse' as const },
  { id: 'card', label: 'Cards', icon: 'card' as const },
  { id: 'device', label: 'Devices', icon: 'phone-portrait' as const },
  { id: 'other', label: 'Other', icon: 'cube' as const },
];

/**
 * Calculate calibration result from reference object and drawn dimensions
 */
export function calculateCalibration(
  referenceObject: ReferenceObject,
  drawnWidthPx: number,
  drawnHeightPx: number
): CalibrationResult {
  const pixelsPerCmX = drawnWidthPx / referenceObject.widthCm;
  const pixelsPerCmY = drawnHeightPx / referenceObject.heightCm;
  const scaleFactor = (pixelsPerCmX + pixelsPerCmY) / 2;

  // Check aspect ratio match for confidence
  const expectedAspectRatio = referenceObject.widthCm / referenceObject.heightCm;
  const actualAspectRatio = drawnWidthPx / drawnHeightPx;
  const aspectRatioDiff = Math.abs(expectedAspectRatio - actualAspectRatio);

  let confidence: 'high' | 'medium' | 'low';
  if (aspectRatioDiff < 0.1) {
    confidence = 'high';
  } else if (aspectRatioDiff < 0.25) {
    confidence = 'medium';
  } else {
    confidence = 'low';
  }

  return {
    referenceObject,
    pixelsPerCmX,
    pixelsPerCmY,
    scaleFactor,
    timestamp: new Date(),
    confidence,
  };
}

/**
 * Convert pixel measurement to centimeters using calibration
 */
export function pixelsToCm(pixels: number, scaleFactor: number): number {
  return pixels / scaleFactor;
}

/**
 * Convert centimeters to pixels using calibration
 */
export function cmToPixels(cm: number, scaleFactor: number): number {
  return cm * scaleFactor;
}

export function ReferenceObjectCalibration({
  onCalibrate,
  onCancel,
  drawnWidth = 0,
  drawnHeight = 0,
  initialObjectId,
}: ReferenceObjectCalibrationProps): React.ReactElement {
  const [selectedCategory, setSelectedCategory] = useState<string>('coin');
  const [selectedObject, setSelectedObject] = useState<ReferenceObject | null>(
    initialObjectId
      ? REFERENCE_OBJECTS.find((obj) => obj.id === initialObjectId) || null
      : null
  );

  const filteredObjects = REFERENCE_OBJECTS.filter(
    (obj) => obj.category === selectedCategory
  );

  const handleSelectObject = useCallback((object: ReferenceObject) => {
    setSelectedObject(object);
  }, []);

  const handleConfirm = useCallback(() => {
    if (!selectedObject) return;

    // If no drawn dimensions provided, use default screen-based estimate
    const width = drawnWidth || SCREEN_WIDTH * 0.3;
    const height = drawnHeight || SCREEN_WIDTH * 0.3;

    const result = calculateCalibration(selectedObject, width, height);
    onCalibrate(result);
  }, [selectedObject, drawnWidth, drawnHeight, onCalibrate]);

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={onCancel} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.title}>Calibrate with Reference</Text>
        <View style={styles.placeholder} />
      </View>

      {/* Instructions */}
      <View style={styles.instructions}>
        <Ionicons name="information-circle" size={24} color={colors.primary.main} />
        <Text style={styles.instructionsText}>
          Place a reference object next to your food for more accurate size estimation
        </Text>
      </View>

      {/* Category tabs */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.categoryScroll}
        contentContainerStyle={styles.categoryContainer}
      >
        {CATEGORIES.map((category) => (
          <TouchableOpacity
            key={category.id}
            style={[
              styles.categoryTab,
              selectedCategory === category.id && styles.categoryTabActive,
            ]}
            onPress={() => setSelectedCategory(category.id)}
          >
            <Ionicons
              name={category.icon}
              size={18}
              color={
                selectedCategory === category.id
                  ? colors.text.primary
                  : colors.text.tertiary
              }
            />
            <Text
              style={[
                styles.categoryText,
                selectedCategory === category.id && styles.categoryTextActive,
              ]}
            >
              {category.label}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Object list */}
      <ScrollView style={styles.objectList}>
        {filteredObjects.map((object) => (
          <TouchableOpacity
            key={object.id}
            style={[
              styles.objectItem,
              selectedObject?.id === object.id && styles.objectItemSelected,
            ]}
            onPress={() => handleSelectObject(object)}
          >
            <View style={styles.objectIcon}>
              <Ionicons
                name={object.icon}
                size={28}
                color={
                  selectedObject?.id === object.id
                    ? colors.primary.main
                    : colors.text.secondary
                }
              />
            </View>
            <View style={styles.objectInfo}>
              <Text
                style={[
                  styles.objectName,
                  selectedObject?.id === object.id && styles.objectNameSelected,
                ]}
              >
                {object.name}
              </Text>
              <Text style={styles.objectDescription}>{object.description}</Text>
            </View>
            {selectedObject?.id === object.id && (
              <Ionicons
                name="checkmark-circle"
                size={24}
                color={colors.primary.main}
              />
            )}
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Selected object preview */}
      {selectedObject && (
        <View style={styles.preview}>
          <Text style={styles.previewTitle}>Selected Reference</Text>
          <View style={styles.previewContent}>
            <View style={styles.previewDimensions}>
              <Text style={styles.previewLabel}>Width</Text>
              <Text style={styles.previewValue}>
                {selectedObject.widthCm.toFixed(2)} cm
              </Text>
            </View>
            <View style={styles.previewDivider} />
            <View style={styles.previewDimensions}>
              <Text style={styles.previewLabel}>Height</Text>
              <Text style={styles.previewValue}>
                {selectedObject.heightCm.toFixed(2)} cm
              </Text>
            </View>
          </View>
        </View>
      )}

      {/* Confirm button */}
      <View style={styles.footer}>
        <TouchableOpacity
          style={[
            styles.confirmButton,
            !selectedObject && styles.confirmButtonDisabled,
          ]}
          onPress={handleConfirm}
          disabled={!selectedObject}
        >
          <Ionicons
            name="checkmark"
            size={20}
            color={selectedObject ? colors.text.primary : colors.text.tertiary}
          />
          <Text
            style={[
              styles.confirmButtonText,
              !selectedObject && styles.confirmButtonTextDisabled,
            ]}
          >
            Use This Reference
          </Text>
        </TouchableOpacity>
      </View>
    </View>
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
  title: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  placeholder: {
    width: 40,
  },
  instructions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    padding: spacing.md,
    backgroundColor: colors.special.highlight,
    marginHorizontal: spacing.md,
    marginTop: spacing.md,
    borderRadius: borderRadius.md,
  },
  instructionsText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    lineHeight: 20,
  },
  categoryScroll: {
    flexGrow: 0,
    marginTop: spacing.md,
  },
  categoryContainer: {
    paddingHorizontal: spacing.md,
    gap: spacing.sm,
  },
  categoryTab: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.full,
    backgroundColor: colors.background.tertiary,
  },
  categoryTabActive: {
    backgroundColor: colors.primary.main,
  },
  categoryText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    fontWeight: typography.fontWeight.medium,
  },
  categoryTextActive: {
    color: colors.text.primary,
  },
  objectList: {
    flex: 1,
    marginTop: spacing.md,
    paddingHorizontal: spacing.md,
  },
  objectItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    marginBottom: spacing.sm,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  objectItemSelected: {
    borderColor: colors.primary.main,
    backgroundColor: colors.special.highlight,
  },
  objectIcon: {
    width: 48,
    height: 48,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.md,
    marginRight: spacing.md,
  },
  objectInfo: {
    flex: 1,
  },
  objectName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  objectNameSelected: {
    color: colors.primary.main,
  },
  objectDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  preview: {
    marginHorizontal: spacing.md,
    marginBottom: spacing.md,
    padding: spacing.md,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.primary.main,
  },
  previewTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
    marginBottom: spacing.sm,
  },
  previewContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  previewDimensions: {
    alignItems: 'center',
    flex: 1,
  },
  previewLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  previewValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  previewDivider: {
    width: 1,
    height: 40,
    backgroundColor: colors.border.secondary,
    marginHorizontal: spacing.lg,
  },
  footer: {
    padding: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  confirmButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.md,
  },
  confirmButtonDisabled: {
    backgroundColor: colors.background.tertiary,
  },
  confirmButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  confirmButtonTextDisabled: {
    color: colors.text.tertiary,
  },
});

export default ReferenceObjectCalibration;
