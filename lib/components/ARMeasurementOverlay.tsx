/**
 * AR Measurement Overlay Component
 *
 * Interactive overlay for measuring food dimensions using AR/LiDAR.
 * Users tap to set measurement points, and the component calculates
 * real-world dimensions using depth data.
 *
 * Features:
 * - Tap-to-place measurement points
 * - Visual guides for measurement
 * - Real-time dimension calculations
 * - Depth-based distance estimation
 * - Plane detection feedback
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  PanResponder,
  Animated,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import {
  colors,
  spacing,
  borderRadius,
  typography,
} from '@/lib/theme/colors';
import LiDARModule from '@/lib/modules/LiDARModule';
import type { ARMeasurement } from '@/lib/types/food-analysis';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

/**
 * Measurement point on screen
 */
interface MeasurementPoint {
  id: string;
  x: number; // Screen X coordinate
  y: number; // Screen Y coordinate
  depth: number; // Estimated depth in cm (from LiDAR or estimation)
  worldX?: number; // Real-world X coordinate in cm
  worldY?: number; // Real-world Y coordinate in cm
  worldZ?: number; // Real-world Z coordinate in cm
}

/**
 * Measurement state - tracks the corners of the food bounding box
 */
interface MeasurementState {
  /** First corner point (top-left of food) */
  point1: MeasurementPoint | null;
  /** Second corner point (bottom-right of food) */
  point2: MeasurementPoint | null;
  /** Back point for depth measurement */
  point3: MeasurementPoint | null;
  /** Current step in measurement flow */
  step: 'place-first' | 'place-second' | 'place-depth' | 'complete';
  /** Is plane detected for reliable measurement */
  planeDetected: boolean;
  /** Average depth to food */
  averageDepth: number;
}

export interface ARMeasurementOverlayProps {
  /** Whether LiDAR/depth is available */
  hasLiDAR: boolean;
  /** Called when measurement is complete */
  onMeasurementComplete: (measurement: ARMeasurement) => void;
  /** Called when measurement is cancelled */
  onCancel: () => void;
  /** Optional: function to get depth at screen coordinate */
  getDepthAtPoint?: (x: number, y: number) => Promise<number>;
  /** Whether the view is active */
  isActive: boolean;
}

/**
 * Default depth estimation based on typical phone distance
 */
const DEFAULT_DEPTH_CM = 30;

/**
 * Pixels per cm at typical phone distance (rough estimate)
 * This is calibrated for typical arm's length viewing distance
 */
const PIXELS_PER_CM_AT_30CM = 12;

/**
 * Calculate real-world distance from screen pixels
 */
function screenToWorld(
  pixelDistance: number,
  depthCm: number
): number {
  // Scale factor increases with depth
  const scaleFactor = depthCm / 30; // Normalized to 30cm reference distance
  return (pixelDistance / PIXELS_PER_CM_AT_30CM) * scaleFactor;
}

/**
 * AR Measurement Overlay Component
 */
export function ARMeasurementOverlay({
  hasLiDAR,
  onMeasurementComplete,
  onCancel,
  getDepthAtPoint,
  isActive,
}: ARMeasurementOverlayProps): React.ReactElement {
  const [measurementState, setMeasurementState] = useState<MeasurementState>({
    point1: null,
    point2: null,
    point3: null,
    step: 'place-first',
    planeDetected: false,
    averageDepth: DEFAULT_DEPTH_CM,
  });

  // Animated values for point indicators
  const point1Scale = useRef(new Animated.Value(0)).current;
  const point2Scale = useRef(new Animated.Value(0)).current;
  const point3Scale = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Start pulse animation for guidance
  useEffect(() => {
    if (isActive && measurementState.step !== 'complete') {
      const pulse = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.2,
            duration: 800,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 800,
            useNativeDriver: true,
          }),
        ])
      );
      pulse.start();
      return () => pulse.stop();
    }
  }, [isActive, measurementState.step, pulseAnim]);

  // Check for plane detection (simulated for now)
  useEffect(() => {
    if (hasLiDAR && isActive) {
      // In a real implementation, this would come from ARKit plane detection
      // For now, we simulate plane detection after a short delay
      const timer = setTimeout(() => {
        setMeasurementState(prev => ({
          ...prev,
          planeDetected: true,
        }));
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [hasLiDAR, isActive]);

  /**
   * Get depth at a screen point
   */
  const getDepth = useCallback(async (x: number, y: number): Promise<number> => {
    if (getDepthAtPoint) {
      try {
        const depth = await getDepthAtPoint(x, y);
        return depth > 0 ? depth : DEFAULT_DEPTH_CM;
      } catch {
        return DEFAULT_DEPTH_CM;
      }
    }
    return DEFAULT_DEPTH_CM;
  }, [getDepthAtPoint]);

  /**
   * Handle tap to place measurement point
   */
  const handleTap = useCallback(async (event: { nativeEvent: { locationX: number; locationY: number } }) => {
    if (!isActive) return;

    const { locationX, locationY } = event.nativeEvent;
    const depth = await getDepth(locationX, locationY);

    const createPoint = (id: string): MeasurementPoint => ({
      id,
      x: locationX,
      y: locationY,
      depth,
    });

    switch (measurementState.step) {
      case 'place-first':
        const p1 = createPoint('p1');
        setMeasurementState(prev => ({
          ...prev,
          point1: p1,
          step: 'place-second',
          averageDepth: depth,
        }));
        Animated.spring(point1Scale, {
          toValue: 1,
          friction: 5,
          useNativeDriver: true,
        }).start();
        break;

      case 'place-second':
        const p2 = createPoint('p2');
        const avgDepth = (measurementState.averageDepth + depth) / 2;
        setMeasurementState(prev => ({
          ...prev,
          point2: p2,
          step: 'place-depth',
          averageDepth: avgDepth,
        }));
        Animated.spring(point2Scale, {
          toValue: 1,
          friction: 5,
          useNativeDriver: true,
        }).start();
        break;

      case 'place-depth':
        const p3 = createPoint('p3');
        setMeasurementState(prev => ({
          ...prev,
          point3: p3,
          step: 'complete',
        }));
        Animated.spring(point3Scale, {
          toValue: 1,
          friction: 5,
          useNativeDriver: true,
        }).start();
        break;

      default:
        break;
    }
  }, [isActive, measurementState.step, measurementState.averageDepth, getDepth, point1Scale, point2Scale, point3Scale]);

  /**
   * Calculate final measurement from placed points
   */
  const calculateMeasurement = useCallback((): ARMeasurement | null => {
    const { point1, point2, point3, averageDepth, planeDetected } = measurementState;

    if (!point1 || !point2) return null;

    // Calculate width and height from screen coordinates
    const pixelWidth = Math.abs(point2.x - point1.x);
    const pixelHeight = Math.abs(point2.y - point1.y);

    const width = screenToWorld(pixelWidth, averageDepth);
    const height = screenToWorld(pixelHeight, averageDepth);

    // Calculate depth from third point or estimate
    let depth: number;
    if (point3) {
      // Use vertical distance from point3 to the baseline
      const pixelDepth = Math.abs(point3.y - (point1.y + point2.y) / 2);
      depth = screenToWorld(pixelDepth, averageDepth);
    } else {
      // Estimate depth as smaller of width/height
      depth = Math.min(width, height) * 0.7;
    }

    // Determine confidence based on measurement quality
    let confidence: 'high' | 'medium' | 'low';
    if (hasLiDAR && planeDetected && point3) {
      confidence = 'high';
    } else if (hasLiDAR || (planeDetected && point3)) {
      confidence = 'medium';
    } else {
      confidence = 'low';
    }

    return {
      width: Math.round(width * 10) / 10, // Round to 1 decimal
      height: Math.round(height * 10) / 10,
      depth: Math.round(depth * 10) / 10,
      distance: Math.round(averageDepth),
      confidence,
      planeDetected,
      timestamp: new Date(),
    };
  }, [measurementState, hasLiDAR]);

  /**
   * Handle confirm measurement
   */
  const handleConfirm = useCallback(() => {
    const measurement = calculateMeasurement();
    if (measurement) {
      onMeasurementComplete(measurement);
    }
  }, [calculateMeasurement, onMeasurementComplete]);

  /**
   * Reset measurement
   */
  const handleReset = useCallback(() => {
    setMeasurementState({
      point1: null,
      point2: null,
      point3: null,
      step: 'place-first',
      planeDetected: measurementState.planeDetected,
      averageDepth: DEFAULT_DEPTH_CM,
    });
    point1Scale.setValue(0);
    point2Scale.setValue(0);
    point3Scale.setValue(0);
  }, [measurementState.planeDetected, point1Scale, point2Scale, point3Scale]);

  /**
   * Get instruction text based on current step
   */
  const getInstructionText = (): string => {
    switch (measurementState.step) {
      case 'place-first':
        return 'Tap the top-left corner of the food';
      case 'place-second':
        return 'Tap the bottom-right corner';
      case 'place-depth':
        return 'Tap the back edge for depth';
      case 'complete':
        return 'Measurement complete!';
      default:
        return 'Tap to place measurement points';
    }
  };

  const { point1, point2, point3 } = measurementState;
  const measurement = measurementState.step === 'complete' ? calculateMeasurement() : null;

  return (
    <View style={styles.container} pointerEvents="box-none">
      {/* Touch area for placing points */}
      <TouchableOpacity
        style={styles.touchArea}
        activeOpacity={1}
        onPress={handleTap}
        disabled={measurementState.step === 'complete'}
      >
        {/* Crosshair guide */}
        {measurementState.step !== 'complete' && (
          <Animated.View
            style={[
              styles.crosshair,
              { transform: [{ scale: pulseAnim }] },
            ]}
          >
            <Ionicons
              name="add-circle-outline"
              size={48}
              color={colors.primary.main}
            />
          </Animated.View>
        )}

        {/* Measurement points */}
        {point1 && (
          <Animated.View
            style={[
              styles.measurePoint,
              { left: point1.x - 12, top: point1.y - 12 },
              { transform: [{ scale: point1Scale }] },
            ]}
          >
            <View style={styles.measurePointInner} />
            <Text style={styles.pointLabel}>1</Text>
          </Animated.View>
        )}

        {point2 && (
          <Animated.View
            style={[
              styles.measurePoint,
              { left: point2.x - 12, top: point2.y - 12 },
              { transform: [{ scale: point2Scale }] },
            ]}
          >
            <View style={styles.measurePointInner} />
            <Text style={styles.pointLabel}>2</Text>
          </Animated.View>
        )}

        {point3 && (
          <Animated.View
            style={[
              styles.measurePoint,
              styles.depthPoint,
              { left: point3.x - 12, top: point3.y - 12 },
              { transform: [{ scale: point3Scale }] },
            ]}
          >
            <View style={[styles.measurePointInner, styles.depthPointInner]} />
            <Text style={styles.pointLabel}>D</Text>
          </Animated.View>
        )}

        {/* Measurement lines */}
        {point1 && point2 && (
          <View style={styles.linesContainer} pointerEvents="none">
            {/* Bounding box */}
            <View
              style={[
                styles.boundingBox,
                {
                  left: Math.min(point1.x, point2.x),
                  top: Math.min(point1.y, point2.y),
                  width: Math.abs(point2.x - point1.x),
                  height: Math.abs(point2.y - point1.y),
                },
              ]}
            />

            {/* Width label */}
            {measurement && (
              <>
                <View
                  style={[
                    styles.dimensionLabel,
                    {
                      left: (point1.x + point2.x) / 2 - 30,
                      top: Math.max(point1.y, point2.y) + 8,
                    },
                  ]}
                >
                  <Text style={styles.dimensionText}>
                    {measurement.width.toFixed(1)} cm
                  </Text>
                </View>

                {/* Height label */}
                <View
                  style={[
                    styles.dimensionLabel,
                    {
                      left: Math.max(point1.x, point2.x) + 8,
                      top: (point1.y + point2.y) / 2 - 10,
                    },
                  ]}
                >
                  <Text style={styles.dimensionText}>
                    {measurement.height.toFixed(1)} cm
                  </Text>
                </View>
              </>
            )}
          </View>
        )}
      </TouchableOpacity>

      {/* Top bar with status */}
      <View style={styles.topBar}>
        <TouchableOpacity onPress={onCancel} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>

        <View style={styles.statusContainer}>
          {/* LiDAR indicator */}
          <View style={styles.statusItem}>
            <Ionicons
              name={hasLiDAR ? 'hardware-chip' : 'hardware-chip-outline'}
              size={16}
              color={hasLiDAR ? colors.status.success : colors.text.tertiary}
            />
            <Text
              style={[
                styles.statusText,
                { color: hasLiDAR ? colors.status.success : colors.text.tertiary },
              ]}
            >
              {hasLiDAR ? 'LiDAR' : 'No LiDAR'}
            </Text>
          </View>

          {/* Plane detection indicator */}
          <View style={styles.statusItem}>
            <Ionicons
              name={measurementState.planeDetected ? 'layers' : 'layers-outline'}
              size={16}
              color={measurementState.planeDetected ? colors.status.success : colors.text.tertiary}
            />
            <Text
              style={[
                styles.statusText,
                { color: measurementState.planeDetected ? colors.status.success : colors.text.tertiary },
              ]}
            >
              {measurementState.planeDetected ? 'Surface' : 'Detecting...'}
            </Text>
          </View>
        </View>
      </View>

      {/* Instruction banner */}
      <View style={styles.instructionBanner}>
        <Ionicons
          name="information-circle"
          size={20}
          color={colors.primary.main}
        />
        <Text style={styles.instructionText}>{getInstructionText()}</Text>
      </View>

      {/* Bottom controls */}
      <View style={styles.bottomControls}>
        {/* Measurement summary */}
        {measurement && (
          <View style={styles.summaryCard}>
            <Text style={styles.summaryTitle}>Measured Dimensions</Text>
            <View style={styles.summaryRow}>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>Width</Text>
                <Text style={styles.summaryValue}>{measurement.width} cm</Text>
              </View>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>Height</Text>
                <Text style={styles.summaryValue}>{measurement.height} cm</Text>
              </View>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>Depth</Text>
                <Text style={styles.summaryValue}>{measurement.depth} cm</Text>
              </View>
            </View>
            <View style={styles.confidenceRow}>
              <Ionicons
                name={
                  measurement.confidence === 'high'
                    ? 'checkmark-circle'
                    : measurement.confidence === 'medium'
                    ? 'alert-circle'
                    : 'warning'
                }
                size={16}
                color={
                  measurement.confidence === 'high'
                    ? colors.status.success
                    : measurement.confidence === 'medium'
                    ? colors.status.warning
                    : colors.status.error
                }
              />
              <Text style={styles.confidenceText}>
                {measurement.confidence.charAt(0).toUpperCase() +
                  measurement.confidence.slice(1)}{' '}
                confidence measurement
              </Text>
            </View>
          </View>
        )}

        {/* Action buttons */}
        <View style={styles.buttonRow}>
          {measurementState.step !== 'place-first' && (
            <TouchableOpacity
              style={[styles.button, styles.secondaryButton]}
              onPress={handleReset}
            >
              <Ionicons name="refresh" size={20} color={colors.text.primary} />
              <Text style={styles.secondaryButtonText}>Reset</Text>
            </TouchableOpacity>
          )}

          {measurementState.step === 'complete' && measurement && (
            <TouchableOpacity
              style={[styles.button, styles.primaryButton]}
              onPress={handleConfirm}
            >
              <Ionicons name="checkmark" size={20} color={colors.text.primary} />
              <Text style={styles.primaryButtonText}>Use Measurement</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 10,
  },
  touchArea: {
    flex: 1,
  },
  crosshair: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    marginTop: -24,
    marginLeft: -24,
    opacity: 0.8,
  },
  measurePoint: {
    position: 'absolute',
    width: 24,
    height: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  measurePointInner: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: colors.primary.main,
    borderWidth: 3,
    borderColor: colors.background.primary,
  },
  depthPoint: {},
  depthPointInner: {
    backgroundColor: colors.secondary.main,
  },
  pointLabel: {
    position: 'absolute',
    top: -16,
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 4,
    borderRadius: 2,
  },
  linesContainer: {
    ...StyleSheet.absoluteFillObject,
  },
  boundingBox: {
    position: 'absolute',
    borderWidth: 2,
    borderColor: colors.primary.main,
    borderStyle: 'dashed',
    backgroundColor: 'rgba(139, 92, 246, 0.1)',
  },
  dimensionLabel: {
    position: 'absolute',
    backgroundColor: 'rgba(0,0,0,0.8)',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
  },
  dimensionText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
  },
  topBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: 60,
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.md,
    backgroundColor: 'rgba(0,0,0,0.6)',
  },
  backButton: {
    padding: spacing.xs,
  },
  statusContainer: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: spacing.md,
  },
  statusItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  statusText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
  },
  instructionBanner: {
    position: 'absolute',
    top: 120,
    left: spacing.md,
    right: spacing.md,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.8)',
    padding: spacing.md,
    borderRadius: borderRadius.md,
    gap: spacing.sm,
  },
  instructionText: {
    flex: 1,
    color: colors.text.primary,
    fontSize: typography.fontSize.sm,
  },
  bottomControls: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: spacing.md,
    paddingBottom: spacing.xl,
    backgroundColor: 'rgba(0,0,0,0.8)',
  },
  summaryCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.md,
  },
  summaryTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.sm,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.sm,
  },
  summaryItem: {
    alignItems: 'center',
    flex: 1,
  },
  summaryLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginBottom: 2,
  },
  summaryValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.primary.main,
  },
  confidenceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingTop: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  confidenceText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.secondary,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  button: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    gap: spacing.xs,
  },
  primaryButton: {
    backgroundColor: colors.primary.main,
  },
  primaryButtonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
  },
  secondaryButton: {
    backgroundColor: colors.background.elevated,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  secondaryButtonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
  },
});

export default ARMeasurementOverlay;
