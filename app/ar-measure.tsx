/**
 * AR Measurement Screen
 *
 * Full-screen modal for measuring food portion size using AR/LiDAR
 * or manual size picker fallback.
 *
 * Features:
 * - AR measurement with LiDAR (when available)
 * - Manual size picker fallback for non-LiDAR devices
 * - Camera preview with measurement overlay
 * - Returns measurement data to calling screen
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Platform,
  ActivityIndicator,
  DeviceEventEmitter,
  Dimensions,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { colors, gradients, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import { ARMeasurementOverlay } from '@/lib/components/ARMeasurementOverlay';
import { ManualSizePicker } from '@/lib/components/ManualSizePicker';
import {
  ReferenceObjectCalibration,
  type CalibrationResult,
} from '@/lib/components/ReferenceObjectCalibration';
import LiDARModule from '@/lib/modules/LiDARModule';
import type { ARMeasurement } from '@/lib/types/food-analysis';
import type { LiDARCapture } from '@/lib/types/ar-data';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

type MeasurementMode = 'ar' | 'manual' | 'loading' | 'calibration';

export default function ARMeasureScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{ foodName?: string; returnTo?: string }>();
  const [permission, requestPermission] = useCameraPermissions();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  const [mode, setMode] = useState<MeasurementMode>('loading');
  const [previousMode, setPreviousMode] = useState<'ar' | 'manual' | null>(null);
  const [hasLiDAR, setHasLiDAR] = useState(false);
  const [supportsSceneDepth, setSupportsSceneDepth] = useState(false);
  const [isARAvailable, setIsARAvailable] = useState(false);
  const [measurement, setMeasurement] = useState<ARMeasurement | null>(null);
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null);
  const [isARSessionActive, setIsARSessionActive] = useState(false);
  const lastDepthCapture = useRef<LiDARCapture | null>(null);
  const depthCaptureInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // Check LiDAR availability on mount
  useEffect(() => {
    async function checkCapabilities() {
      if (Platform.OS !== 'ios') {
        setMode('manual');
        return;
      }

      try {
        // Check if native module is available
        if (!LiDARModule || !(LiDARModule as any).isAvailable) {
          setMode('manual');
          return;
        }

        const capabilities = await LiDARModule.getDeviceCapabilities();
        setHasLiDAR(capabilities.hasLiDAR);
        setSupportsSceneDepth(capabilities.supportsSceneDepth);

        // AR is available if device has LiDAR OR supports ARKit Scene Depth (iPhone 12+ with A14 chip)
        const arAvailable = capabilities.hasLiDAR || capabilities.supportsSceneDepth;
        setIsARAvailable(arAvailable);

        // Default to AR mode if AR depth is available (LiDAR or Scene Depth), otherwise manual
        if (arAvailable) {
          setMode('ar');
        } else {
          setMode('manual');
        }
      } catch (error) {
        console.error('Error checking LiDAR capabilities:', error);
        setMode('manual');
      }
    }

    checkCapabilities();
  }, []);

  // Request camera permission
  useEffect(() => {
    if (permission && !permission.granted && mode === 'ar') {
      requestPermission();
    }
  }, [permission, requestPermission, mode]);

  // Start/stop AR session based on mode
  useEffect(() => {
    async function startARSession() {
      // Start AR session if AR mode is selected and device supports depth capture (LiDAR or Scene Depth)
      if (mode === 'ar' && isARAvailable && LiDARModule.isAvailable && !isARSessionActive) {
        try {
          // Use 'lidar' mode for LiDAR devices, 'ar_depth' for Scene Depth (non-LiDAR iPhone 12+)
          const captureMode = hasLiDAR ? 'lidar' : 'ar_depth';
          console.log(`Starting AR session with ${captureMode} mode...`);
          await LiDARModule.startARSession(captureMode);
          setIsARSessionActive(true);

          // Start capturing depth frames periodically
          // Use higher interval for Scene Depth (ML-based) to reduce CPU load
          const captureInterval = hasLiDAR ? 100 : 200; // 10 FPS for LiDAR, 5 FPS for Scene Depth
          depthCaptureInterval.current = setInterval(async () => {
            try {
              const capture = await LiDARModule.captureDepthFrame();
              lastDepthCapture.current = capture;
            } catch (err) {
              // Silently handle capture errors (e.g., session not ready)
              console.debug('Depth capture error:', err);
            }
          }, captureInterval);

          console.log(`AR session started successfully (${captureMode})`);
        } catch (err) {
          console.error('Failed to start AR session:', err);
        }
      }
    }

    async function stopARSession() {
      if (isARSessionActive) {
        try {
          // Stop depth capture interval
          if (depthCaptureInterval.current) {
            clearInterval(depthCaptureInterval.current);
            depthCaptureInterval.current = null;
          }

          await LiDARModule.stopARSession();
          setIsARSessionActive(false);
          lastDepthCapture.current = null;
          console.log('AR session stopped');
        } catch (err) {
          console.error('Failed to stop AR session:', err);
        }
      }
    }

    if (mode === 'ar' && isARAvailable && LiDARModule.isAvailable) {
      startARSession();
    } else {
      stopARSession();
    }

    // Cleanup on unmount
    return () => {
      stopARSession();
    };
  }, [mode, hasLiDAR, isARAvailable, isARSessionActive]);

  /**
   * Get depth at a specific screen coordinate (in cm)
   * Uses the latest depth capture from LiDAR
   */
  const getDepthAtPoint = useCallback(async (screenX: number, screenY: number): Promise<number> => {
    const capture = lastDepthCapture.current;

    if (!capture || !capture.depthBuffer.data || capture.depthBuffer.data.length === 0) {
      // No depth data available, return default distance
      return 30; // Default 30cm
    }

    const { depthBuffer } = capture;
    const { width, height, data } = depthBuffer;

    // Map screen coordinates to depth buffer coordinates
    const depthX = Math.floor((screenX / SCREEN_WIDTH) * width);
    const depthY = Math.floor((screenY / SCREEN_HEIGHT) * height);

    // Clamp to valid range
    const clampedX = Math.max(0, Math.min(width - 1, depthX));
    const clampedY = Math.max(0, Math.min(height - 1, depthY));

    // Get depth value (in meters)
    const index = clampedY * width + clampedX;
    const depthMeters = data[index];

    // Convert to centimeters
    const depthCm = depthMeters * 100;

    // Validate depth value (LiDAR range is typically 0.2m - 5m)
    if (depthCm > 0 && depthCm < 500) {
      return depthCm;
    }

    // Invalid depth, return default
    return 30;
  }, []);

  /**
   * Handle AR measurement completion
   */
  const handleARMeasurementComplete = useCallback((result: ARMeasurement) => {
    setMeasurement(result);
  }, []);

  /**
   * Handle manual size picker selection
   */
  const handleManualSelect = useCallback((result: ARMeasurement) => {
    setMeasurement(result);
  }, []);

  /**
   * Handle cancellation
   */
  const handleCancel = useCallback(() => {
    router.back();
  }, [router]);

  /**
   * Handle mode switch
   */
  const handleSwitchMode = useCallback(() => {
    setMode(current => current === 'ar' ? 'manual' : 'ar');
    setMeasurement(null);
  }, []);

  /**
   * Handle confirm measurement
   */
  const handleConfirm = useCallback(() => {
    if (!measurement) return;

    // Navigate back with measurement data
    // The measurement will be available via global state or params
    router.back();

    // Use a slight delay to ensure navigation completes
    setTimeout(() => {
      // Emit event with measurement data using React Native's DeviceEventEmitter
      // This can be picked up by the calling screen
      DeviceEventEmitter.emit('ar-measurement-complete', measurement);
    }, 100);
  }, [measurement, router]);

  /**
   * Clear current measurement
   */
  const handleClearMeasurement = useCallback(() => {
    setMeasurement(null);
  }, []);

  /**
   * Open calibration screen
   */
  const handleRequestCalibration = useCallback(() => {
    // Save current mode to return to after calibration
    if (mode === 'ar' || mode === 'manual') {
      setPreviousMode(mode);
    }
    setMode('calibration');
  }, [mode]);

  /**
   * Handle calibration complete
   */
  const handleCalibrationComplete = useCallback((result: CalibrationResult) => {
    setCalibration(result);
    // Return to previous mode
    setMode(previousMode || 'manual');
    setPreviousMode(null);
  }, [previousMode]);

  /**
   * Handle calibration cancel
   */
  const handleCalibrationCancel = useCallback(() => {
    // Return to previous mode without saving
    setMode(previousMode || 'manual');
    setPreviousMode(null);
  }, [previousMode]);

  // Loading state
  if (mode === 'loading') {
    return (
      <SafeAreaView style={styles.container} testID="ar-measure-screen">
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Checking device capabilities...</Text>
        </View>
      </SafeAreaView>
    );
  }

  // Calibration mode
  if (mode === 'calibration') {
    return (
      <SafeAreaView style={styles.container} edges={['top']} testID="ar-measure-screen">
        <ReferenceObjectCalibration
          onCalibrate={handleCalibrationComplete}
          onCancel={handleCalibrationCancel}
        />
      </SafeAreaView>
    );
  }

  // Manual mode - show size picker
  if (mode === 'manual') {
    return (
      <SafeAreaView style={styles.container} edges={['top']} testID="ar-measure-screen">
        {/* Mode switcher (only if AR depth is available - LiDAR or Scene Depth) */}
        {isARAvailable && (
          <View style={styles.modeSwitcher}>
            <TouchableOpacity
              style={styles.modeSwitchButton}
              onPress={handleSwitchMode}
              testID="ar-measure-switch-to-ar-button"
            >
              <Ionicons name="camera-outline" size={20} color={colors.primary.main} />
              <Text style={styles.modeSwitchText}>Switch to AR Mode</Text>
            </TouchableOpacity>
          </View>
        )}

        {measurement ? (
          // Show measurement result
          <View style={[
            styles.resultContainer,
            { paddingHorizontal: responsiveSpacing.horizontal },
            isTablet && styles.resultContainerTablet
          ]}>
            <View style={[styles.resultCard, isTablet && styles.resultCardTablet]}>
              <Text style={styles.resultTitle}>Measurement Ready</Text>

              <View style={styles.dimensionsRow}>
                <View style={styles.dimensionItem}>
                  <Text style={styles.dimensionValue}>{measurement.width}</Text>
                  <Text style={styles.dimensionLabel}>Width (cm)</Text>
                </View>
                <View style={styles.dimensionItem}>
                  <Text style={styles.dimensionValue}>{measurement.height}</Text>
                  <Text style={styles.dimensionLabel}>Height (cm)</Text>
                </View>
                <View style={styles.dimensionItem}>
                  <Text style={styles.dimensionValue}>{measurement.depth}</Text>
                  <Text style={styles.dimensionLabel}>Depth (cm)</Text>
                </View>
              </View>

              <View style={styles.confidenceBadge}>
                <Ionicons
                  name={
                    measurement.confidence === 'high' ? 'checkmark-circle' :
                    measurement.confidence === 'medium' ? 'alert-circle' : 'warning'
                  }
                  size={16}
                  color={
                    measurement.confidence === 'high' ? colors.status.success :
                    measurement.confidence === 'medium' ? colors.status.warning : colors.status.error
                  }
                />
                <Text style={styles.confidenceText}>
                  {measurement.confidence.charAt(0).toUpperCase() + measurement.confidence.slice(1)} confidence
                </Text>
              </View>
            </View>

            <View style={styles.resultActions}>
              <TouchableOpacity
                style={[styles.actionButton, styles.secondaryButton]}
                onPress={handleClearMeasurement}
                testID="ar-measure-remeasure-button"
              >
                <Text style={styles.secondaryButtonText}>Remeasure</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.actionButton, styles.primaryButton]}
                onPress={handleConfirm}
                testID="ar-measure-confirm-button"
              >
                <LinearGradient
                  colors={gradients.primary}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.primaryButtonGradient}
                >
                  <Text style={styles.primaryButtonText}>Use Measurement</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </View>
        ) : (
          // Show manual size picker
          <ManualSizePicker
            foodName={params.foodName}
            onSelect={handleManualSelect}
            onCancel={handleCancel}
            onRequestCalibration={handleRequestCalibration}
            calibration={calibration}
          />
        )}
      </SafeAreaView>
    );
  }

  // AR mode - show camera with measurement overlay
  // Check camera permission
  if (!permission?.granted) {
    return (
      <SafeAreaView style={styles.container} testID="ar-measure-screen">
        <View style={styles.permissionContainer}>
          <Ionicons name="camera-outline" size={64} color={colors.text.tertiary} />
          <Text style={styles.permissionTitle}>Camera Access Required</Text>
          <Text style={styles.permissionText}>
            AR measurement requires camera access to detect surfaces and measure your food.
          </Text>
          <TouchableOpacity
            style={styles.permissionButton}
            onPress={requestPermission}
          >
            <LinearGradient
              colors={gradients.primary}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.permissionButtonGradient}
            >
              <Text style={styles.permissionButtonText}>Grant Permission</Text>
            </LinearGradient>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.fallbackButton}
            onPress={() => setMode('manual')}
            testID="ar-measure-use-manual-button"
          >
            <Text style={styles.fallbackButtonText}>Use Manual Mode Instead</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.container} testID="ar-measure-screen">
      {/* Camera preview */}
      <CameraView style={styles.camera} facing="back">
        {/* AR Measurement Overlay */}
        {!measurement && (
          <ARMeasurementOverlay
            hasLiDAR={hasLiDAR}
            hasSceneDepth={supportsSceneDepth && !hasLiDAR}
            onMeasurementComplete={handleARMeasurementComplete}
            onCancel={handleCancel}
            isActive={true}
            getDepthAtPoint={isARAvailable && isARSessionActive ? getDepthAtPoint : undefined}
            lidarCapture={isARAvailable && isARSessionActive ? lastDepthCapture.current : null}
            foodName={params.foodName}
          />
        )}

        {/* Measurement result overlay */}
        {measurement && (
          <View style={styles.measurementResultOverlay}>
            <SafeAreaView style={styles.resultOverlayContent}>
              <View style={styles.resultOverlayCard}>
                <Text style={styles.resultOverlayTitle}>Measurement Complete</Text>

                <View style={styles.dimensionsRow}>
                  <View style={styles.dimensionItem}>
                    <Text style={styles.dimensionValueLight}>{measurement.width}</Text>
                    <Text style={styles.dimensionLabelLight}>Width (cm)</Text>
                  </View>
                  <View style={styles.dimensionItem}>
                    <Text style={styles.dimensionValueLight}>{measurement.height}</Text>
                    <Text style={styles.dimensionLabelLight}>Height (cm)</Text>
                  </View>
                  <View style={styles.dimensionItem}>
                    <Text style={styles.dimensionValueLight}>{measurement.depth}</Text>
                    <Text style={styles.dimensionLabelLight}>Depth (cm)</Text>
                  </View>
                </View>

                {/* Volume and Weight estimates */}
                <View style={styles.estimatesRowOverlay}>
                  <View style={styles.estimateItemOverlay}>
                    <Ionicons name="cube-outline" size={18} color={colors.text.tertiary} />
                    <Text style={styles.estimateLabelOverlay}>
                      {measurement.volumeAdjusted ?? measurement.volume ?? 0} cm³
                    </Text>
                  </View>
                  <View style={styles.estimateDividerOverlay} />
                  <View style={styles.estimateItemOverlay}>
                    <Ionicons name="scale-outline" size={18} color={colors.text.tertiary} />
                    <Text style={styles.estimateLabelOverlay}>
                      ~{measurement.estimatedWeight ?? 0}g
                    </Text>
                  </View>
                </View>

                <View style={styles.resultOverlayActions}>
                  <TouchableOpacity
                    style={[styles.actionButton, styles.outlineButton]}
                    onPress={handleClearMeasurement}
                    testID="ar-measure-remeasure-overlay-button"
                  >
                    <Text style={styles.outlineButtonText}>Remeasure</Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={[styles.actionButton, styles.primaryButton]}
                    onPress={handleConfirm}
                    testID="ar-measure-confirm-overlay-button"
                  >
                    <LinearGradient
                      colors={gradients.primary}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 0 }}
                      style={styles.primaryButtonGradient}
                    >
                      <Text style={styles.primaryButtonText}>Use This</Text>
                    </LinearGradient>
                  </TouchableOpacity>
                </View>
              </View>
            </SafeAreaView>
          </View>
        )}
      </CameraView>

      {/* Mode switcher at bottom (only when not measuring) */}
      {!measurement && (
        <View style={styles.bottomBar}>
          <SafeAreaView edges={['bottom']}>
            <TouchableOpacity
              style={styles.manualModeButton}
              onPress={handleSwitchMode}
              testID="ar-measure-switch-to-manual-button"
            >
              <Ionicons name="resize-outline" size={20} color={colors.text.secondary} />
              <Text style={styles.manualModeText}>Switch to Manual Mode</Text>
            </TouchableOpacity>
          </SafeAreaView>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: spacing.md,
    color: colors.text.secondary,
    fontSize: typography.fontSize.md,
  },
  camera: {
    flex: 1,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
  },
  permissionTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  permissionText: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing.xl,
    lineHeight: 24,
  },
  permissionButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    marginBottom: spacing.md,
  },
  permissionButtonGradient: {
    paddingHorizontal: spacing['2xl'],
    paddingVertical: spacing.md,
  },
  permissionButtonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
  },
  fallbackButton: {
    padding: spacing.md,
  },
  fallbackButtonText: {
    color: colors.text.tertiary,
    fontSize: typography.fontSize.sm,
  },
  modeSwitcher: {
    padding: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  modeSwitchButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    padding: spacing.sm,
  },
  modeSwitchText: {
    color: colors.primary.main,
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
  },
  resultContainer: {
    flex: 1,
    padding: spacing.lg,
  },
  resultContainerTablet: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  resultCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    ...shadows.md,
  },
  resultCardTablet: {
    maxWidth: FORM_MAX_WIDTH,
    width: '100%',
  },
  resultTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  dimensionsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: spacing.lg,
  },
  dimensionItem: {
    alignItems: 'center',
  },
  dimensionValue: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.primary.main,
  },
  dimensionLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  dimensionValueLight: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  dimensionLabelLight: {
    fontSize: typography.fontSize.xs,
    color: colors.text.secondary,
    marginTop: spacing.xs,
  },
  confidenceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  confidenceText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },
  resultActions: {
    flexDirection: 'row',
    gap: spacing.md,
    marginTop: spacing.xl,
  },
  actionButton: {
    flex: 1,
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  primaryButton: {},
  primaryButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
  },
  secondaryButton: {
    backgroundColor: colors.background.elevated,
    paddingVertical: spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  secondaryButtonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    textAlign: 'center',
  },
  outlineButton: {
    backgroundColor: 'transparent',
    paddingVertical: spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.text.primary,
  },
  outlineButtonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    textAlign: 'center',
  },
  measurementResultOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: colors.overlay.medium,
    justifyContent: 'flex-end',
  },
  resultOverlayContent: {
    padding: spacing.md,
  },
  resultOverlayCard: {
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
    ...shadows.lg,
  },
  resultOverlayTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: spacing.lg,
  },
  resultOverlayActions: {
    flexDirection: 'row',
    gap: spacing.md,
    marginTop: spacing.lg,
  },
  estimatesRowOverlay: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.lg,
    paddingVertical: spacing.sm,
    marginTop: spacing.sm,
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.sm,
  },
  estimateItemOverlay: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  estimateLabelOverlay: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    fontWeight: typography.fontWeight.medium,
  },
  estimateDividerOverlay: {
    width: 1,
    height: 20,
    backgroundColor: colors.border.secondary,
  },
  bottomBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: colors.overlay.light,
  },
  manualModeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    padding: spacing.md,
  },
  manualModeText: {
    color: colors.text.secondary,
    fontSize: typography.fontSize.sm,
  },
});
