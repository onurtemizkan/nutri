import { useState, useRef, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Image,
  Animated,
  Platform,
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import * as ImageManipulator from 'expo-image-manipulator';
import { Ionicons } from '@expo/vector-icons';
import { foodAnalysisApi } from '@/lib/api/food-analysis';
import { showAlert } from '@/lib/utils/alert';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import LiDARModule from '@/lib/modules/LiDARModule';
import {
  estimateWeightFromMeasurement,
  formatWeight,
  formatDimensions,
} from '@/lib/utils/portion-estimation';
import type {
  FoodScanResult,
  ARMeasurement,
} from '@/lib/types/food-analysis';

export default function ScanFoodScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<CameraType>('back');
  const [flash, setFlash] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [scanResult, setScanResult] = useState<FoodScanResult | null>(null);
  const [showGuide, setShowGuide] = useState(true);
  const [arMeasurement, setArMeasurement] = useState<ARMeasurement | null>(null);
  const [hasARSupport, setHasARSupport] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const router = useRouter();

  // Check AR support on mount
  useEffect(() => {
    async function checkARSupport() {
      if (Platform.OS !== 'ios') return;

      try {
        if (LiDARModule && (LiDARModule as any).isAvailable) {
          const capabilities = await LiDARModule.getDeviceCapabilities();
          setHasARSupport(capabilities.hasARKit || capabilities.hasLiDAR);
        }
      } catch (error) {
        console.log('AR support check failed:', error);
      }
    }

    checkARSupport();
  }, []);

  // Listen for AR measurement results from the ar-measure screen
  useEffect(() => {
    const handleMeasurement = (event: CustomEvent<ARMeasurement>) => {
      setArMeasurement(event.detail);
    };

    if (typeof window !== 'undefined') {
      window.addEventListener('ar-measurement-complete', handleMeasurement as EventListener);
      return () => {
        window.removeEventListener('ar-measurement-complete', handleMeasurement as EventListener);
      };
    }
  }, []);

  // Pulsing animation for the analyzing spinner
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (isAnalyzing) {
      // Start pulsing animation
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.2,
            duration: 1000,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 1000,
            useNativeDriver: true,
          }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [isAnalyzing, pulseAnim]);

  // Hide guide after 3 seconds
  useEffect(() => {
    if (showGuide) {
      const timer = setTimeout(() => setShowGuide(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [showGuide]);

  // Request permission on mount
  useEffect(() => {
    if (permission && !permission.granted) {
      requestPermission();
    }
  }, [permission, requestPermission]);

  const handleCapturePhoto = async () => {
    if (!cameraRef.current) return;

    try {
      // Capture photo
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: false,
        exif: false,
      });

      if (!photo) {
        showAlert('Error', 'Failed to capture photo');
        return;
      }

      // Compress and process image
      const manipulatedImage = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 1024 } }], // Resize for faster upload
        { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG }
      );

      setCapturedImage(manipulatedImage.uri);
    } catch (error) {
      console.error('Error capturing photo:', error);
      showAlert('Error', 'Failed to capture photo. Please try again.');
    }
  };

  const handleMeasurePortion = useCallback(() => {
    router.push('/ar-measure' as any);
  }, [router]);

  const handleAnalyzeFood = async () => {
    if (!capturedImage) return;

    setIsAnalyzing(true);

    try {
      // Use AR measurement if available, otherwise undefined
      const measurements: ARMeasurement | undefined = arMeasurement || undefined;

      // Call ML service
      const response = await foodAnalysisApi.analyzeFood({
        imageUri: capturedImage,
        measurements,
      });

      const result: FoodScanResult = {
        ...response,
        imageUri: capturedImage,
        timestamp: new Date(),
      };

      setScanResult(result);
    } catch (error) {
      console.error('Food analysis error:', error);
      showAlert(
        'Analysis Failed',
        'Could not analyze the food. Please ensure the ML service is running and try again.',
        [
          {
            text: 'Retry',
            onPress: handleAnalyzeFood,
          },
          {
            text: 'Cancel',
            style: 'cancel',
            onPress: handleRetake,
          },
        ]
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleRetake = () => {
    setCapturedImage(null);
    setScanResult(null);
    setArMeasurement(null);
  };

  const handleUseScan = () => {
    if (!scanResult || scanResult.foodItems.length === 0) {
      showAlert('Error', 'No food items detected');
      return;
    }

    // Navigate to add meal screen with pre-filled data
    const primaryFood = scanResult.foodItems[0];
    router.push({
      pathname: '/add-meal',
      params: {
        name: primaryFood.name,
        calories: primaryFood.nutrition.calories.toString(),
        protein: primaryFood.nutrition.protein.toString(),
        carbs: primaryFood.nutrition.carbs.toString(),
        fat: primaryFood.nutrition.fat.toString(),
        fiber: primaryFood.nutrition.fiber?.toString() || '',
        servingSize: primaryFood.portionSize,
        fromScan: 'true',
      },
    });
  };

  const toggleFlash = () => {
    setFlash((prev) => !prev);
  };

  const toggleCameraFacing = () => {
    setFacing((current) => (current === 'back' ? 'front' : 'back'));
  };

  // Permission handling
  if (!permission) {
    return (
      <View style={styles.permissionContainer}>
        <ActivityIndicator size="large" color="#3b5998" />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer} testID="scan-food-permission-screen">
        <Ionicons name="camera-outline" size={64} color="#ccc" />
        <Text style={styles.permissionText}>Camera access is required</Text>
        <Text style={styles.permissionSubtext}>
          We need camera access to scan and analyze your food
        </Text>
        <TouchableOpacity
          style={styles.permissionButton}
          onPress={requestPermission}
          activeOpacity={0.8}
          testID="scan-food-grant-permission-button"
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
      </View>
    );
  }

  // Show captured image with analysis results
  if (capturedImage) {
    return (
      <SafeAreaView style={styles.container} testID="scan-food-preview-screen">
        <View style={styles.header}>
          <TouchableOpacity onPress={handleRetake} testID="scan-food-retake-button">
            <Text style={styles.headerButton}>Retake</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Food Scan</Text>
          <View style={{ width: 60 }} />
        </View>

        <View style={styles.previewContainer}>
          <Image source={{ uri: capturedImage }} style={styles.previewImage} />

          {isAnalyzing && (
            <BlurView intensity={80} tint="dark" style={styles.analyzingOverlay}>
              <View style={styles.analyzingCard}>
                <View style={styles.spinnerWrapper}>
                  <Animated.View
                    style={[
                      styles.pulseCircle,
                      {
                        transform: [{ scale: pulseAnim }],
                      },
                    ]}
                  >
                    <LinearGradient
                      colors={[colors.primary.main, colors.primary.light]}
                      start={{ x: 0, y: 0 }}
                      end={{ x: 1, y: 1 }}
                      style={styles.pulseGradient}
                    />
                  </Animated.View>

                  <View style={styles.spinnerContainer}>
                    <ActivityIndicator size="large" color={colors.primary.main} />
                  </View>
                </View>

                <View style={styles.analyzingTextContainer}>
                  <Text style={styles.analyzingTitle}>Analyzing Food</Text>
                  <Text style={styles.analyzingSubtext}>
                    Using AI to identify nutrition...
                  </Text>
                </View>
              </View>
            </BlurView>
          )}

          {scanResult && (
            <View style={styles.resultsContainer}>
              <Text style={styles.resultsTitle}>Detected Food</Text>

              {scanResult.foodItems.map((item: any, index: number) => (
                <View key={index} style={styles.foodItem}>
                  <View style={styles.foodItemHeader}>
                    <Text style={styles.foodName}>{item.name}</Text>
                    <Text style={styles.confidence}>
                      {Math.round(item.confidence * 100)}% confident
                    </Text>
                  </View>

                  <Text style={styles.portionSize}>{item.portionSize}</Text>

                  <View style={styles.nutritionGrid}>
                    <View style={styles.nutritionItem}>
                      <Text style={styles.nutritionValue}>
                        {Math.round(item.nutrition.calories)}
                      </Text>
                      <Text style={styles.nutritionLabel}>cal</Text>
                    </View>
                    <View style={styles.nutritionItem}>
                      <Text style={styles.nutritionValue}>
                        {Math.round(item.nutrition.protein)}g
                      </Text>
                      <Text style={styles.nutritionLabel}>protein</Text>
                    </View>
                    <View style={styles.nutritionItem}>
                      <Text style={styles.nutritionValue}>
                        {Math.round(item.nutrition.carbs)}g
                      </Text>
                      <Text style={styles.nutritionLabel}>carbs</Text>
                    </View>
                    <View style={styles.nutritionItem}>
                      <Text style={styles.nutritionValue}>
                        {Math.round(item.nutrition.fat)}g
                      </Text>
                      <Text style={styles.nutritionLabel}>fat</Text>
                    </View>
                  </View>
                </View>
              ))}

              {scanResult.suggestions && scanResult.suggestions.length > 0 && (
                <View style={styles.suggestions}>
                  <Text style={styles.suggestionsTitle}>Suggestions:</Text>
                  {scanResult.suggestions.map((suggestion: string, index: number) => (
                    <Text key={index} style={styles.suggestionText}>
                      • {suggestion}
                    </Text>
                  ))}
                </View>
              )}
            </View>
          )}
        </View>

        <View style={styles.actionsContainer}>
          {/* AR Measurement display */}
          {arMeasurement && !scanResult && (
            <View style={styles.measurementInfo}>
              <View style={styles.measurementHeader}>
                <Ionicons name="resize-outline" size={20} color={colors.status.success} />
                <Text style={styles.measurementTitle}>Portion Measured</Text>
              </View>
              <Text style={styles.measurementDimensions}>
                {formatDimensions(arMeasurement.width, arMeasurement.height, arMeasurement.depth)}
              </Text>
              <Text style={styles.measurementConfidence}>
                {arMeasurement.confidence} confidence
                {arMeasurement.planeDetected ? ' • Surface detected' : ''}
              </Text>
            </View>
          )}

          {!isAnalyzing && !scanResult && (
            <>
              {/* Measure Portion button - only show on iOS with AR support */}
              {Platform.OS === 'ios' && hasARSupport && !arMeasurement && (
                <TouchableOpacity
                  style={styles.measureButton}
                  onPress={handleMeasurePortion}
                  activeOpacity={0.8}
                >
                  <View style={styles.measureButtonContent}>
                    <Ionicons name="cube-outline" size={24} color={colors.primary.main} />
                    <View style={styles.measureButtonText}>
                      <Text style={styles.measureButtonTitle}>Measure Portion Size</Text>
                      <Text style={styles.measureButtonSubtitle}>
                        Use AR to get accurate dimensions
                      </Text>
                    </View>
                    <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
                  </View>
                </TouchableOpacity>
              )}

              {/* Re-measure button if already measured */}
              {arMeasurement && (
                <TouchableOpacity
                  style={styles.remeasureButton}
                  onPress={handleMeasurePortion}
                  activeOpacity={0.8}
                >
                  <Ionicons name="refresh-outline" size={18} color={colors.text.secondary} />
                  <Text style={styles.remeasureButtonText}>Re-measure</Text>
                </TouchableOpacity>
              )}

              {/* Analyze button */}
              <TouchableOpacity
                style={styles.analyzeButton}
                onPress={handleAnalyzeFood}
                activeOpacity={0.8}
                testID="scan-food-analyze-button"
              >
                <LinearGradient
                  colors={gradients.primary}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.analyzeButtonGradient}
                >
                  <Text style={styles.analyzeButtonText}>
                    {arMeasurement ? 'Analyze with Measurements' : 'Analyze Food'}
                  </Text>
                </LinearGradient>
              </TouchableOpacity>
            </>
          )}

          {scanResult && (
            <TouchableOpacity
              style={styles.analyzeButton}
              onPress={handleUseScan}
              activeOpacity={0.8}
              testID="scan-food-use-scan-button"
            >
              <LinearGradient
                colors={gradients.primary}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.analyzeButtonGradient}
              >
                <Text style={styles.analyzeButtonText}>Use This Scan</Text>
              </LinearGradient>
            </TouchableOpacity>
          )}
        </View>
      </SafeAreaView>
    );
  }

  // Camera view
  return (
    <View style={styles.container} testID="scan-food-camera-screen">
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing={facing}
        flash={flash ? 'on' : 'off'}
      >
        {/* Header */}
        <SafeAreaView style={styles.cameraHeader}>
          <TouchableOpacity
            style={styles.closeButton}
            onPress={() => router.back()}
            testID="scan-food-close-button"
          >
            <Ionicons name="close" size={32} color="#fff" />
          </TouchableOpacity>

          <View style={styles.cameraControls}>
            <TouchableOpacity
              style={styles.controlButton}
              onPress={toggleFlash}
            >
              <Ionicons
                name={flash ? 'flash' : 'flash-off'}
                size={24}
                color="#fff"
              />
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.controlButton}
              onPress={toggleCameraFacing}
            >
              <Ionicons name="camera-reverse" size={24} color="#fff" />
            </TouchableOpacity>
          </View>
        </SafeAreaView>

        {/* Guide overlay */}
        {showGuide && (
          <View style={styles.guideOverlay}>
            <View style={styles.guideBox}>
              <Text style={styles.guideText}>
                📸 Center your food in the frame
              </Text>
              <Text style={styles.guideSubtext}>
                Include a reference object like your hand or a coin for better
                size estimation
              </Text>
            </View>
          </View>
        )}

        {/* Capture button */}
        <View style={styles.captureContainer}>
          <TouchableOpacity
            style={styles.captureButton}
            onPress={handleCapturePhoto}
            testID="scan-food-capture-button"
          >
            <View style={styles.captureButtonInner} />
          </TouchableOpacity>
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.background.primary,
    padding: spacing.xl,
  },
  permissionText: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  permissionSubtext: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    textAlign: 'center',
    marginBottom: spacing['2xl'],
  },
  permissionButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
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
  camera: {
    flex: 1,
  },
  cameraHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 8,
  },
  closeButton: {
    width: 44,
    height: 44,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraControls: {
    flexDirection: 'row',
    gap: 16,
  },
  controlButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  guideOverlay: {
    position: 'absolute',
    top: '40%',
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  guideBox: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 20,
    borderRadius: 16,
  },
  guideText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    textAlign: 'center',
    marginBottom: 8,
  },
  guideSubtext: {
    color: '#ccc',
    fontSize: 14,
    textAlign: 'center',
  },
  captureContainer: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#fff',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    backgroundColor: colors.background.secondary,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  headerButton: {
    fontSize: typography.fontSize.md,
    color: colors.primary.main,
    fontWeight: typography.fontWeight.semibold,
  },
  headerTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  previewContainer: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  previewImage: {
    width: '100%',
    height: '50%',
    resizeMode: 'contain',
  },
  analyzingOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
  analyzingCard: {
    alignItems: 'center',
  },
  spinnerWrapper: {
    width: 120,
    height: 120,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  pulseCircle: {
    position: 'absolute',
    width: 120,
    height: 120,
    justifyContent: 'center',
    alignItems: 'center',
    opacity: 0.2,
  },
  pulseGradient: {
    width: '100%',
    height: '100%',
    borderRadius: 60,
  },
  spinnerContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  analyzingTextContainer: {
    alignItems: 'center',
  },
  analyzingTitle: {
    color: colors.text.primary,
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    marginBottom: spacing.xs,
    letterSpacing: -0.5,
  },
  analyzingSubtext: {
    color: colors.text.tertiary,
    fontSize: typography.fontSize.sm,
    textAlign: 'center',
  },
  resultsContainer: {
    flex: 1,
    backgroundColor: colors.background.primary,
    padding: spacing.lg,
  },
  resultsTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    marginBottom: spacing.md,
    color: colors.text.primary,
  },
  foodItem: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.md,
  },
  foodItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  foodName: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    flex: 1,
  },
  confidence: {
    fontSize: typography.fontSize.xs,
    color: colors.primary.main,
    fontWeight: typography.fontWeight.semibold,
  },
  portionSize: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.md,
  },
  nutritionGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  nutritionItem: {
    alignItems: 'center',
  },
  nutritionValue: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  nutritionLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  suggestions: {
    marginTop: spacing.md,
    padding: spacing.md,
    backgroundColor: colors.special.highlight,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.focus,
  },
  suggestionsTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
    marginBottom: spacing.sm,
  },
  suggestionText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.secondary,
    marginBottom: spacing.xs,
  },
  actionsContainer: {
    padding: spacing.lg,
    backgroundColor: colors.background.secondary,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  analyzeButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    ...shadows.md,
  },
  analyzeButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  analyzeButtonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
  },
  measurementInfo: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.status.success,
  },
  measurementHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.xs,
  },
  measurementTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.status.success,
  },
  measurementDimensions: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  measurementConfidence: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  measureButton: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  measureButtonContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  measureButtonText: {
    flex: 1,
  },
  measureButtonTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  measureButtonSubtitle: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  remeasureButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.sm,
    marginBottom: spacing.sm,
  },
  remeasureButtonText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },
});
