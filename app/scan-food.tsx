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
  ScrollView,
  DeviceEventEmitter,
  TextInput,
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import * as ImageManipulator from 'expo-image-manipulator';
import { Ionicons } from '@expo/vector-icons';
import { foodAnalysisApi } from '@/lib/api/food-analysis';
import { foodsApi } from '@/lib/api/foods';
import { showAlert } from '@/lib/utils/alert';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import FeedbackCorrectionModal from '@/lib/components/FeedbackCorrectionModal';
import { ClassificationResult } from '@/lib/components/ClassificationResult';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import LiDARModule from '@/lib/modules/LiDARModule';
import {
  formatDimensions,
} from '@/lib/utils/portion-estimation';
import type {
  FoodScanResult,
  ARMeasurement,
  USDAClassificationResult,
  USDAFoodMatch,
} from '@/lib/types/food-analysis';
import type { USDAFood, FoodClassification } from '@/lib/types/foods';

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
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  // USDA Classification state
  const [usdaResult, setUsdaResult] = useState<USDAClassificationResult | null>(null);
  const [useUSDASearch, setUseUSDASearch] = useState(true); // Default to USDA-enhanced search
  // Editable state for detected foods
  const [editedFoodName, setEditedFoodName] = useState<string>('');
  const [editedPortionMultiplier, setEditedPortionMultiplier] = useState<number>(1);
  const [isEditingName, setIsEditingName] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

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
    const subscription = DeviceEventEmitter.addListener(
      'ar-measurement-complete',
      (measurement: ARMeasurement) => {
        setArMeasurement(measurement);
      }
    );

    return () => {
      subscription.remove();
    };
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

  // Initialize editable state when scan result changes
  useEffect(() => {
    if (scanResult && scanResult.foodItems.length > 0) {
      setEditedFoodName(scanResult.foodItems[0].name);
      setEditedPortionMultiplier(1);
      setIsEditingName(false);
    }
  }, [scanResult]);

  // Select an alternative food
  const handleSelectAlternative = (alternativeName: string) => {
    setEditedFoodName(alternativeName);
  };

  // Adjust portion multiplier
  const handlePortionAdjust = (delta: number) => {
    setEditedPortionMultiplier(prev => Math.max(0.25, Math.min(10, prev + delta)));
  };

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

      // Auto-navigate to AR measure if LiDAR is available
      if (Platform.OS === 'ios' && hasARSupport) {
        // Small delay to let the preview render first
        setTimeout(() => {
          router.push('/ar-measure' as any);
        }, 300);
      }
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

      if (useUSDASearch) {
        // Use USDA-enhanced classification and search
        try {
          const usdaResponse = await foodAnalysisApi.classifyAndSearch(
            capturedImage,
            measurements
          );
          setUsdaResult(usdaResponse);
          // Also set basic scan result for compatibility
          setScanResult(null);
        } catch (usdaError) {
          console.error('USDA classification failed, falling back to basic analysis:', usdaError);
          // Fallback to basic ML analysis
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
          setUsdaResult(null);
        }
      } else {
        // Use basic ML analysis only
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
        setUsdaResult(null);
      }
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
    setUsdaResult(null);
    setArMeasurement(null);
  };

  // Handle USDA food confirmation from ClassificationResult
  const handleUSDAConfirm = useCallback((food: USDAFood) => {
    // Record the selection for recent foods
    foodsApi.recordFoodSelection(food.fdcId).catch(() => {
      // Ignore errors - this is a non-critical operation
    });

    // Navigate to add meal with USDA nutrition data
    router.push({
      pathname: '/add-meal',
      params: {
        name: food.description,
        calories: Math.round(food.calories).toString(),
        protein: Math.round(food.protein).toString(),
        carbs: Math.round(food.carbs).toString(),
        fat: Math.round(food.fat).toString(),
        fiber: food.fiber ? Math.round(food.fiber).toString() : '',
        servingSize: food.servingSize && food.servingSizeUnit
          ? `${food.servingSize} ${food.servingSizeUnit}`
          : '100g',
        fromScan: 'true',
        fdcId: food.fdcId.toString(),
      },
    });
  }, [router]);

  // Handle search instead from ClassificationResult
  const handleSearchInstead = useCallback((query: string) => {
    router.push({
      pathname: '/food-search' as any,
      params: {
        initialQuery: query,
        fromClassification: 'true',
      },
    });
  }, [router]);

  // Handle report incorrect from ClassificationResult
  const handleReportIncorrect = useCallback(() => {
    setShowFeedbackModal(true);
  }, []);

  const handleUseScan = () => {
    if (!scanResult || scanResult.foodItems.length === 0) {
      showAlert('Error', 'No food items detected');
      return;
    }

    // Navigate to add meal screen with pre-filled data
    // Use edited values if available
    const primaryFood = scanResult.foodItems[0];
    const displayName = editedFoodName || primaryFood.name;
    const multiplier = editedPortionMultiplier;

    router.push({
      pathname: '/add-meal',
      params: {
        name: displayName.replace(/_/g, ' ').split(' ')
          .map((word: string) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
          .join(' '),
        calories: Math.round(primaryFood.nutrition.calories * multiplier).toString(),
        protein: Math.round(primaryFood.nutrition.protein * multiplier).toString(),
        carbs: Math.round(primaryFood.nutrition.carbs * multiplier).toString(),
        fat: Math.round(primaryFood.nutrition.fat * multiplier).toString(),
        fiber: primaryFood.nutrition.fiber
          ? Math.round(primaryFood.nutrition.fiber * multiplier).toString()
          : '',
        servingSize: multiplier === 1
          ? primaryFood.portionSize
          : `${multiplier}x ${primaryFood.portionSize}`,
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
        <ActivityIndicator size="large" color={colors.primary.main} />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer} testID="scan-food-permission-screen">
        <Ionicons name="camera-outline" size={64} color={colors.text.disabled} />
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
          <Image
            source={{ uri: capturedImage }}
            style={[
              styles.previewImage,
              (scanResult || usdaResult) && styles.previewImageSmall,
            ]}
          />

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

          {/* USDA Classification Results */}
          {usdaResult && (
            <ScrollView
              style={styles.resultsScrollView}
              contentContainerStyle={[
                styles.resultsContainer,
                { paddingHorizontal: responsiveSpacing.horizontal },
                isTablet && styles.resultsContainerTablet
              ]}
              showsVerticalScrollIndicator={false}
            >
              <ClassificationResult
                classification={usdaResult.classification as FoodClassification}
                usdaMatches={usdaResult.searchResults.foods as unknown as USDAFood[]}
                portionEstimate={usdaResult.portionEstimate}
                onConfirm={handleUSDAConfirm}
                onSearchInstead={handleSearchInstead}
                onReportIncorrect={handleReportIncorrect}
              />
            </ScrollView>
          )}

          {/* Legacy ML-only scan results (fallback) */}
          {scanResult && !usdaResult && (
            <ScrollView
              style={styles.resultsScrollView}
              contentContainerStyle={[
                styles.resultsContainer,
                { paddingHorizontal: responsiveSpacing.horizontal },
                isTablet && styles.resultsContainerTablet
              ]}
              showsVerticalScrollIndicator={false}
            >
              <Text style={styles.resultsTitle}>Detected Food</Text>

              {scanResult.foodItems.map((item: any, index: number) => (
                <View key={index} style={styles.foodItem}>
                  {/* Editable Food Name */}
                  <View style={styles.foodItemHeader}>
                    {index === 0 && isEditingName ? (
                      <TextInput
                        style={styles.foodNameInput}
                        value={editedFoodName.replace(/_/g, ' ')}
                        onChangeText={(text) => setEditedFoodName(text.toLowerCase().replace(/ /g, '_'))}
                        onBlur={() => setIsEditingName(false)}
                        autoFocus
                        selectTextOnFocus
                      />
                    ) : (
                      <TouchableOpacity
                        onPress={() => index === 0 && setIsEditingName(true)}
                        style={styles.foodNameTouchable}
                      >
                        <Text style={styles.foodName}>
                          {index === 0
                            ? (editedFoodName || item.name).replace(/_/g, ' ').split(' ')
                                .map((word: string) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                                .join(' ')
                            : item.name}
                        </Text>
                        {index === 0 && (
                          <Ionicons name="pencil" size={14} color={colors.text.tertiary} style={{ marginLeft: 4 }} />
                        )}
                      </TouchableOpacity>
                    )}
                    <View style={styles.confidenceContainer}>
                      <Text style={styles.confidence}>
                        {Math.round(item.confidence * 100)}% confident
                      </Text>
                    </View>
                  </View>

                  {/* Alternatives Selection - Only for primary food */}
                  {index === 0 && item.alternatives && item.alternatives.length > 0 && (
                    <View style={styles.alternativesSection}>
                      <Text style={styles.alternativesLabel}>Or select:</Text>
                      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.alternativesScroll}>
                        {item.alternatives.slice(0, 5).map((alt: any, altIndex: number) => (
                          <TouchableOpacity
                            key={altIndex}
                            style={[
                              styles.alternativeChip,
                              editedFoodName === alt.name && styles.alternativeChipSelected,
                            ]}
                            onPress={() => handleSelectAlternative(alt.name)}
                          >
                            <Text
                              style={[
                                styles.alternativeChipText,
                                editedFoodName === alt.name && styles.alternativeChipTextSelected,
                              ]}
                            >
                              {(alt.display_name || alt.name).replace(/_/g, ' ').split(' ')
                                .map((word: string) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                                .join(' ')}
                            </Text>
                            {editedFoodName === alt.name && (
                              <Ionicons name="checkmark" size={14} color={colors.primary.main} />
                            )}
                          </TouchableOpacity>
                        ))}
                      </ScrollView>
                    </View>
                  )}

                  {/* Portion Size with Adjustment Controls - Only for primary food */}
                  {index === 0 ? (
                    <View style={styles.portionSection}>
                      <Text style={styles.portionLabel}>Portion:</Text>
                      <View style={styles.portionControls}>
                        <TouchableOpacity
                          style={styles.portionButton}
                          onPress={() => handlePortionAdjust(-0.5)}
                        >
                          <Ionicons name="remove" size={18} color={colors.text.primary} />
                        </TouchableOpacity>
                        <Text style={styles.portionValue}>
                          {editedPortionMultiplier === 1
                            ? item.portionSize
                            : `${editedPortionMultiplier}x (${item.portionSize})`}
                        </Text>
                        <TouchableOpacity
                          style={styles.portionButton}
                          onPress={() => handlePortionAdjust(0.5)}
                        >
                          <Ionicons name="add" size={18} color={colors.text.primary} />
                        </TouchableOpacity>
                      </View>
                    </View>
                  ) : (
                    <Text style={styles.portionSize}>{item.portionSize}</Text>
                  )}

                  {/* Nutrition Grid - Apply multiplier for primary food */}
                  <View style={styles.nutritionGrid}>
                    <View style={styles.nutritionItem}>
                      <Text style={styles.nutritionValue}>
                        {Math.round(item.nutrition.calories * (index === 0 ? editedPortionMultiplier : 1))}
                      </Text>
                      <Text style={styles.nutritionLabel}>cal</Text>
                    </View>
                    <View style={styles.nutritionItem}>
                      <Text style={styles.nutritionValue}>
                        {Math.round(item.nutrition.protein * (index === 0 ? editedPortionMultiplier : 1))}g
                      </Text>
                      <Text style={styles.nutritionLabel}>protein</Text>
                    </View>
                    <View style={styles.nutritionItem}>
                      <Text style={styles.nutritionValue}>
                        {Math.round(item.nutrition.carbs * (index === 0 ? editedPortionMultiplier : 1))}g
                      </Text>
                      <Text style={styles.nutritionLabel}>carbs</Text>
                    </View>
                    <View style={styles.nutritionItem}>
                      <Text style={styles.nutritionValue}>
                        {Math.round(item.nutrition.fat * (index === 0 ? editedPortionMultiplier : 1))}g
                      </Text>
                      <Text style={styles.nutritionLabel}>fat</Text>
                    </View>
                  </View>

                  {/* Feedback Button */}
                  {index === 0 && (
                    <TouchableOpacity
                      style={styles.feedbackButton}
                      onPress={() => setShowFeedbackModal(true)}
                      activeOpacity={0.7}
                    >
                      <Ionicons
                        name="help-circle-outline"
                        size={18}
                        color={colors.text.tertiary}
                      />
                      <Text style={styles.feedbackButtonText}>Report misclassification</Text>
                    </TouchableOpacity>
                  )}
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
            </ScrollView>
          )}
        </View>

        <View style={[
          styles.actionsContainer,
          { paddingHorizontal: responsiveSpacing.horizontal },
          isTablet && styles.actionsContainerTablet
        ]}>
          {/* LiDAR Prompt Banner - Show when AR supported but measurement was skipped */}
          {Platform.OS === 'ios' && hasARSupport && !arMeasurement && !scanResult && !usdaResult && !isAnalyzing && (
            <TouchableOpacity
              style={styles.lidarPromptBanner}
              onPress={handleMeasurePortion}
              activeOpacity={0.8}
            >
              <View style={styles.lidarPromptIcon}>
                <Ionicons name="cube-outline" size={24} color={colors.status.warning} />
              </View>
              <View style={styles.lidarPromptContent}>
                <Text style={styles.lidarPromptTitle}>LiDAR measurement skipped</Text>
                <Text style={styles.lidarPromptSubtitle}>
                  Tap to measure portion size for better accuracy
                </Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color={colors.status.warning} />
            </TouchableOpacity>
          )}

          {/* AR Measurement display */}
          {arMeasurement && !scanResult && !usdaResult && (
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

          {!isAnalyzing && !scanResult && !usdaResult && (
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

          {/* Legacy scan result button - only show for non-USDA scans */}
          {scanResult && !usdaResult && (
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

        {/* Feedback Correction Modal - works with both USDA and legacy results */}
        {(usdaResult || (scanResult && scanResult.foodItems.length > 0)) && (
          <FeedbackCorrectionModal
            visible={showFeedbackModal}
            onClose={() => setShowFeedbackModal(false)}
            originalPrediction={
              usdaResult
                ? usdaResult.classification.category
                : scanResult?.foodItems[0].name || ''
            }
            originalConfidence={
              usdaResult
                ? usdaResult.classification.confidence
                : scanResult?.foodItems[0].confidence || 0
            }
            alternatives={
              usdaResult
                ? usdaResult.classification.alternatives.map((alt) => ({
                    name: alt.category,
                    confidence: alt.confidence,
                  }))
                : scanResult?.foodItems[0].alternatives
            }
            imageHash={usdaResult?.imageHash || scanResult?.imageHash || ''}
            onFeedbackSubmitted={() => {
              // Could show a toast or update UI here
            }}
          />
        )}
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
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
  },
  closeButton: {
    width: 44,
    height: 44,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraControls: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  controlButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.overlay.light,
    justifyContent: 'center',
    alignItems: 'center',
  },
  guideOverlay: {
    position: 'absolute',
    top: '40%',
    left: spacing.lg,
    right: spacing.lg,
    alignItems: 'center',
  },
  guideBox: {
    backgroundColor: colors.overlay.medium,
    padding: spacing.lg,
    borderRadius: borderRadius.lg,
  },
  guideText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    textAlign: 'center',
    marginBottom: spacing.sm,
  },
  guideSubtext: {
    color: colors.text.tertiary,
    fontSize: typography.fontSize.sm,
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
    backgroundColor: colors.camera.button,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: colors.camera.buttonInner,
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
  previewImageSmall: {
    height: '30%',
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
  resultsScrollView: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  resultsContainer: {
    flexGrow: 1,
    padding: spacing.lg,
  },
  resultsContainerTablet: {
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
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
  foodNameTouchable: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  foodNameInput: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    flex: 1,
    borderBottomWidth: 1,
    borderBottomColor: colors.primary.main,
    paddingVertical: spacing.xs,
  },
  alternativesSection: {
    marginBottom: spacing.md,
  },
  alternativesLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  alternativesScroll: {
    flexDirection: 'row',
  },
  alternativeChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.full,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    marginRight: spacing.sm,
  },
  alternativeChipSelected: {
    backgroundColor: colors.special.highlight,
    borderColor: colors.primary.main,
  },
  alternativeChipText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.secondary,
  },
  alternativeChipTextSelected: {
    color: colors.primary.main,
    fontWeight: typography.fontWeight.semibold,
  },
  portionSection: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  portionLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginRight: spacing.sm,
  },
  portionControls: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  portionButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  portionValue: {
    fontSize: typography.fontSize.sm,
    color: colors.text.primary,
    fontWeight: typography.fontWeight.semibold,
    marginHorizontal: spacing.md,
    flex: 1,
    textAlign: 'center',
  },
  confidenceContainer: {
    backgroundColor: colors.primary.main + '20',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
  },
  confidence: {
    fontSize: typography.fontSize.xs,
    color: colors.primary.main,
    fontWeight: typography.fontWeight.semibold,
  },
  feedbackButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
    marginTop: spacing.md,
    paddingVertical: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  feedbackButtonText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
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
  actionsContainerTablet: {
    alignItems: 'center',
  },
  analyzeButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    width: '100%',
    maxWidth: FORM_MAX_WIDTH,
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
    width: '100%',
    maxWidth: FORM_MAX_WIDTH,
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
    width: '100%',
    maxWidth: FORM_MAX_WIDTH,
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
  lidarPromptBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.status.warning + '15',
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.status.warning + '40',
    width: '100%',
    maxWidth: FORM_MAX_WIDTH,
  },
  lidarPromptIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.status.warning + '20',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  lidarPromptContent: {
    flex: 1,
  },
  lidarPromptTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  lidarPromptSubtitle: {
    fontSize: typography.fontSize.xs,
    color: colors.text.secondary,
  },
});
