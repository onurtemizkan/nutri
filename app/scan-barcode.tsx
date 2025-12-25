/**
 * Barcode Scanner Screen
 *
 * Scans food product barcodes and fetches nutrition information
 * from the Open Food Facts database.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Image,
  Animated,
  TextInput,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from 'react-native';
import { CameraView, useCameraPermissions, BarcodeScanningResult } from 'expo-camera';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import { Ionicons } from '@expo/vector-icons';
import {
  fetchProductByBarcode,
  calculateServingNutrition,
  formatNutritionGrade,
  getNutritionGradeColor,
} from '@/lib/api/openfoodfacts';
import { showAlert } from '@/lib/utils/alert';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import type { BarcodeProduct, BarcodeScannerState } from '@/lib/types/barcode';

// Supported barcode types for food products
const SUPPORTED_BARCODE_TYPES = ['ean13', 'ean8', 'upc_a', 'upc_e'] as const;

// Cooldown between scans (ms)
const SCAN_COOLDOWN = 2000;

export default function ScanBarcodeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [state, setState] = useState<BarcodeScannerState>({
    isScanning: true,
    hasPermission: null,
    lastScannedBarcode: null,
    isLoading: false,
    error: null,
  });
  const [product, setProduct] = useState<BarcodeProduct | null>(null);
  const [servingAmount, setServingAmount] = useState<string>('100');
  const [manualBarcode, setManualBarcode] = useState<string>('');
  const [showManualEntry, setShowManualEntry] = useState(false);

  const lastScanTime = useRef<number>(0);
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  // Scanning animation
  const scanLineAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    let animation: Animated.CompositeAnimation | null = null;

    if (state.isScanning && !state.isLoading && !product) {
      animation = Animated.loop(
        Animated.sequence([
          Animated.timing(scanLineAnim, {
            toValue: 1,
            duration: 2000,
            useNativeDriver: true,
          }),
          Animated.timing(scanLineAnim, {
            toValue: 0,
            duration: 2000,
            useNativeDriver: true,
          }),
        ])
      );
      animation.start();
    } else {
      scanLineAnim.setValue(0);
    }

    // Cleanup: stop animation on unmount or when dependencies change
    return () => {
      animation?.stop();
    };
  }, [state.isScanning, state.isLoading, product, scanLineAnim]);

  // Request permission on mount
  useEffect(() => {
    if (permission && !permission.granted) {
      requestPermission();
    }
  }, [permission, requestPermission]);

  /**
   * Handles barcode detection from camera
   */
  const handleBarcodeScanned = useCallback(
    async (result: BarcodeScanningResult) => {
      // Ignore if loading, already have product, or in cooldown
      if (state.isLoading || product) return;

      const now = Date.now();
      if (now - lastScanTime.current < SCAN_COOLDOWN) return;

      // Check if barcode type is supported
      const barcodeType = result.type.toLowerCase();
      if (
        !SUPPORTED_BARCODE_TYPES.includes(barcodeType as (typeof SUPPORTED_BARCODE_TYPES)[number])
      ) {
        return;
      }

      lastScanTime.current = now;

      // Haptic feedback on scan
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

      setState((prev) => ({
        ...prev,
        isScanning: false,
        isLoading: true,
        lastScannedBarcode: result.data,
        error: null,
      }));

      // Fetch product info
      const scanResult = await fetchProductByBarcode(result.data);

      if (scanResult.success && scanResult.product) {
        setProduct(scanResult.product);
        setServingAmount(scanResult.product.servingQuantity?.toString() || '100');
        await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      } else {
        setState((prev) => ({
          ...prev,
          error: scanResult.error || {
            type: 'API_ERROR',
            message: 'Failed to fetch product',
          },
        }));
        await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      }

      setState((prev) => ({
        ...prev,
        isLoading: false,
      }));
    },
    [state.isLoading, product]
  );

  /**
   * Handles manual barcode entry
   */
  const handleManualSubmit = async () => {
    if (!manualBarcode.trim()) {
      showAlert('Error', 'Please enter a barcode');
      return;
    }

    setState((prev) => ({
      ...prev,
      isScanning: false,
      isLoading: true,
      lastScannedBarcode: manualBarcode.trim(),
      error: null,
    }));

    const scanResult = await fetchProductByBarcode(manualBarcode.trim());

    if (scanResult.success && scanResult.product) {
      setProduct(scanResult.product);
      setServingAmount(scanResult.product.servingQuantity?.toString() || '100');
      setShowManualEntry(false);
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } else {
      setState((prev) => ({
        ...prev,
        error: scanResult.error || {
          type: 'API_ERROR',
          message: 'Failed to fetch product',
        },
      }));
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    }

    setState((prev) => ({
      ...prev,
      isLoading: false,
    }));
  };

  /**
   * Resets scanner to scan again
   */
  const handleScanAgain = () => {
    setProduct(null);
    setManualBarcode('');
    setState({
      isScanning: true,
      hasPermission: permission?.granted ?? null,
      lastScannedBarcode: null,
      isLoading: false,
      error: null,
    });
    lastScanTime.current = 0;
  };

  /**
   * Adds scanned product to meal
   */
  const handleAddToMeal = () => {
    if (!product) return;

    const serving = parseFloat(servingAmount) || 100;
    const nutrition = calculateServingNutrition(product, serving);

    // Build params with all available nutrition data including micronutrients
    const params: Record<string, string> = {
      name: product.brand ? `${product.name} (${product.brand})` : product.name,
      calories: nutrition.calories.toString(),
      protein: nutrition.protein.toString(),
      carbs: nutrition.carbs.toString(),
      fat: nutrition.fat.toString(),
      servingSize: `${serving}g`,
      fromScan: 'true',
      barcode: product.barcode,
    };

    // Add optional nutrients only if they have values
    if (nutrition.fiber !== undefined) params.fiber = nutrition.fiber.toString();
    if (nutrition.sugar !== undefined) params.sugar = nutrition.sugar.toString();

    // Fat breakdown
    if (nutrition.saturatedFat !== undefined)
      params.saturatedFat = nutrition.saturatedFat.toString();
    if (nutrition.transFat !== undefined) params.transFat = nutrition.transFat.toString();
    if (nutrition.cholesterol !== undefined) params.cholesterol = nutrition.cholesterol.toString();

    // Minerals
    if (nutrition.sodium !== undefined) params.sodium = nutrition.sodium.toString();
    if (nutrition.potassium !== undefined) params.potassium = nutrition.potassium.toString();
    if (nutrition.calcium !== undefined) params.calcium = nutrition.calcium.toString();
    if (nutrition.iron !== undefined) params.iron = nutrition.iron.toString();
    if (nutrition.magnesium !== undefined) params.magnesium = nutrition.magnesium.toString();
    if (nutrition.zinc !== undefined) params.zinc = nutrition.zinc.toString();
    if (nutrition.phosphorus !== undefined) params.phosphorus = nutrition.phosphorus.toString();

    // Vitamins
    if (nutrition.vitaminA !== undefined) params.vitaminA = nutrition.vitaminA.toString();
    if (nutrition.vitaminC !== undefined) params.vitaminC = nutrition.vitaminC.toString();
    if (nutrition.vitaminD !== undefined) params.vitaminD = nutrition.vitaminD.toString();
    if (nutrition.vitaminE !== undefined) params.vitaminE = nutrition.vitaminE.toString();
    if (nutrition.vitaminK !== undefined) params.vitaminK = nutrition.vitaminK.toString();
    if (nutrition.vitaminB6 !== undefined) params.vitaminB6 = nutrition.vitaminB6.toString();
    if (nutrition.vitaminB12 !== undefined) params.vitaminB12 = nutrition.vitaminB12.toString();
    if (nutrition.folate !== undefined) params.folate = nutrition.folate.toString();
    if (nutrition.thiamin !== undefined) params.thiamin = nutrition.thiamin.toString();
    if (nutrition.riboflavin !== undefined) params.riboflavin = nutrition.riboflavin.toString();
    if (nutrition.niacin !== undefined) params.niacin = nutrition.niacin.toString();

    router.push({
      pathname: '/add-meal',
      params,
    });
  };

  // Permission loading
  if (!permission) {
    return (
      <View style={styles.permissionContainer}>
        <ActivityIndicator size="large" color={colors.primary.main} />
      </View>
    );
  }

  // Permission denied
  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer} testID="scan-barcode-permission-screen">
        <Ionicons name="barcode-outline" size={64} color={colors.text.tertiary} />
        <Text style={styles.permissionText}>Camera access is required</Text>
        <Text style={styles.permissionSubtext}>We need camera access to scan product barcodes</Text>
        <TouchableOpacity
          style={styles.permissionButton}
          onPress={requestPermission}
          activeOpacity={0.8}
          testID="scan-barcode-grant-permission-button"
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

  // Product found - show details
  if (product) {
    const serving = parseFloat(servingAmount) || 100;
    const nutrition = calculateServingNutrition(product, serving);
    const nutriscoreGrade = formatNutritionGrade(product.nutriscoreGrade);
    const nutriscoreColor = getNutritionGradeColor(product.nutriscoreGrade);

    return (
      <SafeAreaView style={styles.container} testID="scan-barcode-result-screen">
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => router.back()}
            accessibilityLabel="Go back"
            accessibilityRole="button"
            style={styles.headerBackButton}
          >
            <Ionicons name="chevron-back" size={24} color={colors.primary.main} />
            <Text style={styles.headerButton}>Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Product Found</Text>
          <TouchableOpacity
            onPress={handleScanAgain}
            testID="scan-barcode-scan-again-button"
            accessibilityLabel="Scan another barcode"
            accessibilityRole="button"
          >
            <Text style={styles.headerButton}>Scan Again</Text>
          </TouchableOpacity>
        </View>

        <ScrollView
          style={styles.resultScrollView}
          contentContainerStyle={[
            styles.resultContainer,
            { paddingHorizontal: responsiveSpacing.horizontal },
            isTablet && styles.resultContainerTablet,
          ]}
          showsVerticalScrollIndicator={false}
        >
          {/* Product image and basic info */}
          <View style={styles.productHeader}>
            {product.imageUrl ? (
              <Image
                source={{ uri: product.imageUrl }}
                style={styles.productImage}
                resizeMode="contain"
              />
            ) : (
              <View style={styles.productImagePlaceholder}>
                <Ionicons name="cube-outline" size={48} color={colors.text.tertiary} />
              </View>
            )}
            <View style={styles.productInfo}>
              <Text style={styles.productName}>{product.name}</Text>
              {product.brand && <Text style={styles.productBrand}>{product.brand}</Text>}
              <Text style={styles.barcodeText}>
                <Ionicons name="barcode-outline" size={14} color={colors.text.tertiary} />{' '}
                {product.barcode}
              </Text>
            </View>
          </View>

          {/* Nutriscore badge */}
          {nutriscoreGrade && (
            <View style={styles.nutriscoreContainer}>
              <Text style={styles.nutriscoreLabel}>Nutri-Score</Text>
              <View style={[styles.nutriscoreBadge, { backgroundColor: nutriscoreColor }]}>
                <Text style={styles.nutriscoreGrade}>{nutriscoreGrade}</Text>
              </View>
            </View>
          )}

          {/* Serving size input */}
          <View style={styles.servingContainer}>
            <Text style={styles.servingLabel}>Serving Size</Text>
            <View style={styles.servingInputRow}>
              <TextInput
                style={styles.servingInput}
                value={servingAmount}
                onChangeText={setServingAmount}
                keyboardType="numeric"
                placeholder="100"
                placeholderTextColor={colors.text.tertiary}
                testID="scan-barcode-serving-input"
              />
              <Text style={styles.servingUnit}>grams</Text>
            </View>
            {product.servingSize && (
              <Text style={styles.servingSuggestion}>Suggested serving: {product.servingSize}</Text>
            )}
          </View>

          {/* Nutrition info */}
          <View style={styles.nutritionCard}>
            <Text style={styles.nutritionTitle}>Nutrition for {serving}g serving</Text>

            <View style={styles.nutritionGrid}>
              <View style={styles.nutritionItem}>
                <Text style={styles.nutritionValue}>{nutrition.calories}</Text>
                <Text style={styles.nutritionLabel}>Calories</Text>
              </View>
              <View style={styles.nutritionItem}>
                <Text style={styles.nutritionValue}>{nutrition.protein}g</Text>
                <Text style={styles.nutritionLabel}>Protein</Text>
              </View>
              <View style={styles.nutritionItem}>
                <Text style={styles.nutritionValue}>{nutrition.carbs}g</Text>
                <Text style={styles.nutritionLabel}>Carbs</Text>
              </View>
              <View style={styles.nutritionItem}>
                <Text style={styles.nutritionValue}>{nutrition.fat}g</Text>
                <Text style={styles.nutritionLabel}>Fat</Text>
              </View>
            </View>

            {(nutrition.fiber !== undefined || nutrition.sugar !== undefined) && (
              <View style={styles.nutritionSecondaryGrid}>
                {nutrition.fiber !== undefined && (
                  <View style={styles.nutritionSecondaryItem}>
                    <Text style={styles.nutritionSecondaryValue}>{nutrition.fiber}g</Text>
                    <Text style={styles.nutritionSecondaryLabel}>Fiber</Text>
                  </View>
                )}
                {nutrition.sugar !== undefined && (
                  <View style={styles.nutritionSecondaryItem}>
                    <Text style={styles.nutritionSecondaryValue}>{nutrition.sugar}g</Text>
                    <Text style={styles.nutritionSecondaryLabel}>Sugar</Text>
                  </View>
                )}
              </View>
            )}
          </View>

          {/* Allergens */}
          {product.allergens && product.allergens.length > 0 && (
            <View style={styles.allergensContainer}>
              <View style={styles.allergensHeader}>
                <Ionicons name="warning-outline" size={18} color={colors.status.warning} />
                <Text style={styles.allergensTitle}>Allergens</Text>
              </View>
              <Text style={styles.allergensText}>{product.allergens.join(', ')}</Text>
            </View>
          )}

          {/* Ingredients */}
          {product.ingredients && (
            <View style={styles.ingredientsContainer}>
              <Text style={styles.ingredientsTitle}>Ingredients</Text>
              <Text style={styles.ingredientsText}>{product.ingredients}</Text>
            </View>
          )}
        </ScrollView>

        {/* Add to meal button */}
        <View
          style={[
            styles.actionsContainer,
            { paddingHorizontal: responsiveSpacing.horizontal },
            isTablet && styles.actionsContainerTablet,
          ]}
        >
          <TouchableOpacity
            style={styles.addButton}
            onPress={handleAddToMeal}
            activeOpacity={0.8}
            testID="scan-barcode-add-button"
          >
            <LinearGradient
              colors={gradients.primary}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.addButtonGradient}
            >
              <Text style={styles.addButtonText}>Add to Meal</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // Camera view with barcode scanner
  return (
    <View style={styles.container} testID="scan-barcode-camera-screen">
      <CameraView
        style={styles.camera}
        facing="back"
        barcodeScannerSettings={{
          barcodeTypes: ['ean13', 'ean8', 'upc_a', 'upc_e'],
        }}
        onBarcodeScanned={state.isScanning ? handleBarcodeScanned : undefined}
      >
        {/* Header */}
        <SafeAreaView style={styles.cameraHeader}>
          <TouchableOpacity
            style={styles.closeButton}
            onPress={() => router.back()}
            testID="scan-barcode-close-button"
            accessibilityLabel="Go back"
            accessibilityRole="button"
          >
            <Ionicons name="chevron-back" size={28} color={colors.text.primary} />
          </TouchableOpacity>

          <Text style={styles.cameraTitle}>Scan Barcode</Text>

          <TouchableOpacity
            style={styles.manualButton}
            onPress={() => setShowManualEntry(true)}
            testID="scan-barcode-manual-button"
          >
            <Ionicons name="keypad-outline" size={24} color={colors.text.primary} />
          </TouchableOpacity>
        </SafeAreaView>

        {/* Scanning frame */}
        <View style={styles.scannerFrame}>
          <View style={styles.scannerBox}>
            {/* Corner decorations */}
            <View style={[styles.corner, styles.cornerTopLeft]} />
            <View style={[styles.corner, styles.cornerTopRight]} />
            <View style={[styles.corner, styles.cornerBottomLeft]} />
            <View style={[styles.corner, styles.cornerBottomRight]} />

            {/* Animated scan line */}
            {state.isScanning && !state.isLoading && (
              <Animated.View
                style={[
                  styles.scanLine,
                  {
                    transform: [
                      {
                        translateY: scanLineAnim.interpolate({
                          inputRange: [0, 1],
                          outputRange: [0, 180],
                        }),
                      },
                    ],
                  },
                ]}
              >
                <LinearGradient
                  colors={['transparent', colors.primary.main, 'transparent']}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={styles.scanLineGradient}
                />
              </Animated.View>
            )}
          </View>

          <Text style={styles.scannerHint}>Position the barcode within the frame</Text>
        </View>
      </CameraView>

      {/* Loading overlay - OUTSIDE CameraView for proper touch handling */}
      {state.isLoading && (
        <View style={styles.overlayContainer}>
          <BlurView intensity={80} tint="dark" style={styles.loadingOverlay}>
            <View style={styles.loadingCard}>
              <ActivityIndicator size="large" color={colors.primary.main} />
              <Text style={styles.loadingText}>Looking up product...</Text>
              <Text style={styles.loadingBarcode}>{state.lastScannedBarcode}</Text>
            </View>
          </BlurView>
        </View>
      )}

      {/* Error overlay - OUTSIDE CameraView for proper touch handling */}
      {state.error && !state.isLoading && (
        <View style={styles.overlayContainer}>
          <BlurView intensity={80} tint="dark" style={styles.errorOverlay}>
            <View style={styles.errorCard}>
              <Ionicons
                name={
                  state.error.type === 'PRODUCT_NOT_FOUND'
                    ? 'search-outline'
                    : 'alert-circle-outline'
                }
                size={48}
                color={
                  state.error.type === 'PRODUCT_NOT_FOUND'
                    ? colors.status.warning
                    : colors.status.error
                }
              />
              <Text style={styles.errorTitle}>
                {state.error.type === 'PRODUCT_NOT_FOUND'
                  ? 'Product Not Found'
                  : state.error.type === 'NETWORK_ERROR'
                    ? 'Connection Error'
                    : 'Something Went Wrong'}
              </Text>
              <Text style={styles.errorMessage}>
                {state.error.type === 'PRODUCT_NOT_FOUND'
                  ? "This product isn't in the Open Food Facts database yet. You can add it manually or try scanning a different product."
                  : state.error.type === 'NETWORK_ERROR'
                    ? 'Please check your internet connection and try again.'
                    : state.error.message}
              </Text>
              {state.error.barcode && (
                <Text style={styles.errorBarcode}>Barcode: {state.error.barcode}</Text>
              )}
              <TouchableOpacity
                style={styles.errorButton}
                onPress={handleScanAgain}
                activeOpacity={0.8}
                testID="scan-barcode-try-again-button"
              >
                <Text style={styles.errorButtonText}>Scan Another</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.errorSecondaryButton}
                onPress={() => {
                  router.replace('/add-meal');
                }}
                activeOpacity={0.8}
                testID="scan-barcode-enter-manually-button"
              >
                <Text style={styles.errorSecondaryButtonText}>Enter Manually Instead</Text>
              </TouchableOpacity>
            </View>
          </BlurView>
        </View>
      )}

      {/* Manual entry modal - OUTSIDE CameraView for proper touch handling */}
      {showManualEntry && (
        <View style={styles.overlayContainer}>
          <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            style={styles.manualEntryOverlay}
          >
            <BlurView intensity={90} tint="dark" style={styles.manualEntryBlur}>
              <View style={styles.manualEntryCard}>
                <View style={styles.manualEntryHeader}>
                  <Text style={styles.manualEntryTitle}>Enter Barcode</Text>
                  <TouchableOpacity
                    onPress={() => setShowManualEntry(false)}
                    testID="scan-barcode-close-manual-button"
                  >
                    <Ionicons name="close" size={24} color={colors.text.secondary} />
                  </TouchableOpacity>
                </View>

                <TextInput
                  style={styles.manualEntryInput}
                  value={manualBarcode}
                  onChangeText={setManualBarcode}
                  keyboardType="numeric"
                  placeholder="Enter barcode number"
                  placeholderTextColor={colors.text.tertiary}
                  autoFocus
                  testID="scan-barcode-manual-input"
                />

                <TouchableOpacity
                  style={styles.manualEntrySubmit}
                  onPress={handleManualSubmit}
                  activeOpacity={0.8}
                  disabled={!manualBarcode.trim()}
                  testID="scan-barcode-manual-submit-button"
                >
                  <LinearGradient
                    colors={
                      manualBarcode.trim()
                        ? gradients.primary
                        : [colors.background.tertiary, colors.background.tertiary]
                    }
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={styles.manualEntrySubmitGradient}
                  >
                    <Text
                      style={[
                        styles.manualEntrySubmitText,
                        !manualBarcode.trim() && styles.manualEntrySubmitTextDisabled,
                      ]}
                    >
                      Look Up Product
                    </Text>
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            </BlurView>
          </KeyboardAvoidingView>
        </View>
      )}
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
  cameraTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  manualButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.overlay.light,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scannerFrame: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scannerBox: {
    width: 280,
    height: 200,
    position: 'relative',
  },
  corner: {
    position: 'absolute',
    width: 30,
    height: 30,
    borderColor: colors.primary.main,
  },
  cornerTopLeft: {
    top: 0,
    left: 0,
    borderTopWidth: 3,
    borderLeftWidth: 3,
    borderTopLeftRadius: 8,
  },
  cornerTopRight: {
    top: 0,
    right: 0,
    borderTopWidth: 3,
    borderRightWidth: 3,
    borderTopRightRadius: 8,
  },
  cornerBottomLeft: {
    bottom: 0,
    left: 0,
    borderBottomWidth: 3,
    borderLeftWidth: 3,
    borderBottomLeftRadius: 8,
  },
  cornerBottomRight: {
    bottom: 0,
    right: 0,
    borderBottomWidth: 3,
    borderRightWidth: 3,
    borderBottomRightRadius: 8,
  },
  scanLine: {
    position: 'absolute',
    left: 10,
    right: 10,
    height: 2,
    top: 10,
  },
  scanLineGradient: {
    flex: 1,
  },
  scannerHint: {
    marginTop: spacing.xl,
    fontSize: typography.fontSize.sm,
    color: colors.camera.textLight,
    textAlign: 'center',
  },
  overlayContainer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 100,
  },
  loadingOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingCard: {
    backgroundColor: colors.overlay.medium,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    alignItems: 'center',
    minWidth: 200,
  },
  loadingText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    marginTop: spacing.md,
  },
  loadingBarcode: {
    color: colors.text.tertiary,
    fontSize: typography.fontSize.sm,
    marginTop: spacing.sm,
  },
  errorOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
  },
  errorCard: {
    backgroundColor: colors.overlay.heavy,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    alignItems: 'center',
    maxWidth: 320,
    width: '100%',
  },
  errorTitle: {
    color: colors.text.primary,
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    marginTop: spacing.md,
    marginBottom: spacing.sm,
  },
  errorMessage: {
    color: colors.text.secondary,
    fontSize: typography.fontSize.sm,
    textAlign: 'center',
    marginBottom: spacing.sm,
  },
  errorBarcode: {
    color: colors.camera.textDim,
    fontSize: typography.fontSize.xs,
    marginBottom: spacing.lg,
  },
  errorButton: {
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.md,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.xl,
    marginBottom: spacing.sm,
    width: '100%',
  },
  errorButtonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    textAlign: 'center',
  },
  errorSecondaryButton: {
    paddingVertical: spacing.sm,
    width: '100%',
  },
  errorSecondaryButtonText: {
    color: colors.primary.light,
    fontSize: typography.fontSize.sm,
    textAlign: 'center',
  },
  manualEntryOverlay: {
    flex: 1,
  },
  manualEntryBlur: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
  },
  manualEntryCard: {
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    width: '100%',
    maxWidth: 360,
  },
  manualEntryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.lg,
  },
  manualEntryTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  manualEntryInput: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    fontSize: typography.fontSize.lg,
    color: colors.text.primary,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    marginBottom: spacing.lg,
    textAlign: 'center',
    letterSpacing: 2,
  },
  manualEntrySubmit: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  manualEntrySubmitGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  manualEntrySubmitText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
  },
  manualEntrySubmitTextDisabled: {
    color: colors.text.tertiary,
  },
  // Result screen styles
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
  headerBackButton: {
    flexDirection: 'row',
    alignItems: 'center',
    marginLeft: -spacing.xs,
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
  resultScrollView: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  resultContainer: {
    flexGrow: 1,
    padding: spacing.lg,
  },
  resultContainerTablet: {
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
  },
  productHeader: {
    flexDirection: 'row',
    marginBottom: spacing.lg,
  },
  productImage: {
    width: 100,
    height: 100,
    borderRadius: borderRadius.md,
    backgroundColor: colors.background.tertiary,
  },
  productImagePlaceholder: {
    width: 100,
    height: 100,
    borderRadius: borderRadius.md,
    backgroundColor: colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  productInfo: {
    flex: 1,
    marginLeft: spacing.md,
    justifyContent: 'center',
  },
  productName: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  productBrand: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginBottom: spacing.xs,
  },
  barcodeText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  nutriscoreContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.lg,
  },
  nutriscoreLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginRight: spacing.sm,
  },
  nutriscoreBadge: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  nutriscoreGrade: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: '#fff',
  },
  servingContainer: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  servingLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
  },
  servingInputRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  servingInput: {
    flex: 1,
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.sm,
    padding: spacing.sm,
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    marginRight: spacing.sm,
  },
  servingUnit: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
  },
  servingSuggestion: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.sm,
  },
  nutritionCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.md,
  },
  nutritionTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.md,
    textAlign: 'center',
  },
  nutritionGrid: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: spacing.md,
  },
  nutritionItem: {
    alignItems: 'center',
  },
  nutritionValue: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  nutritionLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  nutritionSecondaryGrid: {
    flexDirection: 'row',
    justifyContent: 'center',
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
    gap: spacing.xl,
  },
  nutritionSecondaryItem: {
    alignItems: 'center',
  },
  nutritionSecondaryValue: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
  },
  nutritionSecondaryLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  allergensContainer: {
    backgroundColor: colors.special.warningLight,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.status.warning,
  },
  allergensHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  allergensTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.status.warning,
    marginLeft: spacing.sm,
  },
  allergensText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.primary,
    textTransform: 'capitalize',
  },
  ingredientsContainer: {
    marginBottom: spacing.lg,
  },
  ingredientsTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
  },
  ingredientsText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    lineHeight: 20,
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
  addButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    width: '100%',
    maxWidth: FORM_MAX_WIDTH,
    ...shadows.md,
  },
  addButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  addButtonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
  },
});
