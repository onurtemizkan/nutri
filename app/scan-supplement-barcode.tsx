/**
 * Supplement Barcode Scanner Screen
 *
 * Scans supplement product barcodes and extracts dosage information
 * for one serving (tablet/capsule/scoop) rather than whole bottle.
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
import {
  CameraView,
  useCameraPermissions,
  BarcodeScanningResult,
} from 'expo-camera';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';
import { Ionicons } from '@expo/vector-icons';
import { fetchProductByBarcode } from '@/lib/api/openfoodfacts';
import { showAlert } from '@/lib/utils/alert';
import {
  colors,
  gradients,
  spacing,
  borderRadius,
  typography,
} from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import type { BarcodeProduct, BarcodeScannerState } from '@/lib/types/barcode';

// Supported barcode types
const SUPPORTED_BARCODE_TYPES = ['ean13', 'ean8', 'upc_a', 'upc_e'] as const;

// Cooldown between scans (ms)
const SCAN_COOLDOWN = 2000;

/**
 * Parses serving size string to extract serving type and dosage
 */
function parseSupplementServing(servingSize?: string): {
  servingType: string;
  servingAmount: number;
  dosageAmount?: number;
  dosageUnit?: string;
} {
  if (!servingSize) {
    return { servingType: 'serving', servingAmount: 1 };
  }

  const lowerServing = servingSize.toLowerCase();

  // Detect serving type
  const servingTypes = [
    { keywords: ['tablet', 'tab'], type: 'tablet', unitPlural: 'tablets' },
    { keywords: ['capsule', 'cap', 'vcap'], type: 'capsule', unitPlural: 'capsules' },
    { keywords: ['softgel', 'soft gel'], type: 'softgel', unitPlural: 'softgels' },
    { keywords: ['gummy', 'gummies'], type: 'gummy', unitPlural: 'gummies' },
    { keywords: ['scoop'], type: 'scoop', unitPlural: 'scoops' },
    { keywords: ['dropper', 'drop'], type: 'drop', unitPlural: 'drops' },
    { keywords: ['lozenge'], type: 'lozenge', unitPlural: 'lozenges' },
    { keywords: ['chewable', 'chew'], type: 'chewable', unitPlural: 'chewables' },
  ];

  let servingType = 'serving';
  for (const st of servingTypes) {
    if (st.keywords.some(kw => lowerServing.includes(kw))) {
      servingType = st.type;
      break;
    }
  }

  // Extract serving amount (e.g., "2 tablets" -> 2)
  const amountMatch = servingSize.match(/^(\d+(?:\.\d+)?)\s*/);
  const servingAmount = amountMatch ? parseFloat(amountMatch[1]) : 1;

  // Extract dosage from parentheses (e.g., "(500mg)" or "(5g)")
  let dosageAmount: number | undefined;
  let dosageUnit: string | undefined;

  // Try different patterns to extract dosage
  const patterns = [
    /\((\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|iu|ml)\)/i,
    /(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|iu|ml)(?:\s|$)/i,
  ];

  for (const pattern of patterns) {
    const match = servingSize.match(pattern);
    if (match) {
      dosageAmount = parseFloat(match[1]);
      dosageUnit = match[2].toLowerCase();
      if (dosageUnit === 'μg') dosageUnit = 'mcg';
      if (dosageUnit === 'iu') dosageUnit = 'IU';
      break;
    }
  }

  // If serving is "2 tablets (1000mg)", calculate per-tablet dosage
  if (dosageAmount && servingAmount > 1) {
    dosageAmount = Math.round((dosageAmount / servingAmount) * 10) / 10;
  }

  return { servingType, servingAmount, dosageAmount, dosageUnit };
}

/**
 * Extracts supplement name from product name
 * Removes common brand suffixes and cleans up
 */
function cleanSupplementName(name: string, brand?: string): string {
  let cleanName = name;

  // Remove brand from name if it appears at start
  if (brand && cleanName.toLowerCase().startsWith(brand.toLowerCase())) {
    cleanName = cleanName.substring(brand.length).trim();
    // Remove leading dash or colon
    cleanName = cleanName.replace(/^[-:]\s*/, '');
  }

  // Common patterns to clean
  const patternsToRemove = [
    /\s*-\s*\d+\s*(tablets?|capsules?|softgels?|count)/i,
    /\s*\(\d+\s*(tablets?|capsules?|softgels?|count)\)/i,
    /\s*,\s*\d+\s*(tablets?|capsules?|softgels?|count)/i,
  ];

  for (const pattern of patternsToRemove) {
    cleanName = cleanName.replace(pattern, '');
  }

  return cleanName.trim() || name;
}

interface SupplementScanResult {
  name: string;
  brand?: string;
  dosageAmount?: number;
  dosageUnit?: string;
  servingType: string;
  imageUrl?: string;
  barcode: string;
}

export default function ScanSupplementBarcodeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [state, setState] = useState<BarcodeScannerState>({
    isScanning: true,
    hasPermission: null,
    lastScannedBarcode: null,
    isLoading: false,
    error: null,
  });
  const [result, setResult] = useState<SupplementScanResult | null>(null);
  const [manualBarcode, setManualBarcode] = useState<string>('');
  const [showManualEntry, setShowManualEntry] = useState(false);

  const lastScanTime = useRef<number>(0);
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();

  // Scanning animation
  const scanLineAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (state.isScanning && !state.isLoading && !result) {
      Animated.loop(
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
      ).start();
    } else {
      scanLineAnim.setValue(0);
    }
  }, [state.isScanning, state.isLoading, result, scanLineAnim]);

  // Request permission on mount
  useEffect(() => {
    if (permission && !permission.granted) {
      requestPermission();
    }
  }, [permission, requestPermission]);

  /**
   * Process scanned product for supplement data
   */
  const processSupplementProduct = (product: BarcodeProduct): SupplementScanResult => {
    const parsed = parseSupplementServing(product.servingSize);
    const cleanName = cleanSupplementName(product.name, product.brand);

    return {
      name: cleanName,
      brand: product.brand,
      dosageAmount: parsed.dosageAmount,
      dosageUnit: parsed.dosageUnit,
      servingType: parsed.servingType,
      imageUrl: product.imageUrl,
      barcode: product.barcode,
    };
  };

  /**
   * Handles barcode detection from camera
   */
  const handleBarcodeScanned = useCallback(
    async (scanResult: BarcodeScanningResult) => {
      if (state.isLoading || result) return;

      const now = Date.now();
      if (now - lastScanTime.current < SCAN_COOLDOWN) return;

      const barcodeType = scanResult.type.toLowerCase();
      if (
        !SUPPORTED_BARCODE_TYPES.includes(
          barcodeType as (typeof SUPPORTED_BARCODE_TYPES)[number]
        )
      ) {
        return;
      }

      lastScanTime.current = now;

      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

      setState((prev) => ({
        ...prev,
        isScanning: false,
        isLoading: true,
        lastScannedBarcode: scanResult.data,
        error: null,
      }));

      const fetchResult = await fetchProductByBarcode(scanResult.data);

      if (fetchResult.success && fetchResult.product) {
        const supplementData = processSupplementProduct(fetchResult.product);
        setResult(supplementData);
        await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      } else {
        setState((prev) => ({
          ...prev,
          error: fetchResult.error || {
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
    [state.isLoading, result]
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

    const fetchResult = await fetchProductByBarcode(manualBarcode.trim());

    if (fetchResult.success && fetchResult.product) {
      const supplementData = processSupplementProduct(fetchResult.product);
      setResult(supplementData);
      setShowManualEntry(false);
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } else {
      setState((prev) => ({
        ...prev,
        error: fetchResult.error || {
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
    setResult(null);
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
   * Confirms and navigates back with supplement data
   */
  const handleUseThisSupplement = () => {
    if (!result) return;

    router.replace({
      pathname: '/supplements',
      params: {
        fromScan: 'true',
        name: result.name,
        brand: result.brand || '',
        dosageAmount: result.dosageAmount?.toString() || '',
        dosageUnit: result.dosageUnit || 'mg',
        servingType: result.servingType,
        barcode: result.barcode,
      },
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
      <View style={styles.permissionContainer}>
        <Ionicons name="barcode-outline" size={64} color={colors.text.tertiary} />
        <Text style={styles.permissionText}>Camera access is required</Text>
        <Text style={styles.permissionSubtext}>
          We need camera access to scan supplement barcodes
        </Text>
        <TouchableOpacity
          style={styles.permissionButton}
          onPress={requestPermission}
          activeOpacity={0.8}
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

  // Supplement found - show details
  if (result) {
    return (
      <SafeAreaView style={styles.container}>
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
          <Text style={styles.headerTitle}>Supplement Found</Text>
          <TouchableOpacity onPress={handleScanAgain}>
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
            {result.imageUrl ? (
              <Image
                source={{ uri: result.imageUrl }}
                style={styles.productImage}
                resizeMode="contain"
              />
            ) : (
              <View style={styles.productImagePlaceholder}>
                <Ionicons
                  name="medical-outline"
                  size={48}
                  color={colors.text.tertiary}
                />
              </View>
            )}
            <View style={styles.productInfo}>
              <Text style={styles.productName}>{result.name}</Text>
              {result.brand && (
                <Text style={styles.productBrand}>{result.brand}</Text>
              )}
              <Text style={styles.barcodeText}>
                <Ionicons name="barcode-outline" size={14} color={colors.text.tertiary} />{' '}
                {result.barcode}
              </Text>
            </View>
          </View>

          {/* Extracted supplement info */}
          <View style={styles.infoCard}>
            <Text style={styles.infoCardTitle}>Detected Supplement Info</Text>
            <Text style={styles.infoCardSubtitle}>
              Pre-filled for 1 {result.servingType}
            </Text>

            <View style={styles.infoRow}>
              <View style={styles.infoItem}>
                <Text style={styles.infoLabel}>Serving Type</Text>
                <Text style={styles.infoValue}>{result.servingType}</Text>
              </View>
              {result.dosageAmount && result.dosageUnit && (
                <View style={styles.infoItem}>
                  <Text style={styles.infoLabel}>Dosage per {result.servingType}</Text>
                  <Text style={styles.infoValue}>
                    {result.dosageAmount} {result.dosageUnit}
                  </Text>
                </View>
              )}
            </View>

            {!result.dosageAmount && (
              <View style={styles.warningBox}>
                <Ionicons name="information-circle-outline" size={18} color={colors.status.warning} />
                <Text style={styles.warningText}>
                  Dosage not detected. You can enter it manually on the next screen.
                </Text>
              </View>
            )}
          </View>

          {/* Help text */}
          <View style={styles.helpCard}>
            <Ionicons name="bulb-outline" size={20} color={colors.primary.main} />
            <Text style={styles.helpText}>
              We've extracted the per-serving dosage, not the whole bottle.
              You can adjust the values on the next screen.
            </Text>
          </View>
        </ScrollView>

        {/* Action buttons */}
        <View
          style={[
            styles.actionsContainer,
            { paddingHorizontal: responsiveSpacing.horizontal },
            isTablet && styles.actionsContainerTablet,
          ]}
        >
          <TouchableOpacity
            style={styles.useButton}
            onPress={handleUseThisSupplement}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={gradients.primary}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.useButtonGradient}
            >
              <Text style={styles.useButtonText}>Use This Supplement</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // Camera view with barcode scanner
  return (
    <View style={styles.container}>
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
            accessibilityLabel="Go back"
            accessibilityRole="button"
          >
            <Ionicons name="chevron-back" size={28} color={colors.text.primary} />
          </TouchableOpacity>

          <Text style={styles.cameraTitle}>Scan Supplement</Text>

          <TouchableOpacity
            style={styles.manualButton}
            onPress={() => setShowManualEntry(true)}
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

          <Text style={styles.scannerHint}>
            Scan the barcode on your supplement bottle
          </Text>
        </View>
      </CameraView>

      {/* Loading overlay */}
      {state.isLoading && (
        <View style={styles.overlayContainer}>
          <BlurView intensity={80} tint="dark" style={styles.loadingOverlay}>
            <View style={styles.loadingCard}>
              <ActivityIndicator size="large" color={colors.primary.main} />
              <Text style={styles.loadingText}>Looking up supplement...</Text>
              <Text style={styles.loadingBarcode}>
                {state.lastScannedBarcode}
              </Text>
            </View>
          </BlurView>
        </View>
      )}

      {/* Error overlay */}
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
                  ? 'Supplement Not Found'
                  : state.error.type === 'NETWORK_ERROR'
                  ? 'Connection Error'
                  : 'Something Went Wrong'}
              </Text>
              <Text style={styles.errorMessage}>
                {state.error.type === 'PRODUCT_NOT_FOUND'
                  ? "This supplement isn't in the database. You can add it manually."
                  : state.error.type === 'NETWORK_ERROR'
                  ? 'Please check your internet connection and try again.'
                  : state.error.message}
              </Text>
              <TouchableOpacity
                style={styles.errorButton}
                onPress={handleScanAgain}
                activeOpacity={0.8}
              >
                <Text style={styles.errorButtonText}>Scan Another</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.errorSecondaryButton}
                onPress={() => router.back()}
                activeOpacity={0.8}
              >
                <Text style={styles.errorSecondaryButtonText}>
                  Enter Manually Instead
                </Text>
              </TouchableOpacity>
            </View>
          </BlurView>
        </View>
      )}

      {/* Manual entry modal */}
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
                  <TouchableOpacity onPress={() => setShowManualEntry(false)}>
                    <Ionicons
                      name="close"
                      size={24}
                      color={colors.text.secondary}
                    />
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
                />

                <TouchableOpacity
                  style={styles.manualEntrySubmit}
                  onPress={handleManualSubmit}
                  activeOpacity={0.8}
                  disabled={!manualBarcode.trim()}
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
                      Look Up Supplement
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
  infoCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  infoCardTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.xs,
  },
  infoCardSubtitle: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.md,
  },
  infoRow: {
    flexDirection: 'row',
    gap: spacing.lg,
  },
  infoItem: {
    flex: 1,
  },
  infoLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
    textTransform: 'uppercase',
  },
  infoValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.primary.main,
    textTransform: 'capitalize',
  },
  warningBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.special.warningLight,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginTop: spacing.md,
    gap: spacing.sm,
  },
  warningText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },
  helpCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: colors.special.highlight,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    gap: spacing.sm,
  },
  helpText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
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
  useButton: {
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    width: '100%',
    maxWidth: FORM_MAX_WIDTH,
  },
  useButtonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  useButtonText: {
    color: colors.text.primary,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
  },
});
