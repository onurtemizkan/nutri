import { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
  Image,
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImageManipulator from 'expo-image-manipulator';
import { Ionicons } from '@expo/vector-icons';
import { foodAnalysisApi } from '@/lib/api/food-analysis';
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
  const cameraRef = useRef<CameraView>(null);
  const router = useRouter();

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
        Alert.alert('Error', 'Failed to capture photo');
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
      Alert.alert('Error', 'Failed to capture photo. Please try again.');
    }
  };

  const handleAnalyzeFood = async () => {
    if (!capturedImage) return;

    setIsAnalyzing(true);

    try {
      // TODO: In Phase 2, add AR measurements here
      const mockMeasurements: ARMeasurement | undefined = undefined;

      // Call ML service
      const response = await foodAnalysisApi.analyzeFood({
        imageUri: capturedImage,
        measurements: mockMeasurements,
      });

      const result: FoodScanResult = {
        ...response,
        imageUri: capturedImage,
        timestamp: new Date(),
      };

      setScanResult(result);
    } catch (error) {
      console.error('Food analysis error:', error);
      Alert.alert(
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
  };

  const handleUseScan = () => {
    if (!scanResult || scanResult.foodItems.length === 0) {
      Alert.alert('Error', 'No food items detected');
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
      <View style={styles.permissionContainer}>
        <Ionicons name="camera-outline" size={64} color="#ccc" />
        <Text style={styles.permissionText}>Camera access is required</Text>
        <Text style={styles.permissionSubtext}>
          We need camera access to scan and analyze your food
        </Text>
        <TouchableOpacity
          style={styles.permissionButton}
          onPress={requestPermission}
        >
          <Text style={styles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Show captured image with analysis results
  if (capturedImage) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={handleRetake}>
            <Text style={styles.headerButton}>Retake</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Food Scan</Text>
          <View style={{ width: 60 }} />
        </View>

        <View style={styles.previewContainer}>
          <Image source={{ uri: capturedImage }} style={styles.previewImage} />

          {isAnalyzing && (
            <View style={styles.analyzingOverlay}>
              <ActivityIndicator size="large" color="#fff" />
              <Text style={styles.analyzingText}>Analyzing food...</Text>
            </View>
          )}

          {scanResult && (
            <View style={styles.resultsContainer}>
              <Text style={styles.resultsTitle}>Detected Food</Text>

              {scanResult.foodItems.map((item, index) => (
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
                  {scanResult.suggestions.map((suggestion, index) => (
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
          {!isAnalyzing && !scanResult && (
            <TouchableOpacity
              style={styles.analyzeButton}
              onPress={handleAnalyzeFood}
            >
              <Text style={styles.analyzeButtonText}>Analyze Food</Text>
            </TouchableOpacity>
          )}

          {scanResult && (
            <TouchableOpacity
              style={styles.analyzeButton}
              onPress={handleUseScan}
            >
              <Text style={styles.analyzeButtonText}>Use This Scan</Text>
            </TouchableOpacity>
          )}
        </View>
      </SafeAreaView>
    );
  }

  // Camera view
  return (
    <View style={styles.container}>
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
    backgroundColor: '#fff',
    padding: 20,
  },
  permissionText: {
    fontSize: 20,
    fontWeight: '600',
    marginTop: 20,
    marginBottom: 8,
  },
  permissionSubtext: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 24,
  },
  permissionButton: {
    backgroundColor: '#3b5998',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
  },
  permissionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
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
    paddingHorizontal: 20,
    paddingVertical: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerButton: {
    fontSize: 16,
    color: '#3b5998',
    fontWeight: '600',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
  },
  previewContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  previewImage: {
    width: '100%',
    height: '50%',
    resizeMode: 'contain',
  },
  analyzingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  analyzingText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
  },
  resultsContainer: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  resultsTitle: {
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 16,
    color: '#000',
  },
  foodItem: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  foodItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  foodName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
    flex: 1,
  },
  confidence: {
    fontSize: 12,
    color: '#3b5998',
    fontWeight: '600',
  },
  portionSize: {
    fontSize: 14,
    color: '#666',
    marginBottom: 12,
  },
  nutritionGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  nutritionItem: {
    alignItems: 'center',
  },
  nutritionValue: {
    fontSize: 16,
    fontWeight: '700',
    color: '#000',
  },
  nutritionLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  suggestions: {
    marginTop: 16,
    padding: 16,
    backgroundColor: '#e8f4fd',
    borderRadius: 12,
  },
  suggestionsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#3b5998',
    marginBottom: 8,
  },
  suggestionText: {
    fontSize: 13,
    color: '#666',
    marginBottom: 4,
  },
  actionsContainer: {
    padding: 20,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  analyzeButton: {
    backgroundColor: '#3b5998',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
