/**
 * Expo App Configuration
 *
 * This file allows dynamic configuration including environment variables.
 * See: https://docs.expo.dev/workflow/configuration/
 */

// Load environment variables from .env file in development
// Note: In production builds, these should be set in your CI/CD pipeline
const API_URL = process.env.API_URL;
const PRODUCTION_API_URL = process.env.PRODUCTION_API_URL;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL;

module.exports = {
  expo: {
    name: 'nutri',
    slug: 'nutri',
    version: '1.0.0',
    orientation: 'portrait',
    icon: './assets/images/icon.png',
    scheme: 'myapp',
    userInterfaceStyle: 'automatic',
    newArchEnabled: true, // Required by react-native-reanimated and other modern dependencies
    ios: {
      supportsTablet: true,
      requireFullScreen: false, // Allow iPad multitasking (Split View, Slide Over)
      bundleIdentifier: 'com.anonymous.nutri',
      bitcode: false,
      infoPlist: {
        NSCameraUsageDescription:
          'Nutri needs camera access to scan and analyze your food for automatic nutrition tracking.',
        NSPhotoLibraryUsageDescription:
          'Nutri needs photo library access to save food images for your meal history.',
        NSPhotoLibraryAddUsageDescription:
          'Nutri would like to save food photos to your photo library.',
        NSMicrophoneUsageDescription:
          'Microphone access is required for video recording features.',
        NSLocationWhenInUseUsageDescription:
          'Nutri uses ARKit for 3D food scanning, which requires location services for AR features.',
        NSHealthShareUsageDescription:
          'Nutri needs access to your health data to track your activity, sleep, and heart rate for personalized nutrition recommendations.',
        NSHealthUpdateUsageDescription:
          'Nutri needs permission to save nutrition data to Apple Health.',
        // ARKit world sensing for LiDAR depth capture
        NSWorldSensingUsageDescription:
          'Nutri uses LiDAR and ARKit to measure food portions accurately for nutrition tracking.',
      },
      // Enable ARKit capability
      config: {
        usesIAD: false,
      },
      entitlements: {
        'com.apple.developer.healthkit': true,
        'com.apple.developer.healthkit.access': [],
      },
    },
    android: {
      adaptiveIcon: {
        foregroundImage: './assets/images/adaptive-icon.png',
        backgroundColor: '#ffffff',
      },
    },
    web: {
      bundler: 'metro',
      output: 'static',
      favicon: './assets/images/favicon.png',
    },
    plugins: [
      'expo-router',
      [
        'expo-screen-orientation',
        {
          // Default to portrait for iPhones, but allow runtime control for iPads
          initialOrientation: 'PORTRAIT',
        },
      ],
      [
        'expo-splash-screen',
        {
          image: './assets/images/splash-icon.png',
          imageWidth: 200,
          resizeMode: 'contain',
          backgroundColor: '#ffffff',
        },
      ],
      [
        'expo-camera',
        {
          cameraPermission:
            'Allow Nutri to access your camera to scan and analyze food for automatic nutrition tracking.',
          microphonePermission:
            'Allow Nutri to access your microphone for video features.',
          recordAudioAndroid: true,
        },
      ],
      [
        'expo-image-picker',
        {
          photosPermission:
            'Nutri needs access to your photos to save food images.',
        },
      ],
      [
        'react-native-health',
        {
          isClinicalDataEnabled: false,
          healthSharePermission:
            'Nutri needs to read your health data (heart rate, HRV, sleep, activity) to provide personalized nutrition insights and track how your diet affects your health metrics.',
          healthUpdatePermission:
            'Nutri would like to save nutrition and health data to Apple Health to keep your health records in sync.',
        },
      ],
    ],
    experiments: {
      typedRoutes: true,
    },
    /**
     * Extra configuration for runtime access via Constants.expoConfig.extra
     *
     * Environment variables:
     * - API_URL: Override API URL for any environment (highest priority)
     * - PRODUCTION_API_URL: API URL for production builds
     *
     * For development with physical devices:
     *   API_URL=http://192.168.1.100:3000/api npx expo start
     *
     * For production:
     *   PRODUCTION_API_URL=https://api.yourapp.com/api eas build
     */
    extra: {
      // Custom API URL override (works in dev and prod)
      apiUrl: API_URL || undefined,
      // Production API URL (only used in production builds when apiUrl is not set)
      productionApiUrl: PRODUCTION_API_URL || undefined,
      // ML Service URL (for food analysis)
      mlServiceUrl: ML_SERVICE_URL || undefined,
      // EAS configuration
      eas: {
        projectId: 'your-project-id', // Replace with your EAS project ID
      },
    },
  },
};
