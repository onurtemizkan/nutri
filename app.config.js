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
const BETA_API_URL = process.env.BETA_API_URL;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL;
const APP_ENV = process.env.APP_ENV || 'production'; // 'production', 'beta', or 'development'

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
        ITSAppUsesNonExemptEncryption: false,
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
        '@kingstinct/react-native-healthkit',
        {
          NSHealthShareUsageDescription:
            'Nutri needs to read your health data (heart rate, HRV, sleep, activity) to provide personalized nutrition insights and track how your diet affects your health metrics.',
          NSHealthUpdateUsageDescription:
            'Nutri would like to save nutrition and health data to Apple Health to keep your health records in sync.',
          background: false,
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
     * - BETA_API_URL: API URL for beta builds
     * - APP_ENV: Environment type ('production', 'beta', 'development')
     *
     * For development with physical devices:
     *   API_URL=http://192.168.1.100:3000/api npx expo start
     *
     * For beta builds:
     *   APP_ENV=beta BETA_API_URL=https://beta-api.yourapp.com/api eas build --profile beta
     *
     * For production:
     *   PRODUCTION_API_URL=https://api.yourapp.com/api eas build --profile production
     */
    extra: {
      // Current environment: 'production', 'beta', or 'development'
      appEnv: APP_ENV,
      // Custom API URL override (works in dev and prod)
      // For physical device development, set this to your computer's IP
      // Set to undefined to use environment-specific URLs
      apiUrl: API_URL || undefined,
      // Production API URL (used when APP_ENV=production)
      productionApiUrl: PRODUCTION_API_URL || 'https://z8cg8kkg4o0wg8044c8g0s0o.195.201.228.58.sslip.io/api',
      // Beta API URL (used when APP_ENV=beta)
      // Set this in EAS secrets or eas.json for beta builds
      betaApiUrl: BETA_API_URL || undefined,
      // ML Service URL (for food analysis)
      // In production/beta, ML service is accessed through the backend proxy
      // For local dev, set ML_SERVICE_URL to your computer's IP:8000
      mlServiceUrl: ML_SERVICE_URL || 'https://z8cg8kkg4o0wg8044c8g0s0o.195.201.228.58.sslip.io',
      // EAS configuration
      eas: {
        projectId: '543d7d98-c664-4db6-bffe-fceabff545d0',
      },
    },
  },
};
