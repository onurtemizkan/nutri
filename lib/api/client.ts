import axios from 'axios';
import * as SecureStore from 'expo-secure-store';
import Constants from 'expo-constants';
import { Platform } from 'react-native';

/**
 * Get the appropriate API URL based on the environment and platform
 *
 * Configuration priority:
 * 1. Custom API URL from expo config (app.config.js extra.apiUrl)
 * 2. Environment-specific defaults (production vs development)
 * 3. Platform-specific development defaults (iOS/Android/Web)
 */
function getApiBaseUrl(): string {
  // Check for custom API URL in app.config.js extra config (highest priority)
  // This allows overriding in both dev and production
  const customApiUrl = Constants.expoConfig?.extra?.apiUrl;
  if (customApiUrl && typeof customApiUrl === 'string' && customApiUrl.trim() !== '') {
    return customApiUrl;
  }

  // Production environment
  if (!__DEV__) {
    // Read production API URL from expo config
    const productionApiUrl = Constants.expoConfig?.extra?.productionApiUrl;
    if (productionApiUrl && typeof productionApiUrl === 'string' && productionApiUrl.trim() !== '') {
      return productionApiUrl;
    }

    // Fail safely with a clear error message if production URL is not configured
    console.error(
      '🚨 Production API URL not configured!\n' +
      'Please set "extra.productionApiUrl" in app.config.js:\n' +
      '  extra: {\n' +
      '    productionApiUrl: "https://api.yourapp.com/api"\n' +
      '  }'
    );
    // Return a placeholder that will fail with a clear error
    return 'https://api-url-not-configured.invalid/api';
  }

  // Development environment - platform-specific defaults
  if (Platform.OS === 'ios') {
    // iOS Simulator - localhost works directly
    return 'http://localhost:3000/api';
  } else if (Platform.OS === 'android') {
    // Android Emulator needs special IP (10.0.2.2 maps to host's localhost)
    return 'http://10.0.2.2:3000/api';
  } else if (Platform.OS === 'web') {
    return 'http://localhost:3000/api';
  }

  // Fallback for physical devices - use localhost (user should configure apiUrl)
  // Run `./scripts/start-all.sh` to see your local IP and configure in app.config.js
  return 'http://localhost:3000/api';
}

const API_BASE_URL = getApiBaseUrl();

// Log API URL in development for debugging
if (__DEV__) {
  console.log('🌐 API Base URL:', API_BASE_URL);
  console.log('📱 Platform:', Platform.OS);
}

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  // Add timeout to prevent hanging requests
  // 30 seconds is more appropriate for mobile networks and cold starts
  timeout: 30000, // 30 seconds
});

// Add auth token to requests
api.interceptors.request.use(
  async (config) => {
    const token = await SecureStore.getItemAsync('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Handle token expiration
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      await SecureStore.deleteItemAsync('authToken');
      // You might want to redirect to login here
    }
    return Promise.reject(error);
  }
);

export default api;
