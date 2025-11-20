import axios from 'axios';
import * as SecureStore from 'expo-secure-store';
import Constants from 'expo-constants';
import { Platform } from 'react-native';

/**
 * Get the appropriate API URL based on the environment and platform
 */
function getApiBaseUrl(): string {
  if (!__DEV__) {
    // Production
    return 'https://your-production-api.com/api';
  }

  // Development environment
  // Check for custom API URL in app.json/app.config.js extra config
  const customApiUrl = Constants.expoConfig?.extra?.apiUrl;
  if (customApiUrl) {
    return customApiUrl;
  }

  // Default development URLs based on platform
  if (Platform.OS === 'ios') {
    // iOS Simulator can use localhost
    return 'http://localhost:3000/api';
  } else if (Platform.OS === 'android') {
    // Android Emulator needs special IP (10.0.2.2 maps to host's localhost)
    return 'http://10.0.2.2:3000/api';
  } else if (Platform.OS === 'web') {
    return 'http://localhost:3000/api';
  }

  // Fallback for physical devices - CHANGE THIS to your local IP
  // Run `./scripts/start-all.sh` to see your local IP
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
  timeout: 10000, // 10 seconds
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
