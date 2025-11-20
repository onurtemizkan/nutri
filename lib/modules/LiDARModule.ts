/**
 * Native iOS LiDAR Module
 *
 * Provides access to iPhone LiDAR sensor and ARKit Scene Depth API
 * for capturing RGB-D data for nutrition estimation.
 *
 * Features:
 * - LiDAR capability detection
 * - ARKit session management
 * - Real-time depth data capture (256x192 @ 60Hz for LiDAR)
 * - Confidence maps
 * - Camera intrinsics
 * - Permission handling
 */

import { Platform, NativeModules } from 'react-native';
import type {
  DeviceCapabilities,
  LiDARCapture,
  CaptureMode,
} from '@/lib/types/ar-data';

/**
 * Permission status types
 */
export type PermissionStatus = 'granted' | 'denied' | 'restricted' | 'undetermined';

/**
 * Native module interface
 */
interface LiDARModuleInterface {
  /**
   * Check if device has LiDAR sensor
   * Returns true for iPhone 12 Pro, 13 Pro, 14 Pro, 15 Pro and newer
   */
  hasLiDAR(): Promise<boolean>;

  /**
   * Check if device supports ARKit Scene Depth
   * Available on devices with A12 Bionic or newer
   */
  supportsSceneDepth(): Promise<boolean>;

  /**
   * Get complete device AR/LiDAR capabilities
   */
  getDeviceCapabilities(): Promise<DeviceCapabilities>;

  /**
   * Start ARKit session with specified capture mode
   * @param mode - 'lidar' for LiDAR devices, 'ar_depth' for fallback
   */
  startARSession(mode: CaptureMode): Promise<void>;

  /**
   * Stop active ARKit session
   */
  stopARSession(): Promise<void>;

  /**
   * Check if AR session is currently running
   */
  isARSessionRunning(): Promise<boolean>;

  /**
   * Capture current depth frame from ARKit
   * Must be called after startARSession()
   * Returns LiDAR capture with depth buffer, confidence map, and camera intrinsics
   */
  captureDepthFrame(): Promise<LiDARCapture>;

  /**
   * Request camera permission
   * Required for AR session
   */
  requestCameraPermission(): Promise<PermissionStatus>;

  /**
   * Get current camera permission status
   */
  getCameraPermissionStatus(): Promise<PermissionStatus>;
}

/**
 * Native module instance (may be undefined in Expo Go / non-iOS)
 */
const nativeModule = Platform.OS === 'ios' ? (NativeModules as any).LiDARModule : null;

// Graceful JS fallback implementation so the app doesn't crash when running
// in environments where the native module isn't available (Expo Go, Android,
// or in unit tests). The fallback returns conservative defaults and no-ops.
const fallback: LiDARModuleInterface = {
  hasLiDAR: async () => false,
  supportsSceneDepth: async () => false,
  getDeviceCapabilities: async () => {
    return {
      lidar: false,
      sceneDepth: false,
      sensorModel: 'unknown',
      maxDepthResolution: { width: 0, height: 0 },
      recommendedCaptureMode: 'ar_depth',
    } as any;
  },
  startARSession: async (_mode: CaptureMode) => {
    console.warn('LiDARModule.startARSession: native module unavailable — no-op');
    return;
  },
  stopARSession: async () => {
    console.warn('LiDARModule.stopARSession: native module unavailable — no-op');
    return;
  },
  isARSessionRunning: async () => false,
  captureDepthFrame: async () => {
    // This operation isn't available in the fallback — surface a clear error
    throw new Error('LiDARModule.captureDepthFrame: native LiDAR not available');
  },
  requestCameraPermission: async () => 'undetermined',
  getCameraPermissionStatus: async () => 'undetermined',
};

// Export a single object that either proxies to the native module or uses the
// fallback. Consumers can check `isAvailable` to determine real native support.
const LiDARModule: LiDARModuleInterface & { isAvailable: boolean } = (
  nativeModule
    ? ({ ...(nativeModule as LiDARModuleInterface), isAvailable: true } as any)
    : ({ ...fallback, isAvailable: false } as any)
);

if (!nativeModule && Platform.OS === 'ios') {
  // Keep a helpful console message, but avoid throwing — fallback will handle calls.
  console.warn(
    'LiDARModule native module is not available. Using JS fallback. ' +
      'If you need native LiDAR functionality, run a custom dev client or EAS build.'
  );
}

export default LiDARModule;
