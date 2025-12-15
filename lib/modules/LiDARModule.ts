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

import { Platform } from 'react-native';
import { requireNativeModule } from 'expo-modules-core';
import type {
  DeviceCapabilities,
  LiDARCapture,
  CaptureMode,
  DepthBuffer,
  ConfidenceMap,
  CameraIntrinsics,
  DepthQuality,
} from '@/lib/types/ar-data';
import { decode as decodeBase64 } from 'base-64';

/**
 * Permission status types
 */
export type PermissionStatus = 'granted' | 'denied' | 'restricted' | 'undetermined';

/**
 * Raw depth buffer data from native (before processing)
 */
interface RawDepthBuffer {
  data: string; // Base64 encoded
  width: number;
  height: number;
  format: 'float32';
  unit: 'meters';
}

/**
 * Raw confidence map data from native (before processing)
 */
interface RawConfidenceMap {
  data: string; // Base64 encoded
  width: number;
  height: number;
  levels: {
    low: number;
    medium: number;
    high: number;
  };
}

/**
 * Raw capture data from native module
 */
interface RawLiDARCapture {
  depthBuffer: RawDepthBuffer;
  confidenceMap: RawConfidenceMap;
  timestamp: number;
  depthQuality: string;
  cameraIntrinsics: CameraIntrinsics;
}

/**
 * Native module interface
 */
interface LiDARModuleInterface {
  hasLiDAR(): Promise<boolean>;
  supportsSceneDepth(): Promise<boolean>;
  getDeviceCapabilities(): Promise<DeviceCapabilities>;
  startARSession(mode: CaptureMode): Promise<void>;
  stopARSession(): Promise<void>;
  isARSessionRunning(): Promise<boolean>;
  captureDepthFrame(): Promise<RawLiDARCapture>;
  requestCameraPermission(): Promise<PermissionStatus>;
  getCameraPermissionStatus(): Promise<PermissionStatus>;
}

/**
 * Try to load the native module
 */
let nativeModule: LiDARModuleInterface | null = null;
let moduleAvailable = false;

if (Platform.OS === 'ios') {
  try {
    nativeModule = requireNativeModule('LiDARModule') as LiDARModuleInterface;
    moduleAvailable = true;
  } catch (error) {
    console.warn(
      'LiDARModule native module is not available. Using JS fallback. ' +
        'If you need native LiDAR functionality, run a custom dev client or EAS build.',
      error
    );
    moduleAvailable = false;
  }
}

/**
 * Decode Base64 string to Float32Array
 */
function decodeBase64ToFloat32Array(base64: string): Float32Array {
  if (!base64 || base64.length === 0) {
    return new Float32Array(0);
  }

  try {
    // Decode base64 to binary string
    const binaryString = decodeBase64(base64);

    // Convert to Uint8Array
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    // Create Float32Array from buffer
    return new Float32Array(bytes.buffer);
  } catch (error) {
    console.error('Failed to decode depth buffer:', error);
    return new Float32Array(0);
  }
}

/**
 * Decode Base64 string to Uint8Array
 */
function decodeBase64ToUint8Array(base64: string): Uint8Array {
  if (!base64 || base64.length === 0) {
    return new Uint8Array(0);
  }

  try {
    // Decode base64 to binary string
    const binaryString = decodeBase64(base64);

    // Convert to Uint8Array
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    return bytes;
  } catch (error) {
    console.error('Failed to decode confidence map:', error);
    return new Uint8Array(0);
  }
}

/**
 * Process raw capture data from native module
 */
function processCapture(raw: RawLiDARCapture): LiDARCapture {
  const depthBuffer: DepthBuffer = {
    data: decodeBase64ToFloat32Array(raw.depthBuffer.data),
    width: raw.depthBuffer.width,
    height: raw.depthBuffer.height,
    format: 'float32',
    unit: 'meters',
  };

  const confidenceMap: ConfidenceMap = {
    data: decodeBase64ToUint8Array(raw.confidenceMap.data),
    width: raw.confidenceMap.width,
    height: raw.confidenceMap.height,
    levels: raw.confidenceMap.levels,
  };

  return {
    depthBuffer,
    confidenceMap,
    timestamp: raw.timestamp,
    depthQuality: raw.depthQuality as DepthQuality,
    cameraIntrinsics: raw.cameraIntrinsics,
  };
}

/**
 * Fallback implementation for non-native environments
 */
const fallback: LiDARModuleInterface = {
  hasLiDAR: async () => false,
  supportsSceneDepth: async () => false,
  getDeviceCapabilities: async () => ({
    hasLiDAR: false,
    hasARKit: false,
    supportsSceneDepth: false,
    maxDepthResolution: { width: 0, height: 0 },
    maxRGBResolution: { width: 0, height: 0 },
    depthRange: { min: 0, max: 0 },
    frameRate: 0,
  }),
  startARSession: async (_mode: CaptureMode) => {
    console.warn('LiDARModule.startARSession: native module unavailable');
  },
  stopARSession: async () => {
    console.warn('LiDARModule.stopARSession: native module unavailable');
  },
  isARSessionRunning: async () => false,
  captureDepthFrame: async () => {
    throw new Error('LiDARModule.captureDepthFrame: native LiDAR not available');
  },
  requestCameraPermission: async () => 'undetermined',
  getCameraPermissionStatus: async () => 'undetermined',
};

/**
 * Select implementation (native or fallback)
 */
const implementation = nativeModule || fallback;

/**
 * Exported LiDAR Module API
 */
const LiDARModule = {
  /**
   * Whether the native module is available
   */
  isAvailable: moduleAvailable,

  /**
   * Check if device has LiDAR sensor
   */
  hasLiDAR: () => implementation.hasLiDAR(),

  /**
   * Check if device supports ARKit Scene Depth
   */
  supportsSceneDepth: () => implementation.supportsSceneDepth(),

  /**
   * Get complete device AR/LiDAR capabilities
   */
  getDeviceCapabilities: () => implementation.getDeviceCapabilities(),

  /**
   * Start ARKit session with specified capture mode
   * @param mode - 'lidar' for LiDAR devices, 'ar_depth' for fallback
   */
  startARSession: (mode: CaptureMode) => implementation.startARSession(mode),

  /**
   * Stop active ARKit session
   */
  stopARSession: () => implementation.stopARSession(),

  /**
   * Check if AR session is currently running
   */
  isARSessionRunning: () => implementation.isARSessionRunning(),

  /**
   * Capture current depth frame from ARKit
   * Must be called after startARSession()
   * Returns LiDAR capture with depth buffer, confidence map, and camera intrinsics
   */
  captureDepthFrame: async (): Promise<LiDARCapture> => {
    const raw = await implementation.captureDepthFrame();
    return processCapture(raw);
  },

  /**
   * Request camera permission
   */
  requestCameraPermission: () => implementation.requestCameraPermission(),

  /**
   * Get current camera permission status
   */
  getCameraPermissionStatus: () => implementation.getCameraPermissionStatus(),
};

export default LiDARModule;
