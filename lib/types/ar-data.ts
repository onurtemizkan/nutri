/**
 * AR Data Types
 * 
 * Type definitions for RGB-D capture and AR food scanning.
 */

/**
 * Single RGB-D frame captured from device
 */
export interface RGBDFrame {
  id: string;
  timestamp: number;
  rgbData: Uint8Array;
  depthData: Float32Array;
  confidenceData?: Uint8Array;
  width: number;
  height: number;
  intrinsics?: CameraIntrinsics;
}

/**
 * Camera intrinsic parameters
 */
export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

/**
 * Capture session containing multiple frames
 */
export interface CaptureSession {
  sessionId: string;
  startTime: number;
  endTime?: number;
  frames: RGBDFrame[];
  deviceInfo: DeviceCapabilities;
  metadata?: Record<string, unknown>;
}

/**
 * Device capabilities for AR capture
 */
export interface DeviceCapabilities {
  hasLiDAR: boolean;
  hasARKit: boolean;
  maxDepthRange: number;
  supportedResolutions: string[];
  deviceModel: string;
}

/**
 * Capture mode
 */
export type CaptureMode = 'photo' | 'video';

/**
 * Export configuration for ML processing
 */
export interface ExportConfig {
  outputDir: string;
  format: 'rgbd_normalized' | 'rgbd_raw' | 'point_cloud';
  includeConfidenceMaps: boolean;
  normalizeDepth: boolean;
  normalizationMethod: 'min_max' | 'z_score';
  compressRGB: boolean;
  rgbFormat: 'jpeg' | 'png';
  rgbQuality: number;
}

/**
 * Exported ML data
 */
export interface MLExportData {
  sessionId: string;
  format: string;
  frameCount: number;
  outputPath: string;
  metadata: Record<string, unknown>;
}
