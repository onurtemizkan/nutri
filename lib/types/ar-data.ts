/**
 * AR Data Types
 *
 * Type definitions for RGB-D capture and AR food scanning.
 */

/**
 * Resolution dimensions
 */
export interface Resolution {
  width: number;
  height: number;
}

/**
 * Depth range configuration
 */
export interface DepthRange {
  min: number;
  max: number;
}

/**
 * Device capabilities for AR capture
 */
export interface DeviceCapabilities {
  hasLiDAR: boolean;
  supportsSceneDepth: boolean;
  maxDepthResolution: Resolution;
  maxRGBResolution?: Resolution;
  depthRange: DepthRange;
  frameRate: number;
  sensorModel?: string;
  recommendedCaptureMode?: CaptureMode;
  maxDepthRange?: number;
  hasARKit?: boolean;
  supportedResolutions?: string[];
  deviceModel?: string;
}

/**
 * Depth buffer for storing depth data
 */
export interface DepthBuffer {
  data: Float32Array;
  width: number;
  height: number;
  format: 'float32' | 'uint16';
  unit: 'meters' | 'millimeters';
}

/**
 * Confidence map for depth quality
 */
export interface ConfidenceMap {
  data: Uint8Array;
  width: number;
  height: number;
  levels: {
    low: number;
    medium: number;
    high: number;
  };
}

/**
 * Camera intrinsic parameters
 */
export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  focalLength?: { x: number; y: number };
  principalPoint?: { x: number; y: number };
  radialDistortion?: number[];
  width?: number;
  height?: number;
  imageResolution?: Resolution;
}

/**
 * Depth quality levels
 */
export type DepthQuality = 'low' | 'medium' | 'high';

/**
 * Capture mode
 */
export type CaptureMode = 'lidar' | 'ar_depth' | 'photo' | 'video';

/**
 * RGB image data
 */
export interface RGBImage {
  uri: string;
  width: number;
  height: number;
  format: 'jpeg' | 'png';
}

/**
 * LiDAR capture data from native module
 */
export interface LiDARCapture {
  depthBuffer: DepthBuffer;
  confidenceMap?: ConfidenceMap;
  intrinsics: CameraIntrinsics;
  timestamp: number;
  quality: DepthQuality;
  depthQuality?: DepthQuality;
}

/**
 * Single RGB-D frame captured from device
 */
export interface RGBDFrame {
  // Core identifiers
  id?: string;
  frameId?: string;

  // Timestamps
  timestamp: number;

  // Image data - support both formats
  rgbData?: Uint8Array;
  rgbImage?: RGBImage;

  // Depth data - support both formats
  depthData: Float32Array | LiDARCapture;

  // Dimensions (for raw data format)
  width?: number;
  height?: number;

  // Optional data
  confidenceData?: Uint8Array;
  intrinsics?: CameraIntrinsics;

  // Metadata
  metadata?: {
    deviceModel?: string;
    osVersion?: string;
    captureMode?: CaptureMode;
    [key: string]: unknown;
  };
}

/**
 * Capture session containing multiple frames
 */
export interface CaptureSession {
  sessionId: string;
  startTime: number;
  endTime?: number;
  frames: RGBDFrame[];
  deviceCapabilities?: DeviceCapabilities;
  deviceInfo?: DeviceCapabilities;
  metadata?: Record<string, unknown>;
}

/**
 * ML-ready processed data
 */
export interface MLReadyData {
  sessionId: string;
  format: string;
  frameCount: number;
  outputPath?: string;
  frames?: ProcessedFrame[];
  metadata: Record<string, unknown>;
  version?: string;
  rgbImages?: string[];
  depthMaps?: string[];
  confidenceMaps?: string[];
}

/**
 * Exported ML data (alias for backward compatibility)
 */
export type MLExportData = MLReadyData;

/**
 * Processed frame for ML consumption
 */
export interface ProcessedFrame {
  id: string;
  rgbPath: string;
  depthPath: string;
  confidencePath?: string;
  intrinsics: CameraIntrinsics;
}

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
