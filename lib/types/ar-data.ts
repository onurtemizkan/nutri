/**
 * AR Data Types
 *
 * Type definitions for RGB-D capture and AR food scanning.
 */

/**
 * Camera intrinsic parameters
 */
export interface CameraIntrinsics {
  focalLength: { x: number; y: number };
  principalPoint: { x: number; y: number };
  imageResolution: { width: number; height: number };
  radialDistortion: number[];
  tangentialDistortion: number[];
}

/**
 * Depth quality levels
 */
export type DepthQuality = 'low' | 'medium' | 'high';

/**
 * Depth buffer data
 */
export interface DepthBuffer {
  data: Float32Array;
  width: number;
  height: number;
  format: 'float32';
  unit: 'meters';
}

/**
 * Confidence map for depth data
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
 * LiDAR capture data from a single frame
 */
export interface LiDARCapture {
  depthBuffer: DepthBuffer;
  confidenceMap: ConfidenceMap;
  timestamp: number;
  depthQuality: DepthQuality;
  cameraIntrinsics: CameraIntrinsics;
}

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
 * Single RGB-D frame captured from device
 */
export interface RGBDFrame {
  rgbImage: RGBImage;
  depthData: LiDARCapture;
  timestamp: number;
  frameId: string;
  metadata: {
    deviceModel: string;
    osVersion: string;
    captureMode: CaptureMode;
  };
}

/**
 * Device capabilities for AR capture
 */
export interface DeviceCapabilities {
  hasLiDAR: boolean;
  hasARKit: boolean;
  supportsSceneDepth: boolean;
  maxDepthResolution: { width: number; height: number };
  maxRGBResolution: { width: number; height: number };
  depthRange: { min: number; max: number };
  frameRate: number;
}

/**
 * Capture mode
 */
export type CaptureMode = 'lidar' | 'ar_depth';

/**
 * Capture session containing multiple frames
 */
export interface CaptureSession {
  sessionId: string;
  startTime: number;
  endTime?: number;
  frames: RGBDFrame[];
  deviceCapabilities: DeviceCapabilities;
  metadata?: Record<string, unknown>;
}

/**
 * ML image file metadata
 */
export interface MLImageFile {
  path: string;
  width: number;
  height: number;
  channels: number;
  dtype: 'uint8' | 'float32';
  normalization?: {
    min: number;
    max: number;
    method: 'min_max' | 'z_score';
  };
}

/**
 * Exported ML-ready data format
 */
export interface MLReadyData {
  sessionId: string;
  version: string;
  format: string;
  rgbImages: MLImageFile[];
  depthMaps: MLImageFile[];
  confidenceMaps: MLImageFile[];
  metadata: {
    deviceModel: string;
    osVersion: string;
    captureMode: CaptureMode;
    timestamp: number;
    frameRate?: number;
    duration?: number;
    cameraIntrinsics: CameraIntrinsics;
  };
  annotations?: Record<string, unknown>;
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
