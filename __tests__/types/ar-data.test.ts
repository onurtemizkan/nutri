/**
 * Tests for AR/LiDAR data structure types
 * These types define the format for ML-ready RGB-D data capture
 */

import {
  RGBDFrame,
  LiDARCapture,
  DepthBuffer,
  ConfidenceMap,
  CameraIntrinsics,
  CaptureSession,
  MLReadyData,
  DeviceCapabilities,
  DepthQuality,
} from '@/lib/types/ar-data';

describe('AR Data Type Structures', () => {
  describe('DeviceCapabilities', () => {
    it('should define LiDAR capability correctly', () => {
      const capabilities: DeviceCapabilities = {
        hasLiDAR: true,
        supportsSceneDepth: true,
        maxDepthResolution: { width: 256, height: 192 },
        maxRGBResolution: { width: 1920, height: 1440 },
        depthRange: { min: 0.3, max: 5.0 },
        frameRate: 60,
      };

      expect(capabilities.hasLiDAR).toBe(true);
      expect(capabilities.maxDepthResolution.width).toBe(256);
      expect(capabilities.maxDepthResolution.height).toBe(192);
      expect(capabilities.frameRate).toBe(60);
    });

    it('should define non-LiDAR fallback capabilities', () => {
      const capabilities: DeviceCapabilities = {
        hasLiDAR: false,
        supportsSceneDepth: true,
        maxDepthResolution: { width: 640, height: 480 },
        maxRGBResolution: { width: 1920, height: 1080 },
        depthRange: { min: 0.5, max: 3.0 },
        frameRate: 30,
      };

      expect(capabilities.hasLiDAR).toBe(false);
      expect(capabilities.supportsSceneDepth).toBe(true);
    });
  });

  describe('DepthBuffer', () => {
    it('should store float32 depth values with correct dimensions', () => {
      const depthBuffer: DepthBuffer = {
        data: new Float32Array(256 * 192), // 256x192 for LiDAR
        width: 256,
        height: 192,
        format: 'float32',
        unit: 'meters',
      };

      expect(depthBuffer.data.length).toBe(256 * 192);
      expect(depthBuffer.data).toBeInstanceOf(Float32Array);
      expect(depthBuffer.format).toBe('float32');
      expect(depthBuffer.unit).toBe('meters');
    });

    it('should handle variable resolution depth data', () => {
      const depthBuffer: DepthBuffer = {
        data: new Float32Array(640 * 480),
        width: 640,
        height: 480,
        format: 'float32',
        unit: 'meters',
      };

      expect(depthBuffer.data.length).toBe(640 * 480);
    });
  });

  describe('ConfidenceMap', () => {
    it('should store uint8 confidence values (1-3)', () => {
      const confidenceMap: ConfidenceMap = {
        data: new Uint8Array(256 * 192),
        width: 256,
        height: 192,
        levels: {
          low: 1,
          medium: 2,
          high: 3,
        },
      };

      expect(confidenceMap.data).toBeInstanceOf(Uint8Array);
      expect(confidenceMap.levels.low).toBe(1);
      expect(confidenceMap.levels.medium).toBe(2);
      expect(confidenceMap.levels.high).toBe(3);
    });
  });

  describe('CameraIntrinsics', () => {
    it('should store camera calibration parameters', () => {
      const intrinsics: CameraIntrinsics = {
        focalLength: { x: 1000.5, y: 1000.5 },
        principalPoint: { x: 640.0, y: 360.0 },
        imageResolution: { width: 1920, height: 1440 },
        radialDistortion: [0.1, 0.01, 0.001],
        tangentialDistortion: [0.001, 0.001],
      };

      expect(intrinsics.focalLength.x).toBeCloseTo(1000.5);
      expect(intrinsics.principalPoint.x).toBe(640.0);
      expect(intrinsics.radialDistortion).toHaveLength(3);
    });
  });

  describe('LiDARCapture', () => {
    it('should capture complete LiDAR frame data', () => {
      const capture: LiDARCapture = {
        depthBuffer: {
          data: new Float32Array(256 * 192),
          width: 256,
          height: 192,
          format: 'float32',
          unit: 'meters',
        },
        confidenceMap: {
          data: new Uint8Array(256 * 192),
          width: 256,
          height: 192,
          levels: { low: 1, medium: 2, high: 3 },
        },
        timestamp: Date.now(),
        depthQuality: 'high' as DepthQuality,
        cameraIntrinsics: {
          focalLength: { x: 1000.5, y: 1000.5 },
          principalPoint: { x: 640.0, y: 360.0 },
          imageResolution: { width: 1920, height: 1440 },
          radialDistortion: [0.1, 0.01, 0.001],
          tangentialDistortion: [0.001, 0.001],
        },
      };

      expect(capture.depthBuffer.data.length).toBe(256 * 192);
      expect(capture.confidenceMap.data.length).toBe(256 * 192);
      expect(capture.depthQuality).toBe('high');
      expect(capture.timestamp).toBeGreaterThan(0);
    });
  });

  describe('RGBDFrame', () => {
    it('should combine RGB image with depth data', () => {
      const frame: RGBDFrame = {
        rgbImage: {
          uri: 'file:///path/to/image.jpg',
          width: 1920,
          height: 1440,
          format: 'jpeg',
        },
        depthData: {
          depthBuffer: {
            data: new Float32Array(256 * 192),
            width: 256,
            height: 192,
            format: 'float32',
            unit: 'meters',
          },
          confidenceMap: {
            data: new Uint8Array(256 * 192),
            width: 256,
            height: 192,
            levels: { low: 1, medium: 2, high: 3 },
          },
          timestamp: Date.now(),
          depthQuality: 'high' as DepthQuality,
          cameraIntrinsics: {
            focalLength: { x: 1000.5, y: 1000.5 },
            principalPoint: { x: 640.0, y: 360.0 },
            imageResolution: { width: 1920, height: 1440 },
            radialDistortion: [0.1, 0.01, 0.001],
            tangentialDistortion: [0.001, 0.001],
          },
        },
        timestamp: Date.now(),
        frameId: 'frame_001',
        metadata: {
          deviceModel: 'iPhone 15 Pro',
          osVersion: 'iOS 17.0',
          captureMode: 'lidar',
        },
      };

      expect(frame.rgbImage.width).toBe(1920);
      expect(frame.rgbImage.height).toBe(1440);
      expect(frame.depthData.depthBuffer.width).toBe(256);
      expect(frame.depthData.depthBuffer.height).toBe(192);
      expect(frame.frameId).toBe('frame_001');
      expect(frame.metadata.captureMode).toBe('lidar');
    });

    it('should handle video frame sequence', () => {
      const frames: RGBDFrame[] = [];
      const baseTimestamp = Date.now();

      for (let i = 0; i < 5; i++) {
        frames.push({
          rgbImage: {
            uri: `file:///path/to/frame_${i}.jpg`,
            width: 1920,
            height: 1440,
            format: 'jpeg',
          },
          depthData: {
            depthBuffer: {
              data: new Float32Array(256 * 192),
              width: 256,
              height: 192,
              format: 'float32',
              unit: 'meters',
            },
            confidenceMap: {
              data: new Uint8Array(256 * 192),
              width: 256,
              height: 192,
              levels: { low: 1, medium: 2, high: 3 },
            },
            timestamp: baseTimestamp + i * 16, // ~60fps
            depthQuality: 'high' as DepthQuality,
            cameraIntrinsics: {
              focalLength: { x: 1000.5, y: 1000.5 },
              principalPoint: { x: 640.0, y: 360.0 },
              imageResolution: { width: 1920, height: 1440 },
              radialDistortion: [0.1, 0.01, 0.001],
              tangentialDistortion: [0.001, 0.001],
            },
          },
          timestamp: baseTimestamp + i * 16,
          frameId: `frame_${String(i).padStart(3, '0')}`,
          metadata: {
            deviceModel: 'iPhone 15 Pro',
            osVersion: 'iOS 17.0',
            captureMode: 'lidar',
          },
        });
      }

      expect(frames).toHaveLength(5);
      expect(frames[0].frameId).toBe('frame_000');
      expect(frames[4].frameId).toBe('frame_004');
    });
  });

  describe('CaptureSession', () => {
    it('should manage complete capture session', () => {
      const session: CaptureSession = {
        sessionId: 'session_123',
        startTime: Date.now(),
        endTime: Date.now() + 5000,
        frames: [],
        deviceCapabilities: {
          hasLiDAR: true,
          supportsSceneDepth: true,
          maxDepthResolution: { width: 256, height: 192 },
          maxRGBResolution: { width: 1920, height: 1440 },
          depthRange: { min: 0.3, max: 5.0 },
          frameRate: 60,
        },
        metadata: {
          foodItem: 'Grilled Chicken Salad',
          captureType: 'single_frame',
          notes: 'Test capture',
        },
      };

      expect(session.sessionId).toBe('session_123');
      expect(session.deviceCapabilities.hasLiDAR).toBe(true);
      expect(session.frames).toHaveLength(0);
      expect(session.metadata.captureType).toBe('single_frame');
    });
  });

  describe('MLReadyData', () => {
    it('should structure data for ML export', () => {
      const mlData: MLReadyData = {
        sessionId: 'session_123',
        version: '1.0.0',
        format: 'rgbd_normalized',
        rgbImages: [
          {
            path: 'rgb/frame_000.jpg',
            width: 1920,
            height: 1440,
            channels: 3,
            dtype: 'uint8',
          },
        ],
        depthMaps: [
          {
            path: 'depth/frame_000.bin',
            width: 256,
            height: 192,
            channels: 1,
            dtype: 'float32',
            normalization: {
              min: 0.0,
              max: 5.0,
              method: 'min_max',
            },
          },
        ],
        confidenceMaps: [
          {
            path: 'confidence/frame_000.bin',
            width: 256,
            height: 192,
            channels: 1,
            dtype: 'uint8',
          },
        ],
        metadata: {
          deviceModel: 'iPhone 15 Pro',
          osVersion: 'iOS 17.0',
          captureMode: 'lidar',
          timestamp: Date.now(),
          cameraIntrinsics: {
            focalLength: { x: 1000.5, y: 1000.5 },
            principalPoint: { x: 640.0, y: 360.0 },
            imageResolution: { width: 1920, height: 1440 },
            radialDistortion: [0.1, 0.01, 0.001],
            tangentialDistortion: [0.001, 0.001],
          },
        },
        annotations: {
          foodItem: 'Grilled Chicken Salad',
          portionSize: 'medium',
        },
      };

      expect(mlData.format).toBe('rgbd_normalized');
      expect(mlData.rgbImages).toHaveLength(1);
      expect(mlData.depthMaps).toHaveLength(1);
      expect(mlData.confidenceMaps).toHaveLength(1);
      expect(mlData.rgbImages[0].channels).toBe(3);
      expect(mlData.depthMaps[0].channels).toBe(1);
      expect(mlData.depthMaps[0].normalization?.method).toBe('min_max');
    });

    it('should support multi-frame video sequences', () => {
      const mlData: MLReadyData = {
        sessionId: 'session_video_001',
        version: '1.0.0',
        format: 'rgbd_sequence',
        rgbImages: Array(30).fill(null).map((_, i) => ({
          path: `rgb/frame_${String(i).padStart(3, '0')}.jpg`,
          width: 1920,
          height: 1440,
          channels: 3,
          dtype: 'uint8' as const,
        })),
        depthMaps: Array(30).fill(null).map((_, i) => ({
          path: `depth/frame_${String(i).padStart(3, '0')}.bin`,
          width: 256,
          height: 192,
          channels: 1,
          dtype: 'float32' as const,
          normalization: {
            min: 0.0,
            max: 5.0,
            method: 'min_max' as const,
          },
        })),
        confidenceMaps: Array(30).fill(null).map((_, i) => ({
          path: `confidence/frame_${String(i).padStart(3, '0')}.bin`,
          width: 256,
          height: 192,
          channels: 1,
          dtype: 'uint8' as const,
        })),
        metadata: {
          deviceModel: 'iPhone 15 Pro',
          osVersion: 'iOS 17.0',
          captureMode: 'lidar',
          timestamp: Date.now(),
          frameRate: 60,
          duration: 500, // 30 frames at 60fps = 500ms
          cameraIntrinsics: {
            focalLength: { x: 1000.5, y: 1000.5 },
            principalPoint: { x: 640.0, y: 360.0 },
            imageResolution: { width: 1920, height: 1440 },
            radialDistortion: [0.1, 0.01, 0.001],
            tangentialDistortion: [0.001, 0.001],
          },
        },
        annotations: {
          foodItem: 'Rotating food plate',
          portionSize: 'large',
        },
      };

      expect(mlData.rgbImages).toHaveLength(30);
      expect(mlData.depthMaps).toHaveLength(30);
      expect(mlData.metadata.frameRate).toBe(60);
      expect(mlData.metadata.duration).toBe(500);
    });
  });
});
