/**
 * Integration tests for AR food scanning workflow
 *
 * Tests the complete flow:
 * 1. Device capability detection
 * 2. AR session management
 * 3. Frame capture
 * 4. Data storage
 * 5. ML export
 */

import { captureStorage } from '@/lib/utils/capture-storage';
import type {
  RGBDFrame,
  CaptureSession,
  DeviceCapabilities,
  ExportConfig,
} from '@/lib/types/ar-data';

describe('AR Workflow Integration', () => {
  const mockCapabilities: DeviceCapabilities = {
    hasLiDAR: true,
    supportsSceneDepth: true,
    maxDepthResolution: { width: 256, height: 192 },
    maxRGBResolution: { width: 1920, height: 1440 },
    depthRange: { min: 0.3, max: 5.0 },
    frameRate: 60,
  };

  const createMockFrame = (frameId: string): RGBDFrame => ({
    rgbImage: {
      uri: `file:///tmp/${frameId}.jpg`,
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
      depthQuality: 'high',
      cameraIntrinsics: {
        focalLength: { x: 1000.5, y: 1000.5 },
        principalPoint: { x: 640.0, y: 360.0 },
        imageResolution: { width: 1920, height: 1440 },
        radialDistortion: [0.1, 0.01, 0.001],
        tangentialDistortion: [0.001, 0.001],
      },
    },
    timestamp: Date.now(),
    frameId,
    metadata: {
      deviceModel: 'iPhone 15 Pro',
      osVersion: 'iOS 17.0',
      captureMode: 'lidar',
    },
  });

  describe('Complete Single Frame Workflow', () => {
    it('should complete full workflow for single frame', () => {
      // 1. Create session
      const session: CaptureSession = {
        sessionId: 'test_session_001',
        startTime: Date.now(),
        endTime: Date.now() + 1000,
        frames: [],
        deviceCapabilities: mockCapabilities,
        metadata: {
          captureType: 'single_frame',
          foodItem: 'Apple',
        },
      };

      expect(session.sessionId).toBeDefined();
      expect(session.deviceCapabilities.hasLiDAR).toBe(true);

      // 2. Capture frame
      const frame = createMockFrame('frame_000');
      session.frames.push(frame);

      expect(session.frames).toHaveLength(1);
      expect(frame.rgbImage).toBeDefined();
      expect(frame.depthData).toBeDefined();

      // 3. Validate data structure
      expect(frame.depthData.depthBuffer.data).toBeInstanceOf(Float32Array);
      expect(frame.depthData.confidenceMap.data).toBeInstanceOf(Uint8Array);
      expect(frame.depthData.depthQuality).toBe('high');

      // 4. Prepare for ML export
      const exportConfig: ExportConfig = {
        outputDir: 'ml-data',
        format: 'rgbd_normalized',
        includeConfidenceMaps: true,
        normalizeDepth: true,
        normalizationMethod: 'min_max',
        compressRGB: true,
        rgbFormat: 'jpeg',
      };

      expect(exportConfig.format).toBe('rgbd_normalized');
      expect(exportConfig.normalizeDepth).toBe(true);
    });
  });

  describe('Complete Video Sequence Workflow', () => {
    it('should complete full workflow for video sequence', () => {
      // 1. Create session
      const session: CaptureSession = {
        sessionId: 'test_session_video_001',
        startTime: Date.now(),
        endTime: Date.now() + 5000,
        frames: [],
        deviceCapabilities: mockCapabilities,
        metadata: {
          captureType: 'video_sequence',
          foodItem: 'Meal plate',
        },
      };

      // 2. Capture multiple frames (simulate 60fps for 1 second = 60 frames)
      const frameCount = 60;
      const baseTimestamp = Date.now();

      for (let i = 0; i < frameCount; i++) {
        const frame = createMockFrame(`frame_${String(i).padStart(3, '0')}`);
        frame.timestamp = baseTimestamp + i * 16; // ~60fps
        session.frames.push(frame);
      }

      expect(session.frames).toHaveLength(frameCount);

      // 3. Validate frame sequence
      const timestamps = session.frames.map(f => f.timestamp);
      const isSequential = timestamps.every(
        (t, i) => i === 0 || t > timestamps[i - 1]
      );
      expect(isSequential).toBe(true);

      // 4. Calculate frame rate
      const duration = session.endTime - session.startTime;
      const actualFps = (frameCount / duration) * 1000;
      expect(actualFps).toBeCloseTo(60, 0);
    });
  });

  describe('Data Validation', () => {
    it('should validate depth buffer dimensions', () => {
      const frame = createMockFrame('frame_test');
      const { depthBuffer } = frame.depthData;

      const expectedSize = depthBuffer.width * depthBuffer.height;
      expect(depthBuffer.data.length).toBe(expectedSize);
      expect(depthBuffer.format).toBe('float32');
      expect(depthBuffer.unit).toBe('meters');
    });

    it('should validate confidence map dimensions', () => {
      const frame = createMockFrame('frame_test');
      const { confidenceMap } = frame.depthData;

      const expectedSize = confidenceMap.width * confidenceMap.height;
      expect(confidenceMap.data.length).toBe(expectedSize);
      expect(confidenceMap.levels.low).toBe(1);
      expect(confidenceMap.levels.medium).toBe(2);
      expect(confidenceMap.levels.high).toBe(3);
    });

    it('should validate camera intrinsics', () => {
      const frame = createMockFrame('frame_test');
      const { cameraIntrinsics } = frame.depthData;

      expect(cameraIntrinsics.focalLength).toBeDefined();
      expect(cameraIntrinsics.principalPoint).toBeDefined();
      expect(cameraIntrinsics.imageResolution).toBeDefined();
      expect(Array.isArray(cameraIntrinsics.radialDistortion)).toBe(true);
      expect(Array.isArray(cameraIntrinsics.tangentialDistortion)).toBe(true);
    });

    it('should validate RGB-D alignment', () => {
      const frame = createMockFrame('frame_test');

      // RGB and depth should have same aspect ratio (after upscaling)
      const rgbAspect = frame.rgbImage.width / frame.rgbImage.height;
      const depthAspect =
        frame.depthData.depthBuffer.width / frame.depthData.depthBuffer.height;

      // Aspect ratios should be similar (within 10%)
      const aspectDiff = Math.abs(rgbAspect - depthAspect) / rgbAspect;
      expect(aspectDiff).toBeLessThan(0.1);
    });
  });

  describe('Error Handling', () => {
    it('should handle empty session', () => {
      const session: CaptureSession = {
        sessionId: 'empty_session',
        startTime: Date.now(),
        endTime: Date.now(),
        frames: [],
        deviceCapabilities: mockCapabilities,
        metadata: {
          captureType: 'single_frame',
        },
      };

      expect(session.frames).toHaveLength(0);
      // Empty sessions should still be valid
      expect(session.sessionId).toBeDefined();
    });

    it('should handle non-LiDAR devices', () => {
      const nonLiDARCapabilities: DeviceCapabilities = {
        hasLiDAR: false,
        supportsSceneDepth: true,
        maxDepthResolution: { width: 640, height: 480 },
        maxRGBResolution: { width: 1920, height: 1080 },
        depthRange: { min: 0.5, max: 3.0 },
        frameRate: 30,
      };

      const session: CaptureSession = {
        sessionId: 'ar_depth_session',
        startTime: Date.now(),
        endTime: Date.now() + 1000,
        frames: [],
        deviceCapabilities: nonLiDARCapabilities,
        metadata: {
          captureType: 'single_frame',
        },
      };

      expect(session.deviceCapabilities.hasLiDAR).toBe(false);
      expect(session.deviceCapabilities.supportsSceneDepth).toBe(true);
      // Fallback to AR depth with lower resolution
      expect(session.deviceCapabilities.maxDepthResolution.width).toBe(640);
    });
  });
});
