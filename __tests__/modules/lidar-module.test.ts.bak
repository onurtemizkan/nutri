/**
 * Tests for native iOS LiDAR module
 * This module provides access to ARKit Scene Depth API and LiDAR sensor data
 */

import { Platform } from 'react-native';
import LiDARModule from '@/lib/modules/LiDARModule';
import type {
  DeviceCapabilities,
  LiDARCapture,
  CaptureMode,
} from '@/lib/types/ar-data';

// Mock Platform to simulate iOS
jest.mock('react-native/Libraries/Utilities/Platform', () => ({
  OS: 'ios',
  select: jest.fn((options) => options.ios),
}));

describe('LiDARModule', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Device Capabilities', () => {
    it('should check if device has LiDAR', async () => {
      const hasLiDAR = await LiDARModule.hasLiDAR();
      expect(typeof hasLiDAR).toBe('boolean');
    });

    it('should check if device supports Scene Depth', async () => {
      const supportsSceneDepth = await LiDARModule.supportsSceneDepth();
      expect(typeof supportsSceneDepth).toBe('boolean');
    });

    it('should get complete device capabilities', async () => {
      const capabilities = await LiDARModule.getDeviceCapabilities();

      expect(capabilities).toBeDefined();
      expect(typeof capabilities.hasLiDAR).toBe('boolean');
      expect(typeof capabilities.supportsSceneDepth).toBe('boolean');
      expect(capabilities.maxDepthResolution).toBeDefined();
      expect(capabilities.maxRGBResolution).toBeDefined();
      expect(capabilities.depthRange).toBeDefined();
      expect(typeof capabilities.frameRate).toBe('number');

      // Validate resolution structure
      expect(typeof capabilities.maxDepthResolution.width).toBe('number');
      expect(typeof capabilities.maxDepthResolution.height).toBe('number');

      // Validate depth range structure
      expect(typeof capabilities.depthRange.min).toBe('number');
      expect(typeof capabilities.depthRange.max).toBe('number');
      expect(capabilities.depthRange.min).toBeGreaterThan(0);
      expect(capabilities.depthRange.max).toBeGreaterThan(capabilities.depthRange.min);
    });

    it('should return appropriate capabilities for LiDAR devices', async () => {
      const capabilities = await LiDARModule.getDeviceCapabilities();

      if (capabilities.hasLiDAR) {
        // LiDAR devices should have 256x192 depth resolution
        expect(capabilities.maxDepthResolution.width).toBe(256);
        expect(capabilities.maxDepthResolution.height).toBe(192);
        expect(capabilities.frameRate).toBe(60);
        expect(capabilities.depthRange.min).toBeCloseTo(0.3);
        expect(capabilities.depthRange.max).toBeCloseTo(5.0);
      }
    });
  });

  describe('AR Session Management', () => {
    it('should start AR session successfully', async () => {
      const capabilities = await LiDARModule.getDeviceCapabilities();
      const mode: CaptureMode = capabilities.hasLiDAR ? 'lidar' : 'ar_depth';

      await expect(LiDARModule.startARSession(mode)).resolves.not.toThrow();
    });

    it('should stop AR session successfully', async () => {
      await expect(LiDARModule.stopARSession()).resolves.not.toThrow();
    });

    it('should check if AR session is running', async () => {
      // Session should not be running initially
      let isRunning = await LiDARModule.isARSessionRunning();
      expect(isRunning).toBe(false);

      // Start session
      const capabilities = await LiDARModule.getDeviceCapabilities();
      const mode: CaptureMode = capabilities.hasLiDAR ? 'lidar' : 'ar_depth';
      await LiDARModule.startARSession(mode);

      // Session should be running
      isRunning = await LiDARModule.isARSessionRunning();
      expect(isRunning).toBe(true);

      // Stop session
      await LiDARModule.stopARSession();

      // Session should not be running
      isRunning = await LiDARModule.isARSessionRunning();
      expect(isRunning).toBe(false);
    });

    it('should handle multiple start calls gracefully', async () => {
      const capabilities = await LiDARModule.getDeviceCapabilities();
      const mode: CaptureMode = capabilities.hasLiDAR ? 'lidar' : 'ar_depth';

      await LiDARModule.startARSession(mode);
      // Starting again should not throw
      await expect(LiDARModule.startARSession(mode)).resolves.not.toThrow();

      await LiDARModule.stopARSession();
    });

    it('should handle stop without start gracefully', async () => {
      await expect(LiDARModule.stopARSession()).resolves.not.toThrow();
    });
  });

  describe('Depth Data Capture', () => {
    beforeEach(async () => {
      const capabilities = await LiDARModule.getDeviceCapabilities();
      const mode: CaptureMode = capabilities.hasLiDAR ? 'lidar' : 'ar_depth';
      await LiDARModule.startARSession(mode);
    });

    afterEach(async () => {
      await LiDARModule.stopARSession();
    });

    it('should capture current depth frame', async () => {
      const depthCapture = await LiDARModule.captureDepthFrame();

      expect(depthCapture).toBeDefined();
      expect(depthCapture.depthBuffer).toBeDefined();
      expect(depthCapture.confidenceMap).toBeDefined();
      expect(typeof depthCapture.timestamp).toBe('number');
      expect(['low', 'medium', 'high']).toContain(depthCapture.depthQuality);
      expect(depthCapture.cameraIntrinsics).toBeDefined();
    });

    it('should validate depth buffer structure', async () => {
      const depthCapture = await LiDARModule.captureDepthFrame();
      const { depthBuffer } = depthCapture;

      expect(depthBuffer.data).toBeInstanceOf(Float32Array);
      expect(typeof depthBuffer.width).toBe('number');
      expect(typeof depthBuffer.height).toBe('number');
      expect(depthBuffer.format).toBe('float32');
      expect(depthBuffer.unit).toBe('meters');
      expect(depthBuffer.data.length).toBe(depthBuffer.width * depthBuffer.height);

      // Validate depth values are in reasonable range
      const depthValues = Array.from(depthBuffer.data);
      const validDepths = depthValues.filter(d => d > 0 && d < 10);
      expect(validDepths.length).toBeGreaterThan(0);
    });

    it('should validate confidence map structure', async () => {
      const depthCapture = await LiDARModule.captureDepthFrame();
      const { confidenceMap } = depthCapture;

      expect(confidenceMap.data).toBeInstanceOf(Uint8Array);
      expect(typeof confidenceMap.width).toBe('number');
      expect(typeof confidenceMap.height).toBe('number');
      expect(confidenceMap.data.length).toBe(confidenceMap.width * confidenceMap.height);

      // Validate confidence levels
      expect(confidenceMap.levels.low).toBe(1);
      expect(confidenceMap.levels.medium).toBe(2);
      expect(confidenceMap.levels.high).toBe(3);

      // Validate confidence values are within range
      const confidenceValues = Array.from(confidenceMap.data);
      const validConfidence = confidenceValues.every(c => c >= 0 && c <= 3);
      expect(validConfidence).toBe(true);
    });

    it('should validate camera intrinsics', async () => {
      const depthCapture = await LiDARModule.captureDepthFrame();
      const { cameraIntrinsics } = depthCapture;

      expect(typeof cameraIntrinsics.focalLength.x).toBe('number');
      expect(typeof cameraIntrinsics.focalLength.y).toBe('number');
      expect(typeof cameraIntrinsics.principalPoint.x).toBe('number');
      expect(typeof cameraIntrinsics.principalPoint.y).toBe('number');

      expect(cameraIntrinsics.imageResolution).toBeDefined();
      expect(typeof cameraIntrinsics.imageResolution.width).toBe('number');
      expect(typeof cameraIntrinsics.imageResolution.height).toBe('number');

      expect(Array.isArray(cameraIntrinsics.radialDistortion)).toBe(true);
      expect(Array.isArray(cameraIntrinsics.tangentialDistortion)).toBe(true);
    });

    it('should capture multiple frames with different timestamps', async () => {
      const frame1 = await LiDARModule.captureDepthFrame();

      // Wait a bit
      await new Promise(resolve => setTimeout(resolve, 50));

      const frame2 = await LiDARModule.captureDepthFrame();

      expect(frame2.timestamp).toBeGreaterThan(frame1.timestamp);
    });
  });

  describe('Camera Access', () => {
    it('should request camera permission', async () => {
      const permission = await LiDARModule.requestCameraPermission();
      expect(['granted', 'denied', 'restricted']).toContain(permission);
    });

    it('should check camera permission status', async () => {
      const status = await LiDARModule.getCameraPermissionStatus();
      expect(['granted', 'denied', 'restricted', 'undetermined']).toContain(status);
    });
  });

  describe('Error Handling', () => {
    it('should throw error when capturing without AR session', async () => {
      await expect(LiDARModule.captureDepthFrame()).rejects.toThrow();
    });

    it('should provide meaningful error messages', async () => {
      try {
        await LiDARModule.captureDepthFrame();
        fail('Should have thrown error');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
        expect((error as Error).message).toBeTruthy();
      }
    });
  });

  describe('Platform Compatibility', () => {
    it('should only be available on iOS', () => {
      if (Platform.OS !== 'ios') {
        expect(LiDARModule).toBeUndefined();
      } else {
        expect(LiDARModule).toBeDefined();
      }
    });
  });
});
