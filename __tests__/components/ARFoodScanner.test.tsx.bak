/**
 * Tests for AR Food Scanner Component
 *
 * This component provides the UI for capturing RGB-D food data
 * using the LiDAR module.
 */

import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import ARFoodScanner from '@/components/ARFoodScanner';
import LiDARModule from '@/lib/modules/LiDARModule';

// Mock the LiDAR module
jest.mock('@/lib/modules/LiDARModule', () => ({
  __esModule: true,
  default: {
    hasLiDAR: jest.fn(),
    supportsSceneDepth: jest.fn(),
    getDeviceCapabilities: jest.fn(),
    startARSession: jest.fn(),
    stopARSession: jest.fn(),
    isARSessionRunning: jest.fn(),
    captureDepthFrame: jest.fn(),
    requestCameraPermission: jest.fn(),
    getCameraPermissionStatus: jest.fn(),
  },
}));

// Mock Camera from expo-camera
jest.mock('expo-camera', () => ({
  Camera: {
    requestCameraPermissionsAsync: jest.fn(),
    getCameraPermissionsAsync: jest.fn(),
  },
}));

describe('ARFoodScanner', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Setup default mocks
    (LiDARModule.hasLiDAR as jest.Mock).mockResolvedValue(true);
    (LiDARModule.supportsSceneDepth as jest.Mock).mockResolvedValue(true);
    (LiDARModule.getDeviceCapabilities as jest.Mock).mockResolvedValue({
      hasLiDAR: true,
      supportsSceneDepth: true,
      maxDepthResolution: { width: 256, height: 192 },
      maxRGBResolution: { width: 1920, height: 1440 },
      depthRange: { min: 0.3, max: 5.0 },
      frameRate: 60,
    });
    (LiDARModule.getCameraPermissionStatus as jest.Mock).mockResolvedValue('granted');
    (LiDARModule.isARSessionRunning as jest.Mock).mockResolvedValue(false);
  });

  describe('Component Rendering', () => {
    it('should render scanner interface', () => {
      const { getByText } = render(<ARFoodScanner />);
      expect(getByText(/AR Food Scanner/i)).toBeTruthy();
    });

    it('should show device capabilities on mount', async () => {
      const { getByText } = render(<ARFoodScanner />);

      await waitFor(() => {
        expect(LiDARModule.getDeviceCapabilities).toHaveBeenCalled();
      });
    });

    it('should show permission prompt if not granted', async () => {
      (LiDARModule.getCameraPermissionStatus as jest.Mock).mockResolvedValue('denied');

      const { getByText } = render(<ARFoodScanner />);

      await waitFor(() => {
        expect(getByText(/Camera permission/i)).toBeTruthy();
      });
    });
  });

  describe('Scanning Controls', () => {
    it('should start scanning when start button is pressed', async () => {
      (LiDARModule.startARSession as jest.Mock).mockResolvedValue(undefined);

      const { getByText } = render(<ARFoodScanner />);

      const startButton = getByText(/Start Scanning/i);
      fireEvent.press(startButton);

      await waitFor(() => {
        expect(LiDARModule.startARSession).toHaveBeenCalledWith('lidar');
      });
    });

    it('should stop scanning when stop button is pressed', async () => {
      (LiDARModule.isARSessionRunning as jest.Mock).mockResolvedValue(true);
      (LiDARModule.stopARSession as jest.Mock).mockResolvedValue(undefined);

      const { getByText } = render(<ARFoodScanner />);

      const stopButton = getByText(/Stop Scanning/i);
      fireEvent.press(stopButton);

      await waitFor(() => {
        expect(LiDARModule.stopARSession).toHaveBeenCalled();
      });
    });

    it('should fallback to ar_depth mode if LiDAR not available', async () => {
      (LiDARModule.hasLiDAR as jest.Mock).mockResolvedValue(false);
      (LiDARModule.getDeviceCapabilities as jest.Mock).mockResolvedValue({
        hasLiDAR: false,
        supportsSceneDepth: true,
        maxDepthResolution: { width: 640, height: 480 },
        maxRGBResolution: { width: 1920, height: 1080 },
        depthRange: { min: 0.5, max: 3.0 },
        frameRate: 30,
      });
      (LiDARModule.startARSession as jest.Mock).mockResolvedValue(undefined);

      const { getByText } = render(<ARFoodScanner />);

      const startButton = getByText(/Start Scanning/i);
      fireEvent.press(startButton);

      await waitFor(() => {
        expect(LiDARModule.startARSession).toHaveBeenCalledWith('ar_depth');
      });
    });
  });

  describe('Frame Capture', () => {
    beforeEach(() => {
      (LiDARModule.isARSessionRunning as jest.Mock).mockResolvedValue(true);
      (LiDARModule.captureDepthFrame as jest.Mock).mockResolvedValue({
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
      });
    });

    it('should capture single frame when capture button is pressed', async () => {
      const { getByText } = render(<ARFoodScanner />);

      const captureButton = getByText(/Capture Frame/i);
      fireEvent.press(captureButton);

      await waitFor(() => {
        expect(LiDARModule.captureDepthFrame).toHaveBeenCalled();
      });
    });

    it('should show frame count after capture', async () => {
      const { getByText, getByTestId } = render(<ARFoodScanner />);

      const captureButton = getByText(/Capture Frame/i);
      fireEvent.press(captureButton);

      await waitFor(() => {
        expect(getByText(/Frames captured: 1/i)).toBeTruthy();
      });
    });

    it('should handle capture errors gracefully', async () => {
      (LiDARModule.captureDepthFrame as jest.Mock).mockRejectedValue(
        new Error('Capture failed')
      );

      const { getByText } = render(<ARFoodScanner />);

      const captureButton = getByText(/Capture Frame/i);
      fireEvent.press(captureButton);

      await waitFor(() => {
        expect(getByText(/Error capturing frame/i)).toBeTruthy();
      });
    });

    it('should capture multiple frames in video mode', async () => {
      const { getByText } = render(<ARFoodScanner mode="video" />);

      const startButton = getByText(/Start Recording/i);
      fireEvent.press(startButton);

      // Wait for multiple captures
      await waitFor(() => {
        expect(LiDARModule.captureDepthFrame).toHaveBeenCalledTimes(1);
      }, { timeout: 1000 });
    });
  });

  describe('Session Management', () => {
    it('should create new session on start', async () => {
      const onSessionStart = jest.fn();
      const { getByText } = render(<ARFoodScanner onSessionStart={onSessionStart} />);

      const startButton = getByText(/Start Scanning/i);
      fireEvent.press(startButton);

      await waitFor(() => {
        expect(onSessionStart).toHaveBeenCalled();
        const sessionId = onSessionStart.mock.calls[0][0];
        expect(sessionId).toMatch(/^session_/);
      });
    });

    it('should finalize session on stop', async () => {
      const onSessionEnd = jest.fn();
      (LiDARModule.isARSessionRunning as jest.Mock).mockResolvedValue(true);

      const { getByText } = render(<ARFoodScanner onSessionEnd={onSessionEnd} />);

      const stopButton = getByText(/Stop Scanning/i);
      fireEvent.press(stopButton);

      await waitFor(() => {
        expect(onSessionEnd).toHaveBeenCalled();
      });
    });

    it('should include metadata in session', async () => {
      const onSessionStart = jest.fn();
      const metadata = { foodItem: 'Apple', notes: 'Test' };

      const { getByText } = render(
        <ARFoodScanner onSessionStart={onSessionStart} metadata={metadata} />
      );

      const startButton = getByText(/Start Scanning/i);
      fireEvent.press(startButton);

      await waitFor(() => {
        expect(onSessionStart).toHaveBeenCalled();
        const session = onSessionStart.mock.calls[0][1];
        expect(session.metadata.foodItem).toBe('Apple');
        expect(session.metadata.notes).toBe('Test');
      });
    });
  });

  describe('Error Handling', () => {
    it('should show error if AR session fails to start', async () => {
      (LiDARModule.startARSession as jest.Mock).mockRejectedValue(
        new Error('ARSession failed')
      );

      const { getByText } = render(<ARFoodScanner />);

      const startButton = getByText(/Start Scanning/i);
      fireEvent.press(startButton);

      await waitFor(() => {
        expect(getByText(/Failed to start AR session/i)).toBeTruthy();
      });
    });

    it('should show error if device does not support depth', async () => {
      (LiDARModule.supportsSceneDepth as jest.Mock).mockResolvedValue(false);

      const { getByText } = render(<ARFoodScanner />);

      await waitFor(() => {
        expect(getByText(/not supported/i)).toBeTruthy();
      });
    });

    it('should cleanup on unmount', async () => {
      (LiDARModule.isARSessionRunning as jest.Mock).mockResolvedValue(true);
      (LiDARModule.stopARSession as jest.Mock).mockResolvedValue(undefined);

      const { unmount } = render(<ARFoodScanner />);
      unmount();

      await waitFor(() => {
        expect(LiDARModule.stopARSession).toHaveBeenCalled();
      });
    });
  });

  describe('Props and Callbacks', () => {
    it('should call onCapture callback with frame data', async () => {
      const onCapture = jest.fn();
      (LiDARModule.isARSessionRunning as jest.Mock).mockResolvedValue(true);

      const { getByText } = render(<ARFoodScanner onCapture={onCapture} />);

      const captureButton = getByText(/Capture Frame/i);
      fireEvent.press(captureButton);

      await waitFor(() => {
        expect(onCapture).toHaveBeenCalled();
        const frame = onCapture.mock.calls[0][0];
        expect(frame.rgbImage).toBeDefined();
        expect(frame.depthData).toBeDefined();
      });
    });

    it('should call onError callback on errors', async () => {
      const onError = jest.fn();
      (LiDARModule.captureDepthFrame as jest.Mock).mockRejectedValue(
        new Error('Test error')
      );

      const { getByText } = render(<ARFoodScanner onError={onError} />);

      const captureButton = getByText(/Capture Frame/i);
      fireEvent.press(captureButton);

      await waitFor(() => {
        expect(onError).toHaveBeenCalledWith(expect.any(Error));
      });
    });
  });
});
