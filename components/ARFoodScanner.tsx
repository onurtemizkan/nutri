/**
 * AR Food Scanner Component
 *
 * Provides UI and logic for capturing RGB-D food data using iPhone LiDAR and ARKit.
 *
 * Features:
 * - LiDAR/AR depth capture
 * - Single frame and video sequence modes
 * - Real-time preview
 * - Session management
 * - RGB-D data export
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'expo-router';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Platform,
  AppState,
  AppStateStatus,
} from 'react-native';
import { Camera, CameraView } from 'expo-camera';
import LiDARModule from '@/lib/modules/LiDARModule';
import type {
  RGBDFrame,
  CaptureSession,
  DeviceCapabilities,
  CaptureMode,
} from '@/lib/types/ar-data';

/**
 * Scanner mode: single frame or video sequence
 */
type ScannerMode = 'photo' | 'video';

/**
 * Scanner state
 */
interface ScannerState {
  isScanning: boolean;
  isRecording: boolean;
  frameCount: number;
  error: string | null;
  permissionGranted: boolean;
  capabilities: DeviceCapabilities | null;
  currentSession: CaptureSession | null;
  // true when running a developer simulation fallback (no native LiDAR)
  simulated?: boolean;
}

/**
 * Component props
 */
interface ARFoodScannerProps {
  mode?: ScannerMode;
  metadata?: {
    foodItem?: string;
    notes?: string;
    [key: string]: unknown;
  };
  onCapture?: (frame: RGBDFrame) => void;
  onSessionStart?: (sessionId: string, session: CaptureSession) => void;
  onSessionEnd?: (session: CaptureSession) => void;
  onError?: (error: Error) => void;
}

const ARFoodScanner: React.FC<ARFoodScannerProps> = ({
  mode = 'photo',
  metadata = {},
  onCapture,
  onSessionStart,
  onSessionEnd,
  onError,
}) => {
  const [state, setState] = useState<ScannerState>({
    isScanning: false,
    isRecording: false,
    frameCount: 0,
    error: null,
    permissionGranted: false,
    capabilities: null,
    currentSession: null,
  });

  const cameraRef = useRef<CameraView>(null);
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const expectedFrameCountRef = useRef<number>(0);
  const actualFrameCountRef = useRef<number>(0);

  // Check for iOS platform
  useEffect(() => {
    if (Platform.OS !== 'ios') {
      setState(prev => ({
        ...prev,
        error: 'AR scanning is only supported on iOS devices',
      }));
      return;
    }

    // Check if LiDAR native module is available. The module exports an
    // `isAvailable` flag (JS fallback present) — use that to decide whether to
    // initialize the real scanner or enter a safe simulated mode for Expo Go.
    if (!LiDARModule || (LiDARModule as any).isAvailable === false) {
      // Enter simulated/fallback mode so the app remains usable in Expo Go.
      setState(prev => ({
        ...prev,
        capabilities: {
          hasLiDAR: false,
          supportsSceneDepth: true,
          sensorModel: 'simulator',
          maxDepthResolution: { width: 256, height: 192 },
          frameRate: 30,
          depthRange: { min: 0.2, max: 5 },
          recommendedCaptureMode: 'ar_depth',
        } as any,
        permissionGranted: true,
        simulated: true,
        error: null,
      }));
      return;
    }

    initializeScanner();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, []);

  // Handle app state changes (interruptions)
  useEffect(() => {
    const handleAppStateChange = async (nextAppState: AppStateStatus) => {
      if (nextAppState === 'background' || nextAppState === 'inactive') {
        // App backgrounded or interrupted (e.g., phone call)
        if (state.isRecording || state.isScanning) {
          console.log('App interrupted during scanning, stopping session');

          // Stop recording
          if (state.isRecording) {
            stopRecording();
          }

          // Stop scanning and notify user
          await stopScanning();

          Alert.alert(
            'Scanning Interrupted',
            'Recording stopped due to app interruption. Session data has been saved.',
            [{ text: 'OK' }]
          );
        }
      }
    };

    const subscription = AppState.addEventListener('change', handleAppStateChange);

    return () => {
      subscription.remove();
    };
  }, [state.isRecording, state.isScanning]);

  const initializeScanner = async () => {
    if (!LiDARModule || (LiDARModule as any).isAvailable === false) {
      setState(prev => ({ ...prev, error: 'LiDAR native module not available' }));
      return;
    }

    try {
      // Check device capabilities
      const capabilities = await LiDARModule.getDeviceCapabilities();

      if (!capabilities.supportsSceneDepth) {
        setState(prev => ({
          ...prev,
          error: 'AR depth scanning not supported on this device',
        }));
        return;
      }

      // Check camera permission
      const permissionStatus = await LiDARModule.getCameraPermissionStatus();

      if (permissionStatus !== 'granted') {
        const result = await LiDARModule.requestCameraPermission();
        if (result !== 'granted') {
          setState(prev => ({
            ...prev,
            error: 'Camera permission required for AR scanning',
            permissionGranted: false,
          }));
          return;
        }
      }

      setState(prev => ({
        ...prev,
        capabilities,
        permissionGranted: true,
        error: null,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to initialize scanner';
      setState(prev => ({ ...prev, error: errorMessage }));
      onError?.(error instanceof Error ? error : new Error(errorMessage));
    }
  };

  const startScanning = async () => {
    if (!state.capabilities || !LiDARModule) {
      return;
    }

    try {
      // Determine capture mode
      const captureMode: CaptureMode = state.capabilities.hasLiDAR ? 'lidar' : 'ar_depth';

      // Start AR session
      await LiDARModule.startARSession(captureMode);

      // Create new capture session (frames will be saved to disk, not in memory)
      const sessionId = `session_${Date.now()}`;
      const session: CaptureSession = {
        sessionId,
        startTime: Date.now(),
        endTime: 0,
        frames: [], // Empty - frames saved to disk to prevent memory leaks
        deviceCapabilities: state.capabilities,
        metadata: {
          ...metadata,
          captureType: mode === 'video' ? 'video_sequence' : 'single_frame',
        },
      };

      setState(prev => ({
        ...prev,
        isScanning: true,
        currentSession: session,
        frameCount: 0,
        error: null,
      }));

      onSessionStart?.(sessionId, session);

      // Start recording if in video mode
      if (mode === 'video') {
        startRecording();
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to start AR session';
      setState(prev => ({ ...prev, error: errorMessage }));
      onError?.(error instanceof Error ? error : new Error(errorMessage));
      Alert.alert('Error', 'Failed to start AR session');
    }
  };

  // Developer helper: simulated capture flow for when native LiDAR isn't available
  const simulateCaptureDepthFrame = async () => {
    try {
      // Create a fake depthData payload similar to native shape expected by app
      const fakeDepth = {
        width: state.capabilities?.maxDepthResolution.width || 256,
        height: state.capabilities?.maxDepthResolution.height || 192,
        buffer: new Uint8Array((state.capabilities?.maxDepthResolution.width || 256) * (state.capabilities?.maxDepthResolution.height || 192)).fill(128),
        confidenceMap: null,
        intrinsics: null,
      } as any;

      // Simulate camera capture if CameraView available
      let fakePhoto: any = { uri: 'data:image/png;base64,', width: 1024, height: 768 };
      if (cameraRef.current && (cameraRef.current as any).takePictureAsync) {
        try {
          fakePhoto = await (cameraRef.current as any).takePictureAsync?.({ quality: 0.6, skipProcessing: true });
        } catch (e) {
          // ignore and use placeholder
        }
      }

      const frame = {
        rgbImage: {
          uri: fakePhoto.uri,
          width: fakePhoto.width,
          height: fakePhoto.height,
          format: 'jpeg',
        },
        depthData: fakeDepth,
        timestamp: Date.now(),
        frameId: `sim_frame_${Date.now()}`,
        metadata: {
          deviceModel: 'simulator',
          osVersion: Platform.Version.toString(),
          // Cast to CaptureMode by using a supported mode string
          captureMode: ('ar_depth' as CaptureMode),
        },
      } as RGBDFrame;

      // call handler and increment frame count to exercise capture logic
      onCapture?.(frame);
      setState(prev => ({ ...prev, frameCount: prev.frameCount + 1 }));
      console.log('Simulated capture produced frame:', frame.frameId);
    } catch (err) {
      console.error('Simulated capture error', err);
    }
  };

  const stopScanning = async () => {
    try {
      // Stop recording if active
      if (state.isRecording) {
        stopRecording();
      }

      // Stop AR session
      if (LiDARModule) {
        await LiDARModule.stopARSession();
      }

      // Finalize session (frames array will be empty - actual frames are saved to disk)
      if (state.currentSession) {
        const finalSession: CaptureSession = {
          ...state.currentSession,
          endTime: Date.now(),
          // Note: frames array is empty - actual frame data is on disk
        };
        onSessionEnd?.(finalSession);
      }

      setState(prev => ({
        ...prev,
        isScanning: false,
        isRecording: false,
        currentSession: null,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to stop AR session';
      onError?.(error instanceof Error ? error : new Error(errorMessage));
    }
  };

  const captureFrame = async (): Promise<void> => {
    if (!state.isScanning || !LiDARModule || !cameraRef.current) {
      throw new Error('Scanner not active');
    }

    try {
      // Capture RGB image from camera
      const photo = await cameraRef.current.takePictureAsync({
        quality: 1,
        skipProcessing: false,
      });

      if (!photo) {
        throw new Error('Failed to capture RGB image');
      }

      // Capture depth data from LiDAR
      const depthData = await LiDARModule.captureDepthFrame();

      // Create RGB-D frame
      const frame: RGBDFrame = {
        rgbImage: {
          uri: photo.uri,
          width: photo.width,
          height: photo.height,
          format: 'jpeg',
        },
        depthData,
        timestamp: Date.now(),
        frameId: `frame_${String(state.frameCount).padStart(3, '0')}`,
        metadata: {
          deviceModel: state.capabilities?.hasLiDAR ? 'iPhone with LiDAR' : 'iPhone',
          osVersion: Platform.Version.toString(),
          captureMode: state.capabilities?.hasLiDAR ? 'lidar' : 'ar_depth',
        },
      };

      // Update frame count only - frames are saved to disk via onCapture callback
      // Don't accumulate frames in memory to prevent memory leaks
      setState(prev => ({
        ...prev,
        frameCount: prev.frameCount + 1,
      }));

      // Call capture callback - caller is responsible for saving to disk
      onCapture?.(frame);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to capture frame';
      setState(prev => ({ ...prev, error: errorMessage }));
      onError?.(error instanceof Error ? error : new Error(errorMessage));
      Alert.alert('Error', 'Error capturing frame');
    }
  };

  const startRecording = () => {
    setState(prev => ({ ...prev, isRecording: true }));

    // Reset frame counters
    expectedFrameCountRef.current = 0;
    actualFrameCountRef.current = 0;

    // Capture frames at ~60fps (for LiDAR) or ~30fps (for AR depth)
    const fps = state.capabilities?.frameRate || 30;
    const interval = 1000 / fps;

    recordingIntervalRef.current = setInterval(() => {
      expectedFrameCountRef.current++;

      captureFrame().then(() => {
        actualFrameCountRef.current++;

        // Calculate frame drop rate
        const dropRate =
          1 - actualFrameCountRef.current / expectedFrameCountRef.current;

        // Warn if drop rate exceeds 10%
        if (dropRate > 0.1 && expectedFrameCountRef.current % 60 === 0) {
          console.warn(
            `Frame drop rate: ${(dropRate * 100).toFixed(1)}% ` +
            `(${actualFrameCountRef.current}/${expectedFrameCountRef.current} frames)`
          );
        }
      }).catch(error => {
        console.error('Frame capture error:', error);
      });
    }, interval);
  };

  const stopRecording = () => {
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }
    setState(prev => ({ ...prev, isRecording: false }));
  };

  const cleanup = async () => {
    // Clear interval first
    if (recordingIntervalRef.current) {
      clearInterval(recordingIntervalRef.current);
      recordingIntervalRef.current = null;
    }

    // Then stop AR session if running
    if (LiDARModule) {
      try {
        const isRunning = await LiDARModule.isARSessionRunning();
        if (isRunning) {
          await LiDARModule.stopARSession();
        }
      } catch (error) {
        console.error('Cleanup error:', error);
      }
    }
  };

  // Show error state
  if (state.error) {
    return (
      <View style={styles.container}>
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>{state.error}</Text>
          {state.error.includes('permission') && (
            <TouchableOpacity style={styles.button} onPress={initializeScanner}>
              <Text style={styles.buttonText}>Request Permission</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    );
  }

  // Show loading state
  if (!state.capabilities) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Initializing AR scanner...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {state.simulated && (
        <View style={styles.simulationBanner}>
          <Text style={styles.simulationText}>
            Running in simulated AR mode — native LiDAR not available. For
            accurate depth capture, run a development build on a compatible
            iOS device.
          </Text>
        </View>
      )}
      {/* Dev-only controls */}
      {__DEV__ && (
        <View style={styles.devControls}>
          <TouchableOpacity style={[styles.button, { backgroundColor: '#444' }]} onPress={simulateCaptureDepthFrame}>
            <Text style={styles.buttonText}>Simulate Capture</Text>
          </TouchableOpacity>
        </View>
      )}
      <View style={styles.header}>
        <Text style={styles.title}>AR Food Scanner</Text>
        <Text style={styles.subtitle}>
          {state.capabilities.hasLiDAR ? 'LiDAR Mode' : 'AR Depth Mode'}
        </Text>
      </View>

      <View style={styles.cameraContainer}>
        {state.permissionGranted && (
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing="back"
          />
        )}

        {state.isScanning && (
          <View style={styles.overlay}>
            <View style={styles.frameGuide} />
            <Text style={styles.frameCount}>Frames captured: {state.frameCount}</Text>
          </View>
        )}
      </View>

      <View style={styles.controls}>
        {!state.isScanning ? (
          <TouchableOpacity
            style={[styles.button, styles.startButton]}
            onPress={startScanning}
          >
            <Text style={styles.buttonText}>
              {mode === 'video' ? 'Start Recording' : 'Start Scanning'}
            </Text>
          </TouchableOpacity>
        ) : (
          <>
            {mode === 'photo' && (
              <TouchableOpacity
                style={[styles.button, styles.captureButton]}
                onPress={captureFrame}
              >
                <Text style={styles.buttonText}>Capture Frame</Text>
              </TouchableOpacity>
            )}

            <TouchableOpacity
              style={[styles.button, styles.stopButton]}
              onPress={stopScanning}
            >
              <Text style={styles.buttonText}>
                {mode === 'video' ? 'Stop Recording' : 'Stop Scanning'}
              </Text>
            </TouchableOpacity>
          </>
        )}
      </View>

      <View style={styles.info}>
        <Text style={styles.infoText}>
          Resolution: {state.capabilities.maxDepthResolution.width}x
          {state.capabilities.maxDepthResolution.height}
        </Text>
        <Text style={styles.infoText}>
          Frame rate: {state.capabilities.frameRate} Hz
        </Text>
        <Text style={styles.infoText}>
          Range: {state.capabilities.depthRange.min}m - {state.capabilities.depthRange.max}m
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  header: {
    padding: 20,
    paddingTop: 60,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#aaa',
  },
  cameraContainer: {
    flex: 1,
    position: 'relative',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
  frameGuide: {
    width: 280,
    height: 280,
    borderWidth: 2,
    borderColor: '#007AFF',
    borderRadius: 20,
  },
  frameCount: {
    position: 'absolute',
    top: 40,
    color: '#fff',
    fontSize: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'center',
    padding: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    gap: 12,
  },
  button: {
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    minWidth: 120,
  },
  startButton: {
    backgroundColor: '#007AFF',
  },
  captureButton: {
    backgroundColor: '#34C759',
  },
  stopButton: {
    backgroundColor: '#FF3B30',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  info: {
    padding: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
  },
  infoText: {
    color: '#aaa',
    fontSize: 12,
    marginBottom: 4,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
  },
  loadingText: {
    color: '#aaa',
    fontSize: 16,
    marginTop: 16,
  },
  simulationBanner: {
    backgroundColor: '#333',
    padding: 8,
  },
  simulationText: {
    color: '#ffd966',
    fontSize: 12,
    textAlign: 'center',
  },
  devControls: {
    position: 'absolute',
    right: 12,
    top: 16,
    zIndex: 1000,
  },
});

export default ARFoodScanner;
