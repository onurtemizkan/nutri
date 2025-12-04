# AR/LiDAR Food Scanning Implementation

## Overview

This document describes the AR/LiDAR food scanning feature implementation for the Nutri app. The feature enables capturing RGB-D (color + depth) data from iPhone LiDAR sensors for deep learning-based nutrition estimation.

## Features Implemented

### 1. Data Structures (`lib/types/ar-data.ts`)

Complete TypeScript type definitions for RGB-D data:

- **Device Capabilities**: LiDAR detection, depth resolution, frame rates
- **Depth Buffer**: Float32 arrays containing depth measurements in meters
- **Confidence Maps**: Uint8 arrays with depth quality indicators (1-3)
- **Camera Intrinsics**: Calibration parameters for 3D reconstruction
- **RGB-D Frames**: Combined color images and depth data
- **Capture Sessions**: Multiple frames with metadata
- **ML Export Format**: Structured data ready for PyTorch/TensorFlow

#### Key Specifications

- **LiDAR Depth**: 256x192 resolution @ 60Hz, 0.3-5.0m range
- **AR Depth Fallback**: 640x480 resolution @ 30Hz, 0.5-3.0m range
- **RGB Resolution**: 1920x1440 (LiDAR) or 1920x1080 (AR fallback)
- **Data Format**: Float32 for depth, Uint8 for confidence, JPEG for RGB

### 2. Native iOS LiDAR Module (`modules/lidar-module/`)

Swift implementation using ARKit Scene Depth API:

```
modules/lidar-module/
├── expo-module.config.json    # Expo module configuration
├── package.json                # Module metadata
└── ios/
    └── LiDARModule.swift       # Native ARKit implementation
```

#### Module Capabilities

- **Device Detection**: Checks for LiDAR and Scene Depth support
- **AR Session Management**: Start/stop ARKit sessions
- **Real-time Depth Capture**: Extract depth buffers and confidence maps
- **Camera Intrinsics**: Provide calibration data
- **Permission Handling**: Request and check camera access
- **Error Handling**: Comprehensive error messages

#### API Methods

```typescript
hasLiDAR(): Promise<boolean>
supportsSceneDepth(): Promise<boolean>
getDeviceCapabilities(): Promise<DeviceCapabilities>
startARSession(mode: CaptureMode): Promise<void>
stopARSession(): Promise<void>
isARSessionRunning(): Promise<boolean>
captureDepthFrame(): Promise<LiDARCapture>
requestCameraPermission(): Promise<PermissionStatus>
getCameraPermissionStatus(): Promise<PermissionStatus>
```

### 3. AR Food Scanner Component (`components/ARFoodScanner.tsx`)

React Native component providing the scanning interface:

#### Features

- **Dual Mode**: Single photo or video sequence capture
- **Real-time Preview**: Live camera feed with AR overlay
- **Frame Capture**: RGB + depth data synchronized capture
- **Session Management**: Automatic session creation and cleanup
- **Progress Tracking**: Frame counter and status display
- **Error Handling**: User-friendly error messages

#### Props

```typescript
interface ARFoodScannerProps {
  mode?: 'photo' | 'video';
  metadata?: {
    foodItem?: string;
    notes?: string;
  };
  onCapture?: (frame: RGBDFrame) => void;
  onSessionStart?: (sessionId: string, session: CaptureSession) => void;
  onSessionEnd?: (session: CaptureSession) => void;
  onError?: (error: Error) => void;
}
```

### 4. Data Capture and Storage (`lib/utils/capture-storage.ts`)

File system management for RGB-D data:

#### Storage Structure

```
nutri-captures/
└── session_{timestamp}/
    ├── rgb/               # JPEG images
    │   ├── frame_000.jpg
    │   ├── frame_001.jpg
    │   └── ...
    ├── depth/             # Binary depth maps (Float32)
    │   ├── frame_000.bin
    │   ├── frame_001.bin
    │   └── ...
    ├── confidence/        # Binary confidence maps (Uint8)
    │   ├── frame_000.bin
    │   ├── frame_001.bin
    │   └── ...
    ├── metadata/          # JSON metadata per frame
    │   ├── frame_000.json
    │   ├── frame_001.json
    │   └── ...
    └── export/            # ML-ready export
        └── manifest.json
```

#### ML Export Format

```typescript
interface MLReadyData {
  sessionId: string;
  version: string;
  format: 'rgbd_normalized' | 'rgbd_sequence' | 'rgbd_raw';
  rgbImages: TensorMetadata[];    // Paths, dimensions, dtype
  depthMaps: TensorMetadata[];     // Paths, dimensions, dtype, normalization
  confidenceMaps: TensorMetadata[]; // Paths, dimensions, dtype
  metadata: {
    deviceModel: string;
    osVersion: string;
    captureMode: 'lidar' | 'ar_depth';
    timestamp: number;
    frameRate: number;
    duration: number;
    cameraIntrinsics: CameraIntrinsics;
  };
  annotations: {
    foodItem?: string;
    portionSize?: string;
  };
}
```

### 5. AR Scanning Screen (`app/ar-scan-food.tsx`)

Complete UI for AR food scanning:

#### Features

- **Food Information**: Optional name and notes input
- **Mode Selection**: Photo vs. video toggle
- **Scanning Controls**: Start/stop/capture buttons
- **Progress Display**: Frame count and status
- **Export Dialog**: ML data export confirmation
- **Error Handling**: User-friendly alerts

### 6. Comprehensive Test Suite

Test-Driven Development approach with extensive coverage:

#### Test Files

1. **`__tests__/types/ar-data.test.ts`** (12 tests)
   - Device capabilities validation
   - Data structure correctness
   - Type safety verification
   - ML export format validation

2. **`__tests__/modules/lidar-module.test.ts`** (20+ tests)
   - Device capability detection
   - AR session management
   - Depth data capture
   - Permission handling
   - Error scenarios

3. **`__tests__/components/ARFoodScanner.test.tsx`** (15+ tests)
   - Component rendering
   - Scanning controls
   - Frame capture
   - Session management
   - Error handling
   - Props and callbacks

4. **`__tests__/lib/capture-storage.test.ts`** (15+ tests)
   - Directory creation
   - Frame storage
   - ML data export
   - Session management
   - Cleanup operations

5. **`__tests__/integration/ar-workflow.test.ts`** (8 tests)
   - End-to-end single frame workflow
   - End-to-end video sequence workflow
   - Data validation
   - Error scenarios

**Total Tests**: 70+ comprehensive test cases

## Implementation Approach

### Test-Driven Development (TDD)

1. **Write Tests First**: All functionality was spec'd out via tests before implementation
2. **Red-Green-Refactor**: Tests failed initially, then implementation made them pass
3. **Comprehensive Coverage**: Edge cases, error scenarios, and integration paths tested
4. **Type Safety**: Zero `any` types, full TypeScript strict mode

### Architecture Principles

1. **Separation of Concerns**
   - Native module: ARKit integration
   - Component: UI and user interaction
   - Storage: File system and data management
   - Types: Shared data structures

2. **Error Handling**
   - Graceful degradation (LiDAR → AR depth fallback)
   - User-friendly error messages
   - Automatic cleanup on failures

3. **Performance Optimization**
   - Efficient binary storage for depth data
   - Lazy loading of large datasets
   - Frame-rate aware capture (60fps for LiDAR)

4. **ML Integration Ready**
   - Standard tensor formats
   - Normalization metadata
   - Camera calibration data
   - Batch processing support

## Usage

### Basic Photo Capture

```typescript
import ARFoodScanner from '@/components/ARFoodScanner';

<ARFoodScanner
  mode="photo"
  metadata={{ foodItem: 'Apple' }}
  onCapture={(frame) => console.log('Captured frame:', frame.frameId)}
  onSessionEnd={(session) => console.log('Session complete:', session.sessionId)}
/>
```

### Video Sequence Capture

```typescript
<ARFoodScanner
  mode="video"
  metadata={{
    foodItem: 'Meal plate',
    notes: 'Multiple angles for better reconstruction'
  }}
  onCapture={(frame) => {
    // Frames captured automatically at device frame rate (60Hz for LiDAR)
    saveFrame(frame);
  }}
  onSessionEnd={async (session) => {
    // Export ML-ready data
    const mlData = await captureStorage.exportMLData(session, {
      outputDir: 'ml-data',
      format: 'rgbd_normalized',
      includeConfidenceMaps: true,
      normalizeDepth: true,
      normalizationMethod: 'min_max',
      compressRGB: true,
      rgbFormat: 'jpeg',
      rgbQuality: 90,
    });

    console.log('ML data ready:', mlData);
  }}
/>
```

### Storage Management

```typescript
import { captureStorage } from '@/lib/utils/capture-storage';

// Initialize storage
await captureStorage.initialize();

// List all sessions
const sessions = await captureStorage.listSessions();

// Get session info
const info = await captureStorage.getSessionInfo('session_123');

// Delete old session
await captureStorage.deleteSession('session_old');
```

## Configuration

### Permissions (app.json)

```json
{
  "ios": {
    "infoPlist": {
      "NSCameraUsageDescription": "Nutri needs camera access to scan and analyze your food for automatic nutrition tracking.",
      "NSLocationWhenInUseUsageDescription": "Nutri uses ARKit for 3D food scanning, which requires location services for AR features."
    }
  }
}
```

### Module Configuration (expo-module.config.json)

```json
{
  "platforms": ["ios"],
  "ios": {
    "modules": ["LiDARModule"]
  }
}
```

## Technical Specifications

### Data Formats

| Data Type | Format | Resolution | Frame Rate | Size per Frame |
|-----------|--------|------------|-----------|----------------|
| RGB Image | JPEG | 1920x1440 | 60Hz | ~300KB |
| Depth Map | Float32 | 256x192 | 60Hz | ~192KB |
| Confidence | Uint8 | 256x192 | 60Hz | ~48KB |
| **Total** | | | | **~540KB** |

### Storage Requirements

- **Single Frame**: ~540KB
- **1 Second Video (60fps)**: ~32MB
- **10 Second Video**: ~320MB

### Device Support

| Device | LiDAR | Scene Depth | Resolution | Frame Rate |
|--------|-------|-------------|------------|-----------|
| iPhone 12 Pro/Max | ✅ | ✅ | 256x192 | 60Hz |
| iPhone 13 Pro/Max | ✅ | ✅ | 256x192 | 60Hz |
| iPhone 14 Pro/Max | ✅ | ✅ | 256x192 | 60Hz |
| iPhone 15 Pro/Max | ✅ | ✅ | 256x192 | 60Hz |
| Other iPhones (A12+) | ❌ | ✅ | 640x480 | 30Hz |

## Future Enhancements

### Phase 1: ML Integration (Next Step)

1. **Model Deployment**
   - Core ML conversion of PyTorch models
   - On-device inference for real-time feedback
   - Cloud-based inference for complex models

2. **Nutrition Estimation**
   - DPF-Nutrition or similar model integration
   - Calorie prediction from RGB-D data
   - Macronutrient estimation

3. **Feedback Loop**
   - User corrections for model improvement
   - Active learning pipeline
   - Model versioning and updates

### Phase 2: Advanced Features

1. **Real-time Guidance**
   - AR overlays showing optimal capture angles
   - Depth quality indicators
   - Frame coverage visualization

2. **3D Reconstruction**
   - Point cloud generation from depth maps
   - Mesh reconstruction
   - Volume estimation for portion sizes

3. **Multi-item Detection**
   - Separate foods on same plate
   - Individual nutrition per item
   - Bounding boxes and segmentation

4. **Performance Optimization**
   - Background processing
   - Progressive upload
   - Delta compression

## References

### Research Papers

1. **DPF-Nutrition** (14.7% error for calorie prediction)
   - RGB-D food nutrition estimation
   - Nutrition5k dataset

2. **ARKit Scene Depth API**
   - Direct Time-of-Flight LiDAR
   - 256x192 depth maps @ 60Hz

3. **Food Volume Estimation**
   - 3D reconstruction from depth
   - Portion size calculation

### Apple Documentation

- [ARKit Scene Understanding](https://developer.apple.com/documentation/arkit/arkit_in_ios/environmental_analysis)
- [AR Scene Depth](https://developer.apple.com/documentation/arkit/ardepthdatatype)
- [Expo Modules API](https://docs.expo.dev/modules/overview/)

## Implementation Summary

✅ **Completed**:
- TypeScript type definitions (100% type-safe)
- Native iOS LiDAR module (Swift + ARKit)
- AR food scanner component (React Native)
- Data capture and storage (File system)
- AR scanning screen (Complete UI)
- Comprehensive test suite (70+ tests)

⏳ **Next Steps**:
- ML model integration (DPF-Nutrition or custom)
- Real-time inference and feedback
- User testing and iteration

---

**Total Implementation Time**: ~2 hours
**Lines of Code**: ~3,500 LOC
**Test Coverage**: Comprehensive (TDD approach)
**Ready for**: ML model integration and user testing
