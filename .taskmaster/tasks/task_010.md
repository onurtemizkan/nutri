# Task ID: 10

**Title:** Implement AR Portion Size Measurement

**Status:** done

**Dependencies:** 2 ✓

**Priority:** low

**Description:** Add AR capability to measure food portion dimensions and improve nutrition estimation accuracy.

**Details:**

1. Install AR dependencies:
   - expo-three (already installed - three.js is in dependencies)
   - expo-gl (already installed)
   - @react-three/fiber for React Native

2. Create AR measurement component in `lib/components/ARPortionMeasure.tsx`:
   - Initialize AR session with plane detection
   - Render measurement guides on detected surfaces
   - Allow user to place measurement points
   - Calculate bounding box dimensions (width, height, depth)
   - Return dimensions in centimeters

3. Update food scanning flow (`app/scan-food.tsx`):
   - Add 'Measure with AR' button after capturing photo
   - Launch AR measurement overlay
   - Pass dimensions to food analysis API
   - Update `mockMeasurements` with real AR data

4. AR measurement flow:
   1. User captures food photo
   2. User taps 'Measure Portion'
   3. AR view opens with plane detection
   4. User taps to place corner points (4 points for bounding box)
   5. App calculates volume and converts to portion weight
   6. Dimensions sent to /api/food/analyze

5. Dimension to weight conversion (in food_analysis_service.py):
   - Already implemented in `_estimate_portion_from_dimensions()`
   - Uses food density estimates
   - Returns estimated weight in grams

6. Calibration feature:
   - Include reference object option (credit card, hand)
   - Use known dimensions to calibrate scale
   - Improve accuracy for subsequent measurements

7. Fallback handling:
   - If AR not supported (older devices), show manual size picker
   - Options: Small, Medium, Large with example photos

**Test Strategy:**

1. Unit tests for dimension calculation
2. Test AR component mounting/unmounting
3. Test plane detection callbacks
4. Integration test with mock AR data
5. Test fallback to manual size picker
6. Manual testing on physical device with AR support
7. Test calibration accuracy with known objects

## Subtasks

### 10.1. Install AR Dependencies and Configure Native Modules

**Status:** done  
**Dependencies:** None  

Add required AR dependencies for React Native/Expo including @react-three/fiber, react-three-fiber, and configure native AR capabilities for iOS (ARKit) and Android (ARCore).

**Details:**

1. Install @react-three/fiber and react-three-fiber packages
2. Configure expo plugins for AR in app.json (expo-camera already configured)
3. Set up iOS ARKit permissions in Info.plist (NSCameraUsageDescription already exists)
4. Configure Android ARCore requirements in AndroidManifest.xml
5. Verify expo-gl and three.js integration
6. Create basic AR session test to verify setup
7. Document AR capability requirements for devices (iOS 11+, ARCore-compatible Android)

### 10.2. Create Interactive AR Measurement Component

**Status:** done  
**Dependencies:** 10.1  

Build the core ARPortionMeasure component that allows users to tap 4 corner points to create a bounding box and measure food portion dimensions in real-world coordinates.

**Details:**

1. Create lib/components/ARPortionMeasure.tsx component
2. Initialize AR session with plane detection enabled
3. Implement tap-to-place point placement (4 corners for bounding box)
4. Convert screen coordinates to world coordinates using AR raycasting
5. Calculate real-world dimensions (width, height, depth) from placed points
6. Display visual guides showing detected plane surface
7. Render bounding box overlay with dimension labels
8. Add point placement indicators and connection lines
9. Implement reset/undo functionality for point placement
10. Return ARMeasurement type with confidence scoring based on plane detection quality
11. Handle edge cases: insufficient plane detection, invalid point placement

### 10.3. Build AR Measurement Modal/Overlay Screen

**Status:** done  
**Dependencies:** 10.2  

Create the modal screen (app/ar-measure-portion.tsx) that launches the AR measurement experience with user instructions and controls.

**Details:**

1. Create app/ar-measure-portion.tsx as a modal screen
2. Integrate ARPortionMeasure component into modal
3. Design instruction UI:
   - Step-by-step guide for users (detect plane, place 4 corners)
   - Visual indicators for current step
   - Progress indicator during plane detection
4. Add control buttons:
   - Confirm measurement (validates 4 points placed)
   - Cancel and return to scan screen
   - Reset measurement (clear all points)
5. Display real-time measurement quality indicator
6. Show current dimensions as user places points
7. Handle AR session lifecycle (start on mount, cleanup on unmount)
8. Add error states: no plane detected, AR not supported
9. Implement navigation: return measured dimensions to caller

### 10.4. Integrate AR Measurement into Food Scanning Flow

**Status:** done  
**Dependencies:** 10.3  

Update scan-food.tsx to include 'Measure with AR' button after photo capture, launch the AR measurement modal, and pass captured dimensions to the food analysis API.

**Details:**

1. Update scan-food.tsx after photo capture (line 114 area)
2. Add 'Measure with AR' button alongside 'Analyze Food' button
3. Implement AR measurement flow:
   - Launch ar-measure-portion modal
   - Receive ARMeasurement result from modal
   - Store measurements in component state
4. Update foodAnalysisApi.analyzeFood() call to include measurements
5. Replace mockMeasurements with real AR data
6. Display measurement quality indicator in UI (high/medium/low badge)
7. Show captured dimensions in preview (width x height x depth)
8. Allow re-measurement before final analysis
9. Handle AR not available gracefully (hide button, show alternative)
10. Update UI flow: Photo → Measure (optional) → Analyze → Results

### 10.5. Implement Reference Object Calibration Feature

**Status:** done  
**Dependencies:** 10.2  

Build calibration wizard allowing users to use a credit card or other reference object to improve AR measurement accuracy.

**Details:**

1. Create lib/components/ARCalibration.tsx component
2. Implement credit card calibration mode:
   - Standard dimensions: 85.60mm × 53.98mm
   - AR measurement of credit card
   - Calculate calibration factor: measured/actual
3. Build calibration wizard UI:
   - Introduction screen explaining calibration
   - Place credit card on surface instructions
   - Measure card with AR (4 corner points)
   - Validation: check if dimensions are reasonable (within 20% of standard)
   - Success/failure feedback
4. Store calibration factor in AsyncStorage/SecureStore
5. Apply calibration to subsequent measurements (multiply by factor)
6. Add calibration status indicator in AR measurement screen
7. Optional: Allow re-calibration from settings
8. Optional: Support other reference objects (smartphone, hand span)
9. Create lib/utils/calibration.ts for storage and retrieval

### 10.6. Create Fallback Manual Size Picker for Non-AR Devices

**Status:** done  
**Dependencies:** None  

Build a manual size selection UI for devices without AR/LiDAR support, providing Small/Medium/Large presets with visual references.

**Details:**

1. Create lib/components/ManualSizePicker.tsx component
2. Implement size selector options:
   - Small (e.g., 5cm × 5cm × 5cm → ~87g assuming 0.7 density)
   - Medium (e.g., 10cm × 10cm × 10cm → ~700g)
   - Large (e.g., 15cm × 15cm × 15cm → ~2.3kg)
   - Custom (slider input for each dimension)
3. Add visual reference images for each size:
   - Small: Size of a golf ball
   - Medium: Size of a baseball
   - Large: Size of a grapefruit
4. Implement custom slider:
   - Width slider (1-30cm)
   - Height slider (1-30cm)
   - Depth slider (1-30cm)
   - Real-time volume calculation display
5. Convert selected size to ARMeasurement type:
   - Set confidence: 'low' (manual estimate)
   - Set planeDetected: false
   - Set distance, width, height, depth
6. Integrate into scan-food.tsx as fallback when AR unavailable
7. Show manual picker when device lacks AR support or user declines AR permissions

### 10.7. Add Dimension-to-Weight Conversion Utilities

**Status:** done  
**Dependencies:** None  

Create client-side utilities for volume calculation, food density lookup, and weight estimation to complement the ML service's backend estimation.

**Details:**

1. Create lib/utils/portion-estimation.ts utility file
2. Implement volume calculation:
   - volumeFromDimensions(width, height, depth): cm³
   - applyShapeFactor(volume, shapeFactor): adjusted cm³
3. Create food density lookup table:
   - Common foods with g/cm³ density values
   - Categorized by food type (fruits, vegetables, proteins, grains)
   - Default density for unknown foods
4. Implement weight estimation:
   - estimateWeight(volume, foodType): grams
   - Confidence score based on food type match
   - Apply min/max bounds (1g - 5000g)
5. Add unit conversion helpers:
   - cmToInches(cm), inchesToCm(inches)
   - gramsToOz(grams), ozToGrams(oz)
   - volumeCm3ToMl(cm3), mlToVolumeCm3(ml)
6. Create TypeScript interfaces for density data
7. Export utility functions for use in components

### 10.8. Write Comprehensive Tests for AR Measurement System

**Status:** done  
**Dependencies:** 10.2, 10.3, 10.4, 10.5, 10.6, 10.7  

Create unit, component, and integration tests covering the entire AR measurement feature including edge cases and device compatibility.

**Details:**

1. Unit tests for dimension calculations:
   - Test bounding box calculation from 4 points
   - Test coordinate conversion (screen to world)
   - Test volume and weight calculations
   - Test calibration factor application
2. Component tests for ARPortionMeasure:
   - Test point placement state management
   - Test plane detection callbacks
   - Test measurement completion validation
   - Test reset functionality
3. Component tests for ar-measure-portion modal:
   - Test modal lifecycle (mount, unmount)
   - Test navigation with params
   - Test instruction UI state transitions
4. Integration tests for measurement flow:
   - Test full flow: scan → measure → analyze
   - Test with calibration applied
   - Test fallback to manual picker
   - Test error handling (no plane, invalid points)
5. Mock tests for devices without AR:
   - Mock AR availability check
   - Test manual picker display
   - Test manual measurements passed to API
6. Test calibration accuracy:
   - Test with known reference object dimensions
   - Test calibration persistence
   - Test validation logic
7. Add test fixtures:
   - Mock ARMeasurement data
   - Mock AR session responses
   - Mock plane detection results
8. Create test documentation in README or docs/testing.md
