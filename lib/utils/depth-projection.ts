/**
 * Depth Projection Utilities
 *
 * Converts screen coordinates to real-world measurements using
 * LiDAR depth data and camera intrinsics.
 *
 * The key formula for projecting screen coordinates to 3D:
 *   realX = (screenX - cx) * depth / fx
 *   realY = (screenY - cy) * depth / fy
 *   realZ = depth
 *
 * Where:
 *   cx, cy = principal point (optical center in pixels)
 *   fx, fy = focal lengths in pixels
 *   depth = distance from camera in the same unit as desired output
 */

import { Dimensions } from 'react-native';
import type { CameraIntrinsics, LiDARCapture } from '@/lib/types/ar-data';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

/**
 * 3D point in real-world coordinates
 */
export interface Point3D {
  x: number; // cm
  y: number; // cm
  z: number; // cm (depth from camera)
}

/**
 * Screen point with depth
 */
export interface ScreenPoint {
  x: number; // screen pixels
  y: number; // screen pixels
  depth: number; // cm
}

/**
 * Default camera intrinsics for estimation when LiDAR unavailable
 * Based on typical iPhone camera specs
 */
export const DEFAULT_INTRINSICS: CameraIntrinsics = {
  focalLength: {
    x: 1500, // Typical iPhone focal length in pixels
    y: 1500,
  },
  principalPoint: {
    x: SCREEN_WIDTH / 2,
    y: SCREEN_HEIGHT / 2,
  },
  imageResolution: {
    width: SCREEN_WIDTH,
    height: SCREEN_HEIGHT,
  },
  radialDistortion: [0, 0, 0],
  tangentialDistortion: [0, 0],
};

/**
 * Fallback constants for non-LiDAR estimation
 */
export const FALLBACK_DEPTH_CM = 30; // Default assumed distance
export const FALLBACK_PIXELS_PER_CM_AT_30CM = 15; // Empirical estimate

/**
 * Project a screen point to 3D real-world coordinates using camera intrinsics
 *
 * @param screenX - X coordinate on screen (pixels)
 * @param screenY - Y coordinate on screen (pixels)
 * @param depthCm - Depth in centimeters
 * @param intrinsics - Camera intrinsics (optional, uses defaults if not provided)
 * @returns Point3D in centimeters
 */
export function screenTo3D(
  screenX: number,
  screenY: number,
  depthCm: number,
  intrinsics?: CameraIntrinsics
): Point3D {
  const intr = intrinsics || DEFAULT_INTRINSICS;

  // Scale intrinsics to screen resolution if needed
  const scaleX = SCREEN_WIDTH / intr.imageResolution.width;
  const scaleY = SCREEN_HEIGHT / intr.imageResolution.height;

  const fx = intr.focalLength.x * scaleX;
  const fy = intr.focalLength.y * scaleY;
  const cx = intr.principalPoint.x * scaleX;
  const cy = intr.principalPoint.y * scaleY;

  // Project screen coordinates to 3D using pinhole camera model
  // Convert depth from cm to same unit for calculation
  const x = ((screenX - cx) * depthCm) / fx;
  const y = ((screenY - cy) * depthCm) / fy;
  const z = depthCm;

  return { x, y, z };
}

/**
 * Calculate real-world distance between two screen points
 *
 * @param point1 - First screen point with depth
 * @param point2 - Second screen point with depth
 * @param intrinsics - Camera intrinsics
 * @returns Distance in centimeters
 */
export function measureDistance(
  point1: ScreenPoint,
  point2: ScreenPoint,
  intrinsics?: CameraIntrinsics
): number {
  const p1_3d = screenTo3D(point1.x, point1.y, point1.depth, intrinsics);
  const p2_3d = screenTo3D(point2.x, point2.y, point2.depth, intrinsics);

  // Euclidean distance in 3D space
  const dx = p2_3d.x - p1_3d.x;
  const dy = p2_3d.y - p1_3d.y;
  const dz = p2_3d.z - p1_3d.z;

  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Calculate width and height from two corner points (bounding box)
 * Uses the horizontal and vertical distances at average depth
 *
 * @param topLeft - Top-left corner point
 * @param bottomRight - Bottom-right corner point
 * @param intrinsics - Camera intrinsics
 * @returns { width, height } in centimeters
 */
export function measureBoundingBox(
  topLeft: ScreenPoint,
  bottomRight: ScreenPoint,
  intrinsics?: CameraIntrinsics
): { width: number; height: number } {
  const avgDepth = (topLeft.depth + bottomRight.depth) / 2;

  // Project both points to 3D at average depth for more stable measurement
  const tl3d = screenTo3D(topLeft.x, topLeft.y, avgDepth, intrinsics);
  const br3d = screenTo3D(bottomRight.x, bottomRight.y, avgDepth, intrinsics);

  // Width is the horizontal distance (x difference)
  const width = Math.abs(br3d.x - tl3d.x);

  // Height is the vertical distance (y difference)
  const height = Math.abs(br3d.y - tl3d.y);

  return { width, height };
}

/**
 * Measure food dimensions from three points:
 * - Point 1: Top-left corner of food
 * - Point 2: Bottom-right corner of food
 * - Point 3: Back edge for depth measurement
 *
 * @param point1 - Top-left corner
 * @param point2 - Bottom-right corner
 * @param point3 - Back edge point (optional)
 * @param intrinsics - Camera intrinsics
 * @returns { width, height, depth } in centimeters
 */
export function measureFoodDimensions(
  point1: ScreenPoint,
  point2: ScreenPoint,
  point3?: ScreenPoint,
  intrinsics?: CameraIntrinsics
): { width: number; height: number; depth: number } {
  // Get width and height from bounding box
  const { width, height } = measureBoundingBox(point1, point2, intrinsics);

  // Calculate depth
  let depth: number;

  if (point3) {
    // Method 1: Use third point's depth difference from the front plane
    // The food's depth is the difference between the front surface and back point
    const frontDepth = Math.min(point1.depth, point2.depth);
    const backDepth = point3.depth;

    // If back point is farther from camera, depth is the difference
    if (backDepth > frontDepth) {
      depth = backDepth - frontDepth;
    } else {
      // Use vertical distance from point3 to baseline as proxy for depth
      const avgY = (point1.y + point2.y) / 2;
      const avgDepthFront = (point1.depth + point2.depth) / 2;
      const verticalPixels = Math.abs(point3.y - avgY);

      // Convert vertical pixels to cm at front depth
      const p3_3d = screenTo3D(point3.x, point3.y, avgDepthFront, intrinsics);
      const baseline_3d = screenTo3D(point3.x, avgY, avgDepthFront, intrinsics);
      depth = Math.abs(p3_3d.y - baseline_3d.y);
    }
  } else {
    // Estimate depth as a fraction of the smaller dimension (typical food shapes)
    depth = Math.min(width, height) * 0.6;
  }

  // Ensure minimum depth
  depth = Math.max(depth, 0.5);

  return {
    width: Math.round(width * 10) / 10,
    height: Math.round(height * 10) / 10,
    depth: Math.round(depth * 10) / 10,
  };
}

/**
 * Fallback measurement when LiDAR/intrinsics unavailable
 * Uses simple pixel-to-cm scaling based on assumed distance
 *
 * @param pixelWidth - Width in pixels
 * @param pixelHeight - Height in pixels
 * @param assumedDepthCm - Assumed depth in cm (default 30cm)
 * @returns { width, height, depth } in centimeters
 */
export function fallbackMeasurement(
  pixelWidth: number,
  pixelHeight: number,
  assumedDepthCm: number = FALLBACK_DEPTH_CM
): { width: number; height: number; depth: number } {
  // Scale factor increases linearly with depth
  const scaleFactor = assumedDepthCm / 30;
  const pixelsPerCm = FALLBACK_PIXELS_PER_CM_AT_30CM / scaleFactor;

  const width = pixelWidth / pixelsPerCm;
  const height = pixelHeight / pixelsPerCm;
  const depth = Math.min(width, height) * 0.6; // Estimate depth

  return {
    width: Math.round(width * 10) / 10,
    height: Math.round(height * 10) / 10,
    depth: Math.round(depth * 10) / 10,
  };
}

/**
 * Get depth value at a screen coordinate from LiDAR capture
 *
 * @param capture - LiDAR capture data
 * @param screenX - Screen X coordinate
 * @param screenY - Screen Y coordinate
 * @returns Depth in centimeters, or default if unavailable
 */
export function getDepthFromCapture(
  capture: LiDARCapture | null,
  screenX: number,
  screenY: number
): number {
  if (!capture || !capture.depthBuffer.data || capture.depthBuffer.data.length === 0) {
    return FALLBACK_DEPTH_CM;
  }

  const { depthBuffer } = capture;
  const { width, height, data } = depthBuffer;

  // Map screen coordinates to depth buffer coordinates
  const depthX = Math.floor((screenX / SCREEN_WIDTH) * width);
  const depthY = Math.floor((screenY / SCREEN_HEIGHT) * height);

  // Clamp to valid range
  const clampedX = Math.max(0, Math.min(width - 1, depthX));
  const clampedY = Math.max(0, Math.min(height - 1, depthY));

  // Get depth value (in meters)
  const index = clampedY * width + clampedX;
  const depthMeters = data[index];

  // Convert to centimeters
  const depthCm = depthMeters * 100;

  // Validate depth value (LiDAR range is typically 0.2m - 5m)
  if (depthCm > 0 && depthCm < 500) {
    return depthCm;
  }

  return FALLBACK_DEPTH_CM;
}

/**
 * Get average depth in a region around a point
 * More robust than single-pixel lookup
 *
 * @param capture - LiDAR capture data
 * @param screenX - Screen X coordinate (center)
 * @param screenY - Screen Y coordinate (center)
 * @param radius - Radius in depth buffer pixels (default 3)
 * @returns Average depth in centimeters
 */
export function getAverageDepthInRegion(
  capture: LiDARCapture | null,
  screenX: number,
  screenY: number,
  radius: number = 3
): number {
  if (!capture || !capture.depthBuffer.data || capture.depthBuffer.data.length === 0) {
    return FALLBACK_DEPTH_CM;
  }

  const { depthBuffer, confidenceMap } = capture;
  const { width, height, data } = depthBuffer;

  // Map screen coordinates to depth buffer coordinates
  const centerX = Math.floor((screenX / SCREEN_WIDTH) * width);
  const centerY = Math.floor((screenY / SCREEN_HEIGHT) * height);

  let validDepths: number[] = [];

  // Sample depth values in a circular region
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      // Skip corners (approximate circle)
      if (dx * dx + dy * dy > radius * radius) continue;

      const x = Math.max(0, Math.min(width - 1, centerX + dx));
      const y = Math.max(0, Math.min(height - 1, centerY + dy));
      const index = y * width + x;

      const depthMeters = data[index];
      const depthCm = depthMeters * 100;

      // Check confidence if available
      let confidence = 2; // Default high
      if (confidenceMap && confidenceMap.data && confidenceMap.data.length > index) {
        confidence = confidenceMap.data[index];
      }

      // Only include valid depths with medium or high confidence
      if (depthCm > 20 && depthCm < 500 && confidence >= 1) {
        validDepths.push(depthCm);
      }
    }
  }

  if (validDepths.length === 0) {
    return FALLBACK_DEPTH_CM;
  }

  // Use median for robustness against outliers
  validDepths.sort((a, b) => a - b);
  const medianIndex = Math.floor(validDepths.length / 2);

  if (validDepths.length % 2 === 0) {
    return (validDepths[medianIndex - 1] + validDepths[medianIndex]) / 2;
  }

  return validDepths[medianIndex];
}

/**
 * Validate measurement dimensions
 *
 * @param width - Width in cm
 * @param height - Height in cm
 * @param depth - Depth in cm
 * @returns { valid, reason } - Whether dimensions are reasonable for food
 */
export function validateDimensions(
  width: number,
  height: number,
  depth: number
): { valid: boolean; reason?: string } {
  // Minimum size (about 1 cm³)
  if (width < 0.5 || height < 0.5 || depth < 0.5) {
    return { valid: false, reason: 'Measurement too small' };
  }

  // Maximum size (about 50cm - very large food item)
  if (width > 50 || height > 50 || depth > 50) {
    return { valid: false, reason: 'Measurement too large for food' };
  }

  // Aspect ratio check (food items typically have reasonable proportions)
  const maxDim = Math.max(width, height, depth);
  const minDim = Math.min(width, height, depth);
  if (maxDim / minDim > 20) {
    return { valid: false, reason: 'Unrealistic proportions' };
  }

  return { valid: true };
}

/**
 * Calculate measurement confidence based on depth quality and consistency
 *
 * @param capture - LiDAR capture data
 * @param point1 - First measurement point
 * @param point2 - Second measurement point
 * @param point3 - Third measurement point (optional)
 * @returns Confidence level
 */
export function calculateMeasurementConfidence(
  capture: LiDARCapture | null,
  point1: ScreenPoint,
  point2: ScreenPoint,
  point3?: ScreenPoint
): 'high' | 'medium' | 'low' {
  if (!capture) {
    return 'low';
  }

  // Check depth quality from capture
  if (capture.depthQuality === 'high') {
    // Check depth consistency between points
    const depths = [point1.depth, point2.depth];
    if (point3) depths.push(point3.depth);

    const avgDepth = depths.reduce((a, b) => a + b, 0) / depths.length;
    const maxDeviation = Math.max(...depths.map(d => Math.abs(d - avgDepth)));

    // If depths are consistent (within 10%), high confidence
    if (maxDeviation / avgDepth < 0.1) {
      return 'high';
    }
    return 'medium';
  }

  if (capture.depthQuality === 'medium') {
    return 'medium';
  }

  return 'low';
}
