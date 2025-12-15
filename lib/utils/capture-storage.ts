/**
 * Capture Storage Utility
 * 
 * Manages storage for AR capture sessions, including frame data and exports.
 */

import * as FileSystem from 'expo-file-system/legacy';
import type { RGBDFrame, CaptureSession, ExportConfig, MLExportData } from '@/lib/types/ar-data';

const CAPTURE_DIR = `${FileSystem.documentDirectory}captures/`;

/**
 * Capture storage manager
 */
export const captureStorage = {
  /**
   * Initialize storage directory
   */
  async initialize(): Promise<void> {
    const dirInfo = await FileSystem.getInfoAsync(CAPTURE_DIR);
    if (!dirInfo.exists) {
      await FileSystem.makeDirectoryAsync(CAPTURE_DIR, { intermediates: true });
    }
  },

  /**
   * Create a session directory
   */
  async createSessionDirectory(sessionId: string): Promise<string> {
    const sessionDir = `${CAPTURE_DIR}${sessionId}/`;
    await FileSystem.makeDirectoryAsync(sessionDir, { intermediates: true });
    return sessionDir;
  },

  /**
   * Save a captured frame
   */
  async saveFrame(sessionId: string, frame: RGBDFrame): Promise<void> {
    const sessionDir = `${CAPTURE_DIR}${sessionId}/`;
    const frameFile = `${sessionDir}frame_${frame.frameId}.json`;

    // For now, save frame metadata (actual binary data handling would need native module)
    const frameData = {
      frameId: frame.frameId,
      timestamp: frame.timestamp,
      width: frame.rgbImage.width,
      height: frame.rgbImage.height,
      cameraIntrinsics: frame.depthData.cameraIntrinsics,
      metadata: frame.metadata,
    };

    await FileSystem.writeAsStringAsync(frameFile, JSON.stringify(frameData));
  },

  /**
   * Delete a session and its data
   */
  async deleteSession(sessionId: string): Promise<void> {
    const sessionDir = `${CAPTURE_DIR}${sessionId}/`;
    const dirInfo = await FileSystem.getInfoAsync(sessionDir);
    if (dirInfo.exists) {
      await FileSystem.deleteAsync(sessionDir, { idempotent: true });
    }
  },

  /**
   * Export session data for ML processing
   */
  async exportMLData(session: CaptureSession, config: ExportConfig): Promise<MLExportData> {
    const outputDir = `${CAPTURE_DIR}${session.sessionId}/${config.outputDir}/`;
    await FileSystem.makeDirectoryAsync(outputDir, { intermediates: true });

    // Create metadata file
    const metadata = {
      sessionId: session.sessionId,
      exportTime: Date.now(),
      format: config.format,
      frameCount: session.frames.length,
      config,
    };

    await FileSystem.writeAsStringAsync(
      `${outputDir}metadata.json`,
      JSON.stringify(metadata, null, 2)
    );

    return {
      sessionId: session.sessionId,
      format: config.format,
      frameCount: session.frames.length,
      outputPath: outputDir,
      metadata,
    };
  },

  /**
   * Get all sessions
   */
  async getSessions(): Promise<string[]> {
    try {
      const dirInfo = await FileSystem.getInfoAsync(CAPTURE_DIR);
      if (!dirInfo.exists) {
        return [];
      }
      const contents = await FileSystem.readDirectoryAsync(CAPTURE_DIR);
      return contents;
    } catch {
      return [];
    }
  },
};
