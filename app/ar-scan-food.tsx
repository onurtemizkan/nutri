/**
 * AR Food Scanning Screen
 *
 * Allows users to scan food items using AR/LiDAR to capture RGB-D data
 * for automatic nutrition estimation via ML models.
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Alert,
  TouchableOpacity,
  ScrollView,
  TextInput,
} from 'react-native';
import { router } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import ARFoodScanner from '@/components/ARFoodScanner';
import { captureStorage } from '@/lib/utils/capture-storage';
import type {
  RGBDFrame,
  CaptureSession,
  ExportConfig,
} from '@/lib/types/ar-data';

type ScanMode = 'photo' | 'video';

export default function ARScanFoodScreen() {
  const [scanMode, setScanMode] = useState<ScanMode>('photo');
  const [foodName, setFoodName] = useState('');
  const [notes, setNotes] = useState('');
  const [isScanning, setIsScanning] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [capturedFrames, setCapturedFrames] = useState<RGBDFrame[]>([]);
  const [isExporting, setIsExporting] = useState(false);

  const handleSessionStart = useCallback(
    async (sessionId: string, session: CaptureSession) => {
      setCurrentSessionId(sessionId);
      setIsScanning(true);
      setCapturedFrames([]);

      try {
        // Initialize storage and create session directory
        await captureStorage.initialize();
        await captureStorage.createSessionDirectory(sessionId);
      } catch (error) {
        console.error('Failed to initialize session storage:', error);
        Alert.alert('Error', 'Failed to initialize storage');
      }
    },
    []
  );

  const handleSessionEnd = useCallback(
    async (session: CaptureSession) => {
      setIsScanning(false);

      // Ask user if they want to export the data
      Alert.alert(
        'Scan Complete',
        `Captured ${capturedFrames.length} frame(s). Export data for ML processing?`,
        [
          {
            text: 'Cancel',
            style: 'cancel',
            onPress: () => {
              // Optionally delete session
              if (currentSessionId) {
                captureStorage.deleteSession(currentSessionId);
              }
            },
          },
          {
            text: 'Export',
            onPress: () => handleExportData(session),
          },
        ]
      );
    },
    [capturedFrames.length, currentSessionId]
  );

  const handleFrameCapture = useCallback(
    async (frame: RGBDFrame) => {
      try {
        // Save frame to storage
        if (currentSessionId) {
          await captureStorage.saveFrame(currentSessionId, frame);
          setCapturedFrames(prev => [...prev, frame]);
        }
      } catch (error) {
        console.error('Failed to save frame:', error);
        Alert.alert('Error', 'Failed to save frame data');
      }
    },
    [currentSessionId]
  );

  const handleExportData = async (session: CaptureSession) => {
    if (!currentSessionId) {
      return;
    }

    setIsExporting(true);

    try {
      // Configure ML export
      const exportConfig: ExportConfig = {
        outputDir: 'ml-data',
        format: 'rgbd_normalized',
        includeConfidenceMaps: true,
        normalizeDepth: true,
        normalizationMethod: 'min_max',
        compressRGB: true,
        rgbFormat: 'jpeg',
        rgbQuality: 90,
      };

      // Export ML-ready data
      const mlData = await captureStorage.exportMLData(session, exportConfig);

      Alert.alert(
        'Export Complete',
        `Data exported for ${capturedFrames.length} frame(s)\n\n` +
        `Session ID: ${session.sessionId}\n` +
        `Format: ${mlData.format}\n` +
        `Ready for ML processing!`,
        [
          {
            text: 'Done',
            onPress: () => {
              // Reset state
              setCapturedFrames([]);
              setCurrentSessionId(null);
              setFoodName('');
              setNotes('');
            },
          },
        ]
      );
    } catch (error) {
      console.error('Failed to export data:', error);
      Alert.alert('Error', 'Failed to export ML data');
    } finally {
      setIsExporting(false);
    }
  };

  const handleError = useCallback((error: Error) => {
    console.error('Scanner error:', error);
    Alert.alert('Scanner Error', error.message);
  }, []);

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#007AFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>AR Food Scanner</Text>
        <View style={styles.placeholder} />
      </View>

      {/* Scanner Mode Toggle */}
      {!isScanning && (
        <View style={styles.configSection}>
          <ScrollView style={styles.configScroll}>
            <Text style={styles.sectionTitle}>Food Information</Text>

            <TextInput
              style={styles.input}
              placeholder="Food name (optional)"
              value={foodName}
              onChangeText={setFoodName}
              placeholderTextColor="#999"
            />

            <TextInput
              style={[styles.input, styles.textArea]}
              placeholder="Notes (optional)"
              value={notes}
              onChangeText={setNotes}
              multiline
              numberOfLines={3}
              placeholderTextColor="#999"
            />

            <Text style={styles.sectionTitle}>Scan Mode</Text>

            <View style={styles.modeSelector}>
              <TouchableOpacity
                style={[
                  styles.modeButton,
                  scanMode === 'photo' && styles.modeButtonActive,
                ]}
                onPress={() => setScanMode('photo')}
              >
                <Ionicons
                  name="camera"
                  size={24}
                  color={scanMode === 'photo' ? '#fff' : '#007AFF'}
                />
                <Text
                  style={[
                    styles.modeText,
                    scanMode === 'photo' && styles.modeTextActive,
                  ]}
                >
                  Single Photo
                </Text>
                <Text style={styles.modeSubtext}>Capture one frame</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[
                  styles.modeButton,
                  scanMode === 'video' && styles.modeButtonActive,
                ]}
                onPress={() => setScanMode('video')}
              >
                <Ionicons
                  name="videocam"
                  size={24}
                  color={scanMode === 'video' ? '#fff' : '#007AFF'}
                />
                <Text
                  style={[
                    styles.modeText,
                    scanMode === 'video' && styles.modeTextActive,
                  ]}
                >
                  Video
                </Text>
                <Text style={styles.modeSubtext}>Multiple angles</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.infoBox}>
              <Ionicons name="information-circle" size={20} color="#007AFF" />
              <Text style={styles.infoText}>
                {scanMode === 'photo'
                  ? 'Capture a single high-quality RGB-D image of your food using LiDAR.'
                  : 'Record a short video rotating around your food for better 3D reconstruction.'}
              </Text>
            </View>
          </ScrollView>
        </View>
      )}

      {/* AR Scanner */}
      <View style={styles.scannerContainer}>
        <ARFoodScanner
          mode={scanMode}
          metadata={{
            foodItem: foodName || undefined,
            notes: notes || undefined,
          }}
          onSessionStart={handleSessionStart}
          onSessionEnd={handleSessionEnd}
          onCapture={handleFrameCapture}
          onError={handleError}
        />
      </View>

      {/* Status Bar */}
      {isScanning && (
        <View style={styles.statusBar}>
          <Text style={styles.statusText}>
            {scanMode === 'photo' ? 'Position food in frame' : 'Rotate around food'}
          </Text>
          <Text style={styles.frameCount}>{capturedFrames.length} frames</Text>
        </View>
      )}

      {/* Export Overlay */}
      {isExporting && (
        <View style={styles.exportOverlay}>
          <View style={styles.exportCard}>
            <Text style={styles.exportTitle}>Exporting Data</Text>
            <Text style={styles.exportSubtitle}>
              Preparing ML-ready RGB-D data...
            </Text>
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 60,
    paddingBottom: 16,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
  },
  placeholder: {
    width: 40,
  },
  configSection: {
    backgroundColor: '#fff',
    maxHeight: 300,
  },
  configScroll: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
    marginBottom: 12,
    marginTop: 8,
  },
  input: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    marginBottom: 12,
    color: '#000',
  },
  textArea: {
    height: 80,
    textAlignVertical: 'top',
  },
  modeSelector: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  modeButton: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  modeButtonActive: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  modeText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#007AFF',
    marginTop: 8,
  },
  modeTextActive: {
    color: '#fff',
  },
  modeSubtext: {
    fontSize: 12,
    color: '#999',
    marginTop: 4,
  },
  infoBox: {
    flexDirection: 'row',
    backgroundColor: '#E8F4FD',
    borderRadius: 8,
    padding: 12,
    gap: 8,
    marginBottom: 16,
  },
  infoText: {
    flex: 1,
    fontSize: 13,
    color: '#007AFF',
    lineHeight: 18,
  },
  scannerContainer: {
    flex: 1,
  },
  statusBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    padding: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statusText: {
    color: '#fff',
    fontSize: 14,
  },
  frameCount: {
    color: '#34C759',
    fontSize: 14,
    fontWeight: '600',
  },
  exportOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  exportCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    minWidth: 200,
  },
  exportTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
    marginBottom: 8,
  },
  exportSubtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
});
