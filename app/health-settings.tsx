/**
 * Health Settings Screen
 * Manage Apple Health integration and sync settings
 */

import { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import { showAlert } from '@/lib/utils/alert';
import {
  healthKitService,
  SyncStatus,
} from '@/lib/services/healthkit';

/**
 * Format date for display
 */
function formatDate(date: Date | undefined): string {
  if (!date) return 'Never';
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;

  return date.toLocaleDateString();
}

/**
 * Sync category display component
 */
function SyncCategory({
  title,
  icon,
  lastSync,
  isLoading,
}: {
  title: string;
  icon: keyof typeof Ionicons.glyphMap;
  lastSync?: Date;
  isLoading: boolean;
}) {
  return (
    <View style={styles.syncCategory}>
      <View style={styles.syncCategoryLeft}>
        <View style={styles.syncCategoryIcon}>
          <Ionicons name={icon} size={20} color={colors.primary.main} />
        </View>
        <View>
          <Text style={styles.syncCategoryTitle}>{title}</Text>
          <Text style={styles.syncCategoryTime}>
            {isLoading ? 'Syncing...' : `Last sync: ${formatDate(lastSync)}`}
          </Text>
        </View>
      </View>
      {isLoading && (
        <ActivityIndicator size="small" color={colors.primary.main} />
      )}
    </View>
  );
}

export default function HealthSettingsScreen() {
  const router = useRouter();
  const { isTablet, getSpacing } = useResponsive();
  const responsiveSpacing = getSpacing();
  const [isLoading, setIsLoading] = useState(true);
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncProgress, setSyncProgress] = useState(0);
  const [syncMessage, setSyncMessage] = useState('');
  const [status, setStatus] = useState<SyncStatus | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Load initial status
  const loadStatus = useCallback(async () => {
    setIsLoading(true);
    try {
      const healthStatus = await healthKitService.getStatus();
      const syncStatus = await healthKitService.getSyncStatus();
      setStatus(syncStatus);
      setIsConnected(healthStatus.authorized);
    } catch (error) {
      console.warn('Error loading health status:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  // Handle connect to Apple Health
  const handleConnect = async () => {
    if (Platform.OS !== 'ios') {
      showAlert('Not Available', 'Apple Health is only available on iOS devices.');
      return;
    }

    try {
      setIsLoading(true);
      const result = await healthKitService.requestPermissions();
      if (result.success) {
        setIsConnected(true);
        showAlert('Success', 'Connected to Apple Health! Tap "Sync Now" to import your health data.');
        await loadStatus();
      } else {
        showAlert('Permission Required', result.error || 'Please grant access to Apple Health in Settings.');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to connect';
      showAlert('Error', message);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle disconnect from Apple Health
  const handleDisconnect = async () => {
    showAlert(
      'Disconnect Apple Health',
      'This will stop syncing health data. Your existing data will remain.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Disconnect',
          style: 'destructive',
          onPress: async () => {
            try {
              await healthKitService.disconnect();
              setIsConnected(false);
              setStatus(null);
              showAlert('Disconnected', 'Apple Health has been disconnected.');
            } catch (error) {
              showAlert('Error', 'Failed to disconnect');
            }
          },
        },
      ]
    );
  };

  // Handle sync
  const handleSync = async () => {
    if (isSyncing) return;

    setIsSyncing(true);
    setSyncProgress(0);
    setSyncMessage('Starting sync...');

    try {
      const result = await healthKitService.syncAll((message, progress) => {
        setSyncMessage(message);
        setSyncProgress(progress);
      });

      if (result.success) {
        showAlert(
          'Sync Complete',
          `Successfully synced ${result.totalMetrics} health metrics.`
        );
      } else if (result.errors.length > 0) {
        showAlert(
          'Sync Completed with Errors',
          `Synced ${result.totalMetrics} metrics. ${result.errors.length} errors occurred.`
        );
      }

      await loadStatus();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Sync failed';
      showAlert('Sync Error', message);
    } finally {
      setIsSyncing(false);
      setSyncProgress(0);
      setSyncMessage('');
    }
  };

  // Handle full resync
  const handleFullResync = async () => {
    showAlert(
      'Full Resync',
      'This will re-sync all health data from the last 30 days. This may take a while.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Resync',
          onPress: async () => {
            setIsSyncing(true);
            setSyncProgress(0);
            setSyncMessage('Starting full resync...');

            try {
              const result = await healthKitService.forceFullSync(30, (message, progress) => {
                setSyncMessage(message);
                setSyncProgress(progress);
              });

              if (result.success) {
                showAlert(
                  'Resync Complete',
                  `Successfully synced ${result.totalMetrics} health metrics.`
                );
              }

              await loadStatus();
            } catch (error) {
              const message = error instanceof Error ? error.message : 'Resync failed';
              showAlert('Resync Error', message);
            } finally {
              setIsSyncing(false);
              setSyncProgress(0);
              setSyncMessage('');
            }
          },
        },
      ]
    );
  };

  const isIOS = Platform.OS === 'ios';

  return (
    <SafeAreaView style={styles.container} testID="health-settings-screen">
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => router.back()}
          accessibilityLabel="Go back"
          testID="health-settings-back-button"
        >
          <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Health Integration</Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={[
          styles.scrollContent,
          { paddingHorizontal: responsiveSpacing.horizontal },
          isTablet && styles.scrollContentTablet
        ]}
      >
        <View style={styles.content}>
          {/* Apple Health Card */}
          <View style={styles.section}>
            <View style={styles.appleHealthHeader}>
              <View style={styles.appleHealthIcon}>
                <Ionicons name="heart" size={28} color="#FF2D55" />
              </View>
              <View style={styles.appleHealthInfo}>
                <Text style={styles.appleHealthTitle}>Apple Health</Text>
                <Text style={styles.appleHealthSubtitle}>
                  {isConnected ? 'Connected' : 'Not connected'}
                </Text>
              </View>
              {isLoading ? (
                <ActivityIndicator color={colors.primary.main} />
              ) : isConnected ? (
                <View style={styles.connectedBadge}>
                  <Ionicons name="checkmark-circle" size={20} color={colors.status.success} />
                </View>
              ) : null}
            </View>

            {!isIOS && (
              <View style={styles.warningBox}>
                <Ionicons name="information-circle" size={20} color={colors.status.warning} />
                <Text style={styles.warningText}>
                  Apple Health is only available on iOS devices.
                </Text>
              </View>
            )}

            {isIOS && !isConnected && (
              <TouchableOpacity
                style={styles.connectButton}
                onPress={handleConnect}
                disabled={isLoading}
                accessibilityLabel="Connect to Apple Health"
                testID="health-settings-connect-button"
              >
                <Ionicons name="add-circle" size={20} color={colors.text.primary} />
                <Text style={styles.connectButtonText}>Connect Apple Health</Text>
              </TouchableOpacity>
            )}

            {isIOS && isConnected && (
              <>
                {/* Sync Status */}
                <View style={styles.syncStatusSection}>
                  <Text style={styles.syncStatusTitle}>Sync Status</Text>

                  <SyncCategory
                    title="Cardiovascular"
                    icon="heart-outline"
                    lastSync={status?.lastSync?.cardiovascular}
                    isLoading={isSyncing}
                  />
                  <SyncCategory
                    title="Respiratory"
                    icon="fitness-outline"
                    lastSync={status?.lastSync?.respiratory}
                    isLoading={isSyncing}
                  />
                  <SyncCategory
                    title="Sleep"
                    icon="moon-outline"
                    lastSync={status?.lastSync?.sleep}
                    isLoading={isSyncing}
                  />
                  <SyncCategory
                    title="Activity"
                    icon="walk-outline"
                    lastSync={status?.lastSync?.activity}
                    isLoading={isSyncing}
                  />
                </View>

                {/* Sync Progress */}
                {isSyncing && (
                  <View style={styles.syncProgressSection}>
                    <View style={styles.progressBar}>
                      <View
                        style={[styles.progressFill, { width: `${syncProgress}%` }]}
                      />
                    </View>
                    <Text style={styles.syncProgressText}>{syncMessage}</Text>
                  </View>
                )}

                {/* Sync Buttons */}
                <View style={styles.buttonGroup}>
                  <TouchableOpacity
                    style={[styles.syncButton, isSyncing && styles.syncButtonDisabled]}
                    onPress={handleSync}
                    disabled={isSyncing}
                    accessibilityLabel="Sync health data"
                    testID="health-settings-sync-button"
                  >
                    {isSyncing ? (
                      <ActivityIndicator color={colors.text.primary} />
                    ) : (
                      <>
                        <Ionicons name="sync" size={20} color={colors.text.primary} />
                        <Text style={styles.syncButtonText}>Sync Now</Text>
                      </>
                    )}
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={[styles.resyncButton, isSyncing && styles.syncButtonDisabled]}
                    onPress={handleFullResync}
                    disabled={isSyncing}
                    accessibilityLabel="Full resync"
                    testID="health-settings-resync-button"
                  >
                    <Text style={styles.resyncButtonText}>Full Resync</Text>
                  </TouchableOpacity>
                </View>

                {/* Disconnect */}
                <TouchableOpacity
                  style={styles.disconnectButton}
                  onPress={handleDisconnect}
                  accessibilityLabel="Disconnect Apple Health"
                  testID="health-settings-disconnect-button"
                >
                  <Text style={styles.disconnectButtonText}>Disconnect</Text>
                </TouchableOpacity>
              </>
            )}
          </View>

          {/* Info Section */}
          <View style={styles.infoSection}>
            <Text style={styles.infoTitle}>What data is synced?</Text>
            <View style={styles.infoList}>
              <View style={styles.infoItem}>
                <Ionicons name="heart" size={16} color={colors.text.tertiary} />
                <Text style={styles.infoText}>
                  Resting Heart Rate & Heart Rate Variability
                </Text>
              </View>
              <View style={styles.infoItem}>
                <Ionicons name="fitness" size={16} color={colors.text.tertiary} />
                <Text style={styles.infoText}>
                  Respiratory Rate, SpO2 & VO2Max
                </Text>
              </View>
              <View style={styles.infoItem}>
                <Ionicons name="moon" size={16} color={colors.text.tertiary} />
                <Text style={styles.infoText}>
                  Sleep Duration, Deep Sleep & REM
                </Text>
              </View>
              <View style={styles.infoItem}>
                <Ionicons name="walk" size={16} color={colors.text.tertiary} />
                <Text style={styles.infoText}>
                  Steps & Active Calories
                </Text>
              </View>
            </View>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  backButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  headerRight: {
    width: 40,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: spacing.lg,
  },
  scrollContentTablet: {
    maxWidth: FORM_MAX_WIDTH,
    alignSelf: 'center',
    width: '100%',
  },
  content: {
    paddingVertical: spacing.lg,
  },

  // Section
  section: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },

  // Apple Health Header
  appleHealthHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  appleHealthIcon: {
    width: 48,
    height: 48,
    borderRadius: borderRadius.md,
    backgroundColor: colors.special.appleHealthLight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  appleHealthInfo: {
    flex: 1,
  },
  appleHealthTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: 2,
  },
  appleHealthSubtitle: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  connectedBadge: {
    width: 32,
    height: 32,
    justifyContent: 'center',
    alignItems: 'center',
  },

  // Warning Box
  warningBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.special.warningLight,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginTop: spacing.sm,
    gap: spacing.sm,
  },
  warningText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.status.warning,
  },

  // Connect Button
  connectButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary.main,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    marginTop: spacing.md,
    gap: spacing.sm,
    ...shadows.sm,
  },
  connectButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Sync Status
  syncStatusSection: {
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  syncStatusTitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    marginBottom: spacing.md,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  syncCategory: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
  },
  syncCategoryLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  syncCategoryIcon: {
    width: 32,
    height: 32,
    borderRadius: borderRadius.sm,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.sm,
  },
  syncCategoryTitle: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
  },
  syncCategoryTime: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },

  // Sync Progress
  syncProgressSection: {
    marginTop: spacing.md,
  },
  progressBar: {
    height: 4,
    backgroundColor: colors.background.elevated,
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: colors.primary.main,
  },
  syncProgressText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
    textAlign: 'center',
  },

  // Buttons
  buttonGroup: {
    flexDirection: 'row',
    gap: spacing.sm,
    marginTop: spacing.md,
  },
  syncButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary.main,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    gap: spacing.sm,
    ...shadows.sm,
  },
  syncButtonDisabled: {
    opacity: 0.5,
  },
  syncButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  resyncButton: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'transparent',
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  resyncButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
  },
  disconnectButton: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
    marginTop: spacing.lg,
  },
  disconnectButtonText: {
    fontSize: typography.fontSize.sm,
    color: colors.status.error,
  },

  // Info Section
  infoSection: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  infoTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  infoList: {
    gap: spacing.sm,
  },
  infoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  infoText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
});
