import { useEffect, useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  Platform,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useOnboarding } from '@/lib/context/OnboardingContext';
import { healthKitService } from '@/lib/services/healthkit';
import { OnboardingStep3Data } from '@/lib/onboarding/types';
import { colors, gradients, typography, spacing, borderRadius } from '@/lib/theme/colors';

type SyncCategory = 'cardiovascular' | 'respiratory' | 'sleep' | 'activity';

interface CategoryStatus {
  status: 'pending' | 'syncing' | 'complete' | 'error';
  count: number;
}

export default function OnboardingSync() {
  const router = useRouter();
  const { getDraftForStep, savedData } = useOnboarding();

  // Animation values
  const [fadeAnim] = useState(new Animated.Value(0));
  const [pulseAnim] = useState(new Animated.Value(1));
  const pulseRef = useRef<Animated.CompositeAnimation | null>(null);

  // Sync state
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncComplete, setSyncComplete] = useState(false);
  const [currentMessage, setCurrentMessage] = useState('Preparing to sync...');
  const [progress, setProgress] = useState(0);
  const [totalMetrics, setTotalMetrics] = useState(0);
  const [hasError, setHasError] = useState(false);
  const [categoryStatus, setCategoryStatus] = useState<Record<SyncCategory, CategoryStatus>>({
    cardiovascular: { status: 'pending', count: 0 },
    respiratory: { status: 'pending', count: 0 },
    sleep: { status: 'pending', count: 0 },
    activity: { status: 'pending', count: 0 },
  });

  // Check if HealthKit was enabled
  const healthKitEnabled = (() => {
    // Try draft first, then saved data
    const draft = getDraftForStep<OnboardingStep3Data>(3);
    if (draft?.healthKitEnabled !== undefined) {
      return draft.healthKitEnabled;
    }
    return savedData?.permissions?.healthKitEnabled ?? false;
  })();

  useEffect(() => {
    // Entrance animation
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 500,
      useNativeDriver: true,
    }).start();

    // Start pulsing animation
    pulseRef.current = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.1,
          duration: 800,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 800,
          useNativeDriver: true,
        }),
      ])
    );
    pulseRef.current.start();

    return () => {
      pulseRef.current?.stop();
    };
  }, [fadeAnim, pulseAnim]);

  useEffect(() => {
    // Skip sync if HealthKit not enabled or not iOS
    if (!healthKitEnabled || Platform.OS !== 'ios') {
      router.replace('/onboarding/complete');
      return;
    }

    startSync();
  }, [healthKitEnabled]);

  const startSync = async () => {
    setIsSyncing(true);
    setCurrentMessage('Connecting to Apple Health...');

    try {
      // Check if HealthKit is available
      const isAvailable = await healthKitService.isAvailable();
      if (!isAvailable) {
        setCurrentMessage('Apple Health not available');
        setHasError(true);
        setTimeout(() => router.replace('/onboarding/complete'), 2000);
        return;
      }

      // Check if initialized
      const isInitialized = await healthKitService.isInitialized();
      if (!isInitialized) {
        setCurrentMessage('Requesting permissions...');
        const result = await healthKitService.requestPermissions();
        if (!result.success) {
          setCurrentMessage('Could not get permissions');
          setHasError(true);
          setTimeout(() => router.replace('/onboarding/complete'), 2000);
          return;
        }
      }

      // Sync all health data with progress callback
      setCurrentMessage('Syncing health data...');
      const result = await healthKitService.syncAll((message: string, progressValue: number) => {
        setCurrentMessage(message);
        setProgress(progressValue);

        // Update category status based on progress
        if (progressValue >= 10 && progressValue < 25) {
          setCategoryStatus(prev => ({
            ...prev,
            cardiovascular: { status: 'syncing', count: 0 },
          }));
        } else if (progressValue >= 25 && progressValue < 50) {
          setCategoryStatus(prev => ({
            ...prev,
            cardiovascular: { status: 'complete', count: prev.cardiovascular.count },
            respiratory: { status: 'syncing', count: 0 },
          }));
        } else if (progressValue >= 50 && progressValue < 75) {
          setCategoryStatus(prev => ({
            ...prev,
            respiratory: { status: 'complete', count: prev.respiratory.count },
            sleep: { status: 'syncing', count: 0 },
          }));
        } else if (progressValue >= 75 && progressValue < 100) {
          setCategoryStatus(prev => ({
            ...prev,
            sleep: { status: 'complete', count: prev.sleep.count },
            activity: { status: 'syncing', count: 0 },
          }));
        }
      });

      // Update final status
      setCategoryStatus({
        cardiovascular: {
          status: result.results.cardiovascular.success ? 'complete' : 'error',
          count: result.results.cardiovascular.metricsCount,
        },
        respiratory: {
          status: result.results.respiratory.success ? 'complete' : 'error',
          count: result.results.respiratory.metricsCount,
        },
        sleep: {
          status: result.results.sleep.success ? 'complete' : 'error',
          count: result.results.sleep.metricsCount,
        },
        activity: {
          status: result.results.activity.success ? 'complete' : 'error',
          count: result.results.activity.metricsCount,
        },
      });

      setTotalMetrics(result.totalMetrics);
      setProgress(100);
      setSyncComplete(true);
      setCurrentMessage(
        result.totalMetrics > 0
          ? `Synced ${result.totalMetrics} health records`
          : 'No new health data to sync'
      );

      // Stop pulsing animation
      pulseRef.current?.stop();

      // Navigate to complete after brief delay
      setTimeout(() => {
        router.replace('/onboarding/complete');
      }, 2000);
    } catch (error) {
      console.error('Sync error:', error);
      setHasError(true);
      setCurrentMessage('Sync failed - continuing setup');
      setTimeout(() => router.replace('/onboarding/complete'), 2000);
    } finally {
      setIsSyncing(false);
    }
  };

  const handleSkip = () => {
    pulseRef.current?.stop();
    router.replace('/onboarding/complete');
  };

  const getCategoryIcon = (category: SyncCategory): keyof typeof Ionicons.glyphMap => {
    switch (category) {
      case 'cardiovascular':
        return 'heart-outline';
      case 'respiratory':
        return 'fitness-outline';
      case 'sleep':
        return 'moon-outline';
      case 'activity':
        return 'walk-outline';
    }
  };

  const getCategoryLabel = (category: SyncCategory): string => {
    switch (category) {
      case 'cardiovascular':
        return 'Heart Rate & HRV';
      case 'respiratory':
        return 'Respiratory';
      case 'sleep':
        return 'Sleep';
      case 'activity':
        return 'Activity';
    }
  };

  const getStatusIcon = (status: CategoryStatus['status']): keyof typeof Ionicons.glyphMap => {
    switch (status) {
      case 'complete':
        return 'checkmark-circle';
      case 'syncing':
        return 'sync';
      case 'error':
        return 'alert-circle';
      default:
        return 'ellipse-outline';
    }
  };

  const getStatusColor = (status: CategoryStatus['status']): string => {
    switch (status) {
      case 'complete':
        return colors.semantic.success;
      case 'syncing':
        return colors.primary.main;
      case 'error':
        return colors.semantic.error;
      default:
        return colors.text.tertiary;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <LinearGradient colors={gradients.dark} style={styles.gradient}>
        <Animated.View style={[styles.content, { opacity: fadeAnim }]}>
          {/* Header Icon */}
          <Animated.View
            style={[
              styles.iconContainer,
              { transform: [{ scale: syncComplete ? 1 : pulseAnim }] },
            ]}
          >
            <LinearGradient
              colors={syncComplete ? [colors.semantic.success, colors.semantic.success] : gradients.primary}
              style={styles.iconGradient}
            >
              <Ionicons
                name={syncComplete ? 'checkmark-circle' : hasError ? 'alert-circle' : 'heart'}
                size={48}
                color={colors.text.primary}
              />
            </LinearGradient>
          </Animated.View>

          {/* Title */}
          <Text style={styles.title}>
            {syncComplete ? 'Sync Complete!' : 'Syncing Health Data'}
          </Text>
          <Text style={styles.subtitle}>{currentMessage}</Text>

          {/* Progress Bar */}
          <View style={styles.progressContainer}>
            <View style={styles.progressBar}>
              <Animated.View
                style={[
                  styles.progressFill,
                  { width: `${progress}%` },
                ]}
              />
            </View>
            <Text style={styles.progressText}>{Math.round(progress)}%</Text>
          </View>

          {/* Category Status */}
          <View style={styles.categoriesCard}>
            {(Object.keys(categoryStatus) as SyncCategory[]).map((category) => {
              const status = categoryStatus[category];
              return (
                <View key={category} style={styles.categoryRow}>
                  <View style={styles.categoryLeft}>
                    <Ionicons
                      name={getCategoryIcon(category)}
                      size={24}
                      color={getStatusColor(status.status)}
                    />
                    <Text style={styles.categoryLabel}>{getCategoryLabel(category)}</Text>
                  </View>
                  <View style={styles.categoryRight}>
                    {status.status === 'complete' && status.count > 0 && (
                      <Text style={styles.categoryCount}>{status.count}</Text>
                    )}
                    <Ionicons
                      name={getStatusIcon(status.status)}
                      size={20}
                      color={getStatusColor(status.status)}
                    />
                  </View>
                </View>
              );
            })}
          </View>

          {/* Total synced */}
          {syncComplete && totalMetrics > 0 && (
            <View style={styles.totalCard}>
              <Ionicons name="analytics-outline" size={24} color={colors.primary.main} />
              <Text style={styles.totalText}>
                {totalMetrics} health records synced
              </Text>
            </View>
          )}
        </Animated.View>

        {/* Skip Button (only show while syncing) */}
        {!syncComplete && (
          <Animated.View style={[styles.footer, { opacity: fadeAnim }]}>
            <TouchableOpacity style={styles.skipButton} onPress={handleSkip}>
              <Text style={styles.skipButtonText}>Skip for now</Text>
            </TouchableOpacity>
            <Text style={styles.footerHint}>
              You can sync your health data anytime from Settings
            </Text>
          </Animated.View>
        )}
      </LinearGradient>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  gradient: {
    flex: 1,
  },
  content: {
    flex: 1,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing['3xl'],
    alignItems: 'center',
  },
  iconContainer: {
    marginBottom: spacing.xl,
  },
  iconGradient: {
    width: 100,
    height: 100,
    borderRadius: 50,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    ...typography.h2,
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: spacing.sm,
  },
  subtitle: {
    ...typography.body,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  progressContainer: {
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.xl,
    gap: spacing.md,
  },
  progressBar: {
    flex: 1,
    height: 8,
    backgroundColor: colors.surface.card,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: colors.primary.main,
    borderRadius: 4,
  },
  progressText: {
    ...typography.bodySmall,
    color: colors.text.secondary,
    width: 40,
    textAlign: 'right',
  },
  categoriesCard: {
    width: '100%',
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
  },
  categoryRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing.sm,
  },
  categoryLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  categoryLabel: {
    ...typography.body,
    color: colors.text.primary,
  },
  categoryRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  categoryCount: {
    ...typography.bodySmall,
    color: colors.text.secondary,
  },
  totalCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    backgroundColor: colors.surface.elevated,
    borderRadius: borderRadius.md,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
  },
  totalText: {
    ...typography.bodyBold,
    color: colors.text.primary,
  },
  footer: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xl,
    alignItems: 'center',
  },
  skipButton: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
  },
  skipButtonText: {
    ...typography.body,
    color: colors.text.secondary,
    textDecorationLine: 'underline',
  },
  footerHint: {
    ...typography.bodySmall,
    color: colors.text.tertiary,
    textAlign: 'center',
    marginTop: spacing.xs,
  },
});
