/**
 * Activity Detail Screen
 *
 * Displays detailed information about a specific activity
 * with edit and delete functionality.
 */

import { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { activitiesApi } from '@/lib/api/activities';
import {
  Activity,
  ACTIVITY_TYPE_CONFIG,
  INTENSITY_CONFIG,
  SOURCE_CONFIG,
  formatDuration,
  formatDistance,
} from '@/lib/types/activities';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';

export default function ActivityDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [activity, setActivity] = useState<Activity | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const { getResponsiveValue } = useResponsive();

  const contentPadding = getResponsiveValue({
    small: spacing.md,
    medium: spacing.lg,
    large: spacing.lg,
    tablet: spacing.xl,
    default: spacing.lg,
  });

  const loadActivity = useCallback(async () => {
    if (!id) {
      setError('Activity not found');
      setIsLoading(false);
      return;
    }

    try {
      setError(null);
      const data = await activitiesApi.getById(id);
      setActivity(data);
    } catch (err) {
      console.error('Failed to load activity:', err);
      setError('Failed to load activity');
    } finally {
      setIsLoading(false);
    }
  }, [id]);

  useFocusEffect(
    useCallback(() => {
      loadActivity();
    }, [loadActivity])
  );

  const handleDelete = () => {
    Alert.alert(
      'Delete Activity',
      'Are you sure you want to delete this activity? This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            if (!id) return;
            setIsDeleting(true);
            try {
              await activitiesApi.delete(id);
              router.back();
            } catch (err) {
              console.error('Failed to delete activity:', err);
              Alert.alert('Error', 'Failed to delete activity. Please try again.');
            } finally {
              setIsDeleting(false);
            }
          },
        },
      ]
    );
  };

  const handleEdit = () => {
    // Navigate to edit screen (using add screen with pre-filled data)
    router.push({
      pathname: '/activity/add',
      params: { editId: id },
    });
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
          <Text style={styles.loadingText}>Loading activity...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error || !activity) {
    return (
      <SafeAreaView style={styles.container} edges={['top']}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
          </TouchableOpacity>
        </View>
        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle-outline" size={64} color={colors.status.error} />
          <Text style={styles.errorText}>{error || 'Activity not found'}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={loadActivity}>
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  const typeConfig = ACTIVITY_TYPE_CONFIG[activity.activityType];
  const intensityConfig = INTENSITY_CONFIG[activity.intensity];
  const sourceConfig = SOURCE_CONFIG[activity.source];

  const startDate = new Date(activity.startedAt);
  const endDate = new Date(activity.endedAt);

  return (
    <SafeAreaView style={styles.container} edges={['top']} testID="activity-detail-screen">
      {/* Header */}
      <View style={[styles.header, { paddingHorizontal: contentPadding }]}>
        <TouchableOpacity
          onPress={() => router.back()}
          style={styles.backButton}
          accessibilityLabel="Go back"
          testID="activity-detail-back-button"
        >
          <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Activity Details</Text>
        <View style={styles.headerActions}>
          <TouchableOpacity
            onPress={handleEdit}
            style={styles.headerButton}
            accessibilityLabel="Edit activity"
            testID="activity-edit-button"
          >
            <Ionicons name="pencil" size={20} color={colors.primary.main} />
          </TouchableOpacity>
          <TouchableOpacity
            onPress={handleDelete}
            style={styles.headerButton}
            disabled={isDeleting}
            accessibilityLabel="Delete activity"
            testID="activity-delete-button"
          >
            {isDeleting ? (
              <ActivityIndicator size="small" color={colors.status.error} />
            ) : (
              <Ionicons name="trash-outline" size={20} color={colors.status.error} />
            )}
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[styles.scrollContent, { paddingHorizontal: contentPadding }]}
        showsVerticalScrollIndicator={false}
      >
        {/* Activity Type Card */}
        <View style={styles.typeCard}>
          <LinearGradient
            colors={[typeConfig.color + '40', typeConfig.color + '20']}
            style={styles.typeCardGradient}
          >
            <View style={[styles.typeIconContainer, { backgroundColor: typeConfig.color + '30' }]}>
              <Ionicons
                name={typeConfig.icon as keyof typeof Ionicons.glyphMap}
                size={40}
                color={typeConfig.color}
              />
            </View>
            <Text style={styles.typeTitle}>{typeConfig.displayName}</Text>
            <View style={[styles.intensityBadge, { backgroundColor: intensityConfig.color }]}>
              <Ionicons
                name={intensityConfig.icon as keyof typeof Ionicons.glyphMap}
                size={14}
                color={colors.text.primary}
              />
              <Text style={styles.intensityBadgeText}>{intensityConfig.displayName}</Text>
            </View>
          </LinearGradient>
        </View>

        {/* Main Stats */}
        <View style={styles.statsGrid}>
          <View style={styles.statCard}>
            <Ionicons name="time-outline" size={24} color={colors.primary.main} />
            <Text style={styles.statValue}>{formatDuration(activity.duration)}</Text>
            <Text style={styles.statLabel}>Duration</Text>
          </View>
          {activity.caloriesBurned && (
            <View style={styles.statCard}>
              <Ionicons name="flame-outline" size={24} color="#EF4444" />
              <Text style={styles.statValue}>{activity.caloriesBurned}</Text>
              <Text style={styles.statLabel}>Calories</Text>
            </View>
          )}
          {activity.distance && (
            <View style={styles.statCard}>
              <Ionicons name="navigate-outline" size={24} color="#3B82F6" />
              <Text style={styles.statValue}>{formatDistance(activity.distance)}</Text>
              <Text style={styles.statLabel}>Distance</Text>
            </View>
          )}
          {activity.steps && (
            <View style={styles.statCard}>
              <Ionicons name="footsteps-outline" size={24} color="#10B981" />
              <Text style={styles.statValue}>{activity.steps.toLocaleString()}</Text>
              <Text style={styles.statLabel}>Steps</Text>
            </View>
          )}
        </View>

        {/* Heart Rate Section */}
        {(activity.averageHeartRate || activity.maxHeartRate) && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Heart Rate</Text>
            <View style={styles.heartRateCard}>
              {activity.averageHeartRate && (
                <View style={styles.heartRateStat}>
                  <Ionicons name="heart-outline" size={20} color={colors.status.error} />
                  <View>
                    <Text style={styles.heartRateValue}>{activity.averageHeartRate} bpm</Text>
                    <Text style={styles.heartRateLabel}>Average</Text>
                  </View>
                </View>
              )}
              {activity.maxHeartRate && (
                <View style={styles.heartRateStat}>
                  <Ionicons name="heart" size={20} color={colors.status.error} />
                  <View>
                    <Text style={styles.heartRateValue}>{activity.maxHeartRate} bpm</Text>
                    <Text style={styles.heartRateLabel}>Max</Text>
                  </View>
                </View>
              )}
            </View>
          </View>
        )}

        {/* Time Details */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Time</Text>
          <View style={styles.detailCard}>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Date</Text>
              <Text style={styles.detailValue}>
                {startDate.toLocaleDateString('en-US', {
                  weekday: 'long',
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                })}
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Started</Text>
              <Text style={styles.detailValue}>
                {startDate.toLocaleTimeString('en-US', {
                  hour: 'numeric',
                  minute: '2-digit',
                })}
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Ended</Text>
              <Text style={styles.detailValue}>
                {endDate.toLocaleTimeString('en-US', {
                  hour: 'numeric',
                  minute: '2-digit',
                })}
              </Text>
            </View>
          </View>
        </View>

        {/* Notes */}
        {activity.notes && (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Notes</Text>
            <View style={styles.notesCard}>
              <Text style={styles.notesText}>{activity.notes}</Text>
            </View>
          </View>
        )}

        {/* Source */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Source</Text>
          <View style={styles.sourceCard}>
            <Ionicons
              name={sourceConfig.icon as keyof typeof Ionicons.glyphMap}
              size={20}
              color={colors.text.secondary}
            />
            <Text style={styles.sourceText}>{sourceConfig.displayName}</Text>
          </View>
        </View>

        {/* Bottom padding */}
        <View style={{ height: spacing.xl }} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: spacing.md,
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: spacing.xl,
  },
  errorText: {
    fontSize: typography.fontSize.md,
    color: colors.status.error,
    marginTop: spacing.md,
    textAlign: 'center',
  },
  retryButton: {
    marginTop: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.md,
  },
  retryButtonText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.md,
  },
  backButton: {
    padding: spacing.sm,
    marginLeft: -spacing.sm,
  },
  headerTitle: {
    flex: 1,
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
    marginLeft: spacing.sm,
  },
  headerActions: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  headerButton: {
    padding: spacing.sm,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: spacing.xl,
  },
  typeCard: {
    borderRadius: borderRadius.xl,
    overflow: 'hidden',
    marginBottom: spacing.lg,
    ...shadows.md,
  },
  typeCardGradient: {
    alignItems: 'center',
    paddingVertical: spacing.xl,
    paddingHorizontal: spacing.lg,
  },
  typeIconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  typeTitle: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold as '700',
    color: colors.text.primary,
    marginBottom: spacing.sm,
  },
  intensityBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
    gap: spacing.xs,
  },
  intensityBadgeText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.md,
    marginBottom: spacing.lg,
  },
  statCard: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  statValue: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold as '700',
    color: colors.text.primary,
    marginTop: spacing.sm,
  },
  statLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  section: {
    marginBottom: spacing.lg,
  },
  sectionTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.secondary,
    marginBottom: spacing.sm,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  heartRateCard: {
    flexDirection: 'row',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    gap: spacing.xl,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  heartRateStat: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  heartRateValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold as '600',
    color: colors.text.primary,
  },
  heartRateLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  detailCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
  },
  detailLabel: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
  detailValue: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium as '500',
    color: colors.text.primary,
  },
  notesCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  notesText: {
    fontSize: typography.fontSize.md,
    color: colors.text.primary,
    lineHeight: 22,
  },
  sourceCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    gap: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  sourceText: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
  },
});
