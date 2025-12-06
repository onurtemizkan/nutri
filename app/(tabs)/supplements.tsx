import { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import {
  supplementsApi,
  userSupplementsApi,
  supplementLogsApi,
} from '@/lib/api/supplements';
import {
  ScheduledSupplement,
  SupplementWeeklySummary,
  UserSupplement,
} from '@/lib/types/supplements';
import { useAuth } from '@/lib/context/AuthContext';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { getErrorMessage } from '@/lib/utils/errorHandling';

export default function SupplementsScreen() {
  const [scheduled, setScheduled] = useState<ScheduledSupplement[]>([]);
  const [weeklySummary, setWeeklySummary] = useState<SupplementWeeklySummary | null>(null);
  const [userSupplements, setUserSupplements] = useState<UserSupplement[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loggingId, setLoggingId] = useState<string | null>(null);
  const { user } = useAuth();
  const router = useRouter();

  const loadData = useCallback(async () => {
    if (!user) {
      setIsLoading(false);
      return;
    }

    try {
      const [scheduledData, summaryData, supplementsData] = await Promise.all([
        userSupplementsApi.getTodayScheduled(),
        supplementsApi.getWeeklySummary(),
        userSupplementsApi.getActive(),
      ]);
      setScheduled(scheduledData);
      setWeeklySummary(summaryData);
      setUserSupplements(supplementsData);
    } catch (error) {
      console.error('Failed to load supplement data:', error);
      Alert.alert('Error', getErrorMessage(error, 'Failed to load supplements'));
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  }, [loadData]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleLogSupplement = async (item: ScheduledSupplement) => {
    setLoggingId(item.userSupplement.id);
    try {
      await supplementLogsApi.logScheduled(
        item.userSupplement.id,
        item.userSupplement.supplementId,
        item.userSupplement.dosage,
        item.userSupplement.unit
      );
      await loadData();
    } catch (error) {
      Alert.alert('Error', getErrorMessage(error, 'Failed to log supplement'));
    } finally {
      setLoggingId(null);
    }
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
        </View>
      </SafeAreaView>
    );
  }

  const takenCount = scheduled.filter((s) => s.taken).length;
  const totalScheduled = scheduled.length;

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={colors.primary.main}
            colors={[colors.primary.main]}
          />
        }
      >
        <View style={styles.content}>
          {/* Header */}
          <View style={styles.header}>
            <View>
              <Text style={styles.title}>Supplements</Text>
              <Text style={styles.subtitle}>
                {new Date().toLocaleDateString('en-US', {
                  weekday: 'long',
                  month: 'long',
                  day: 'numeric',
                })}
              </Text>
            </View>
            <TouchableOpacity
              style={styles.addButton}
              onPress={() => router.push('/add-supplement' as never)}
            >
              <Ionicons name="add" size={24} color={colors.text.primary} />
            </TouchableOpacity>
          </View>

          {/* Today's Progress Card */}
          <LinearGradient
            colors={gradients.primary}
            style={styles.progressCard}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
          >
            <View style={styles.progressHeader}>
              <Text style={styles.progressTitle}>Today's Progress</Text>
              <Text style={styles.progressCount}>
                {takenCount}/{totalScheduled}
              </Text>
            </View>
            <View style={styles.progressBarContainer}>
              <View
                style={[
                  styles.progressBar,
                  {
                    width: totalScheduled > 0 ? `${(takenCount / totalScheduled) * 100}%` : '0%',
                  },
                ]}
              />
            </View>
            <Text style={styles.progressText}>
              {totalScheduled === 0
                ? 'No supplements scheduled for today'
                : takenCount === totalScheduled
                ? 'All done for today!'
                : `${totalScheduled - takenCount} remaining`}
            </Text>
          </LinearGradient>

          {/* Weekly Adherence */}
          {weeklySummary && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Weekly Adherence</Text>
              <View style={styles.weeklyCard}>
                <View style={styles.weeklyDays}>
                  {weeklySummary.days.map((day) => {
                    const dayName = new Date(day.date).toLocaleDateString('en-US', {
                      weekday: 'short',
                    });
                    const isToday =
                      new Date(day.date).toDateString() === new Date().toDateString();
                    return (
                      <View key={day.date} style={styles.dayColumn}>
                        <Text style={[styles.dayLabel, isToday && styles.todayLabel]}>
                          {dayName}
                        </Text>
                        <View
                          style={[
                            styles.dayIndicator,
                            day.adherencePercentage >= 100 && styles.dayComplete,
                            day.adherencePercentage > 0 &&
                              day.adherencePercentage < 100 &&
                              styles.dayPartial,
                            day.scheduledCount === 0 && styles.dayEmpty,
                            isToday && styles.dayToday,
                          ]}
                        >
                          {day.adherencePercentage >= 100 ? (
                            <Ionicons name="checkmark" size={16} color={colors.text.primary} />
                          ) : (
                            <Text style={styles.dayPercentage}>
                              {day.scheduledCount > 0 ? `${day.adherencePercentage}%` : '-'}
                            </Text>
                          )}
                        </View>
                      </View>
                    );
                  })}
                </View>
                <View style={styles.weeklyStats}>
                  <Text style={styles.weeklyAverage}>
                    {weeklySummary.averageAdherence}% average
                  </Text>
                </View>
              </View>
            </View>
          )}

          {/* Today's Scheduled */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Today's Schedule</Text>
              <TouchableOpacity onPress={() => router.push('/quick-log-supplement' as never)}>
                <Text style={styles.quickLogLink}>Quick Log</Text>
              </TouchableOpacity>
            </View>
            {scheduled.length === 0 ? (
              <View style={styles.emptyState}>
                <Ionicons name="medical-outline" size={48} color={colors.text.disabled} />
                <Text style={styles.emptyText}>No supplements scheduled</Text>
                <TouchableOpacity
                  style={styles.emptyButton}
                  onPress={() => router.push('/add-supplement' as never)}
                >
                  <Text style={styles.emptyButtonText}>Add Supplement</Text>
                </TouchableOpacity>
              </View>
            ) : (
              scheduled.map((item) => (
                <View key={item.userSupplement.id} style={styles.supplementCard}>
                  <View style={styles.supplementInfo}>
                    <View
                      style={[
                        styles.supplementIcon,
                        item.taken && styles.supplementIconTaken,
                      ]}
                    >
                      <Ionicons
                        name={item.taken ? 'checkmark' : 'medical'}
                        size={20}
                        color={item.taken ? colors.text.primary : colors.primary.main}
                      />
                    </View>
                    <View style={styles.supplementDetails}>
                      <Text style={styles.supplementName}>
                        {item.userSupplement.supplement.name}
                      </Text>
                      <Text style={styles.supplementDosage}>
                        {item.userSupplement.dosage} {item.userSupplement.unit}
                        {item.scheduledTimes.length > 1 &&
                          ` x ${item.scheduledTimes.length}`}
                      </Text>
                      {item.scheduledTimes.length > 0 && (
                        <Text style={styles.supplementTimes}>
                          {item.scheduledTimes.join(', ')}
                        </Text>
                      )}
                    </View>
                  </View>
                  {!item.taken && (
                    <TouchableOpacity
                      style={styles.logButton}
                      onPress={() => handleLogSupplement(item)}
                      disabled={loggingId === item.userSupplement.id}
                    >
                      {loggingId === item.userSupplement.id ? (
                        <ActivityIndicator size="small" color={colors.primary.main} />
                      ) : (
                        <Ionicons name="checkmark-circle" size={32} color={colors.primary.main} />
                      )}
                    </TouchableOpacity>
                  )}
                  {item.taken && (
                    <View style={styles.takenBadge}>
                      <Text style={styles.takenText}>
                        {item.takenCount}/{item.scheduledTimes.length}
                      </Text>
                    </View>
                  )}
                </View>
              ))
            )}
          </View>

          {/* My Supplements */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>My Supplements</Text>
              <TouchableOpacity onPress={() => router.push('/supplement-history' as never)}>
                <Text style={styles.viewAllLink}>History</Text>
              </TouchableOpacity>
            </View>
            {userSupplements.length === 0 ? (
              <View style={styles.emptyState}>
                <Text style={styles.emptyText}>No active supplements</Text>
              </View>
            ) : (
              userSupplements.slice(0, 5).map((supp) => (
                <TouchableOpacity
                  key={supp.id}
                  style={styles.mySupplementCard}
                  onPress={() => router.push(`/edit-supplement/${supp.id}` as never)}
                >
                  <View style={styles.mySupplementInfo}>
                    <Text style={styles.mySupplementName}>{supp.supplement.name}</Text>
                    <Text style={styles.mySupplementSchedule}>
                      {supp.dosage} {supp.unit} - {supp.scheduleType.replace('_', ' ').toLowerCase()}
                    </Text>
                  </View>
                  <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
                </TouchableOpacity>
              ))
            )}
            {userSupplements.length > 5 && (
              <TouchableOpacity
                style={styles.viewMoreButton}
                onPress={() => router.push('/my-supplements' as never)}
              >
                <Text style={styles.viewMoreText}>
                  View all {userSupplements.length} supplements
                </Text>
              </TouchableOpacity>
            )}
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: spacing.xl * 2,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.lg,
  },
  title: {
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  addButton: {
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.full,
    padding: spacing.sm,
    ...shadows.sm,
  },
  progressCard: {
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    ...shadows.md,
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  progressTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  progressCount: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  progressBarContainer: {
    height: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderRadius: borderRadius.full,
    marginBottom: spacing.sm,
  },
  progressBar: {
    height: '100%',
    backgroundColor: colors.text.primary,
    borderRadius: borderRadius.full,
  },
  progressText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
    color: 'rgba(255, 255, 255, 0.9)',
  },
  section: {
    marginBottom: spacing.lg,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  sectionTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    letterSpacing: -0.3,
  },
  quickLogLink: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },
  viewAllLink: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },
  weeklyCard: {
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  weeklyDays: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  dayColumn: {
    alignItems: 'center',
  },
  dayLabel: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  todayLabel: {
    color: colors.primary.main,
    fontWeight: typography.fontWeight.bold,
  },
  dayIndicator: {
    width: 36,
    height: 36,
    borderRadius: borderRadius.full,
    backgroundColor: colors.background.tertiary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  dayComplete: {
    backgroundColor: colors.status.success,
  },
  dayPartial: {
    backgroundColor: colors.status.warning,
  },
  dayEmpty: {
    backgroundColor: colors.background.tertiary,
  },
  dayToday: {
    borderWidth: 2,
    borderColor: colors.primary.main,
  },
  dayPercentage: {
    fontSize: 10,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
  },
  weeklyStats: {
    alignItems: 'center',
    paddingTop: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: colors.border.primary,
  },
  weeklyAverage: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
  },
  emptyState: {
    alignItems: 'center',
    padding: spacing.xl,
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  emptyText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.tertiary,
    marginTop: spacing.sm,
  },
  emptyButton: {
    marginTop: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.primary.main,
    borderRadius: borderRadius.md,
  },
  emptyButtonText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  supplementCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  supplementInfo: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
  },
  supplementIcon: {
    width: 44,
    height: 44,
    borderRadius: borderRadius.full,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.md,
  },
  supplementIconTaken: {
    backgroundColor: colors.status.success,
  },
  supplementDetails: {
    flex: 1,
  },
  supplementName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  supplementDosage: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
  },
  supplementTimes: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.regular,
    color: colors.text.tertiary,
    marginTop: 2,
  },
  logButton: {
    padding: spacing.xs,
  },
  takenBadge: {
    backgroundColor: colors.status.success,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
  },
  takenText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  mySupplementCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  mySupplementInfo: {
    flex: 1,
  },
  mySupplementName: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  mySupplementSchedule: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.tertiary,
    marginTop: 2,
    textTransform: 'capitalize',
  },
  viewMoreButton: {
    alignItems: 'center',
    padding: spacing.md,
  },
  viewMoreText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },
});
