import { useState, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Circle, Defs, LinearGradient as SvgLinearGradient, Stop } from 'react-native-svg';
import { waterApi } from '@/lib/api/water';
import { WaterDailySummary, WaterWeeklySummary, WATER_PRESETS } from '@/lib/types';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { showAlert } from '@/lib/utils/alert';
import { getErrorMessage } from '@/lib/utils/errorHandling';

// Water color gradient - blue theme
const WATER_GRADIENT = ['#3B82F6', '#06B6D4'] as const;

export default function WaterTrackingScreen() {
  const [dailySummary, setDailySummary] = useState<WaterDailySummary | null>(null);
  const [weeklySummary, setWeeklySummary] = useState<WaterWeeklySummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [isAdding, setIsAdding] = useState<string | null>(null); // Track which button is loading
  const router = useRouter();
  const { isTablet, getResponsiveValue } = useResponsive();

  const contentPadding = getResponsiveValue({
    small: spacing.md,
    medium: spacing.lg,
    large: spacing.lg,
    tablet: spacing.xl,
    default: spacing.lg,
  });

  const loadData = useCallback(async () => {
    try {
      const [daily, weekly] = await Promise.all([
        waterApi.getDailySummary(),
        waterApi.getWeeklySummary(),
      ]);
      setDailySummary(daily);
      setWeeklySummary(weekly);
    } catch (error) {
      console.error('Failed to load water data:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  }, [loadData]);

  useFocusEffect(
    useCallback(() => {
      loadData();
    }, [loadData])
  );

  const handleQuickAdd = useCallback(
    async (preset: 'glass' | 'bottle' | 'cup') => {
      setIsAdding(preset);
      try {
        await waterApi.quickAddWater({ preset });
        // Refresh data
        await loadData();
      } catch (error) {
        showAlert('Error', getErrorMessage(error, 'Failed to add water intake'));
      } finally {
        setIsAdding(null);
      }
    },
    [loadData]
  );

  const handleCustomAdd = useCallback(() => {
    router.push('/water-add');
  }, [router]);

  const handleDeleteIntake = useCallback(
    async (intakeId: string, amount: number) => {
      showAlert('Delete Water Intake', `Delete ${amount}ml intake?`, [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await waterApi.deleteWaterIntake(intakeId);
              await loadData();
            } catch (error) {
              showAlert('Error', getErrorMessage(error, 'Failed to delete intake'));
            }
          },
        },
      ]);
    },
    [loadData]
  );

  const handleGoalSettings = useCallback(() => {
    router.push('/water-goal');
  }, [router]);

  // Calculate progress
  const progress = useMemo(() => {
    if (!dailySummary) return 0;
    return Math.min(1, dailySummary.totalAmount / dailySummary.goalAmount);
  }, [dailySummary?.totalAmount, dailySummary?.goalAmount]);

  // Format amounts for display
  const formatAmount = (ml: number): string => {
    if (ml >= 1000) {
      return `${(ml / 1000).toFixed(1)}L`;
    }
    return `${ml}ml`;
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.secondary.main} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => router.back()}
          style={styles.backButton}
          accessibilityLabel="Go back"
          accessibilityRole="button"
        >
          <Ionicons name="arrow-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Water Tracking</Text>
        <TouchableOpacity
          onPress={handleGoalSettings}
          style={styles.settingsButton}
          accessibilityLabel="Water goal settings"
          accessibilityRole="button"
        >
          <Ionicons name="settings-outline" size={24} color={colors.text.primary} />
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={colors.secondary.main}
            colors={[colors.secondary.main]}
          />
        }
      >
        <View style={[styles.content, { padding: contentPadding }]}>
          {/* Progress Ring */}
          <View style={[styles.progressCard, isTablet && styles.progressCardTablet]}>
            <View style={styles.progressRingContainer}>
              <Svg width={200} height={200}>
                <Defs>
                  <SvgLinearGradient id="waterGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <Stop offset="0%" stopColor={WATER_GRADIENT[0]} />
                    <Stop offset="100%" stopColor={WATER_GRADIENT[1]} />
                  </SvgLinearGradient>
                </Defs>
                {/* Background circle */}
                <Circle
                  cx={100}
                  cy={100}
                  r={85}
                  stroke={colors.background.elevated}
                  strokeWidth={15}
                  fill="transparent"
                />
                {/* Progress circle */}
                <Circle
                  cx={100}
                  cy={100}
                  r={85}
                  stroke="url(#waterGradient)"
                  strokeWidth={15}
                  fill="transparent"
                  strokeLinecap="round"
                  strokeDasharray={2 * Math.PI * 85}
                  strokeDashoffset={2 * Math.PI * 85 * (1 - progress)}
                  rotation={-90}
                  origin="100, 100"
                />
              </Svg>
              <View style={styles.progressContent}>
                <Ionicons name="water" size={32} color={colors.secondary.main} />
                <Text style={styles.progressValue}>
                  {formatAmount(dailySummary?.totalAmount || 0)}
                </Text>
                <Text style={styles.progressGoal}>
                  of {formatAmount(dailySummary?.goalAmount || 2000)}
                </Text>
                <Text style={styles.progressPercent}>{dailySummary?.percentageComplete || 0}%</Text>
              </View>
            </View>
          </View>

          {/* Quick Add Buttons */}
          <View style={styles.quickAddSection}>
            <Text style={styles.sectionTitle}>Quick Add</Text>
            <View style={styles.quickAddGrid}>
              <TouchableOpacity
                style={styles.quickAddButton}
                onPress={() => handleQuickAdd('glass')}
                disabled={isAdding !== null}
                activeOpacity={0.7}
                accessibilityLabel={`Add ${WATER_PRESETS.glass}ml glass of water`}
                accessibilityRole="button"
                accessibilityHint="Quickly log a glass of water"
              >
                <LinearGradient
                  colors={WATER_GRADIENT}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={styles.quickAddGradient}
                >
                  {isAdding === 'glass' ? (
                    <ActivityIndicator size="small" color={colors.text.primary} />
                  ) : (
                    <>
                      <Ionicons name="water-outline" size={28} color={colors.text.primary} />
                      <Text style={styles.quickAddAmount}>{WATER_PRESETS.glass}ml</Text>
                      <Text style={styles.quickAddLabel}>Glass</Text>
                    </>
                  )}
                </LinearGradient>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.quickAddButton}
                onPress={() => handleQuickAdd('bottle')}
                disabled={isAdding !== null}
                activeOpacity={0.7}
                accessibilityLabel={`Add ${WATER_PRESETS.bottle}ml bottle of water`}
                accessibilityRole="button"
                accessibilityHint="Quickly log a bottle of water"
              >
                <LinearGradient
                  colors={WATER_GRADIENT}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={styles.quickAddGradient}
                >
                  {isAdding === 'bottle' ? (
                    <ActivityIndicator size="small" color={colors.text.primary} />
                  ) : (
                    <>
                      <Ionicons name="flask-outline" size={28} color={colors.text.primary} />
                      <Text style={styles.quickAddAmount}>{WATER_PRESETS.bottle}ml</Text>
                      <Text style={styles.quickAddLabel}>Bottle</Text>
                    </>
                  )}
                </LinearGradient>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.quickAddButton}
                onPress={() => handleQuickAdd('cup')}
                disabled={isAdding !== null}
                activeOpacity={0.7}
                accessibilityLabel={`Add ${WATER_PRESETS.cup}ml cup of water`}
                accessibilityRole="button"
                accessibilityHint="Quickly log a cup of water"
              >
                <LinearGradient
                  colors={WATER_GRADIENT}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={styles.quickAddGradient}
                >
                  {isAdding === 'cup' ? (
                    <ActivityIndicator size="small" color={colors.text.primary} />
                  ) : (
                    <>
                      <Ionicons name="cafe-outline" size={28} color={colors.text.primary} />
                      <Text style={styles.quickAddAmount}>{WATER_PRESETS.cup}ml</Text>
                      <Text style={styles.quickAddLabel}>Cup</Text>
                    </>
                  )}
                </LinearGradient>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.quickAddButton}
                onPress={handleCustomAdd}
                disabled={isAdding !== null}
                activeOpacity={0.7}
                accessibilityLabel="Add custom water amount"
                accessibilityRole="button"
                accessibilityHint="Open screen to enter a custom water amount"
              >
                <View style={styles.customAddButton}>
                  <Ionicons name="add-circle-outline" size={28} color={colors.secondary.main} />
                  <Text style={[styles.quickAddAmount, { color: colors.secondary.main }]}>
                    Custom
                  </Text>
                  <Text style={[styles.quickAddLabel, { color: colors.text.tertiary }]}>
                    Amount
                  </Text>
                </View>
              </TouchableOpacity>
            </View>
          </View>

          {/* Weekly Summary */}
          {weeklySummary && (
            <View style={styles.weeklySection}>
              <Text style={styles.sectionTitle}>7-Day Summary</Text>
              <View style={[styles.weeklyCard, isTablet && styles.weeklyCardTablet]}>
                <View style={styles.weeklyStats}>
                  <View style={styles.weeklyStat}>
                    <Text style={styles.weeklyStatValue}>
                      {formatAmount(weeklySummary.dailyAverage)}
                    </Text>
                    <Text style={styles.weeklyStatLabel}>Daily Avg</Text>
                  </View>
                  <View style={styles.weeklyStatDivider} />
                  <View style={styles.weeklyStat}>
                    <Text style={styles.weeklyStatValue}>{weeklySummary.daysMetGoal}/7</Text>
                    <Text style={styles.weeklyStatLabel}>Days Met</Text>
                  </View>
                  <View style={styles.weeklyStatDivider} />
                  <View style={styles.weeklyStat}>
                    <Text style={styles.weeklyStatValue}>
                      {formatAmount(weeklySummary.totalAmount)}
                    </Text>
                    <Text style={styles.weeklyStatLabel}>Total</Text>
                  </View>
                </View>

                {/* Daily bars */}
                <View style={styles.dailyBars}>
                  {weeklySummary.dailySummaries.map((day, index) => {
                    const dayProgress = Math.min(1, day.totalAmount / day.goalAmount);
                    const dayName = new Date(day.date).toLocaleDateString('en-US', {
                      weekday: 'short',
                    });
                    const isToday = index === weeklySummary.dailySummaries.length - 1;

                    return (
                      <View key={day.date} style={styles.dailyBar}>
                        <View style={styles.dailyBarTrack}>
                          <LinearGradient
                            colors={isToday ? WATER_GRADIENT : ['#374151', '#374151']}
                            start={{ x: 0, y: 1 }}
                            end={{ x: 0, y: 0 }}
                            style={[styles.dailyBarFill, { height: `${dayProgress * 100}%` }]}
                          />
                        </View>
                        <Text style={[styles.dailyBarLabel, isToday && styles.dailyBarLabelToday]}>
                          {dayName}
                        </Text>
                      </View>
                    );
                  })}
                </View>
              </View>
            </View>
          )}

          {/* Today's Intakes */}
          {dailySummary && dailySummary.intakes.length > 0 && (
            <View style={styles.intakesSection}>
              <Text style={styles.sectionTitle}>Today's Intakes</Text>
              {dailySummary.intakes.map((intake) => (
                <TouchableOpacity
                  key={intake.id}
                  style={styles.intakeCard}
                  onPress={() => handleDeleteIntake(intake.id, intake.amount)}
                  activeOpacity={0.7}
                  accessibilityLabel={`${formatAmount(intake.amount)} water intake. Tap to delete`}
                  accessibilityRole="button"
                >
                  <View style={styles.intakeInfo}>
                    <Ionicons name="water" size={20} color={colors.secondary.main} />
                    <Text style={styles.intakeAmount}>{formatAmount(intake.amount)}</Text>
                  </View>
                  <Text style={styles.intakeTime}>
                    {new Date(intake.recordedAt).toLocaleTimeString('en-US', {
                      hour: 'numeric',
                      minute: '2-digit',
                    })}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          )}
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
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.secondary,
  },
  backButton: {
    padding: spacing.xs,
  },
  headerTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  settingsButton: {
    padding: spacing.xs,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    paddingBottom: 100,
  },

  // Progress Card
  progressCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    marginBottom: spacing.xl,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.md,
  },
  progressCardTablet: {
    maxWidth: 400,
    alignSelf: 'center',
    width: '100%',
  },
  progressRingContainer: {
    width: 200,
    height: 200,
    justifyContent: 'center',
    alignItems: 'center',
  },
  progressContent: {
    position: 'absolute',
    alignItems: 'center',
  },
  progressValue: {
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginTop: spacing.xs,
  },
  progressGoal: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  progressPercent: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.secondary.main,
    marginTop: spacing.xs,
  },

  // Quick Add Section
  quickAddSection: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  quickAddGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  quickAddButton: {
    flex: 1,
    minWidth: '45%',
    aspectRatio: 1.2,
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
    ...shadows.sm,
  },
  quickAddGradient: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.md,
  },
  customAddButton: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.md,
    backgroundColor: colors.background.tertiary,
    borderWidth: 2,
    borderColor: colors.secondary.main,
    borderStyle: 'dashed',
    borderRadius: borderRadius.lg,
  },
  quickAddAmount: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginTop: spacing.xs,
  },
  quickAddLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    opacity: 0.9,
  },

  // Weekly Section
  weeklySection: {
    marginBottom: spacing.xl,
  },
  weeklyCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  weeklyCardTablet: {
    maxWidth: 600,
    alignSelf: 'center',
    width: '100%',
  },
  weeklyStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: spacing.lg,
  },
  weeklyStat: {
    alignItems: 'center',
  },
  weeklyStatValue: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  weeklyStatLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  weeklyStatDivider: {
    width: 1,
    height: 40,
    backgroundColor: colors.border.secondary,
  },
  dailyBars: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    height: 100,
    paddingTop: spacing.sm,
  },
  dailyBar: {
    flex: 1,
    alignItems: 'center',
    marginHorizontal: 2,
  },
  dailyBarTrack: {
    flex: 1,
    width: '70%',
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.xs,
    overflow: 'hidden',
    justifyContent: 'flex-end',
  },
  dailyBarFill: {
    width: '100%',
    borderRadius: borderRadius.xs,
  },
  dailyBarLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },
  dailyBarLabelToday: {
    color: colors.secondary.main,
    fontWeight: typography.fontWeight.semibold,
  },

  // Intakes Section
  intakesSection: {
    marginBottom: spacing.xl,
  },
  intakeCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginBottom: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  intakeInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  intakeAmount: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  intakeTime: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
});
