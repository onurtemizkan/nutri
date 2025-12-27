import React, { memo, useCallback } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, gradients, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';

/**
 * Report Section component for the dashboard home screen
 * Provides navigation to weekly and monthly nutrition reports
 */
export const ReportSection = memo(function ReportSection() {
  const router = useRouter();

  const handleWeeklyReportPress = useCallback(() => {
    router.push('/reports/weekly');
  }, [router]);

  const handleMonthlyReportPress = useCallback(() => {
    router.push('/reports/monthly');
  }, [router]);

  return (
    <View style={styles.container}>
      {/* Section Header */}
      <View style={styles.header}>
        <Text style={styles.sectionTitle}>Reports</Text>
      </View>

      {/* Report Cards */}
      <View style={styles.cardsContainer}>
        {/* Weekly Report Card */}
        <TouchableOpacity
          style={styles.reportCard}
          onPress={handleWeeklyReportPress}
          activeOpacity={0.8}
        >
          <LinearGradient
            colors={[colors.background.tertiary, colors.background.elevated]}
            style={styles.cardGradient}
          >
            <View style={styles.cardIconContainer}>
              <Ionicons name="calendar-outline" size={24} color={colors.primary.main} />
            </View>
            <View style={styles.cardContent}>
              <Text style={styles.cardTitle}>Weekly Report</Text>
              <Text style={styles.cardDescription}>Review your week's nutrition</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
          </LinearGradient>
        </TouchableOpacity>

        {/* Monthly Report Card */}
        <TouchableOpacity
          style={styles.reportCard}
          onPress={handleMonthlyReportPress}
          activeOpacity={0.8}
        >
          <LinearGradient
            colors={[colors.background.tertiary, colors.background.elevated]}
            style={styles.cardGradient}
          >
            <View style={styles.cardIconContainer}>
              <Ionicons name="stats-chart-outline" size={24} color={colors.secondary.main} />
            </View>
            <View style={styles.cardContent}>
              <Text style={styles.cardTitle}>Monthly Report</Text>
              <Text style={styles.cardDescription}>Analyze monthly trends</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </View>
  );
});

const styles = StyleSheet.create({
  container: {
    marginBottom: spacing.xl,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  sectionTitle: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    letterSpacing: -0.5,
  },
  cardsContainer: {
    gap: spacing.sm,
  },
  reportCard: {
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  cardGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
  },
  cardIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.background.primary,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing.md,
  },
  cardContent: {
    flex: 1,
  },
  cardTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: 2,
  },
  cardDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
});

export default ReportSection;
