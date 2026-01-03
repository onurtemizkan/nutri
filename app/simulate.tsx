/**
 * What-If Simulation Screen
 *
 * Allows users to simulate nutrition changes and see predicted
 * health metric trajectories over 7/14/30 days.
 */

import { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Slider from '@react-native-community/slider';
import { LineChart } from 'react-native-chart-kit';

import { useAuth } from '@/lib/context/AuthContext';
import { simulateTrajectory, createTrajectoryRequest } from '@/lib/api/simulation';
import {
  NUTRITION_SLIDERS,
  METRIC_CONFIGS,
  getMetricConfig,
  formatChange,
  type PredictionMetric,
  type SimulationDuration,
  type TrajectoryResponse,
  type SimulationTrajectory,
} from '@/lib/types/simulation';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { FORM_MAX_WIDTH } from '@/lib/responsive/breakpoints';
import { getErrorMessage } from '@/lib/utils/errorHandling';

const DURATIONS: SimulationDuration[] = [7, 14, 30];
const AVAILABLE_METRICS: PredictionMetric[] = [
  'RESTING_HEART_RATE',
  'HEART_RATE_VARIABILITY_SDNN',
  'RECOVERY_SCORE',
];

export default function SimulationScreen() {
  const router = useRouter();
  const { user } = useAuth();
  const { isTablet, getSpacing, width: screenWidth } = useResponsive();
  const responsiveSpacing = getSpacing();

  // Calculate chart width
  const scrollPadding = responsiveSpacing.horizontal * 2;
  const cardPadding = spacing.md * 2;
  const maxContentWidth = isTablet ? Math.min(screenWidth, FORM_MAX_WIDTH) : screenWidth;
  const chartWidth = maxContentWidth - scrollPadding - cardPadding;

  // State for nutrition changes
  const [nutritionChanges, setNutritionChanges] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {};
    NUTRITION_SLIDERS.forEach((slider) => {
      initial[slider.featureName] = 0;
    });
    return initial;
  });

  // State for simulation settings
  const [duration, setDuration] = useState<SimulationDuration>(7);
  const [selectedMetrics, setSelectedMetrics] = useState<PredictionMetric[]>([
    'RESTING_HEART_RATE',
    'HEART_RATE_VARIABILITY_SDNN',
  ]);

  // State for simulation results
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TrajectoryResponse | null>(null);

  const handleSliderChange = useCallback((featureName: string, value: number) => {
    setNutritionChanges((prev) => ({
      ...prev,
      [featureName]: value,
    }));
  }, []);

  const toggleMetric = useCallback((metric: PredictionMetric) => {
    setSelectedMetrics((prev) => {
      if (prev.includes(metric)) {
        // Don't allow deselecting all metrics
        if (prev.length === 1) return prev;
        return prev.filter((m) => m !== metric);
      }
      // Max 3 metrics
      if (prev.length >= 3) return prev;
      return [...prev, metric];
    });
  }, []);

  const hasChanges = Object.values(nutritionChanges).some((v) => v !== 0);

  const runSimulation = useCallback(async () => {
    if (!user?.id || !hasChanges) return;

    setIsLoading(true);
    setError(null);

    try {
      const request = createTrajectoryRequest(user.id, nutritionChanges, {
        duration,
        metrics: selectedMetrics,
        includeBaseline: true,
      });

      const response = await simulateTrajectory(request);
      setResult(response);
    } catch (err) {
      setError(getErrorMessage(err, 'Failed to run simulation'));
    } finally {
      setIsLoading(false);
    }
  }, [user?.id, nutritionChanges, duration, selectedMetrics, hasChanges]);

  const resetChanges = useCallback(() => {
    const reset: Record<string, number> = {};
    NUTRITION_SLIDERS.forEach((slider) => {
      reset[slider.featureName] = 0;
    });
    setNutritionChanges(reset);
    setResult(null);
    setError(null);
  }, []);

  const renderSlider = (slider: (typeof NUTRITION_SLIDERS)[0]) => {
    const value = nutritionChanges[slider.featureName] || 0;
    const isPositive = value > 0;
    const isNegative = value < 0;
    const displayValue = slider.formatValue(value);

    return (
      <View key={slider.featureName} style={styles.sliderContainer}>
        <View style={styles.sliderHeader}>
          <Text style={styles.sliderLabel}>{slider.label}</Text>
          <Text
            style={[
              styles.sliderValue,
              isPositive && styles.positiveValue,
              isNegative && styles.negativeValue,
            ]}
          >
            {displayValue}
          </Text>
        </View>
        <Slider
          style={styles.slider}
          minimumValue={slider.min}
          maximumValue={slider.max}
          step={slider.step}
          value={value}
          onValueChange={(v) => handleSliderChange(slider.featureName, v)}
          minimumTrackTintColor={colors.primary.main}
          maximumTrackTintColor={colors.border.primary}
          thumbTintColor={colors.primary.main}
        />
        <View style={styles.sliderRange}>
          <Text style={styles.rangeText}>{slider.formatValue(slider.min)}</Text>
          <Text style={styles.rangeText}>{slider.formatValue(slider.max)}</Text>
        </View>
      </View>
    );
  };

  const renderDurationSelector = () => (
    <View style={styles.sectionContainer}>
      <Text style={styles.sectionTitle}>Simulation Duration</Text>
      <View style={styles.durationContainer}>
        {DURATIONS.map((d) => {
          const isActive = duration === d;
          const label = d === 7 ? '1 Week' : d === 14 ? '2 Weeks' : '1 Month';
          return isActive ? (
            <LinearGradient
              key={d}
              colors={gradients.primary}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.durationButton}
            >
              <Text style={styles.durationTextActive}>{label}</Text>
            </LinearGradient>
          ) : (
            <TouchableOpacity
              key={d}
              style={[styles.durationButton, styles.durationButtonInactive]}
              onPress={() => setDuration(d)}
            >
              <Text style={styles.durationTextInactive}>{label}</Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );

  const renderMetricSelector = () => (
    <View style={styles.sectionContainer}>
      <Text style={styles.sectionTitle}>Metrics to Predict</Text>
      <View style={styles.metricsContainer}>
        {AVAILABLE_METRICS.map((metric) => {
          const config = getMetricConfig(metric);
          const isSelected = selectedMetrics.includes(metric);
          return (
            <TouchableOpacity
              key={metric}
              style={[styles.metricChip, isSelected && styles.metricChipSelected]}
              onPress={() => toggleMetric(metric)}
            >
              <Ionicons
                name={isSelected ? 'checkbox' : 'square-outline'}
                size={20}
                color={isSelected ? colors.primary.main : colors.text.tertiary}
              />
              <Text style={[styles.metricChipText, isSelected && styles.metricChipTextSelected]}>
                {config?.shortLabel || metric}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );

  const renderTrajectoryChart = (
    trajectory: SimulationTrajectory,
    baseline?: SimulationTrajectory
  ) => {
    const config = getMetricConfig(trajectory.metric);
    if (!config) return null;

    const labels = trajectory.trajectory.map((p) => `D${p.day}`);
    const values = trajectory.trajectory.map((p) => p.predicted_value);
    const baselineValues = baseline?.trajectory.map((p) => p.predicted_value) || [];

    const datasets: { data: number[]; color: () => string; strokeWidth: number }[] = [
      {
        data: values,
        color: () => colors.primary.main,
        strokeWidth: 2,
      },
    ];

    if (baselineValues.length > 0) {
      datasets.push({
        data: baselineValues,
        color: () => colors.text.tertiary,
        strokeWidth: 1,
      });
    }

    const changeInfo = formatChange(trajectory.change_from_baseline, config);

    return (
      <View key={trajectory.metric} style={styles.trajectoryCard}>
        <View style={styles.trajectoryHeader}>
          <Text style={styles.trajectoryTitle}>{config.label}</Text>
          <View style={[styles.changeBadge, { backgroundColor: changeInfo.color + '20' }]}>
            <Text style={[styles.changeBadgeText, { color: changeInfo.color }]}>
              {changeInfo.text}
            </Text>
          </View>
        </View>

        <View style={styles.trajectoryStats}>
          <View style={styles.trajectoryStat}>
            <Text style={styles.statLabel}>Start</Text>
            <Text style={styles.statValue}>
              {trajectory.baseline_value.toFixed(1)} {config.unit}
            </Text>
          </View>
          <Ionicons name="arrow-forward" size={20} color={colors.text.tertiary} />
          <View style={styles.trajectoryStat}>
            <Text style={styles.statLabel}>Projected</Text>
            <Text style={[styles.statValue, { color: changeInfo.color }]}>
              {trajectory.projected_final_value.toFixed(1)} {config.unit}
            </Text>
          </View>
        </View>

        <LineChart
          data={{
            labels: labels.filter((_, i) => i % Math.ceil(labels.length / 5) === 0),
            datasets,
          }}
          width={chartWidth}
          height={180}
          chartConfig={{
            backgroundColor: colors.surface.card,
            backgroundGradientFrom: colors.surface.card,
            backgroundGradientTo: colors.surface.card,
            decimalPlaces: 0,
            color: (opacity = 1) => `rgba(139, 92, 246, ${opacity})`,
            labelColor: () => colors.text.tertiary,
            style: { borderRadius: borderRadius.md },
            propsForDots: { r: '3', strokeWidth: '1', stroke: colors.primary.main },
          }}
          bezier
          style={styles.chart}
        />

        {trajectory.lag_description && (
          <View style={styles.lagInfo}>
            <Ionicons name="time-outline" size={14} color={colors.text.tertiary} />
            <Text style={styles.lagText}>{trajectory.lag_description}</Text>
          </View>
        )}
      </View>
    );
  };

  const renderResults = () => {
    if (!result) return null;

    return (
      <View style={styles.resultsContainer}>
        <View style={styles.resultsHeader}>
          <Text style={styles.resultsTitle}>Simulation Results</Text>
          <View style={styles.confidenceBadge}>
            <Text style={styles.confidenceText}>
              {Math.round(result.model_confidence * 100)}% confidence
            </Text>
          </View>
        </View>

        {result.trajectories.map((trajectory) => {
          const baseline = result.baseline_trajectories?.find(
            (b) => b.metric === trajectory.metric
          );
          return renderTrajectoryChart(trajectory, baseline);
        })}

        {/* Summary Card */}
        <View style={styles.summaryCard}>
          <View style={styles.summaryHeader}>
            <Ionicons name="analytics-outline" size={20} color={colors.primary.main} />
            <Text style={styles.summaryTitle}>Summary</Text>
          </View>
          <Text style={styles.summaryText}>{result.summary}</Text>
        </View>

        {/* Recommendation Card */}
        <View style={[styles.summaryCard, styles.recommendationCard]}>
          <View style={styles.summaryHeader}>
            <Ionicons name="bulb-outline" size={20} color={colors.status.warning} />
            <Text style={styles.summaryTitle}>Recommendation</Text>
          </View>
          <Text style={styles.summaryText}>{result.recommendation}</Text>
        </View>

        {/* Warning if present */}
        {result.data_quality_warning && (
          <View style={styles.warningCard}>
            <Ionicons name="warning-outline" size={20} color={colors.status.warning} />
            <Text style={styles.warningText}>{result.data_quality_warning}</Text>
          </View>
        )}

        {/* Disclaimer */}
        <View style={styles.disclaimerContainer}>
          <Text style={styles.disclaimerText}>
            These predictions are estimates based on your historical data and should not replace
            medical advice. Consult a healthcare provider before making significant dietary changes.
          </Text>
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => router.back()}
          accessibilityLabel="Go back"
          accessibilityRole="button"
        >
          <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>What-If Simulation</Text>
        <TouchableOpacity
          style={styles.resetButton}
          onPress={resetChanges}
          accessibilityLabel="Reset all changes"
          accessibilityRole="button"
        >
          <Ionicons name="refresh" size={22} color={colors.text.tertiary} />
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={[
          styles.scrollContent,
          { paddingHorizontal: responsiveSpacing.horizontal },
        ]}
        showsVerticalScrollIndicator={false}
      >
        {/* Introduction */}
        <View style={styles.introCard}>
          <Ionicons name="flask-outline" size={24} color={colors.primary.main} />
          <Text style={styles.introText}>
            Adjust nutrition values below to see how changes might affect your health metrics over
            time.
          </Text>
        </View>

        {/* Nutrition Sliders */}
        <View style={styles.sectionContainer}>
          <Text style={styles.sectionTitle}>Nutrition Adjustments</Text>
          {NUTRITION_SLIDERS.map(renderSlider)}
        </View>

        {/* Duration Selector */}
        {renderDurationSelector()}

        {/* Metric Selector */}
        {renderMetricSelector()}

        {/* Run Simulation Button */}
        <TouchableOpacity
          style={[styles.runButton, !hasChanges && styles.runButtonDisabled]}
          onPress={runSimulation}
          disabled={isLoading || !hasChanges}
          accessibilityLabel="Run simulation"
          accessibilityRole="button"
        >
          <LinearGradient
            colors={hasChanges ? gradients.primary : ['#555', '#555']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
            style={styles.runButtonGradient}
          >
            {isLoading ? (
              <ActivityIndicator size="small" color={colors.text.primary} />
            ) : (
              <>
                <Ionicons name="play" size={20} color={colors.text.primary} />
                <Text style={styles.runButtonText}>Run Simulation</Text>
              </>
            )}
          </LinearGradient>
        </TouchableOpacity>

        {/* Error Message */}
        {error && (
          <View style={styles.errorCard}>
            <Ionicons name="alert-circle" size={20} color={colors.status.error} />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {/* Results */}
        {renderResults()}

        {/* Bottom Spacing */}
        <View style={{ height: spacing['2xl'] }} />
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
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
  },
  backButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'flex-start',
  },
  headerTitle: {
    ...typography.h3,
    color: colors.text.primary,
    flex: 1,
    textAlign: 'center',
  },
  resetButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'flex-end',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingTop: spacing.md,
  },
  introCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface.card,
    padding: spacing.md,
    borderRadius: borderRadius.lg,
    marginBottom: spacing.lg,
    gap: spacing.sm,
  },
  introText: {
    ...typography.body,
    color: colors.text.tertiary,
    flex: 1,
  },
  sectionContainer: {
    marginBottom: spacing.lg,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  sliderContainer: {
    backgroundColor: colors.surface.card,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginBottom: spacing.sm,
  },
  sliderHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  sliderLabel: {
    ...typography.bodyBold,
    color: colors.text.primary,
  },
  sliderValue: {
    ...typography.bodyBold,
    color: colors.text.primary,
    minWidth: 60,
    textAlign: 'right',
  },
  positiveValue: {
    color: colors.status.success,
  },
  negativeValue: {
    color: colors.status.error,
  },
  slider: {
    width: '100%',
    height: 40,
  },
  sliderRange: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: -spacing.xs,
  },
  rangeText: {
    ...typography.caption,
    color: colors.text.tertiary,
  },
  durationContainer: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  durationButton: {
    flex: 1,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  durationButtonInactive: {
    backgroundColor: colors.surface.card,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  durationTextActive: {
    ...typography.bodyBold,
    color: colors.text.primary,
  },
  durationTextInactive: {
    ...typography.body,
    color: colors.text.tertiary,
  },
  metricsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  metricChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    backgroundColor: colors.surface.card,
    borderRadius: borderRadius.full,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  metricChipSelected: {
    borderColor: colors.primary.main,
    backgroundColor: colors.primary.main + '10',
  },
  metricChipText: {
    ...typography.body,
    color: colors.text.tertiary,
  },
  metricChipTextSelected: {
    color: colors.primary.main,
  },
  runButton: {
    marginTop: spacing.md,
    marginBottom: spacing.lg,
    borderRadius: borderRadius.lg,
    overflow: 'hidden',
    ...shadows.md,
  },
  runButtonDisabled: {
    opacity: 0.6,
  },
  runButtonGradient: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
  },
  runButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
  },
  errorCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: colors.status.error + '20',
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginBottom: spacing.lg,
  },
  errorText: {
    ...typography.body,
    color: colors.status.error,
    flex: 1,
  },
  resultsContainer: {
    marginTop: spacing.md,
  },
  resultsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  resultsTitle: {
    ...typography.h3,
    color: colors.text.primary,
  },
  confidenceBadge: {
    backgroundColor: colors.primary.main + '20',
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
    borderRadius: borderRadius.full,
  },
  confidenceText: {
    ...typography.caption,
    color: colors.primary.main,
  },
  trajectoryCard: {
    backgroundColor: colors.surface.card,
    padding: spacing.md,
    borderRadius: borderRadius.lg,
    marginBottom: spacing.md,
    ...shadows.sm,
  },
  trajectoryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  trajectoryTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text.primary,
  },
  changeBadge: {
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.sm,
    borderRadius: borderRadius.full,
  },
  changeBadgeText: {
    fontSize: 12,
    fontWeight: '600',
  },
  trajectoryStats: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.md,
    marginBottom: spacing.md,
    paddingVertical: spacing.sm,
    backgroundColor: colors.background.primary,
    borderRadius: borderRadius.md,
  },
  trajectoryStat: {
    alignItems: 'center',
  },
  statLabel: {
    ...typography.caption,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  statValue: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text.primary,
  },
  chart: {
    marginVertical: spacing.sm,
    borderRadius: borderRadius.md,
  },
  lagInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginTop: spacing.sm,
  },
  lagText: {
    ...typography.caption,
    color: colors.text.tertiary,
  },
  summaryCard: {
    backgroundColor: colors.surface.card,
    padding: spacing.md,
    borderRadius: borderRadius.lg,
    marginBottom: spacing.md,
  },
  recommendationCard: {
    borderLeftWidth: 3,
    borderLeftColor: colors.status.warning,
  },
  summaryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.sm,
  },
  summaryTitle: {
    ...typography.bodyBold,
    color: colors.text.primary,
  },
  summaryText: {
    ...typography.body,
    color: colors.text.tertiary,
    lineHeight: 22,
  },
  warningCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    backgroundColor: colors.status.warning + '15',
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginBottom: spacing.md,
  },
  warningText: {
    ...typography.body,
    color: colors.status.warning,
    flex: 1,
  },
  disclaimerContainer: {
    backgroundColor: colors.surface.card,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  disclaimerText: {
    ...typography.caption,
    color: colors.text.tertiary,
    textAlign: 'center',
    lineHeight: 18,
  },
});
