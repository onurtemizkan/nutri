/**
 * Detailed Prediction View Screen
 *
 * Shows detailed prediction for a specific metric with historical chart,
 * confidence intervals, AI interpretation, and recommendations.
 */

import { useState, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  ActivityIndicator,
  Dimensions,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { predictionsApi } from '@/lib/api/predictions';
import { healthMetricsApi } from '@/lib/api/health-metrics';
import {
  PredictResponse,
  PredictionMetric,
  PREDICTION_METRIC_CONFIG,
  getTomorrowDate,
  formatConfidenceScore,
  getTrendDirection,
} from '@/lib/types/predictions';
import { TimeSeriesDataPoint, HealthMetricType, METRIC_CONFIG } from '@/lib/types/health-metrics';
import { useAuth } from '@/lib/context/AuthContext';
import { colors, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { PredictionChart } from '@/lib/components/PredictionChart';
import { EmptyState } from '@/lib/components/ui/EmptyState';

export default function PredictionDetailScreen() {
  const { metric } = useLocalSearchParams<{ metric: string }>();
  const validMetric = metric as PredictionMetric;

  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [timeSeries, setTimeSeries] = useState<TimeSeriesDataPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { user } = useAuth();
  const router = useRouter();
  const { getResponsiveValue, width } = useResponsive();

  const config = PREDICTION_METRIC_CONFIG[validMetric];
  const healthConfig = METRIC_CONFIG[validMetric as HealthMetricType];

  const contentPadding = getResponsiveValue({
    small: spacing.md,
    medium: spacing.lg,
    large: spacing.lg,
    tablet: spacing.xl,
    default: spacing.lg,
  });

  const chartWidth = width - contentPadding * 2;

  const loadData = useCallback(async () => {
    if (!user || !validMetric) {
      setIsLoading(false);
      return;
    }

    try {
      setError(null);

      // Calculate date range (30 days of history)
      const endDate = new Date();
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - 30);

      // Fetch historical data and prediction in parallel
      const [timeSeriesData, predictionData] = await Promise.all([
        healthMetricsApi.getTimeSeries(
          validMetric as HealthMetricType,
          startDate.toISOString().split('T')[0],
          endDate.toISOString().split('T')[0]
        ),
        predictionsApi.predict(validMetric, getTomorrowDate()),
      ]);

      setTimeSeries(timeSeriesData || []);
      setPrediction(predictionData);
    } catch (err) {
      console.error('Failed to load prediction data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load prediction');
    } finally {
      setIsLoading(false);
    }
  }, [user, validMetric]);

  const onRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await loadData();
    setIsRefreshing(false);
  }, [loadData]);

  useFocusEffect(
    useCallback(() => {
      loadData();
    }, [loadData])
  );

  const handleBackPress = () => {
    router.back();
  };

  // Determine trend
  const trend = useMemo(() => {
    if (!prediction) return 'neutral';
    return getTrendDirection(prediction.prediction.deviation_from_average);
  }, [prediction]);

  // Trend color logic - for RHR, lower is better
  const isLowerBetter = validMetric === 'RESTING_HEART_RATE';
  const trendColor = useMemo(() => {
    if (trend === 'up') {
      return isLowerBetter ? colors.status.error : colors.status.success;
    }
    if (trend === 'down') {
      return isLowerBetter ? colors.status.success : colors.status.error;
    }
    return colors.text.tertiary;
  }, [trend, isLowerBetter]);

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container} testID="prediction-detail-screen">
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
        </View>
      </SafeAreaView>
    );
  }

  const hasData = prediction !== null && timeSeries.length > 0;

  return (
    <SafeAreaView style={styles.container} testID="prediction-detail-screen">
      {/* Header */}
      <View style={[styles.header, { paddingHorizontal: contentPadding }]}>
        <TouchableOpacity
          onPress={handleBackPress}
          style={styles.backButton}
          accessibilityLabel="Go back"
          accessibilityRole="button"
        >
          <Ionicons name="chevron-back" size={28} color={colors.text.primary} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{config?.displayName || 'Prediction'}</Text>
        <View style={styles.headerSpacer} />
      </View>

      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={isRefreshing}
            onRefresh={onRefresh}
            tintColor={colors.primary.main}
            colors={[colors.primary.main]}
          />
        }
      >
        <View style={[styles.content, { padding: contentPadding }]}>
          {/* Error state */}
          {error && (
            <View style={styles.errorContainer}>
              <Ionicons name="alert-circle-outline" size={24} color={colors.status.error} />
              <Text style={styles.errorText}>{error}</Text>
              <TouchableOpacity onPress={loadData} style={styles.retryButton}>
                <Text style={styles.retryText}>Retry</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* No data state */}
          {!error && !hasData && (
            <EmptyState
              icon="analytics-outline"
              title="No prediction available"
              description="We need more health data to generate predictions for this metric."
              actionLabel="Track Health Data"
              onAction={() => router.push('/(tabs)/health')}
              testID="prediction-detail-empty"
            />
          )}

          {/* Prediction data */}
          {!error && hasData && prediction && (
            <>
              {/* Main prediction card */}
              <View style={styles.predictionCard}>
                <View style={styles.predictionHeader}>
                  <View style={styles.iconContainer}>
                    <Ionicons
                      name={config.icon as keyof typeof Ionicons.glyphMap}
                      size={28}
                      color={colors.primary.main}
                    />
                  </View>
                  <View style={styles.confidenceBadge}>
                    <Text style={styles.confidenceText}>
                      {formatConfidenceScore(prediction.prediction.confidence_score)} confidence
                    </Text>
                  </View>
                </View>

                <Text style={styles.predictionLabel}>Predicted for Tomorrow</Text>

                <View style={styles.valueRow}>
                  <Text style={styles.predictionValue}>
                    {prediction.prediction.predicted_value.toFixed(config.decimalPlaces)}
                  </Text>
                  <Text style={styles.unitText}>{config.unit}</Text>
                  <View style={[styles.trendBadge, { backgroundColor: `${trendColor}20` }]}>
                    <Ionicons
                      name={trend === 'up' ? 'trending-up' : trend === 'down' ? 'trending-down' : 'remove'}
                      size={18}
                      color={trendColor}
                    />
                    <Text style={[styles.trendText, { color: trendColor }]}>
                      {prediction.prediction.deviation_from_average > 0 ? '+' : ''}
                      {prediction.prediction.deviation_from_average.toFixed(config.decimalPlaces)}
                    </Text>
                  </View>
                </View>

                <Text style={styles.comparisonText}>
                  Compared to your 30-day average of{' '}
                  {prediction.prediction.historical_average.toFixed(config.decimalPlaces)} {config.unit}
                </Text>
              </View>

              {/* Chart */}
              <View style={styles.chartSection}>
                <Text style={styles.sectionTitle}>30-Day Trend + Prediction</Text>
                <PredictionChart
                  historicalData={timeSeries}
                  predictedValue={prediction.prediction.predicted_value}
                  confidenceIntervalLower={prediction.prediction.confidence_interval_lower}
                  confidenceIntervalUpper={prediction.prediction.confidence_interval_upper}
                  metricType={validMetric as HealthMetricType}
                  width={chartWidth}
                />
              </View>

              {/* Stats cards */}
              <View style={styles.statsRow}>
                <View style={styles.statCard}>
                  <Ionicons name="bar-chart-outline" size={20} color={colors.primary.main} />
                  <Text style={styles.statValue}>
                    {prediction.prediction.predicted_value.toFixed(config.decimalPlaces)}
                  </Text>
                  <Text style={styles.statLabel}>Predicted</Text>
                </View>
                <View style={styles.statCard}>
                  <Ionicons name="shield-checkmark-outline" size={20} color={colors.status.success} />
                  <Text style={styles.statValue}>
                    {formatConfidenceScore(prediction.prediction.confidence_score)}
                  </Text>
                  <Text style={styles.statLabel}>Confidence</Text>
                </View>
                <View style={styles.statCard}>
                  <Ionicons name="time-outline" size={20} color={colors.text.tertiary} />
                  <Text style={styles.statValue}>
                    {prediction.prediction.historical_average.toFixed(config.decimalPlaces)}
                  </Text>
                  <Text style={styles.statLabel}>30-Day Avg</Text>
                </View>
              </View>

              {/* Confidence interval card */}
              <View style={styles.infoCard}>
                <View style={styles.infoHeader}>
                  <Ionicons name="resize-outline" size={20} color={colors.primary.main} />
                  <Text style={styles.infoTitle}>Confidence Interval</Text>
                </View>
                <View style={styles.intervalRow}>
                  <View style={styles.intervalBound}>
                    <Text style={styles.intervalLabel}>Lower (5%)</Text>
                    <Text style={styles.intervalValue}>
                      {prediction.prediction.confidence_interval_lower.toFixed(config.decimalPlaces)}{' '}
                      {config.unit}
                    </Text>
                  </View>
                  <View style={styles.intervalDivider} />
                  <View style={styles.intervalBound}>
                    <Text style={styles.intervalLabel}>Upper (95%)</Text>
                    <Text style={styles.intervalValue}>
                      {prediction.prediction.confidence_interval_upper.toFixed(config.decimalPlaces)}{' '}
                      {config.unit}
                    </Text>
                  </View>
                </View>
              </View>

              {/* AI Interpretation */}
              <View style={styles.infoCard}>
                <View style={styles.infoHeader}>
                  <Ionicons name="bulb-outline" size={20} color={colors.status.warning} />
                  <Text style={styles.infoTitle}>AI Interpretation</Text>
                </View>
                <Text style={styles.interpretationText}>{prediction.interpretation}</Text>
              </View>

              {/* Recommendation */}
              {prediction.recommendation && (
                <View style={[styles.infoCard, styles.recommendationCard]}>
                  <View style={styles.infoHeader}>
                    <Ionicons name="fitness-outline" size={20} color={colors.status.success} />
                    <Text style={styles.infoTitle}>Recommendation</Text>
                  </View>
                  <Text style={styles.recommendationText}>{prediction.recommendation}</Text>
                </View>
              )}

              {/* Feature importance placeholder */}
              <View style={styles.infoCard}>
                <View style={styles.infoHeader}>
                  <Ionicons name="layers-outline" size={20} color={colors.primary.main} />
                  <Text style={styles.infoTitle}>Key Factors</Text>
                </View>
                <View style={styles.factorsList}>
                  <View style={styles.factorItem}>
                    <View style={styles.factorDot} />
                    <Text style={styles.factorText}>Recent sleep patterns</Text>
                  </View>
                  <View style={styles.factorItem}>
                    <View style={styles.factorDot} />
                    <Text style={styles.factorText}>Nutrition quality</Text>
                  </View>
                  <View style={styles.factorItem}>
                    <View style={styles.factorDot} />
                    <Text style={styles.factorText}>Activity levels</Text>
                  </View>
                  <View style={styles.factorItem}>
                    <View style={styles.factorDot} />
                    <Text style={styles.factorText}>Recovery trends</Text>
                  </View>
                </View>
              </View>

              {/* Model info */}
              <View style={styles.modelInfo}>
                <Text style={styles.modelInfoText}>
                  Model: {prediction.prediction.model_version} ({prediction.prediction.architecture})
                </Text>
                <Text style={styles.modelInfoText}>
                  Data quality: {Math.round(prediction.data_quality_score * 100)}% •{' '}
                  {prediction.features_used} features used
                </Text>
                {prediction.cached && (
                  <Text style={styles.cachedText}>Cached prediction</Text>
                )}
              </View>
            </>
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
  scrollView: {
    flex: 1,
  },
  content: {
    paddingBottom: 100,
  },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.md,
  },
  backButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerTitle: {
    flex: 1,
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    textAlign: 'center',
  },
  headerSpacer: {
    width: 40,
  },

  // Error state
  errorContainer: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.status.error,
    marginBottom: spacing.lg,
  },
  errorText: {
    fontSize: typography.fontSize.md,
    color: colors.status.error,
    marginTop: spacing.sm,
    marginBottom: spacing.md,
    textAlign: 'center',
  },
  retryButton: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.special.highlight,
    borderRadius: borderRadius.md,
  },
  retryText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },

  // Main prediction card
  predictionCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  predictionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  iconContainer: {
    width: 52,
    height: 52,
    borderRadius: 26,
    backgroundColor: colors.special.highlight,
    justifyContent: 'center',
    alignItems: 'center',
  },
  confidenceBadge: {
    backgroundColor: colors.special.highlight,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
  },
  confidenceText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },
  predictionLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing.sm,
  },
  valueRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: spacing.sm,
  },
  predictionValue: {
    fontSize: typography.fontSize['4xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  unitText: {
    fontSize: typography.fontSize.xl,
    color: colors.text.tertiary,
    marginLeft: spacing.xs,
  },
  trendBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginLeft: spacing.md,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
  },
  trendText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
  },
  comparisonText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.disabled,
  },

  // Chart section
  chartSection: {
    marginBottom: spacing.lg,
  },
  sectionTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing.md,
  },

  // Stats row
  statsRow: {
    flexDirection: 'row',
    gap: spacing.md,
    marginBottom: spacing.lg,
  },
  statCard: {
    flex: 1,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  statValue: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginTop: spacing.xs,
  },
  statLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginTop: spacing.xs,
  },

  // Info card
  infoCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
  },
  infoHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  infoTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },

  // Confidence interval
  intervalRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  intervalBound: {
    flex: 1,
    alignItems: 'center',
  },
  intervalLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    marginBottom: spacing.xs,
  },
  intervalValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  intervalDivider: {
    width: 1,
    height: 40,
    backgroundColor: colors.border.secondary,
    marginHorizontal: spacing.md,
  },

  // Interpretation
  interpretationText: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
    lineHeight: typography.lineHeight.relaxed * typography.fontSize.md,
  },

  // Recommendation
  recommendationCard: {
    borderColor: colors.status.success,
    backgroundColor: 'rgba(34, 197, 94, 0.05)',
  },
  recommendationText: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
    lineHeight: typography.lineHeight.relaxed * typography.fontSize.md,
  },

  // Factors list
  factorsList: {
    gap: spacing.sm,
  },
  factorItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  factorDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.primary.main,
  },
  factorText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
  },

  // Model info
  modelInfo: {
    alignItems: 'center',
    paddingTop: spacing.lg,
    gap: spacing.xs,
  },
  modelInfoText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
  },
  cachedText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    fontStyle: 'italic',
  },
});
