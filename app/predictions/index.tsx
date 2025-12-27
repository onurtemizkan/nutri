/**
 * Predictions Dashboard Screen
 *
 * Displays ML-powered health predictions for RHR and HRV.
 * Shows prediction cards with confidence scores and trend indicators.
 */

import { useState, useCallback } from 'react';
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
import { predictionsApi } from '@/lib/api/predictions';
import {
  BatchPredictResponse,
  ListModelsResponse,
  PredictionResult,
  PredictionMetric,
  PRIMARY_PREDICTION_METRICS,
  PREDICTION_METRIC_CONFIG,
  getTomorrowDate,
  formatConfidenceScore,
  getTrendDirection,
} from '@/lib/types/predictions';
import { useAuth } from '@/lib/context/AuthContext';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { useResponsive } from '@/hooks/useResponsive';
import { EmptyState } from '@/lib/components/ui/EmptyState';

export default function PredictionsDashboardScreen() {
  const [predictions, setPredictions] = useState<BatchPredictResponse | null>(null);
  const [models, setModels] = useState<ListModelsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { user } = useAuth();
  const router = useRouter();
  const { getResponsiveValue, width, isTablet } = useResponsive();

  // Responsive values
  const contentPadding = getResponsiveValue({
    small: spacing.md,
    medium: spacing.lg,
    large: spacing.lg,
    tablet: spacing.xl,
    default: spacing.lg,
  });

  const cardWidth = isTablet
    ? (width - contentPadding * 2 - spacing.md) / 2
    : width - contentPadding * 2;

  const loadPredictions = useCallback(async () => {
    if (!user) {
      setIsLoading(false);
      return;
    }

    try {
      setError(null);

      // First check if user has any models
      const modelsData = await predictionsApi.listModels();
      setModels(modelsData);

      // If user has models, fetch predictions
      if (modelsData && modelsData.total_models > 0) {
        const predictionsData = await predictionsApi.batchPredict(
          PRIMARY_PREDICTION_METRICS,
          getTomorrowDate()
        );
        setPredictions(predictionsData);
      } else {
        setPredictions(null);
      }
    } catch (err) {
      console.error('Failed to load predictions:', err);
      setError(err instanceof Error ? err.message : 'Failed to load predictions');
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  const onRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await loadPredictions();
    setIsRefreshing(false);
  }, [loadPredictions]);

  useFocusEffect(
    useCallback(() => {
      loadPredictions();
    }, [loadPredictions])
  );

  const handlePredictionPress = (metric: PredictionMetric) => {
    router.push(`/predictions/${metric}`);
  };

  const handleBackPress = () => {
    router.back();
  };

  const renderPredictionCard = (metric: PredictionMetric, prediction?: PredictionResult) => {
    const config = PREDICTION_METRIC_CONFIG[metric];
    const hasData = !!prediction;
    const trend = hasData ? getTrendDirection(prediction.deviation_from_average) : 'neutral';

    // Check if we have a model for this metric
    const hasModel = models?.models.some((m) => m.metric === metric && m.is_active);
    const isProductionReady = models?.models.some(
      (m) => m.metric === metric && m.is_production_ready
    );

    return (
      <TouchableOpacity
        key={metric}
        style={[styles.predictionCard, { width: cardWidth }]}
        onPress={() => handlePredictionPress(metric)}
        activeOpacity={0.7}
        disabled={!hasModel}
      >
        {/* Header with icon and confidence */}
        <View style={styles.cardHeader}>
          <View style={styles.iconContainer}>
            <Ionicons
              name={config.icon as keyof typeof Ionicons.glyphMap}
              size={24}
              color={colors.primary.main}
            />
          </View>
          {hasData && (
            <View style={styles.confidenceBadge}>
              <Text style={styles.confidenceText}>
                {formatConfidenceScore(prediction.confidence_score)}
              </Text>
            </View>
          )}
        </View>

        {/* Metric name */}
        <Text style={styles.metricName}>{config.displayName}</Text>

        {/* Value and trend */}
        {hasData ? (
          <>
            <View style={styles.valueContainer}>
              <Text style={styles.predictionValue}>
                {prediction.predicted_value.toFixed(config.decimalPlaces)}
              </Text>
              <Text style={styles.unitText}>{config.unit}</Text>
              <View
                style={[
                  styles.trendIndicator,
                  trend === 'up' && styles.trendUp,
                  trend === 'down' && styles.trendDown,
                ]}
              >
                <Ionicons
                  name={trend === 'up' ? 'trending-up' : trend === 'down' ? 'trending-down' : 'remove'}
                  size={16}
                  color={
                    trend === 'up'
                      ? colors.status.success
                      : trend === 'down'
                      ? colors.status.error
                      : colors.text.tertiary
                  }
                />
              </View>
            </View>

            {/* Comparison with historical average */}
            <Text style={styles.comparisonText}>
              {prediction.deviation_from_average > 0 ? '+' : ''}
              {prediction.deviation_from_average.toFixed(config.decimalPlaces)} {config.unit} from avg
            </Text>

            {/* Production warning if applicable */}
            {!isProductionReady && (
              <View style={styles.warningBanner}>
                <Ionicons name="warning-outline" size={12} color={colors.status.warning} />
                <Text style={styles.warningText}>Beta prediction</Text>
              </View>
            )}
          </>
        ) : hasModel ? (
          <View style={styles.noDataContainer}>
            <Text style={styles.noDataText}>Prediction pending...</Text>
          </View>
        ) : (
          <View style={styles.noModelContainer}>
            <Ionicons name="analytics-outline" size={32} color={colors.text.disabled} />
            <Text style={styles.noModelText}>Model not trained</Text>
            <Text style={styles.noModelSubtext}>Need more health data</Text>
          </View>
        )}
      </TouchableOpacity>
    );
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container} testID="predictions-dashboard-screen">
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary.main} />
        </View>
      </SafeAreaView>
    );
  }

  const hasModels = models && models.total_models > 0;
  const hasPredictions =
    predictions && predictions.all_predictions_successful && Object.keys(predictions.predictions).length > 0;

  return (
    <SafeAreaView style={styles.container} testID="predictions-dashboard-screen">
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
        <Text style={styles.headerTitle}>Predictions</Text>
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
          {/* Date and status info */}
          <View style={styles.infoSection}>
            <View style={styles.dateContainer}>
              <Ionicons name="calendar-outline" size={18} color={colors.text.tertiary} />
              <Text style={styles.dateText}>
                Predictions for{' '}
                {new Date(getTomorrowDate()).toLocaleDateString('en-US', {
                  weekday: 'long',
                  month: 'short',
                  day: 'numeric',
                })}
              </Text>
            </View>
            {hasPredictions && (
              <View style={styles.qualityBadge}>
                <Ionicons name="shield-checkmark-outline" size={14} color={colors.status.success} />
                <Text style={styles.qualityText}>
                  {Math.round(predictions.overall_data_quality * 100)}% data quality
                </Text>
              </View>
            )}
          </View>

          {/* Error state */}
          {error && (
            <View style={styles.errorContainer}>
              <Ionicons name="alert-circle-outline" size={24} color={colors.status.error} />
              <Text style={styles.errorText}>{error}</Text>
              <TouchableOpacity onPress={loadPredictions} style={styles.retryButton}>
                <Text style={styles.retryText}>Retry</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* No models state */}
          {!error && !hasModels && (
            <EmptyState
              icon="analytics-outline"
              title="No prediction models available"
              description="We need at least 30 days of health data to train a prediction model. Keep tracking your health metrics!"
              actionLabel="Track Health Data"
              onAction={() => router.push('/(tabs)/health')}
              testID="predictions-no-models"
            />
          )}

          {/* Prediction cards */}
          {!error && hasModels && (
            <View style={styles.predictionsGrid}>
              {PRIMARY_PREDICTION_METRICS.map((metric) => {
                const prediction = predictions?.predictions[metric];
                return renderPredictionCard(metric, prediction);
              })}
            </View>
          )}

          {/* Info card */}
          {!error && hasModels && (
            <View style={styles.infoCard}>
              <View style={styles.infoHeader}>
                <Ionicons name="information-circle-outline" size={20} color={colors.primary.main} />
                <Text style={styles.infoTitle}>How predictions work</Text>
              </View>
              <Text style={styles.infoText}>
                Our ML models analyze your nutrition, activity, and health patterns to predict
                tomorrow's health metrics. Predictions improve as you log more data.
              </Text>
            </View>
          )}

          {/* View all models button */}
          {hasModels && (
            <TouchableOpacity style={styles.viewModelsButton} activeOpacity={0.8}>
              <Ionicons name="cog-outline" size={20} color={colors.primary.main} />
              <Text style={styles.viewModelsText}>View trained models</Text>
              <View style={styles.modelCountBadge}>
                <Text style={styles.modelCountText}>{models?.total_models || 0}</Text>
              </View>
              <Ionicons name="chevron-forward" size={20} color={colors.text.tertiary} />
            </TouchableOpacity>
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

  // Info section
  infoSection: {
    marginBottom: spacing.lg,
  },
  dateContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.sm,
  },
  dateText: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
  },
  qualityBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    backgroundColor: 'rgba(34, 197, 94, 0.1)',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
    alignSelf: 'flex-start',
  },
  qualityText: {
    fontSize: typography.fontSize.xs,
    color: colors.status.success,
    fontWeight: typography.fontWeight.medium,
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

  // Predictions grid
  predictionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.md,
    marginBottom: spacing.lg,
  },

  // Prediction card
  predictionCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    ...shadows.sm,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  iconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
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
  metricName: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: spacing.sm,
  },
  valueContainer: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: spacing.xs,
  },
  predictionValue: {
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  unitText: {
    fontSize: typography.fontSize.md,
    color: colors.text.tertiary,
    marginLeft: spacing.xs,
  },
  trendIndicator: {
    marginLeft: spacing.sm,
    padding: spacing.xs,
    borderRadius: borderRadius.sm,
  },
  trendUp: {
    backgroundColor: 'rgba(34, 197, 94, 0.1)',
  },
  trendDown: {
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
  },
  comparisonText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.disabled,
  },
  warningBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginTop: spacing.sm,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    backgroundColor: 'rgba(245, 158, 11, 0.1)',
    borderRadius: borderRadius.sm,
    alignSelf: 'flex-start',
  },
  warningText: {
    fontSize: typography.fontSize.xs,
    color: colors.status.warning,
  },

  // No data states
  noDataContainer: {
    paddingVertical: spacing.lg,
    alignItems: 'center',
  },
  noDataText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
  },
  noModelContainer: {
    paddingVertical: spacing.lg,
    alignItems: 'center',
  },
  noModelText: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
    marginTop: spacing.sm,
  },
  noModelSubtext: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    marginTop: spacing.xs,
  },

  // Info card
  infoCard: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    marginBottom: spacing.lg,
  },
  infoHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.sm,
  },
  infoTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  infoText: {
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    lineHeight: typography.lineHeight.relaxed * typography.fontSize.sm,
  },

  // View models button
  viewModelsButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.tertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.secondary,
    gap: spacing.sm,
  },
  viewModelsText: {
    flex: 1,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
  },
  modelCountBadge: {
    backgroundColor: colors.special.highlight,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
    minWidth: 24,
    alignItems: 'center',
  },
  modelCountText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary.main,
  },
});
