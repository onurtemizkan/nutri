/**
 * PredictionChart Component
 *
 * Displays historical health data with a prediction point and confidence interval.
 * Uses react-native-chart-kit for the line chart visualization.
 */

import React, { useMemo } from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import { colors, spacing, borderRadius, typography, shadows } from '@/lib/theme/colors';
import { TimeSeriesDataPoint } from '@/lib/types/health-metrics';
import { HealthMetricType, METRIC_CONFIG } from '@/lib/types/health-metrics';

interface PredictionChartProps {
  /** Historical data points */
  historicalData: TimeSeriesDataPoint[];
  /** Predicted value for tomorrow */
  predictedValue: number;
  /** 95% confidence interval lower bound */
  confidenceIntervalLower: number;
  /** 95% confidence interval upper bound */
  confidenceIntervalUpper: number;
  /** Metric type for formatting */
  metricType: HealthMetricType;
  /** Chart width (default: screen width - padding) */
  width?: number;
  /** Chart height (default: 220) */
  height?: number;
}

/**
 * Sample data points evenly for display (max 6 points + prediction)
 */
function sampleDataPoints(
  data: TimeSeriesDataPoint[],
  maxPoints: number
): TimeSeriesDataPoint[] {
  if (data.length <= maxPoints) return data;

  const step = Math.floor(data.length / maxPoints);
  const sampled: TimeSeriesDataPoint[] = [];

  for (let i = 0; i < data.length; i += step) {
    sampled.push(data[i]);
    if (sampled.length >= maxPoints) break;
  }

  // Always include the last point
  if (sampled[sampled.length - 1] !== data[data.length - 1]) {
    sampled.push(data[data.length - 1]);
  }

  return sampled;
}

/**
 * Format date label for chart x-axis
 */
function formatDateLabel(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export function PredictionChart({
  historicalData,
  predictedValue,
  confidenceIntervalLower,
  confidenceIntervalUpper,
  metricType,
  width = Dimensions.get('window').width - spacing.lg * 2,
  height = 220,
}: PredictionChartProps) {
  const config = METRIC_CONFIG[metricType];

  // Prepare chart data
  const chartData = useMemo(() => {
    // Sample historical data (max 5 points to leave room for prediction)
    const sampled = sampleDataPoints(historicalData, 5);

    // Create labels and values arrays
    const labels = sampled.map((point) => formatDateLabel(point.date));
    const values = sampled.map((point) => point.value);

    // Add prediction point with "Tomorrow" label
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    labels.push('Tomorrow');
    values.push(predictedValue);

    return {
      labels,
      datasets: [
        {
          data: values,
          color: (opacity = 1) => `rgba(139, 92, 246, ${opacity})`, // Primary purple
          strokeWidth: 2,
        },
      ],
    };
  }, [historicalData, predictedValue]);

  // Calculate min/max for y-axis that includes confidence interval
  const yRange = useMemo(() => {
    const allValues = [
      ...historicalData.map((p) => p.value),
      predictedValue,
      confidenceIntervalLower,
      confidenceIntervalUpper,
    ];
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const padding = (max - min) * 0.1;
    return {
      min: Math.floor(min - padding),
      max: Math.ceil(max + padding),
    };
  }, [historicalData, predictedValue, confidenceIntervalLower, confidenceIntervalUpper]);

  const chartConfig = {
    backgroundColor: colors.background.tertiary,
    backgroundGradientFrom: colors.background.tertiary,
    backgroundGradientTo: colors.background.tertiary,
    decimalPlaces: config.unit === '%' || config.unit === 'bpm' ? 0 : 1,
    color: (opacity = 1) => `rgba(139, 92, 246, ${opacity})`,
    labelColor: (opacity = 1) => `rgba(156, 163, 175, ${opacity})`,
    style: {
      borderRadius: borderRadius.lg,
    },
    propsForDots: {
      r: '4',
      strokeWidth: '2',
      stroke: colors.primary.main,
    },
    propsForBackgroundLines: {
      strokeDasharray: '',
      stroke: colors.border.secondary,
      strokeWidth: 0.5,
    },
  };

  // Decorator for prediction point and confidence interval
  const decorator = () => {
    // Get the position of the last (prediction) point
    const pointIndex = chartData.labels.length - 1;
    const pointX = (width - 64) * (pointIndex / (chartData.labels.length - 1)) + 48;

    // Calculate Y positions for confidence interval
    const yScale = (height - 60) / (yRange.max - yRange.min);
    const predictionY = height - 40 - (predictedValue - yRange.min) * yScale;
    const lowerY = height - 40 - (confidenceIntervalLower - yRange.min) * yScale;
    const upperY = height - 40 - (confidenceIntervalUpper - yRange.min) * yScale;

    return (
      <View style={styles.decoratorContainer}>
        {/* Confidence interval shaded region */}
        <View
          style={[
            styles.confidenceRegion,
            {
              left: pointX - 15,
              top: upperY,
              height: lowerY - upperY,
            },
          ]}
        />
        {/* Prediction point highlight */}
        <View
          style={[
            styles.predictionDot,
            {
              left: pointX - 8,
              top: predictionY - 8,
            },
          ]}
        />
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <LineChart
        data={chartData}
        width={width}
        height={height}
        chartConfig={chartConfig}
        bezier
        style={styles.chart}
        withInnerLines={true}
        withOuterLines={false}
        withVerticalLines={false}
        withHorizontalLabels={true}
        withVerticalLabels={true}
        fromZero={false}
        yAxisInterval={1}
        decorator={decorator}
      />

      {/* Legend */}
      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: colors.primary.main }]} />
          <Text style={styles.legendText}>Historical</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: colors.status.warning }]} />
          <Text style={styles.legendText}>Prediction</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={styles.legendConfidence} />
          <Text style={styles.legendText}>95% CI</Text>
        </View>
      </View>

      {/* Confidence interval values */}
      <View style={styles.confidenceValues}>
        <Text style={styles.confidenceLabel}>Confidence Interval:</Text>
        <Text style={styles.confidenceRange}>
          {confidenceIntervalLower.toFixed(config.unit === '%' ? 0 : 1)} –{' '}
          {confidenceIntervalUpper.toFixed(config.unit === '%' ? 0 : 1)} {config.unit}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    ...shadows.sm,
  },
  chart: {
    marginVertical: spacing.sm,
    borderRadius: borderRadius.lg,
  },
  decoratorContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  confidenceRegion: {
    position: 'absolute',
    width: 30,
    backgroundColor: 'rgba(245, 158, 11, 0.2)',
    borderRadius: borderRadius.sm,
  },
  predictionDot: {
    position: 'absolute',
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: colors.status.warning,
    borderWidth: 3,
    borderColor: colors.background.tertiary,
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.lg,
    marginTop: spacing.sm,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  legendConfidence: {
    width: 12,
    height: 8,
    backgroundColor: 'rgba(245, 158, 11, 0.3)',
    borderRadius: 2,
  },
  legendText: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  confidenceValues: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.sm,
    paddingTop: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: colors.border.secondary,
  },
  confidenceLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
  },
  confidenceRange: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
  },
});

export default PredictionChart;
