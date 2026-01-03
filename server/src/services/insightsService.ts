/**
 * ML Insights Service
 *
 * Generates personalized nutrition insights by analyzing correlations
 * between nutrition/activity patterns and health metrics.
 *
 * This service:
 * 1. Calls ML service correlation endpoints
 * 2. Transforms correlation results into human-readable insights
 * 3. Stores insights in the database
 * 4. Provides retrieval and management methods
 */

import axios from 'axios';
import prisma from '../config/database';
import { logger } from '../config/logger';
import { Prisma, MLInsightType, InsightPriority } from '@prisma/client';
import {
  GetInsightsQuery,
  UpdateInsightInput,
  GenerateInsightsRequest,
  InsightMetadata,
  InsightSummary,
} from '../types';
import { DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT } from '../config/constants';

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// Health metrics to analyze for correlations
const TARGET_HEALTH_METRICS = [
  'RESTING_HEART_RATE',
  'HEART_RATE_VARIABILITY_SDNN',
  'HEART_RATE_VARIABILITY_RMSSD',
  'SLEEP_DURATION',
  'SLEEP_SCORE',
  'RECOVERY_SCORE',
  'VO2_MAX',
  'RESPIRATORY_RATE',
] as const;

// Insight templates for human-readable generation
const INSIGHT_TEMPLATES = {
  positive_correlation: {
    nutrition: {
      title: (feature: string, metric: string) =>
        `Higher ${formatFeatureName(feature)} correlates with better ${formatMetricName(metric)}`,
      description: (feature: string, metric: string, correlation: number, strength: string) =>
        `Your data shows a ${strength} positive correlation (${correlation.toFixed(2)}) between ${formatFeatureName(feature)} and ${formatMetricName(metric)}. When you increase ${formatFeatureName(feature)}, your ${formatMetricName(metric)} tends to improve.`,
      recommendation: (feature: string) =>
        `Consider maintaining or increasing your ${formatFeatureName(feature)} intake to support your health goals.`,
    },
    activity: {
      title: (feature: string, metric: string) =>
        `${formatFeatureName(feature)} positively impacts your ${formatMetricName(metric)}`,
      description: (feature: string, metric: string, correlation: number, strength: string) =>
        `There's a ${strength} positive relationship (${correlation.toFixed(2)}) between your ${formatFeatureName(feature)} and ${formatMetricName(metric)}. More ${formatFeatureName(feature)} is associated with better ${formatMetricName(metric)}.`,
      recommendation: (feature: string) =>
        `Keep up your ${formatFeatureName(feature)} routine - it's having a positive effect on your health metrics.`,
    },
  },
  negative_correlation: {
    nutrition: {
      title: (feature: string, metric: string) =>
        `${formatFeatureName(feature)} may be affecting your ${formatMetricName(metric)}`,
      description: (feature: string, metric: string, correlation: number, strength: string) =>
        `Your data reveals a ${strength} negative correlation (${correlation.toFixed(2)}) between ${formatFeatureName(feature)} and ${formatMetricName(metric)}. Higher ${formatFeatureName(feature)} is associated with lower ${formatMetricName(metric)}.`,
      recommendation: (feature: string, metric: string) =>
        `You might want to experiment with reducing ${formatFeatureName(feature)} and monitoring how your ${formatMetricName(metric)} responds.`,
    },
    activity: {
      title: (feature: string, metric: string) =>
        `High ${formatFeatureName(feature)} may impact ${formatMetricName(metric)}`,
      description: (feature: string, metric: string, correlation: number, strength: string) =>
        `There's a ${strength} negative relationship (${correlation.toFixed(2)}) between ${formatFeatureName(feature)} and ${formatMetricName(metric)}. Consider whether you're overdoing it.`,
      recommendation: (feature: string) =>
        `Consider balancing your ${formatFeatureName(feature)} with adequate recovery time.`,
    },
  },
  delayed_effect: {
    title: (feature: string, metric: string, lagHours: number) =>
      `${formatFeatureName(feature)} affects ${formatMetricName(metric)} after ${formatLagTime(lagHours)}`,
    description: (
      feature: string,
      metric: string,
      lagHours: number,
      correlation: number,
      duration: number | null
    ) =>
      `Your ${formatFeatureName(feature)} has a delayed effect on ${formatMetricName(metric)}, taking about ${formatLagTime(lagHours)} to show impact${duration ? ` and lasting for ${formatLagTime(duration)}` : ''}. Correlation: ${correlation.toFixed(2)}.`,
    recommendation: (feature: string, lagHours: number) =>
      `Plan your ${formatFeatureName(feature)} with this ${formatLagTime(lagHours)} delay in mind for optimal timing.`,
  },
};

// Helper functions for formatting
function formatFeatureName(name: string): string {
  return name
    .replace(/^(nutrition_|activity_|health_|temporal_|interaction_)/, '')
    .replace(/_/g, ' ')
    .replace(/daily$/i, '')
    .trim()
    .toLowerCase();
}

function formatMetricName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .toLowerCase()
    .replace(/sdnn|rmssd/, (match) => match.toUpperCase())
    .replace(/hrv/i, 'HRV')
    .replace(/rhr/i, 'RHR')
    .replace(/vo2/i, 'VO2');
}

function formatLagTime(hours: number): string {
  if (hours < 24) {
    return `${hours} hours`;
  }
  const days = Math.round(hours / 24);
  return days === 1 ? '1 day' : `${days} days`;
}

function interpretCorrelationStrength(correlation: number): string {
  const absCorr = Math.abs(correlation);
  if (absCorr < 0.3) return 'weak';
  if (absCorr < 0.5) return 'moderate';
  if (absCorr < 0.7) return 'strong';
  return 'very strong';
}

function determinePriority(correlation: number, confidence: number): InsightPriority {
  const absCorr = Math.abs(correlation);
  const score = absCorr * confidence;

  if (score >= 0.6) return 'HIGH';
  if (score >= 0.4) return 'MEDIUM';
  return 'LOW';
}

// ML Service Response Types
interface CorrelationResult {
  feature_name: string;
  feature_category: string;
  correlation: number;
  p_value: number;
  sample_size: number;
  method: string;
  is_significant: boolean;
  strength: string;
  direction: string;
  explained_variance?: number;
}

interface CorrelationResponse {
  user_id: string;
  target_metric: string;
  analyzed_at: string;
  lookback_days: number;
  correlations: CorrelationResult[];
  total_features_analyzed: number;
  significant_correlations: number;
  strongest_positive?: CorrelationResult;
  strongest_negative?: CorrelationResult;
  data_quality_score: number;
  missing_days: number;
  warning?: string;
}

interface LagAnalysisResult {
  lag_hours: number;
  correlation: number;
  p_value: number;
  is_significant: boolean;
}

interface LagAnalysisResponse {
  user_id: string;
  target_metric: string;
  feature_name: string;
  analyzed_at: string;
  lag_results: LagAnalysisResult[];
  optimal_lag_hours?: number;
  optimal_correlation?: number;
  immediate_effect: boolean;
  delayed_effect: boolean;
  effect_duration_hours?: number;
  interpretation: string;
}

export class InsightsService {
  /**
   * Generate new insights for a user by analyzing correlations
   */
  async generateInsights(
    userId: string,
    options: GenerateInsightsRequest = {}
  ): Promise<{ generated: number; skipped: number; errors: string[] }> {
    const {
      targetMetrics = [...TARGET_HEALTH_METRICS],
      lookbackDays = 30,
      regenerate = false,
    } = options;

    const result = { generated: 0, skipped: 0, errors: [] as string[] };

    // Check if user has enough data
    const dataCheck = await this.checkUserDataAvailability(userId);
    if (!dataCheck.hasMinimumData) {
      result.errors.push(
        `Insufficient data: need at least ${dataCheck.requiredDays} days of data. Current: ${dataCheck.currentDays} days.`
      );
      return result;
    }

    // Optionally clear existing non-dismissed insights if regenerating
    if (regenerate) {
      await prisma.mLInsight.deleteMany({
        where: {
          userId,
          dismissed: false,
          // Don't delete insights user marked as helpful
          helpful: { not: true },
        },
      });
    }

    // Analyze correlations for each target metric
    for (const targetMetric of targetMetrics) {
      try {
        const insightsFromMetric = await this.analyzeMetricCorrelations(
          userId,
          targetMetric,
          lookbackDays
        );
        result.generated += insightsFromMetric.length;
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Unknown error';
        logger.error({ userId, targetMetric, error }, 'Error analyzing metric correlations');
        result.errors.push(`Failed to analyze ${targetMetric}: ${errorMsg}`);
      }
    }

    // Generate goal progress insights
    try {
      const goalInsights = await this.generateGoalProgressInsights(userId);
      result.generated += goalInsights;
    } catch (error) {
      logger.error({ userId, error }, 'Error generating goal progress insights');
    }

    // Generate pattern detection insights
    try {
      const patternInsights = await this.generatePatternInsights(userId, lookbackDays);
      result.generated += patternInsights;
    } catch (error) {
      logger.error({ userId, error }, 'Error generating pattern insights');
    }

    logger.info({ userId, ...result }, 'Insight generation completed');
    return result;
  }

  /**
   * Analyze correlations for a specific health metric and create insights
   */
  private async analyzeMetricCorrelations(
    userId: string,
    targetMetric: string,
    lookbackDays: number
  ): Promise<Prisma.MLInsightCreateInput[]> {
    const createdInsights: Prisma.MLInsightCreateInput[] = [];

    // Call ML service for correlation analysis
    const correlationResponse = await this.callMLCorrelationAPI(userId, targetMetric, lookbackDays);

    if (!correlationResponse || correlationResponse.correlations.length === 0) {
      return createdInsights;
    }

    // Process significant correlations (top 3 per metric to avoid overwhelming user)
    const topCorrelations = correlationResponse.correlations.slice(0, 3);

    // Batch query: Get all existing insights for these correlations to avoid N+1 queries
    const featureNames = topCorrelations.map((c) => formatFeatureName(c.feature_name));
    const existingInsights = await prisma.mLInsight.findMany({
      where: {
        userId,
        dismissed: false,
        createdAt: { gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) }, // Within last 7 days
        OR: featureNames.map((name) => ({ title: { contains: name } })),
      },
      select: { title: true },
    });
    const existingTitles = new Set(existingInsights.map((i) => i.title));

    for (const correlation of topCorrelations) {
      // Skip if we already have a similar insight (check against pre-fetched titles)
      const featureName = formatFeatureName(correlation.feature_name);
      const hasSimilarInsight = [...existingTitles].some((title) => title.includes(featureName));

      if (hasSimilarInsight) continue;

      // Check for delayed effects on strong correlations
      let lagAnalysis: LagAnalysisResponse | null = null;
      if (Math.abs(correlation.correlation) >= 0.4) {
        try {
          lagAnalysis = await this.callMLLagAnalysisAPI(
            userId,
            targetMetric,
            correlation.feature_name,
            lookbackDays
          );
        } catch {
          // Lag analysis is optional, continue without it
        }
      }

      // Generate insight based on correlation
      const insight = this.createInsightFromCorrelation(
        userId,
        correlation,
        targetMetric,
        correlationResponse.data_quality_score,
        lagAnalysis
      );

      if (insight) {
        await prisma.mLInsight.create({ data: insight });
        createdInsights.push(insight);
      }
    }

    return createdInsights;
  }

  /**
   * Create an insight from a correlation result
   */
  private createInsightFromCorrelation(
    userId: string,
    correlation: CorrelationResult,
    targetMetric: string,
    dataQuality: number,
    lagAnalysis: LagAnalysisResponse | null
  ): Prisma.MLInsightCreateInput | null {
    const { feature_name, feature_category, correlation: corrValue, direction } = correlation;
    const strength = interpretCorrelationStrength(corrValue);
    const isPositive = direction === 'positive';
    const category = feature_category as 'nutrition' | 'activity';

    // Calculate confidence based on data quality and correlation strength
    const confidence = Math.min(dataQuality * (0.5 + Math.abs(corrValue) * 0.5), 1);

    // Determine insight type
    let insightType: MLInsightType = 'CORRELATION';
    let title: string;
    let description: string;
    let recommendation: string;

    // Use lag analysis if there's a delayed effect
    if (lagAnalysis?.delayed_effect && lagAnalysis.optimal_lag_hours) {
      insightType = 'PATTERN_DETECTED';
      title = INSIGHT_TEMPLATES.delayed_effect.title(
        feature_name,
        targetMetric,
        lagAnalysis.optimal_lag_hours
      );
      description = INSIGHT_TEMPLATES.delayed_effect.description(
        feature_name,
        targetMetric,
        lagAnalysis.optimal_lag_hours,
        lagAnalysis.optimal_correlation || corrValue,
        lagAnalysis.effect_duration_hours || null
      );
      recommendation = INSIGHT_TEMPLATES.delayed_effect.recommendation(
        feature_name,
        lagAnalysis.optimal_lag_hours
      );
    } else {
      // Use standard correlation templates
      const templates = isPositive
        ? INSIGHT_TEMPLATES.positive_correlation
        : INSIGHT_TEMPLATES.negative_correlation;

      const categoryTemplates = templates[category] || templates.nutrition;

      title = categoryTemplates.title(feature_name, targetMetric);
      description = categoryTemplates.description(feature_name, targetMetric, corrValue, strength);
      recommendation = categoryTemplates.recommendation(feature_name, targetMetric);
    }

    // Build metadata
    const metadata: InsightMetadata = {
      references: [{ featureName: feature_name, correlation: corrValue }],
    };

    if (lagAnalysis) {
      metadata.lagAnalysis = {
        optimalLagHours: lagAnalysis.optimal_lag_hours || 0,
        immediateEffect: lagAnalysis.immediate_effect,
        delayedEffect: lagAnalysis.delayed_effect,
      };
    }

    return {
      user: { connect: { id: userId } },
      insightType,
      priority: determinePriority(corrValue, confidence),
      title,
      description,
      recommendation,
      correlation: corrValue,
      confidence,
      dataPoints: correlation.sample_size,
      metadata: metadata as unknown as Prisma.InputJsonValue,
      expiresAt: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000), // Expires in 14 days
    };
  }

  /**
   * Generate goal progress insights
   */
  private async generateGoalProgressInsights(userId: string): Promise<number> {
    // Get user's goals
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        goalCalories: true,
        goalProtein: true,
        goalCarbs: true,
        goalFat: true,
        goalWeight: true,
        currentWeight: true,
      },
    });

    if (!user) return 0;

    let insightsCreated = 0;

    // Get recent nutrition data (last 7 days)
    const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    const recentMeals = await prisma.meal.findMany({
      where: {
        userId,
        consumedAt: { gte: weekAgo },
      },
    });

    if (recentMeals.length > 0 && user.goalCalories) {
      // Calculate daily averages
      const mealsByDay = new Map<string, typeof recentMeals>();
      for (const meal of recentMeals) {
        const day = meal.consumedAt.toISOString().split('T')[0];
        if (!mealsByDay.has(day)) mealsByDay.set(day, []);
        mealsByDay.get(day)!.push(meal);
      }

      const dailyCalories: number[] = [];
      const dailyProtein: number[] = [];

      for (const meals of mealsByDay.values()) {
        dailyCalories.push(meals.reduce((sum, m) => sum + m.calories, 0));
        dailyProtein.push(meals.reduce((sum, m) => sum + m.protein, 0));
      }

      const avgCalories = dailyCalories.reduce((a, b) => a + b, 0) / dailyCalories.length;
      const avgProtein = dailyProtein.reduce((a, b) => a + b, 0) / dailyProtein.length;

      // Generate insight if significantly off track
      const calorieVariance = ((avgCalories - user.goalCalories) / user.goalCalories) * 100;

      if (Math.abs(calorieVariance) > 15) {
        const existingGoalInsight = await prisma.mLInsight.findFirst({
          where: {
            userId,
            insightType: 'GOAL_PROGRESS',
            createdAt: { gte: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000) },
          },
        });

        if (!existingGoalInsight) {
          const isOver = calorieVariance > 0;
          await prisma.mLInsight.create({
            data: {
              user: { connect: { id: userId } },
              insightType: 'GOAL_PROGRESS',
              priority: Math.abs(calorieVariance) > 25 ? 'HIGH' : 'MEDIUM',
              title: isOver
                ? `You're ${Math.round(Math.abs(calorieVariance))}% over your calorie goal`
                : `You're ${Math.round(Math.abs(calorieVariance))}% under your calorie goal`,
              description: `Your average daily calories this week is ${Math.round(avgCalories)} compared to your goal of ${user.goalCalories}. ${isOver ? 'Consider reducing portion sizes or choosing lower-calorie options.' : "Make sure you're eating enough to support your energy needs."}`,
              recommendation: isOver
                ? 'Try meal prepping to better control portions and track your intake more consistently.'
                : 'Focus on nutrient-dense foods to meet your calorie goals without feeling overly full.',
              confidence: 0.9,
              dataPoints: dailyCalories.length,
              metadata: {
                trendData: {
                  direction: isOver ? 'up' : 'down',
                  percentChange: calorieVariance,
                },
              } as unknown as Prisma.InputJsonValue,
              expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
            },
          });
          insightsCreated++;
        }
      }

      // Protein goal check
      if (user.goalProtein && Math.abs(avgProtein - user.goalProtein) / user.goalProtein > 0.2) {
        const isLow = avgProtein < user.goalProtein;
        const existingProteinInsight = await prisma.mLInsight.findFirst({
          where: {
            userId,
            insightType: 'GOAL_PROGRESS',
            title: { contains: 'protein' },
            createdAt: { gte: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000) },
          },
        });

        if (!existingProteinInsight && isLow) {
          await prisma.mLInsight.create({
            data: {
              user: { connect: { id: userId } },
              insightType: 'GOAL_PROGRESS',
              priority: 'MEDIUM',
              title: 'Your protein intake is below target',
              description: `Your average daily protein is ${Math.round(avgProtein)}g, which is ${Math.round(((user.goalProtein - avgProtein) / user.goalProtein) * 100)}% below your goal of ${user.goalProtein}g.`,
              recommendation:
                'Consider adding protein-rich foods like lean meats, fish, eggs, legumes, or Greek yogurt to your meals.',
              confidence: 0.85,
              dataPoints: dailyProtein.length,
              metadata: {} as unknown as Prisma.InputJsonValue,
              expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
            },
          });
          insightsCreated++;
        }
      }
    }

    return insightsCreated;
  }

  /**
   * Generate pattern detection insights
   */
  private async generatePatternInsights(userId: string, lookbackDays: number): Promise<number> {
    let insightsCreated = 0;

    // Detect meal timing patterns
    const startDate = new Date(Date.now() - lookbackDays * 24 * 60 * 60 * 1000);
    const meals = await prisma.meal.findMany({
      where: {
        userId,
        consumedAt: { gte: startDate },
      },
      orderBy: { consumedAt: 'asc' },
    });

    if (meals.length < 14) return insightsCreated; // Need at least 2 weeks of data

    // Check for late-night eating pattern
    const lateNightMeals = meals.filter((m) => {
      const hour = m.consumedAt.getHours();
      return hour >= 21 || hour < 5;
    });

    if (lateNightMeals.length > meals.length * 0.2) {
      const existingPatternInsight = await prisma.mLInsight.findFirst({
        where: {
          userId,
          insightType: 'PATTERN_DETECTED',
          title: { contains: 'late-night' },
          createdAt: { gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) },
        },
      });

      if (!existingPatternInsight) {
        await prisma.mLInsight.create({
          data: {
            user: { connect: { id: userId } },
            insightType: 'PATTERN_DETECTED',
            priority: 'MEDIUM',
            title: 'Late-night eating pattern detected',
            description: `About ${Math.round((lateNightMeals.length / meals.length) * 100)}% of your meals are eaten after 9 PM. Late-night eating may affect sleep quality and metabolic health.`,
            recommendation:
              'Try to finish eating at least 2-3 hours before bedtime. If you need a snack, choose something light and easy to digest.',
            confidence: 0.75,
            dataPoints: meals.length,
            metadata: {
              references: [
                {
                  featureName: 'late_night_meals_percentage',
                  correlation: lateNightMeals.length / meals.length,
                },
              ],
            } as unknown as Prisma.InputJsonValue,
            expiresAt: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000),
          },
        });
        insightsCreated++;
      }
    }

    return insightsCreated;
  }

  /**
   * Call ML service correlation API
   */
  private async callMLCorrelationAPI(
    userId: string,
    targetMetric: string,
    lookbackDays: number
  ): Promise<CorrelationResponse | null> {
    try {
      const response = await axios.post<CorrelationResponse>(
        `${ML_SERVICE_URL}/api/correlations/analyze`,
        {
          user_id: userId,
          target_metric: targetMetric,
          methods: ['pearson', 'spearman'],
          lookback_days: lookbackDays,
          significance_threshold: 0.05,
          min_correlation: 0.3,
          top_k: 10,
        },
        { timeout: 30000 }
      );

      return response.data;
    } catch (error) {
      logger.error({ userId, targetMetric, error }, 'Failed to call ML correlation API');
      return null;
    }
  }

  /**
   * Call ML service lag analysis API
   */
  private async callMLLagAnalysisAPI(
    userId: string,
    targetMetric: string,
    featureName: string,
    lookbackDays: number
  ): Promise<LagAnalysisResponse | null> {
    try {
      const response = await axios.post<LagAnalysisResponse>(
        `${ML_SERVICE_URL}/api/correlations/lag-analysis`,
        {
          user_id: userId,
          target_metric: targetMetric,
          feature_name: featureName,
          max_lag_hours: 72,
          lag_step_hours: 6,
          lookback_days: lookbackDays,
          method: 'pearson',
        },
        { timeout: 30000 }
      );

      return response.data;
    } catch (error) {
      logger.warn(
        { userId, targetMetric, featureName, error },
        'Failed to call ML lag analysis API'
      );
      return null;
    }
  }

  /**
   * Check if user has minimum data for insight generation
   */
  private async checkUserDataAvailability(
    userId: string
  ): Promise<{ hasMinimumData: boolean; currentDays: number; requiredDays: number }> {
    const requiredDays = 7;
    const weekAgo = new Date(Date.now() - requiredDays * 24 * 60 * 60 * 1000);

    // Check meals
    const mealCount = await prisma.meal.count({
      where: {
        userId,
        consumedAt: { gte: weekAgo },
      },
    });

    // Check health metrics
    const healthMetricCount = await prisma.healthMetric.count({
      where: {
        userId,
        recordedAt: { gte: weekAgo },
      },
    });

    // Need at least some meals and health metrics
    const hasMinimumData = mealCount >= 7 && healthMetricCount >= 7;

    // Estimate current days of data
    const currentDays = Math.min(mealCount, healthMetricCount);

    return { hasMinimumData, currentDays, requiredDays };
  }

  // ============================================================================
  // CRUD Operations
  // ============================================================================

  /**
   * Get insights for a user with filtering
   */
  async getInsights(userId: string, query: GetInsightsQuery = {}) {
    const {
      insightType,
      priority,
      viewed,
      dismissed = false,
      limit = DEFAULT_PAGE_LIMIT,
      offset = 0,
    } = query;

    const cappedLimit = Math.min(limit, MAX_PAGE_LIMIT);

    const where: Prisma.MLInsightWhereInput = {
      userId,
      dismissed,
      // Don't show expired insights
      OR: [{ expiresAt: null }, { expiresAt: { gt: new Date() } }],
    };

    if (insightType) {
      where.insightType = insightType as MLInsightType;
    }

    if (priority) {
      where.priority = priority as InsightPriority;
    }

    if (viewed !== undefined) {
      where.viewed = viewed;
    }

    // Priority order map (higher number = higher priority)
    const priorityOrder: Record<InsightPriority, number> = {
      LOW: 1,
      MEDIUM: 2,
      HIGH: 3,
      CRITICAL: 4,
    };

    const [insights, total] = await Promise.all([
      prisma.mLInsight.findMany({
        where,
        // Primary sort by createdAt in DB (priority sorted in code due to enum alphabetical issue)
        orderBy: { createdAt: 'desc' },
        take: cappedLimit,
        skip: offset,
      }),
      prisma.mLInsight.count({ where }),
    ]);

    // Sort by priority (descending) then by createdAt (descending)
    const sortedInsights = insights.sort((a, b) => {
      const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];
      if (priorityDiff !== 0) return priorityDiff;
      return b.createdAt.getTime() - a.createdAt.getTime();
    });

    return {
      insights: sortedInsights,
      total,
      limit: cappedLimit,
      offset,
      hasMore: offset + insights.length < total,
    };
  }

  /**
   * Get a single insight by ID
   */
  async getInsightById(userId: string, insightId: string) {
    const insight = await prisma.mLInsight.findFirst({
      where: {
        id: insightId,
        userId,
      },
    });

    if (!insight) {
      throw new Error('Insight not found');
    }

    return insight;
  }

  /**
   * Update an insight (mark as viewed, dismissed, or helpful)
   */
  async updateInsight(userId: string, insightId: string, data: UpdateInsightInput) {
    // Verify insight belongs to user
    await this.getInsightById(userId, insightId);

    const updateData: Prisma.MLInsightUpdateInput = {};

    if (data.viewed !== undefined) {
      updateData.viewed = data.viewed;
      if (data.viewed) {
        updateData.viewedAt = new Date();
      }
    }

    if (data.dismissed !== undefined) {
      updateData.dismissed = data.dismissed;
      if (data.dismissed) {
        updateData.dismissedAt = new Date();
      }
    }

    if (data.helpful !== undefined) {
      updateData.helpful = data.helpful;
    }

    return prisma.mLInsight.update({
      where: { id: insightId },
      data: updateData,
    });
  }

  /**
   * Mark an insight as viewed
   */
  async markAsViewed(userId: string, insightId: string) {
    return this.updateInsight(userId, insightId, { viewed: true });
  }

  /**
   * Dismiss an insight
   */
  async dismissInsight(userId: string, insightId: string) {
    return this.updateInsight(userId, insightId, { dismissed: true });
  }

  /**
   * Provide feedback on an insight
   */
  async provideFeedback(userId: string, insightId: string, helpful: boolean) {
    return this.updateInsight(userId, insightId, { helpful });
  }

  /**
   * Get summary of user's insights
   */
  async getInsightsSummary(userId: string): Promise<InsightSummary> {
    const [total, unviewed, highPriority, byType, latest] = await Promise.all([
      prisma.mLInsight.count({
        where: {
          userId,
          dismissed: false,
          OR: [{ expiresAt: null }, { expiresAt: { gt: new Date() } }],
        },
      }),
      prisma.mLInsight.count({
        where: {
          userId,
          dismissed: false,
          viewed: false,
          OR: [{ expiresAt: null }, { expiresAt: { gt: new Date() } }],
        },
      }),
      prisma.mLInsight.count({
        where: {
          userId,
          dismissed: false,
          priority: { in: ['HIGH', 'CRITICAL'] },
          OR: [{ expiresAt: null }, { expiresAt: { gt: new Date() } }],
        },
      }),
      prisma.mLInsight.groupBy({
        by: ['insightType'],
        where: {
          userId,
          dismissed: false,
          OR: [{ expiresAt: null }, { expiresAt: { gt: new Date() } }],
        },
        _count: true,
      }),
      prisma.mLInsight.findFirst({
        where: { userId },
        orderBy: { createdAt: 'desc' },
        select: { createdAt: true },
      }),
    ]);

    const byTypeMap: Record<string, number> = {
      CORRELATION: 0,
      PREDICTION: 0,
      ANOMALY: 0,
      RECOMMENDATION: 0,
      GOAL_PROGRESS: 0,
      PATTERN_DETECTED: 0,
    };

    for (const item of byType) {
      byTypeMap[item.insightType] = item._count;
    }

    return {
      totalInsights: total,
      unviewedCount: unviewed,
      highPriorityCount: highPriority,
      byType: byTypeMap as Record<MLInsightType, number>,
      lastGeneratedAt: latest?.createdAt,
    };
  }

  /**
   * Delete old or expired insights
   */
  async cleanupOldInsights(userId: string, daysOld: number = 30) {
    const cutoffDate = new Date(Date.now() - daysOld * 24 * 60 * 60 * 1000);

    const result = await prisma.mLInsight.deleteMany({
      where: {
        userId,
        OR: [
          { expiresAt: { lt: new Date() } },
          {
            createdAt: { lt: cutoffDate },
            helpful: { not: true }, // Keep helpful insights
          },
        ],
      },
    });

    return { deleted: result.count };
  }
}

export const insightsService = new InsightsService();
