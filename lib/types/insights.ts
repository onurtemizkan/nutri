/**
 * ML Insights Types
 * Type definitions for ML-generated insights and recommendations
 */

// ============================================================================
// Enums
// ============================================================================

export type MLInsightType =
  | 'CORRELATION'
  | 'PREDICTION'
  | 'ANOMALY'
  | 'RECOMMENDATION'
  | 'GOAL_PROGRESS'
  | 'PATTERN_DETECTED';

export type InsightPriority = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

// ============================================================================
// Core Types
// ============================================================================

export interface MLInsight {
  id: string;
  userId: string;
  insightType: MLInsightType;
  priority: InsightPriority;
  title: string;
  description: string;
  recommendation: string;
  correlation: number | null;
  confidence: number;
  dataPoints: number;
  metadata: InsightMetadata | null;
  viewed: boolean;
  viewedAt: string | null;
  dismissed: boolean;
  dismissedAt: string | null;
  helpful: boolean | null;
  createdAt: string;
  expiresAt: string | null;
}

export interface InsightMetadata {
  chartData?: { date: string; value: number }[];
  references?: { featureName: string; correlation: number }[];
  lagAnalysis?: {
    optimalLagHours: number;
    immediateEffect: boolean;
    delayedEffect: boolean;
  };
  trendData?: {
    direction: 'up' | 'down' | 'stable';
    percentChange: number;
  };
}

// ============================================================================
// API Request/Response Types
// ============================================================================

export interface GetInsightsParams {
  insightType?: MLInsightType;
  priority?: InsightPriority;
  viewed?: boolean;
  dismissed?: boolean;
  limit?: number;
  offset?: number;
}

export interface GetInsightsResponse {
  insights: MLInsight[];
  total: number;
  limit: number;
  offset: number;
  hasMore: boolean;
}

export interface InsightSummary {
  totalInsights: number;
  unviewedCount: number;
  highPriorityCount: number;
  byType: Record<MLInsightType, number>;
  lastGeneratedAt: string | null;
}

export interface GenerateInsightsRequest {
  targetMetrics?: string[];
  lookbackDays?: number;
  regenerate?: boolean;
}

export interface GenerateInsightsResponse {
  message: string;
  generated: number;
  skipped: number;
  errors: string[];
}

export interface UpdateInsightRequest {
  viewed?: boolean;
  dismissed?: boolean;
  helpful?: boolean;
}

export interface FeedbackRequest {
  helpful: boolean;
  feedbackText?: string;
}

// ============================================================================
// UI Helper Types
// ============================================================================

export interface InsightCardProps {
  insight: MLInsight;
  onPress?: () => void;
  onDismiss?: () => void;
  onFeedback?: (helpful: boolean) => void;
}

export interface InsightDetailProps {
  insightId: string;
}

/**
 * Get color for insight priority
 */
export function getPriorityColor(priority: InsightPriority): string {
  switch (priority) {
    case 'CRITICAL':
      return '#DC2626'; // Red
    case 'HIGH':
      return '#F59E0B'; // Amber
    case 'MEDIUM':
      return '#3B82F6'; // Blue
    case 'LOW':
    default:
      return '#6B7280'; // Gray
  }
}

/**
 * Get icon name for insight type
 */
export function getInsightTypeIcon(type: MLInsightType): string {
  switch (type) {
    case 'CORRELATION':
      return 'analytics-outline';
    case 'PREDICTION':
      return 'trending-up-outline';
    case 'ANOMALY':
      return 'alert-circle-outline';
    case 'RECOMMENDATION':
      return 'bulb-outline';
    case 'GOAL_PROGRESS':
      return 'flag-outline';
    case 'PATTERN_DETECTED':
      return 'git-branch-outline';
    default:
      return 'information-circle-outline';
  }
}

/**
 * Get human-readable label for insight type
 */
export function getInsightTypeLabel(type: MLInsightType): string {
  switch (type) {
    case 'CORRELATION':
      return 'Correlation';
    case 'PREDICTION':
      return 'Prediction';
    case 'ANOMALY':
      return 'Anomaly';
    case 'RECOMMENDATION':
      return 'Recommendation';
    case 'GOAL_PROGRESS':
      return 'Goal Progress';
    case 'PATTERN_DETECTED':
      return 'Pattern';
    default:
      return 'Insight';
  }
}

/**
 * Get human-readable label for priority
 */
export function getPriorityLabel(priority: InsightPriority): string {
  switch (priority) {
    case 'CRITICAL':
      return 'Critical';
    case 'HIGH':
      return 'High Priority';
    case 'MEDIUM':
      return 'Medium';
    case 'LOW':
      return 'Low';
    default:
      return priority;
  }
}

/**
 * Format confidence as percentage
 */
export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`;
}

/**
 * Format correlation coefficient
 */
export function formatCorrelation(correlation: number | null): string {
  if (correlation === null) return 'N/A';
  const sign = correlation >= 0 ? '+' : '';
  return `${sign}${correlation.toFixed(2)}`;
}

/**
 * Get correlation strength label
 */
export function getCorrelationStrength(correlation: number | null): string {
  if (correlation === null) return 'Unknown';
  const absCorr = Math.abs(correlation);
  if (absCorr < 0.3) return 'Weak';
  if (absCorr < 0.5) return 'Moderate';
  if (absCorr < 0.7) return 'Strong';
  return 'Very Strong';
}
