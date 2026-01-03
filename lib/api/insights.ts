/**
 * ML Insights API Client
 * API functions for ML-generated insights and recommendations
 */

import api from './client';
import {
  MLInsight,
  GetInsightsParams,
  GetInsightsResponse,
  InsightSummary,
  GenerateInsightsRequest,
  GenerateInsightsResponse,
  UpdateInsightRequest,
  FeedbackRequest,
} from '../types/insights';

export type {
  MLInsight,
  GetInsightsParams,
  GetInsightsResponse,
  InsightSummary,
  GenerateInsightsRequest,
  GenerateInsightsResponse,
};

/**
 * ML Insights API client
 */
export const insightsApi = {
  /**
   * Get user's insights with optional filtering
   * GET /insights
   */
  async getInsights(params?: GetInsightsParams): Promise<GetInsightsResponse> {
    const response = await api.get<GetInsightsResponse>('/insights', { params });
    return response.data;
  },

  /**
   * Get insights summary (counts by type, priority, etc.)
   * GET /insights/summary
   */
  async getSummary(): Promise<InsightSummary> {
    const response = await api.get<InsightSummary>('/insights/summary');
    return response.data;
  },

  /**
   * Get a single insight by ID
   * GET /insights/:id
   */
  async getById(id: string): Promise<MLInsight> {
    const response = await api.get<MLInsight>(`/insights/${id}`);
    return response.data;
  },

  /**
   * Generate new insights for the user
   * POST /insights/generate
   */
  async generate(options?: GenerateInsightsRequest): Promise<GenerateInsightsResponse> {
    const response = await api.post<GenerateInsightsResponse>('/insights/generate', options || {});
    return response.data;
  },

  /**
   * Update an insight (mark as viewed, dismissed, etc.)
   * PATCH /insights/:id
   */
  async update(id: string, data: UpdateInsightRequest): Promise<MLInsight> {
    const response = await api.patch<MLInsight>(`/insights/${id}`, data);
    return response.data;
  },

  /**
   * Mark an insight as viewed
   * POST /insights/:id/view
   */
  async markAsViewed(id: string): Promise<MLInsight> {
    const response = await api.post<MLInsight>(`/insights/${id}/view`);
    return response.data;
  },

  /**
   * Dismiss an insight
   * POST /insights/:id/dismiss
   */
  async dismiss(id: string): Promise<MLInsight> {
    const response = await api.post<MLInsight>(`/insights/${id}/dismiss`);
    return response.data;
  },

  /**
   * Provide feedback on an insight
   * POST /insights/:id/feedback
   */
  async provideFeedback(id: string, feedback: FeedbackRequest): Promise<MLInsight> {
    const response = await api.post<MLInsight>(`/insights/${id}/feedback`, feedback);
    return response.data;
  },

  /**
   * Cleanup old insights
   * DELETE /insights/cleanup
   */
  async cleanup(daysOld?: number): Promise<{ message: string; deleted: number }> {
    const params = daysOld ? { daysOld } : undefined;
    const response = await api.delete<{ message: string; deleted: number }>('/insights/cleanup', {
      params,
    });
    return response.data;
  },

  /**
   * Get unviewed high-priority insights (for notifications/badges)
   */
  async getUnviewedHighPriority(): Promise<MLInsight[]> {
    const response = await this.getInsights({
      viewed: false,
      priority: 'HIGH',
      dismissed: false,
      limit: 10,
    });
    return response.insights;
  },

  /**
   * Get feed data - insights with summary
   * Combines getInsights and getSummary for efficient feed loading
   */
  async getFeed(params?: GetInsightsParams): Promise<{
    insights: MLInsight[];
    summary: InsightSummary;
    hasMore: boolean;
    total: number;
  }> {
    const [insightsResponse, summary] = await Promise.all([
      this.getInsights(params),
      this.getSummary(),
    ]);

    return {
      insights: insightsResponse.insights,
      summary,
      hasMore: insightsResponse.hasMore,
      total: insightsResponse.total,
    };
  },

  /**
   * Batch mark multiple insights as viewed
   */
  async markMultipleAsViewed(ids: string[]): Promise<void> {
    await Promise.all(ids.map((id) => this.markAsViewed(id)));
  },
};

export default insightsApi;
