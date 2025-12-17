/**
 * Activities API Client
 * Unified API functions for activity tracking operations
 */

import api from './client';
import {
  Activity,
  CreateActivityInput,
  UpdateActivityInput,
  ActivityType,
  ActivitySource,
  WeeklySummary,
  DailySummary,
  ActivityStats,
} from '../types/activities';

export type { Activity, CreateActivityInput, UpdateActivityInput, WeeklySummary, DailySummary, ActivityStats };

/**
 * Query parameters for fetching activities
 */
export interface GetActivitiesParams {
  activityType?: ActivityType;
  startDate?: string;
  endDate?: string;
  source?: ActivitySource;
  limit?: number;
  offset?: number;
}

/**
 * Activities API client
 */
export const activitiesApi = {
  /**
   * Create a new activity
   * POST /activities
   */
  async create(data: CreateActivityInput): Promise<Activity> {
    const response = await api.post<Activity>('/activities', data);
    return response.data;
  },

  /**
   * Create multiple activities in bulk
   * POST /activities/bulk
   */
  async createBulk(activities: CreateActivityInput[]): Promise<{
    created: number;
    errors: { index: number; error: string }[];
  }> {
    const response = await api.post('/activities/bulk', { activities });
    return response.data;
  },

  /**
   * Get all activities with optional filters
   * GET /activities
   */
  async getAll(params?: GetActivitiesParams): Promise<Activity[]> {
    const response = await api.get('/activities', { params });
    return response.data.activities || response.data;
  },

  /**
   * Get an activity by ID
   * GET /activities/:id
   */
  async getById(id: string): Promise<Activity> {
    const response = await api.get<Activity>(`/activities/${id}`);
    return response.data;
  },

  /**
   * Update an activity
   * PUT /activities/:id
   */
  async update(id: string, data: UpdateActivityInput): Promise<Activity> {
    const response = await api.put<Activity>(`/activities/${id}`, data);
    return response.data;
  },

  /**
   * Delete an activity
   * DELETE /activities/:id
   */
  async delete(id: string): Promise<void> {
    await api.delete(`/activities/${id}`);
  },

  /**
   * Get daily summary
   * GET /activities/summary/daily
   */
  async getDailySummary(date?: string): Promise<DailySummary> {
    const params: Record<string, string> = {};
    if (date) params.date = date;
    const response = await api.get<DailySummary>('/activities/summary/daily', { params });
    return response.data;
  },

  /**
   * Get weekly summary
   * GET /activities/summary/weekly
   */
  async getWeeklySummary(): Promise<WeeklySummary> {
    const response = await api.get<WeeklySummary>('/activities/summary/weekly');
    return response.data;
  },

  /**
   * Get recovery time recommendation
   * GET /activities/recovery
   */
  async getRecoveryTime(): Promise<{
    recommendedHours: number;
    lastActivityAt: string;
    intensity: string;
    message: string;
  }> {
    const response = await api.get('/activities/recovery');
    return response.data;
  },

  /**
   * Get activity statistics by type
   * GET /activities/stats/:activityType
   */
  async getStatsByType(activityType: ActivityType): Promise<ActivityStats | null> {
    try {
      const response = await api.get<ActivityStats>(`/activities/stats/${activityType}`);
      return response.data;
    } catch (error) {
      if (isNotFoundError(error)) {
        return null;
      }
      throw error;
    }
  },

  /**
   * Get recent activities for a specific type
   * GET /activities?activityType=X&limit=Y
   */
  async getRecentByType(activityType: ActivityType, limit: number = 10): Promise<Activity[]> {
    const response = await api.get('/activities', {
      params: {
        activityType,
        limit,
      },
    });
    return response.data.activities || response.data;
  },

  /**
   * Get activities for a date range
   * Useful for calendar views
   */
  async getForDateRange(startDate: string, endDate: string): Promise<Activity[]> {
    const response = await api.get('/activities', {
      params: {
        startDate,
        endDate,
      },
    });
    return response.data.activities || response.data;
  },

  /**
   * Get today's activities
   */
  async getTodayActivities(): Promise<Activity[]> {
    const today = new Date().toISOString().split('T')[0];
    const tomorrow = new Date(Date.now() + 86400000).toISOString().split('T')[0];
    return this.getForDateRange(today, tomorrow);
  },
};

/**
 * Helper to check if an error is a 404 Not Found
 */
function isNotFoundError(error: unknown): boolean {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { status?: number } }).response;
    return response?.status === 404;
  }
  return false;
}

export default activitiesApi;
