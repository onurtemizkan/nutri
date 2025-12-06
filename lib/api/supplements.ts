/**
 * Supplement API Client
 *
 * Provides type-safe methods for all supplement-related API endpoints:
 * - Master supplement list (read-only)
 * - User supplement schedules (CRUD)
 * - Supplement logs (intake tracking)
 * - Summaries and analytics
 */

import api from './client';
import {
  Supplement,
  UserSupplement,
  SupplementLog,
  CreateUserSupplementInput,
  UpdateUserSupplementInput,
  CreateSupplementLogInput,
  UpdateSupplementLogInput,
  ScheduledSupplement,
  SupplementDailySummary,
  SupplementWeeklySummary,
  SupplementStats,
  GetSupplementsQuery,
  GetSupplementLogsQuery,
  SupplementCategory,
} from '../types/supplements';

// =============================================================================
// MASTER SUPPLEMENT LIST
// =============================================================================

export const supplementsApi = {
  /**
   * Get all supplements with optional filtering
   */
  async getSupplements(query?: GetSupplementsQuery): Promise<Supplement[]> {
    const params: Record<string, string> = {};
    if (query?.category) params.category = query.category;
    if (query?.search) params.search = query.search;

    const response = await api.get<Supplement[]>('/supplements', { params });
    return response.data;
  },

  /**
   * Get supplements by category
   */
  async getSupplementsByCategory(category: SupplementCategory): Promise<Supplement[]> {
    return this.getSupplements({ category });
  },

  /**
   * Search supplements by name
   */
  async searchSupplements(search: string): Promise<Supplement[]> {
    return this.getSupplements({ search });
  },

  /**
   * Get a specific supplement by ID
   */
  async getSupplementById(id: string): Promise<Supplement> {
    const response = await api.get<Supplement>(`/supplements/${id}`);
    return response.data;
  },

  /**
   * Get supplement usage statistics
   */
  async getSupplementStats(supplementId: string, days?: number): Promise<SupplementStats> {
    const params: Record<string, string> = {};
    if (days) params.days = days.toString();

    const response = await api.get<SupplementStats>(`/supplements/${supplementId}/stats`, { params });
    return response.data;
  },

  /**
   * Get daily supplement summary
   */
  async getDailySummary(date?: Date): Promise<SupplementDailySummary> {
    const params: Record<string, string> = {};
    if (date) params.date = date.toISOString();

    const response = await api.get<SupplementDailySummary>('/supplements/summary/daily', { params });
    return response.data;
  },

  /**
   * Get weekly supplement summary
   */
  async getWeeklySummary(): Promise<SupplementWeeklySummary> {
    const response = await api.get<SupplementWeeklySummary>('/supplements/summary/weekly');
    return response.data;
  },

  /**
   * Delete all user supplement data (GDPR/privacy)
   */
  async deleteAllUserData(): Promise<{ message: string }> {
    const response = await api.delete<{ message: string }>('/supplements/user-data');
    return response.data;
  },
};

// =============================================================================
// USER SUPPLEMENT SCHEDULES
// =============================================================================

export const userSupplementsApi = {
  /**
   * Create a new supplement schedule
   */
  async create(data: CreateUserSupplementInput): Promise<UserSupplement> {
    const response = await api.post<UserSupplement>('/user-supplements', data);
    return response.data;
  },

  /**
   * Get all user supplement schedules
   */
  async getAll(includeInactive = false): Promise<UserSupplement[]> {
    const params = includeInactive ? { includeInactive: 'true' } : {};
    const response = await api.get<UserSupplement[]>('/user-supplements', { params });
    return response.data;
  },

  /**
   * Get active supplement schedules only
   */
  async getActive(): Promise<UserSupplement[]> {
    return this.getAll(false);
  },

  /**
   * Get a specific user supplement schedule
   */
  async getById(id: string): Promise<UserSupplement> {
    const response = await api.get<UserSupplement>(`/user-supplements/${id}`);
    return response.data;
  },

  /**
   * Update a user supplement schedule
   */
  async update(id: string, data: UpdateUserSupplementInput): Promise<UserSupplement> {
    const response = await api.put<UserSupplement>(`/user-supplements/${id}`, data);
    return response.data;
  },

  /**
   * Deactivate a user supplement (soft delete)
   */
  async deactivate(id: string): Promise<{ message: string }> {
    const response = await api.delete<{ message: string }>(`/user-supplements/${id}`);
    return response.data;
  },

  /**
   * Get supplements scheduled for a specific date
   */
  async getScheduled(date?: Date): Promise<ScheduledSupplement[]> {
    const params: Record<string, string> = {};
    if (date) params.date = date.toISOString();

    const response = await api.get<ScheduledSupplement[]>('/user-supplements/scheduled', { params });
    return response.data;
  },

  /**
   * Get today's scheduled supplements
   */
  async getTodayScheduled(): Promise<ScheduledSupplement[]> {
    return this.getScheduled(new Date());
  },
};

// =============================================================================
// SUPPLEMENT LOGS (Intake Tracking)
// =============================================================================

export const supplementLogsApi = {
  /**
   * Create a supplement log entry
   */
  async create(data: CreateSupplementLogInput): Promise<SupplementLog> {
    const response = await api.post<SupplementLog>('/supplement-logs', data);
    return response.data;
  },

  /**
   * Create multiple supplement log entries at once
   */
  async createBulk(logs: CreateSupplementLogInput[]): Promise<SupplementLog[]> {
    const response = await api.post<SupplementLog[]>('/supplement-logs/bulk', { logs });
    return response.data;
  },

  /**
   * Get supplement logs with optional filtering
   */
  async getAll(query?: GetSupplementLogsQuery): Promise<SupplementLog[]> {
    const params: Record<string, string> = {};
    if (query?.startDate) params.startDate = query.startDate;
    if (query?.endDate) params.endDate = query.endDate;
    if (query?.supplementId) params.supplementId = query.supplementId;
    if (query?.userSupplementId) params.userSupplementId = query.userSupplementId;

    const response = await api.get<SupplementLog[]>('/supplement-logs', { params });
    return response.data;
  },

  /**
   * Get logs for a specific supplement
   */
  async getBySupplementId(supplementId: string): Promise<SupplementLog[]> {
    return this.getAll({ supplementId });
  },

  /**
   * Get logs for a specific date range
   */
  async getByDateRange(startDate: Date, endDate: Date): Promise<SupplementLog[]> {
    return this.getAll({
      startDate: startDate.toISOString(),
      endDate: endDate.toISOString(),
    });
  },

  /**
   * Get today's logs
   */
  async getToday(): Promise<SupplementLog[]> {
    const today = new Date();
    const startOfDay = new Date(today.getFullYear(), today.getMonth(), today.getDate());
    const endOfDay = new Date(today.getFullYear(), today.getMonth(), today.getDate(), 23, 59, 59);
    return this.getByDateRange(startOfDay, endOfDay);
  },

  /**
   * Get a specific supplement log
   */
  async getById(id: string): Promise<SupplementLog> {
    const response = await api.get<SupplementLog>(`/supplement-logs/${id}`);
    return response.data;
  },

  /**
   * Update a supplement log
   */
  async update(id: string, data: UpdateSupplementLogInput): Promise<SupplementLog> {
    const response = await api.put<SupplementLog>(`/supplement-logs/${id}`, data);
    return response.data;
  },

  /**
   * Delete a supplement log
   */
  async delete(id: string): Promise<{ message: string }> {
    const response = await api.delete<{ message: string }>(`/supplement-logs/${id}`);
    return response.data;
  },

  /**
   * Quick log a supplement (simplified for quick logging)
   */
  async quickLog(supplementId: string, dosage: string, unit: string): Promise<SupplementLog> {
    return this.create({
      supplementId,
      dosage,
      unit,
      takenAt: new Date().toISOString(),
      source: 'QUICK_LOG',
    });
  },

  /**
   * Log a scheduled supplement as taken
   */
  async logScheduled(
    userSupplementId: string,
    supplementId: string,
    dosage: string,
    unit: string,
    scheduledFor?: Date
  ): Promise<SupplementLog> {
    return this.create({
      userSupplementId,
      supplementId,
      dosage,
      unit,
      takenAt: new Date().toISOString(),
      scheduledFor: scheduledFor?.toISOString(),
      source: 'SCHEDULED',
    });
  },
};
