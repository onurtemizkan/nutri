import api from './client';
import {
  Supplement,
  SupplementLog,
  CreateSupplementInput,
  UpdateSupplementInput,
  CreateSupplementLogInput,
  TodaySupplementStatus,
} from '../types';

export const supplementsApi = {
  /**
   * Create a new supplement
   */
  async create(data: CreateSupplementInput): Promise<Supplement> {
    const response = await api.post<Supplement>('/supplements', data);
    return response.data;
  },

  /**
   * Get all supplements
   */
  async getAll(activeOnly: boolean = false): Promise<Supplement[]> {
    const response = await api.get<Supplement[]>('/supplements', {
      params: { activeOnly },
    });
    return response.data;
  },

  /**
   * Get a single supplement by ID
   */
  async getById(id: string): Promise<Supplement & { logs: SupplementLog[] }> {
    const response = await api.get<Supplement & { logs: SupplementLog[] }>(`/supplements/${id}`);
    return response.data;
  },

  /**
   * Update a supplement
   */
  async update(id: string, data: UpdateSupplementInput): Promise<Supplement> {
    const response = await api.put<Supplement>(`/supplements/${id}`, data);
    return response.data;
  },

  /**
   * Delete a supplement
   */
  async delete(id: string): Promise<{ message: string }> {
    const response = await api.delete<{ message: string }>(`/supplements/${id}`);
    return response.data;
  },

  /**
   * Log a supplement intake
   */
  async logIntake(data: CreateSupplementLogInput): Promise<SupplementLog> {
    const response = await api.post<SupplementLog>('/supplements/logs', data);
    return response.data;
  },

  /**
   * Bulk log supplement intakes
   */
  async bulkLogIntake(logs: CreateSupplementLogInput[]): Promise<{ count: number }> {
    const response = await api.post<{ count: number }>('/supplements/logs/bulk', { logs });
    return response.data;
  },

  /**
   * Get logs for a specific day
   */
  async getLogsForDay(date?: string): Promise<SupplementLog[]> {
    const response = await api.get<SupplementLog[]>('/supplements/logs/day', {
      params: { date },
    });
    return response.data;
  },

  /**
   * Get today's supplement status
   */
  async getTodayStatus(): Promise<TodaySupplementStatus> {
    const response = await api.get<TodaySupplementStatus>('/supplements/today');
    return response.data;
  },

  /**
   * Delete a supplement log
   */
  async deleteLog(logId: string): Promise<{ message: string }> {
    const response = await api.delete<{ message: string }>(`/supplements/logs/${logId}`);
    return response.data;
  },

  /**
   * Get supplement history/streak
   */
  async getHistory(
    supplementId: string,
    days: number = 30
  ): Promise<{
    supplement: Supplement;
    days: number;
    logs: SupplementLog[];
    dailyStats: Record<string, { taken: number; skipped: number }>;
    currentStreak: number;
    totalTaken: number;
    totalSkipped: number;
  }> {
    const response = await api.get(`/supplements/${supplementId}/history`, {
      params: { days },
    });
    return response.data;
  },
};
