/**
 * Weight Tracking API Client
 * API functions for weight tracking including CRUD, trends, and progress
 */

import api from './client';
import {
  WeightRecord,
  CreateWeightRecordInput,
  UpdateWeightRecordInput,
  WeightTrendsResult,
  WeightProgressResult,
  WeightSummary,
  GetWeightRecordsParams,
  GetWeightTrendsParams,
} from '../types/weight';

// Re-export types for convenience
export type {
  WeightRecord,
  CreateWeightRecordInput,
  UpdateWeightRecordInput,
  WeightTrendsResult,
  WeightProgressResult,
  WeightSummary,
};

/**
 * Weight API client
 */
export const weightApi = {
  /**
   * Create a new weight record
   * POST /weight
   */
  async create(data: CreateWeightRecordInput): Promise<WeightRecord> {
    const response = await api.post<WeightRecord>('/weight', data);
    return response.data;
  },

  /**
   * Get all weight records with optional filters
   * GET /weight
   */
  async getAll(params?: GetWeightRecordsParams): Promise<WeightRecord[]> {
    const response = await api.get<WeightRecord[]>('/weight', { params });
    return response.data;
  },

  /**
   * Get a specific weight record by ID
   * GET /weight/:id
   */
  async getById(id: string): Promise<WeightRecord> {
    const response = await api.get<WeightRecord>(`/weight/${id}`);
    return response.data;
  },

  /**
   * Get weight records for a specific day
   * GET /weight/day
   */
  async getForDay(date?: string): Promise<WeightRecord[]> {
    const params: Record<string, string> = {};
    if (date) params.date = date;
    const response = await api.get<WeightRecord[]>('/weight/day', { params });
    return response.data;
  },

  /**
   * Update a weight record
   * PUT /weight/:id
   */
  async update(id: string, data: UpdateWeightRecordInput): Promise<WeightRecord> {
    const response = await api.put<WeightRecord>(`/weight/${id}`, data);
    return response.data;
  },

  /**
   * Delete a weight record
   * DELETE /weight/:id
   */
  async delete(id: string): Promise<void> {
    await api.delete(`/weight/${id}`);
  },

  /**
   * Get weight trends with moving averages
   * GET /weight/trends
   */
  async getTrends(params?: GetWeightTrendsParams): Promise<WeightTrendsResult> {
    const response = await api.get<WeightTrendsResult>('/weight/trends', { params });
    return response.data;
  },

  /**
   * Get weight progress towards goal
   * GET /weight/progress
   */
  async getProgress(): Promise<WeightProgressResult> {
    const response = await api.get<WeightProgressResult>('/weight/progress');
    return response.data;
  },

  /**
   * Get weight summary for dashboard widget
   * GET /weight/summary
   */
  async getSummary(): Promise<WeightSummary> {
    const response = await api.get<WeightSummary>('/weight/summary');
    return response.data;
  },

  /**
   * Update goal weight
   * PUT /weight/goal
   */
  async updateGoal(
    goalWeight: number
  ): Promise<{ id: string; goalWeight: number; currentWeight: number | null }> {
    const response = await api.put<{
      id: string;
      goalWeight: number;
      currentWeight: number | null;
    }>('/weight/goal', { goalWeight });
    return response.data;
  },

  /**
   * Get the latest weight record
   * Convenience method that fetches the most recent record
   */
  async getLatest(): Promise<WeightRecord | null> {
    try {
      const records = await this.getAll({ limit: 1 });
      return records.length > 0 ? records[0] : null;
    } catch (error) {
      console.warn('Failed to fetch latest weight record:', error);
      return null;
    }
  },

  /**
   * Get weight records for the past N days
   * Convenience method for charts
   */
  async getForPastDays(days: number): Promise<WeightRecord[]> {
    const endDate = new Date().toISOString();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    return this.getAll({
      startDate: startDate.toISOString(),
      endDate,
      limit: 1000, // Get all records in the period
    });
  },
};

export default weightApi;
