import api from './client';
import {
  WaterIntake,
  CreateWaterIntakeInput,
  UpdateWaterIntakeInput,
  QuickAddWaterInput,
  WaterDailySummary,
  WaterWeeklySummary,
  WaterGoal,
} from '../types';

export const waterApi = {
  /**
   * Create a new water intake record
   */
  async createWaterIntake(data: CreateWaterIntakeInput): Promise<WaterIntake> {
    const response = await api.post<WaterIntake>('/water', data);
    return response.data;
  },

  /**
   * Quick add water using preset amounts
   */
  async quickAddWater(data: QuickAddWaterInput): Promise<WaterIntake> {
    const response = await api.post<WaterIntake>('/water/quick-add', data);
    return response.data;
  },

  /**
   * Get water intakes for a specific day
   */
  async getWaterIntakes(date?: Date): Promise<WaterIntake[]> {
    const params = date ? { date: date.toISOString() } : {};
    const response = await api.get<WaterIntake[]>('/water', { params });
    return response.data;
  },

  /**
   * Get a single water intake by ID
   */
  async getWaterIntakeById(id: string): Promise<WaterIntake> {
    const response = await api.get<WaterIntake>(`/water/${id}`);
    return response.data;
  },

  /**
   * Update a water intake record
   */
  async updateWaterIntake(id: string, data: UpdateWaterIntakeInput): Promise<WaterIntake> {
    const response = await api.put<WaterIntake>(`/water/${id}`, data);
    return response.data;
  },

  /**
   * Delete a water intake record
   */
  async deleteWaterIntake(id: string): Promise<void> {
    await api.delete(`/water/${id}`);
  },

  /**
   * Get daily water summary
   */
  async getDailySummary(date?: Date): Promise<WaterDailySummary> {
    const params = date ? { date: date.toISOString() } : {};
    const response = await api.get<WaterDailySummary>('/water/summary/daily', { params });
    return response.data;
  },

  /**
   * Get weekly water summary
   */
  async getWeeklySummary(): Promise<WaterWeeklySummary> {
    const response = await api.get<WaterWeeklySummary>('/water/summary/weekly');
    return response.data;
  },

  /**
   * Get user's water goal
   */
  async getWaterGoal(): Promise<WaterGoal> {
    const response = await api.get<WaterGoal>('/water/goal');
    return response.data;
  },

  /**
   * Update user's water goal
   */
  async updateWaterGoal(goalWater: number): Promise<WaterGoal> {
    const response = await api.put<WaterGoal>('/water/goal', { goalWater });
    return response.data;
  },
};
