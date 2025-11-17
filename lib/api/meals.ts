import api from './client';
import { Meal, CreateMealInput, DailySummary, WeeklySummary } from '../types';

export const mealsApi = {
  async createMeal(data: CreateMealInput): Promise<Meal> {
    const response = await api.post<Meal>('/meals', data);
    return response.data;
  },

  async getMeals(date?: Date): Promise<Meal[]> {
    const params = date ? { date: date.toISOString() } : {};
    const response = await api.get<Meal[]>('/meals', { params });
    return response.data;
  },

  async getMealById(id: string): Promise<Meal> {
    const response = await api.get<Meal>(`/meals/${id}`);
    return response.data;
  },

  async updateMeal(id: string, data: Partial<CreateMealInput>): Promise<Meal> {
    const response = await api.put<Meal>(`/meals/${id}`, data);
    return response.data;
  },

  async deleteMeal(id: string): Promise<void> {
    await api.delete(`/meals/${id}`);
  },

  async getDailySummary(date?: Date): Promise<DailySummary> {
    const params = date ? { date: date.toISOString() } : {};
    const response = await api.get<DailySummary>('/meals/summary/daily', { params });
    return response.data;
  },

  async getWeeklySummary(): Promise<WeeklySummary[]> {
    const response = await api.get<WeeklySummary[]>('/meals/summary/weekly');
    return response.data;
  },
};
