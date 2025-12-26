import api from './client';

// ============================================================================
// TYPES
// ============================================================================

export interface MacroProgress {
  consumed: number;
  goal: number;
  percentage: number;
  rawPercentage: number;
  isOnTrack: boolean;
  isMet: boolean;
}

export interface DailyGoalProgress {
  date: string;
  calories: MacroProgress;
  protein: MacroProgress;
  carbs: MacroProgress;
  fat: MacroProgress;
  goalsMetCount: number;
  allGoalsMet: boolean;
}

export interface GoalStreak {
  currentStreak: number;
  longestStreak: number;
  lastStreakDate: string | null;
}

export interface WeeklyGoalSummary {
  weekStart: string;
  weekEnd: string;
  daysTracked: number;
  daysGoalsMet: number;
  averageProgress: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
  trend: 'improving' | 'declining' | 'stable';
}

export interface MonthlyGoalSummary {
  month: string;
  daysTracked: number;
  daysGoalsMet: number;
  successRate: number;
  averageProgress: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
}

export interface GoalProgressDashboard {
  today: DailyGoalProgress;
  streak: GoalStreak;
  weeklyAverage: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
  };
  weeklyTrend: 'improving' | 'declining' | 'stable';
}

export interface HistoricalProgress {
  daily: DailyGoalProgress[];
  weeklySummaries: WeeklyGoalSummary[];
  monthlySummaries: MonthlyGoalSummary[];
  streak: GoalStreak;
}

// ============================================================================
// API CLIENT
// ============================================================================

export const goalsApi = {
  /**
   * Get dashboard progress data (today + streak + weekly trends)
   * Optimized for home screen display
   */
  async getDashboardProgress(): Promise<GoalProgressDashboard> {
    const response = await api.get<GoalProgressDashboard>('/goals/dashboard');
    return response.data;
  },

  /**
   * Get today's goal progress
   */
  async getTodayProgress(): Promise<DailyGoalProgress> {
    const response = await api.get<DailyGoalProgress>('/goals/progress/today');
    return response.data;
  },

  /**
   * Get goal progress for a specific date
   */
  async getDailyProgress(date?: Date): Promise<DailyGoalProgress> {
    const params = date ? { date: date.toISOString() } : {};
    const response = await api.get<DailyGoalProgress>('/goals/progress/daily', { params });
    return response.data;
  },

  /**
   * Get goal progress history for a date range
   */
  async getProgressHistory(options?: {
    startDate?: Date;
    endDate?: Date;
    days?: number;
  }): Promise<DailyGoalProgress[]> {
    const params: Record<string, string> = {};
    if (options?.startDate) {
      params.startDate = options.startDate.toISOString();
    }
    if (options?.endDate) {
      params.endDate = options.endDate.toISOString();
    }
    if (options?.days) {
      params.days = options.days.toString();
    }
    const response = await api.get<DailyGoalProgress[]>('/goals/progress/history', { params });
    return response.data;
  },

  /**
   * Get current streak information
   */
  async getStreak(): Promise<GoalStreak> {
    const response = await api.get<GoalStreak>('/goals/streak');
    return response.data;
  },

  /**
   * Get weekly summary with trend
   */
  async getWeeklySummary(offset?: number): Promise<WeeklyGoalSummary> {
    const params = offset !== undefined ? { offset: offset.toString() } : {};
    const response = await api.get<WeeklyGoalSummary>('/goals/summary/weekly', { params });
    return response.data;
  },

  /**
   * Get monthly summary
   */
  async getMonthlySummary(offset?: number): Promise<MonthlyGoalSummary> {
    const params = offset !== undefined ? { offset: offset.toString() } : {};
    const response = await api.get<MonthlyGoalSummary>('/goals/summary/monthly', { params });
    return response.data;
  },

  /**
   * Get full historical progress with all summaries
   */
  async getHistoricalProgress(days?: number): Promise<HistoricalProgress> {
    const params = days !== undefined ? { days: days.toString() } : {};
    const response = await api.get<HistoricalProgress>('/goals/historical', { params });
    return response.data;
  },
};
