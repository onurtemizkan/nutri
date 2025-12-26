/**
 * Reports API Client
 * Fetch weekly and monthly nutrition reports with insights
 */

import api from './client';
import {
  WeeklyReport,
  MonthlyReport,
  ReportExportFormat,
  ReportExportResult,
} from '../types/reports';

/**
 * Reports API client
 */
export const reportsApi = {
  /**
   * Get weekly nutrition report
   * GET /reports/weekly
   * @param date Optional date within the desired week (YYYY-MM-DD). Defaults to current week.
   */
  async getWeeklyReport(date?: string): Promise<WeeklyReport> {
    const params: Record<string, string> = {};
    if (date) params.date = date;

    const response = await api.get<WeeklyReport>('/reports/weekly', { params });
    return response.data;
  },

  /**
   * Get monthly nutrition report
   * GET /reports/monthly
   * @param month Optional month (YYYY-MM). Defaults to current month.
   */
  async getMonthlyReport(month?: string): Promise<MonthlyReport> {
    const params: Record<string, string> = {};
    if (month) params.month = month;

    const response = await api.get<MonthlyReport>('/reports/monthly', { params });
    return response.data;
  },

  /**
   * Export weekly report in specified format
   * GET /reports/weekly/export
   * @param format Export format: 'pdf', 'image', or 'json'
   * @param date Optional date within the desired week (YYYY-MM-DD)
   */
  async exportWeeklyReport(format: ReportExportFormat, date?: string): Promise<ReportExportResult> {
    const params: Record<string, string> = { format };
    if (date) params.date = date;

    const response = await api.get<ReportExportResult>('/reports/weekly/export', { params });
    return response.data;
  },

  /**
   * Export monthly report in specified format
   * GET /reports/monthly/export
   * @param format Export format: 'pdf', 'image', or 'json'
   * @param month Optional month (YYYY-MM)
   */
  async exportMonthlyReport(
    format: ReportExportFormat,
    month?: string
  ): Promise<ReportExportResult> {
    const params: Record<string, string> = { format };
    if (month) params.month = month;

    const response = await api.get<ReportExportResult>('/reports/monthly/export', { params });
    return response.data;
  },

  /**
   * Get the week date range for a given date
   * Useful for navigation UI
   * @param date Any date
   * @returns Start and end dates of the week containing the given date
   */
  getWeekRange(date: Date = new Date()): {
    start: Date;
    end: Date;
    startString: string;
    endString: string;
  } {
    const dayOfWeek = date.getDay();
    const start = new Date(date);
    start.setDate(date.getDate() - dayOfWeek);
    start.setHours(0, 0, 0, 0);

    const end = new Date(start);
    end.setDate(start.getDate() + 6);
    end.setHours(23, 59, 59, 999);

    return {
      start,
      end,
      startString: start.toISOString().split('T')[0],
      endString: end.toISOString().split('T')[0],
    };
  },

  /**
   * Get the month date range for a given month string
   * Useful for navigation UI
   * @param month Month in YYYY-MM format
   * @returns Start and end dates of the month
   */
  getMonthRange(month?: string): { start: Date; end: Date; monthString: string } {
    let year: number;
    let monthIndex: number;

    if (month) {
      const [y, m] = month.split('-').map(Number);
      year = y;
      monthIndex = m - 1;
    } else {
      const now = new Date();
      year = now.getFullYear();
      monthIndex = now.getMonth();
    }

    const start = new Date(year, monthIndex, 1, 0, 0, 0, 0);
    const end = new Date(year, monthIndex + 1, 0, 23, 59, 59, 999);
    const monthString = `${year}-${String(monthIndex + 1).padStart(2, '0')}`;

    return { start, end, monthString };
  },

  /**
   * Navigate to previous week
   * @param currentDate Current week's date
   * @returns Date string for previous week (YYYY-MM-DD)
   */
  getPreviousWeekDate(currentDate: string): string {
    const date = new Date(currentDate);
    date.setDate(date.getDate() - 7);
    return date.toISOString().split('T')[0];
  },

  /**
   * Navigate to next week
   * @param currentDate Current week's date
   * @returns Date string for next week (YYYY-MM-DD)
   */
  getNextWeekDate(currentDate: string): string {
    const date = new Date(currentDate);
    date.setDate(date.getDate() + 7);
    return date.toISOString().split('T')[0];
  },

  /**
   * Navigate to previous month
   * @param currentMonth Current month (YYYY-MM)
   * @returns Month string for previous month (YYYY-MM)
   */
  getPreviousMonth(currentMonth: string): string {
    const [year, month] = currentMonth.split('-').map(Number);
    const date = new Date(year, month - 2, 1);
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
  },

  /**
   * Navigate to next month
   * @param currentMonth Current month (YYYY-MM)
   * @returns Month string for next month (YYYY-MM)
   */
  getNextMonth(currentMonth: string): string {
    const [year, month] = currentMonth.split('-').map(Number);
    const date = new Date(year, month, 1);
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
  },

  /**
   * Format date range for display
   * @param startDate Start date
   * @param endDate End date
   * @returns Formatted string like "Dec 22 - Dec 28, 2024"
   */
  formatDateRange(startDate: string | Date, endDate: string | Date): string {
    const start = typeof startDate === 'string' ? new Date(startDate) : startDate;
    const end = typeof endDate === 'string' ? new Date(endDate) : endDate;

    const options: Intl.DateTimeFormatOptions = { month: 'short', day: 'numeric' };
    const startStr = start.toLocaleDateString('en-US', options);
    const endStr = end.toLocaleDateString('en-US', { ...options, year: 'numeric' });

    return `${startStr} - ${endStr}`;
  },

  /**
   * Format month for display
   * @param month Month in YYYY-MM format
   * @returns Formatted string like "December 2024"
   */
  formatMonth(month: string): string {
    const [year, m] = month.split('-').map(Number);
    const date = new Date(year, m - 1, 1);
    return date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
  },

  /**
   * Check if a week is the current week
   * @param date Date within the week to check
   */
  isCurrentWeek(date: string | Date): boolean {
    const checkDate = typeof date === 'string' ? new Date(date) : date;
    const today = new Date();

    const { startString } = this.getWeekRange(checkDate);
    const { startString: currentWeekStart } = this.getWeekRange(today);

    return startString === currentWeekStart;
  },

  /**
   * Check if a month is the current month
   * @param month Month to check (YYYY-MM)
   */
  isCurrentMonth(month: string): boolean {
    const today = new Date();
    const currentMonth = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}`;
    return month === currentMonth;
  },
};

export default reportsApi;
