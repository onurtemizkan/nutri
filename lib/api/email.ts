/**
 * Email Preferences API Client
 *
 * Handles email preference management:
 * - Get/update preferences
 * - Double opt-in management
 * - Resubscribe functionality
 */

import api from './client';

/**
 * Email category preferences
 */
export interface EmailCategories {
  weekly_reports: boolean;
  health_insights: boolean;
  tips: boolean;
  features: boolean;
  promotions: boolean;
  newsletter: boolean;
}

/**
 * Email frequency options
 */
export type EmailFrequency = 'REALTIME' | 'DAILY_DIGEST' | 'WEEKLY_DIGEST';

/**
 * Email preferences response from API
 */
export interface EmailPreferences {
  categories: EmailCategories;
  frequency: EmailFrequency;
  marketingOptIn: boolean;
  doubleOptInConfirmed: boolean;
  globalUnsubscribed: boolean;
  availableCategories: {
    id: string;
    name: string;
    description: string;
  }[];
}

/**
 * Update preferences request
 */
export interface UpdateEmailPreferencesRequest {
  categories?: Partial<EmailCategories>;
  frequency?: EmailFrequency;
  marketingOptIn?: boolean;
}

/**
 * Email Preferences API
 */
export const emailApi = {
  /**
   * Get current user's email preferences
   */
  async getPreferences(): Promise<EmailPreferences> {
    const response = await api.get<EmailPreferences>('/email/preferences');
    return response.data;
  },

  /**
   * Update email preferences
   */
  async updatePreferences(data: UpdateEmailPreferencesRequest): Promise<EmailPreferences> {
    const response = await api.put<EmailPreferences>('/email/preferences', data);
    return response.data;
  },

  /**
   * Request double opt-in confirmation email
   */
  async requestDoubleOptIn(): Promise<{ message: string }> {
    const response = await api.post<{ message: string }>('/email/opt-in');
    return response.data;
  },

  /**
   * Resubscribe to marketing emails (for users who previously unsubscribed)
   */
  async resubscribe(): Promise<{ message: string }> {
    const response = await api.post<{ message: string }>('/email/resubscribe');
    return response.data;
  },
};
