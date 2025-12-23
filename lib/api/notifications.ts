import api from './client';
import {
  RegisterDeviceInput,
  DeviceInfo,
  NotificationPreferences,
  UpdateNotificationPreferencesInput,
  NotificationHistoryResponse,
  TrackNotificationInput,
} from '../types';

export const notificationsApi = {
  /**
   * Register a device for push notifications
   */
  async registerDevice(data: RegisterDeviceInput): Promise<{ success: boolean; deviceId: string }> {
    const response = await api.post<{ success: boolean; deviceId: string }>(
      '/notifications/register-device',
      data
    );
    return response.data;
  },

  /**
   * Unregister a device from push notifications
   */
  async unregisterDevice(token: string): Promise<{ success: boolean }> {
    const response = await api.delete<{ success: boolean }>('/notifications/unregister-device', {
      data: { token },
    });
    return response.data;
  },

  /**
   * Get all registered devices
   */
  async getDevices(): Promise<DeviceInfo[]> {
    const response = await api.get<DeviceInfo[]>('/notifications/devices');
    return response.data;
  },

  /**
   * Get notification preferences
   */
  async getPreferences(): Promise<NotificationPreferences> {
    const response = await api.get<NotificationPreferences>('/notifications/preferences');
    return response.data;
  },

  /**
   * Update notification preferences
   */
  async updatePreferences(data: UpdateNotificationPreferencesInput): Promise<NotificationPreferences> {
    const response = await api.put<NotificationPreferences>('/notifications/preferences', data);
    return response.data;
  },

  /**
   * Get notification history
   */
  async getHistory(params?: {
    category?: string;
    status?: string;
    startDate?: string;
    endDate?: string;
    page?: number;
    limit?: number;
  }): Promise<NotificationHistoryResponse> {
    const response = await api.get<NotificationHistoryResponse>('/notifications/history', {
      params,
    });
    return response.data;
  },

  /**
   * Track notification delivery or open
   */
  async trackNotification(data: TrackNotificationInput): Promise<{ success: boolean }> {
    const response = await api.post<{ success: boolean }>('/notifications/track', data);
    return response.data;
  },

  /**
   * Send a test notification (development only)
   */
  async sendTestNotification(options?: {
    title?: string;
    body?: string;
    category?: string;
  }): Promise<{ success: boolean; sentTo: number; failed: number }> {
    const response = await api.post<{ success: boolean; sentTo: number; failed: number }>(
      '/notifications/test',
      options || {}
    );
    return response.data;
  },
};
