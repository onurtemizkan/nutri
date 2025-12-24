// Push Notification Types

export type DevicePlatform = 'IOS' | 'ANDROID';

export type NotificationCategory =
  | 'MEAL_REMINDER'
  | 'GOAL_PROGRESS'
  | 'HEALTH_INSIGHT'
  | 'SUPPLEMENT_REMINDER'
  | 'STREAK_ALERT'
  | 'WEEKLY_SUMMARY'
  | 'MARKETING'
  | 'SYSTEM';

export type NotificationStatus = 'PENDING' | 'SENT' | 'DELIVERED' | 'OPENED' | 'FAILED';

export interface RegisterDeviceInput {
  token: string;
  platform: DevicePlatform;
  expoPushToken?: string;
  deviceModel?: string;
  osVersion?: string;
  appVersion?: string;
}

export interface DeviceInfo {
  id: string;
  platform: DevicePlatform;
  deviceModel?: string;
  osVersion?: string;
  appVersion?: string;
  lastActiveAt: string;
  createdAt: string;
}

export interface NotificationPreferences {
  enabled: boolean;
  enabledCategories: NotificationCategory[];
  quietHoursEnabled: boolean;
  quietHoursStart?: string;
  quietHoursEnd?: string;
  mealReminderTimes?: {
    breakfast?: string;
    lunch?: string;
    dinner?: string;
    snack?: string;
  };
  settings?: Record<string, unknown>;
}

export interface UpdateNotificationPreferencesInput {
  enabled?: boolean;
  enabledCategories?: NotificationCategory[];
  quietHoursEnabled?: boolean;
  quietHoursStart?: string | null;
  quietHoursEnd?: string | null;
  mealReminderTimes?: {
    breakfast?: string | null;
    lunch?: string | null;
    dinner?: string | null;
    snack?: string | null;
  };
  settings?: Record<string, unknown>;
}

export interface NotificationLogItem {
  id: string;
  category: NotificationCategory;
  title: string;
  body: string;
  platform: DevicePlatform;
  status: NotificationStatus;
  sentAt: string;
  deliveredAt?: string;
  openedAt?: string;
  actionTaken?: string;
}

export interface NotificationHistoryResponse {
  data: NotificationLogItem[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export interface TrackNotificationInput {
  notificationLogId: string;
  action: 'delivered' | 'opened';
  actionTaken?: string;
}

// Notification data payload structure
export interface NotificationData {
  category?: NotificationCategory;
  screen?: string;
  params?: Record<string, string | number>;
  notificationLogId?: string;
  [key: string]: unknown;
}
