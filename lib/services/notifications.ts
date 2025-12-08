/**
 * Notification Service
 * Handles push notification permissions and scheduling
 */

import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import { Platform } from 'react-native';

// Configure notification behavior
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

export interface NotificationPermissionResult {
  granted: boolean;
  canAskAgain: boolean;
}

export const notificationService = {
  /**
   * Request notification permissions
   */
  async requestPermissions(): Promise<NotificationPermissionResult> {
    if (Platform.OS === 'web') {
      return { granted: false, canAskAgain: false };
    }

    if (!Device.isDevice) {
      console.log('Push notifications are only supported on physical devices');
      return { granted: false, canAskAgain: false };
    }

    const { status: existingStatus } = await Notifications.getPermissionsAsync();

    if (existingStatus === 'granted') {
      return { granted: true, canAskAgain: false };
    }

    const { status, canAskAgain } = await Notifications.requestPermissionsAsync();

    return {
      granted: status === 'granted',
      canAskAgain: canAskAgain ?? false,
    };
  },

  /**
   * Check current permission status
   */
  async getPermissionStatus(): Promise<NotificationPermissionResult> {
    if (Platform.OS === 'web' || !Device.isDevice) {
      return { granted: false, canAskAgain: false };
    }

    const { status, canAskAgain } = await Notifications.getPermissionsAsync();

    return {
      granted: status === 'granted',
      canAskAgain: canAskAgain ?? false,
    };
  },

  /**
   * Get push notification token (for remote notifications)
   */
  async getExpoPushToken(): Promise<string | null> {
    if (Platform.OS === 'web' || !Device.isDevice) {
      return null;
    }

    try {
      const token = await Notifications.getExpoPushTokenAsync();
      return token.data;
    } catch (error) {
      console.error('Error getting push token:', error);
      return null;
    }
  },

  /**
   * Schedule a local notification
   */
  async scheduleNotification(options: {
    title: string;
    body: string;
    data?: Record<string, unknown>;
    trigger?: Notifications.NotificationTriggerInput;
  }): Promise<string> {
    return await Notifications.scheduleNotificationAsync({
      content: {
        title: options.title,
        body: options.body,
        data: options.data,
      },
      trigger: options.trigger || null, // null = immediate
    });
  },

  /**
   * Cancel a scheduled notification
   */
  async cancelNotification(identifier: string): Promise<void> {
    await Notifications.cancelScheduledNotificationAsync(identifier);
  },

  /**
   * Cancel all scheduled notifications
   */
  async cancelAllNotifications(): Promise<void> {
    await Notifications.cancelAllScheduledNotificationsAsync();
  },

  /**
   * Add a notification listener
   */
  addNotificationReceivedListener(
    callback: (notification: Notifications.Notification) => void
  ): Notifications.Subscription {
    return Notifications.addNotificationReceivedListener(callback);
  },

  /**
   * Add a notification response listener (when user taps notification)
   */
  addNotificationResponseReceivedListener(
    callback: (response: Notifications.NotificationResponse) => void
  ): Notifications.Subscription {
    return Notifications.addNotificationResponseReceivedListener(callback);
  },
};
