import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import { Platform } from 'react-native';
import Constants from 'expo-constants';
import { notificationsApi } from '../../api/notifications';
import { DevicePlatform, NotificationCategory, NotificationData } from '../../types';

// Configure how notifications are handled when app is in foreground
Notifications.setNotificationHandler({
  handleNotification: async (notification) => {
    // Check notification category/priority
    const data = notification.request.content.data as NotificationData | undefined;
    const category = data?.category;

    // Time-sensitive notifications (streak alerts, supplement reminders) should always show
    const isTimeSensitive =
      category === 'STREAK_ALERT' ||
      category === 'SUPPLEMENT_REMINDER' ||
      category === 'MEAL_REMINDER';

    return {
      shouldShowAlert: true,
      shouldPlaySound: isTimeSensitive, // Only play sound for important notifications
      shouldSetBadge: false,
      shouldShowBanner: true,
      shouldShowList: true,
      // iOS: Set priority for time-sensitive content
      priority: isTimeSensitive
        ? Notifications.AndroidNotificationPriority.HIGH
        : Notifications.AndroidNotificationPriority.DEFAULT,
    };
  },
});

/**
 * Get the Expo push token for the device
 * Returns null if the device cannot receive push notifications
 */
export async function getExpoPushToken(): Promise<string | null> {
  // Push notifications only work on physical devices
  if (!Device.isDevice) {
    console.warn('Push notifications require a physical device');
    return null;
  }

  try {
    // Get the project ID from the config
    const projectId = Constants.expoConfig?.extra?.eas?.projectId;

    if (!projectId) {
      console.warn('No project ID found in app.config.js for push notifications');
      return null;
    }

    const { data: token } = await Notifications.getExpoPushTokenAsync({
      projectId,
    });

    return token;
  } catch (error) {
    console.error('Error getting Expo push token:', error);
    return null;
  }
}

/**
 * Get the native device token (APNs or FCM)
 */
export async function getDevicePushToken(): Promise<{
  token: string;
  type: 'ios' | 'android';
} | null> {
  if (!Device.isDevice) {
    return null;
  }

  try {
    const tokenData = await Notifications.getDevicePushTokenAsync();
    // The type is typed as a union, so we need to map it
    const type = tokenData.type === 'ios' ? 'ios' : 'android';
    return {
      token: tokenData.data,
      type,
    };
  } catch (error) {
    console.error('Error getting device push token:', error);
    return null;
  }
}

/**
 * Get current notification permission status
 */
export async function getNotificationPermissionStatus(): Promise<
  'granted' | 'denied' | 'undetermined'
> {
  const { status } = await Notifications.getPermissionsAsync();
  return status;
}

/**
 * Request notification permissions
 * Returns the new permission status
 */
export async function requestNotificationPermissions(): Promise<
  'granted' | 'denied' | 'undetermined'
> {
  // Check current status first
  const { status: existingStatus } = await Notifications.getPermissionsAsync();

  if (existingStatus === 'granted') {
    return 'granted';
  }

  // Request permissions
  const { status } = await Notifications.requestPermissionsAsync({
    ios: {
      allowAlert: true,
      allowBadge: true,
      allowSound: true,
      // Enable provisional authorization (silent notifications first)
      allowProvisional: true,
    },
    android: {
      // Android handles permissions differently
    },
  });

  return status;
}

/**
 * Register device for push notifications with the backend
 * This should be called after permissions are granted and token is obtained
 */
export async function registerDeviceForPushNotifications(): Promise<{
  success: boolean;
  deviceId?: string;
  error?: string;
}> {
  try {
    // Get tokens
    const expoPushToken = await getExpoPushToken();
    const deviceToken = await getDevicePushToken();

    if (!deviceToken) {
      return {
        success: false,
        error: 'Could not get device token. Push notifications require a physical device.',
      };
    }

    // Get device info
    const deviceModel = Device.modelName ?? undefined;
    const osVersion = Device.osVersion ?? undefined;
    const appVersion = Constants.expoConfig?.version ?? undefined;

    // Register with backend
    const result = await notificationsApi.registerDevice({
      token: deviceToken.token,
      platform: (Platform.OS === 'ios' ? 'IOS' : 'ANDROID') as DevicePlatform,
      expoPushToken: expoPushToken ?? undefined,
      deviceModel,
      osVersion,
      appVersion,
    });

    return {
      success: result.success,
      deviceId: result.deviceId,
    };
  } catch (error) {
    console.error('Error registering device for push notifications:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Unregister device from push notifications
 */
export async function unregisterDeviceFromPushNotifications(): Promise<boolean> {
  try {
    const deviceToken = await getDevicePushToken();
    if (!deviceToken) {
      return false;
    }

    await notificationsApi.unregisterDevice(deviceToken.token);
    return true;
  } catch (error) {
    console.error('Error unregistering device from push notifications:', error);
    return false;
  }
}

/**
 * Setup notification categories with action buttons (iOS)
 */
export async function setupNotificationCategories(): Promise<void> {
  // Define notification categories with action buttons
  await Notifications.setNotificationCategoryAsync('MEAL_REMINDER', [
    {
      identifier: 'LOG_NOW',
      buttonTitle: 'Log Meal',
      options: {
        opensAppToForeground: true,
      },
    },
    {
      identifier: 'SNOOZE',
      buttonTitle: 'Remind Later',
      options: {
        opensAppToForeground: false,
      },
    },
  ]);

  await Notifications.setNotificationCategoryAsync('SUPPLEMENT_REMINDER', [
    {
      identifier: 'MARK_TAKEN',
      buttonTitle: 'Mark Taken',
      options: {
        opensAppToForeground: false,
      },
    },
    {
      identifier: 'SNOOZE',
      buttonTitle: 'Remind Later',
      options: {
        opensAppToForeground: false,
      },
    },
  ]);

  await Notifications.setNotificationCategoryAsync('STREAK_ALERT', [
    {
      identifier: 'LOG_NOW',
      buttonTitle: 'Log Now',
      options: {
        opensAppToForeground: true,
      },
    },
    {
      identifier: 'DISMISS',
      buttonTitle: 'Dismiss',
      options: {
        opensAppToForeground: false,
        isDestructive: true,
      },
    },
  ]);

  await Notifications.setNotificationCategoryAsync('HEALTH_INSIGHT', [
    {
      identifier: 'VIEW',
      buttonTitle: 'View Details',
      options: {
        opensAppToForeground: true,
      },
    },
    {
      identifier: 'DISMISS',
      buttonTitle: 'Dismiss',
      options: {
        opensAppToForeground: false,
      },
    },
  ]);
}

/**
 * Setup Android notification channels
 * Android 8.0+ requires notification channels
 */
export async function setupAndroidNotificationChannels(): Promise<void> {
  if (Platform.OS !== 'android') {
    return;
  }

  // Meal reminders channel
  await Notifications.setNotificationChannelAsync('meal_reminder', {
    name: 'Meal Reminders',
    importance: Notifications.AndroidImportance.HIGH,
    vibrationPattern: [0, 250, 250, 250],
    lightColor: '#4CAF50',
    sound: 'default',
  });

  // Supplement reminders channel
  await Notifications.setNotificationChannelAsync('supplement_reminder', {
    name: 'Supplement Reminders',
    importance: Notifications.AndroidImportance.HIGH,
    vibrationPattern: [0, 250, 250, 250],
    lightColor: '#2196F3',
    sound: 'default',
  });

  // Health insights channel
  await Notifications.setNotificationChannelAsync('health_insight', {
    name: 'Health Insights',
    importance: Notifications.AndroidImportance.DEFAULT,
    lightColor: '#9C27B0',
    sound: 'default',
  });

  // Goal progress channel
  await Notifications.setNotificationChannelAsync('goal_progress', {
    name: 'Goal Progress',
    importance: Notifications.AndroidImportance.DEFAULT,
    lightColor: '#FF9800',
    sound: 'default',
  });

  // Streak alerts channel
  await Notifications.setNotificationChannelAsync('streak_alert', {
    name: 'Streak Alerts',
    importance: Notifications.AndroidImportance.HIGH,
    vibrationPattern: [0, 250, 250, 250],
    lightColor: '#F44336',
    sound: 'default',
  });

  // Weekly summary channel
  await Notifications.setNotificationChannelAsync('weekly_summary', {
    name: 'Weekly Summary',
    importance: Notifications.AndroidImportance.LOW,
    lightColor: '#607D8B',
    sound: 'default',
  });

  // System notifications channel
  await Notifications.setNotificationChannelAsync('system', {
    name: 'System Notifications',
    importance: Notifications.AndroidImportance.DEFAULT,
    sound: 'default',
  });

  // Marketing channel (opt-in)
  await Notifications.setNotificationChannelAsync('marketing', {
    name: 'Tips & Updates',
    importance: Notifications.AndroidImportance.LOW,
    sound: 'default',
  });
}

/**
 * Initialize all notification settings
 * Should be called when the app starts
 */
export async function initializeNotifications(): Promise<void> {
  // Setup categories and channels
  await Promise.all([
    setupNotificationCategories(),
    setupAndroidNotificationChannels(),
  ]);

  // Clear badge count on app launch
  await Notifications.setBadgeCountAsync(0);
}

/**
 * Schedule a local test notification (development only)
 */
export async function scheduleTestNotification(): Promise<string | null> {
  if (!__DEV__) {
    console.warn('Test notifications are only available in development mode');
    return null;
  }

  const identifier = await Notifications.scheduleNotificationAsync({
    content: {
      title: 'Test Notification',
      body: 'This is a test notification from Nutri',
      data: { test: true },
      sound: true,
    },
    trigger: {
      type: Notifications.SchedulableTriggerInputTypes.TIME_INTERVAL,
      seconds: 2,
    },
  });

  return identifier;
}

/**
 * Cancel all scheduled notifications
 */
export async function cancelAllScheduledNotifications(): Promise<void> {
  await Notifications.cancelAllScheduledNotificationsAsync();
}

/**
 * Get badge count
 */
export async function getBadgeCount(): Promise<number> {
  return await Notifications.getBadgeCountAsync();
}

/**
 * Set badge count
 */
export async function setBadgeCount(count: number): Promise<void> {
  await Notifications.setBadgeCountAsync(count);
}

/**
 * Dismiss all delivered notifications
 */
export async function dismissAllNotifications(): Promise<void> {
  await Notifications.dismissAllNotificationsAsync();
}

// =============================================================================
// iOS-Specific Features
// =============================================================================

/**
 * iOS permission status details
 */
export interface IOSPermissionDetails {
  status: 'granted' | 'denied' | 'undetermined';
  isProvisional: boolean;
  allowsAlert: boolean;
  allowsSound: boolean;
  allowsBadge: boolean;
  allowsAnnouncement: boolean;
  providesAppNotificationSettings: boolean;
  allowsCriticalAlerts: boolean;
}

/**
 * Get detailed iOS notification permission status
 * Includes info about provisional authorization and individual permission types
 */
export async function getIOSPermissionDetails(): Promise<IOSPermissionDetails | null> {
  if (Platform.OS !== 'ios') {
    return null;
  }

  const permissions = await Notifications.getPermissionsAsync();

  return {
    status: permissions.status,
    isProvisional: permissions.ios?.status === Notifications.IosAuthorizationStatus.PROVISIONAL,
    allowsAlert: permissions.ios?.allowsAlert ?? false,
    allowsSound: permissions.ios?.allowsSound ?? false,
    allowsBadge: permissions.ios?.allowsBadge ?? false,
    allowsAnnouncement: permissions.ios?.allowsAnnouncements ?? false,
    providesAppNotificationSettings: permissions.ios?.providesAppNotificationSettings ?? false,
    allowsCriticalAlerts: permissions.ios?.allowsCriticalAlerts ?? false,
  };
}

/**
 * Check if current permission is provisional
 * Provisional permission means notifications are delivered quietly to Notification Center
 */
export async function isProvisionalPermission(): Promise<boolean> {
  if (Platform.OS !== 'ios') {
    return false;
  }

  const permissions = await Notifications.getPermissionsAsync();
  return permissions.ios?.status === Notifications.IosAuthorizationStatus.PROVISIONAL;
}

/**
 * Request full (non-provisional) notification permissions
 * Use this to upgrade from provisional to full authorization
 * The user will see the standard iOS permission dialog
 */
export async function requestFullNotificationPermissions(): Promise<
  'granted' | 'denied' | 'undetermined'
> {
  // Request without provisional flag to get full permissions
  const { status } = await Notifications.requestPermissionsAsync({
    ios: {
      allowAlert: true,
      allowBadge: true,
      allowSound: true,
      allowProvisional: false, // Request full authorization
    },
  });

  return status;
}

/**
 * Get notification settings URL for iOS
 * Opens the app's notification settings in Settings app
 */
export function getNotificationSettingsURL(): string | null {
  if (Platform.OS === 'ios') {
    // This URL opens the app's notification settings
    return 'app-settings:';
  }
  return null;
}

/**
 * Schedule a local notification with iOS-specific options
 * Note: interruptionLevel controls notification priority:
 * - 'passive': silent, won't light up screen
 * - 'active': default behavior with sound
 * - 'timeSensitive': higher priority, may break through Focus
 * - 'critical': highest priority (requires entitlement)
 */
export async function scheduleIOSNotification(options: {
  title: string;
  body: string;
  data?: NotificationData;
  trigger?: Notifications.NotificationTriggerInput;
  categoryIdentifier?: string;
  interruptionLevel?: 'passive' | 'active' | 'timeSensitive' | 'critical';
}): Promise<string> {
  const {
    title,
    body,
    data,
    trigger,
    categoryIdentifier,
    interruptionLevel = 'active',
  } = options;

  // Determine sound and priority based on interruption level
  const shouldPlaySound = interruptionLevel !== 'passive';
  const priority =
    interruptionLevel === 'timeSensitive' || interruptionLevel === 'critical'
      ? Notifications.AndroidNotificationPriority.HIGH
      : Notifications.AndroidNotificationPriority.DEFAULT;

  const identifier = await Notifications.scheduleNotificationAsync({
    content: {
      title,
      body,
      data,
      sound: shouldPlaySound,
      categoryIdentifier,
      priority, // Android priority mapping
    },
    trigger: trigger ?? null, // null = immediate
  });

  return identifier;
}

/**
 * Present a notification immediately (iOS foreground)
 */
export async function presentLocalNotification(options: {
  title: string;
  body: string;
  data?: NotificationData;
  categoryIdentifier?: string;
}): Promise<string> {
  return await scheduleIOSNotification({
    ...options,
    trigger: null, // Immediate
    interruptionLevel: 'active',
  });
}

/**
 * Schedule a time-sensitive notification
 * These can break through Focus Mode on iOS 15+
 */
export async function scheduleTimeSensitiveNotification(options: {
  title: string;
  body: string;
  data?: NotificationData;
  trigger?: Notifications.NotificationTriggerInput;
  categoryIdentifier?: string;
}): Promise<string> {
  return await scheduleIOSNotification({
    ...options,
    interruptionLevel: 'timeSensitive',
  });
}

/**
 * Get all pending scheduled notifications
 */
export async function getPendingNotifications(): Promise<
  Notifications.NotificationRequest[]
> {
  return await Notifications.getAllScheduledNotificationsAsync();
}

/**
 * Cancel a specific scheduled notification by identifier
 */
export async function cancelScheduledNotification(
  identifier: string
): Promise<void> {
  await Notifications.cancelScheduledNotificationAsync(identifier);
}

// Re-export types and listeners for convenience
export {
  Notifications,
};

export type { NotificationData };
