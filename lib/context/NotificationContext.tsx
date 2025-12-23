import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import { AppState, AppStateStatus, Platform } from 'react-native';
import * as Notifications from 'expo-notifications';
import { useRouter } from 'expo-router';
import {
  getNotificationPermissionStatus,
  requestNotificationPermissions,
  registerDeviceForPushNotifications,
  unregisterDeviceFromPushNotifications,
  initializeNotifications,
  setBadgeCount,
  getIOSPermissionDetails,
  isProvisionalPermission,
  requestFullNotificationPermissions,
  IOSPermissionDetails,
} from '../services/notifications';
import { notificationsApi } from '../api/notifications';
import { NotificationData, NotificationCategory } from '../types';

type PermissionStatus = 'granted' | 'denied' | 'undetermined' | 'loading';

interface NotificationContextType {
  // Permission state
  permissionStatus: PermissionStatus;
  isRegistered: boolean;

  // iOS-specific state
  isProvisional: boolean;
  iosPermissionDetails: IOSPermissionDetails | null;

  // Actions
  requestPermission: () => Promise<boolean>;
  requestFullPermission: () => Promise<boolean>; // iOS: upgrade from provisional
  registerDevice: () => Promise<boolean>;
  unregisterDevice: () => Promise<boolean>;
  refreshPermissionStatus: () => Promise<void>;

  // Badge
  badgeCount: number;
  clearBadge: () => Promise<void>;
}

const NotificationContext = createContext<NotificationContextType | null>(null);

export function useNotifications() {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
}

interface NotificationProviderProps {
  children: React.ReactNode;
}

export function NotificationProvider({ children }: NotificationProviderProps) {
  const router = useRouter();
  const [permissionStatus, setPermissionStatus] = useState<PermissionStatus>('loading');
  const [isRegistered, setIsRegistered] = useState(false);
  const [badgeCount, setBadgeCountState] = useState(0);

  // iOS-specific state
  const [isProvisional, setIsProvisional] = useState(false);
  const [iosPermissionDetails, setIosPermissionDetails] = useState<IOSPermissionDetails | null>(null);

  // Refs for notification listeners
  const notificationListener = useRef<Notifications.EventSubscription | null>(null);
  const responseListener = useRef<Notifications.EventSubscription | null>(null);
  const appState = useRef(AppState.currentState);

  // Handle notification tap (when user taps on notification)
  const handleNotificationResponse = useCallback(
    async (response: Notifications.NotificationResponse) => {
      const data = response.notification.request.content.data as NotificationData;
      const actionId = response.actionIdentifier;

      // Track the notification open
      if (data.notificationLogId) {
        try {
          await notificationsApi.trackNotification({
            notificationLogId: data.notificationLogId,
            action: 'opened',
            actionTaken: actionId !== Notifications.DEFAULT_ACTION_IDENTIFIER ? actionId : undefined,
          });
        } catch (error) {
          console.error('Error tracking notification:', error);
        }
      }

      // Handle action buttons
      if (actionId === 'SNOOZE') {
        // Schedule a reminder for 15 minutes later
        await Notifications.scheduleNotificationAsync({
          content: {
            title: response.notification.request.content.title ?? 'Reminder',
            body: response.notification.request.content.body ?? '',
            data: data,
          },
          trigger: {
            type: Notifications.SchedulableTriggerInputTypes.TIME_INTERVAL,
            seconds: 15 * 60, // 15 minutes
          },
        });
        return;
      }

      if (actionId === 'MARK_TAKEN' && data.params?.supplementId) {
        // TODO: Call supplement log API
        return;
      }

      // Navigate based on category/screen
      const screen = data.screen;
      const category = data.category;

      if (screen) {
        // Direct screen navigation
        router.push(screen as `/${string}`);
        return;
      }

      // Category-based navigation
      switch (category) {
        case 'MEAL_REMINDER':
        case 'STREAK_ALERT':
          router.push('/add-meal');
          break;
        case 'GOAL_PROGRESS':
          router.push('/(tabs)/');
          break;
        case 'HEALTH_INSIGHT':
          if (data.params?.metricType) {
            router.push(`/health/${data.params.metricType}` as `/${string}`);
          } else {
            router.push('/(tabs)/health');
          }
          break;
        case 'SUPPLEMENT_REMINDER':
          router.push('/supplements');
          break;
        case 'WEEKLY_SUMMARY':
          router.push('/(tabs)/health');
          break;
        default:
          // Default to dashboard
          router.push('/(tabs)/');
          break;
      }
    },
    [router]
  );

  // Handle foreground notifications
  const handleForegroundNotification = useCallback(
    async (notification: Notifications.Notification) => {
      const data = notification.request.content.data as NotificationData;

      // Track delivery
      if (data.notificationLogId) {
        try {
          await notificationsApi.trackNotification({
            notificationLogId: data.notificationLogId,
            action: 'delivered',
          });
        } catch (error) {
          console.error('Error tracking notification delivery:', error);
        }
      }

      // Increment badge count (if we're tracking it locally)
      setBadgeCountState((prev) => prev + 1);
    },
    []
  );

  // Check permission status (including iOS-specific details)
  const refreshPermissionStatus = useCallback(async () => {
    const status = await getNotificationPermissionStatus();
    setPermissionStatus(status);

    // Get iOS-specific details
    if (Platform.OS === 'ios') {
      const details = await getIOSPermissionDetails();
      setIosPermissionDetails(details);
      setIsProvisional(details?.isProvisional ?? false);
    }
  }, []);

  // Request permission (uses provisional on iOS for less intrusive first request)
  const requestPermission = useCallback(async (): Promise<boolean> => {
    const status = await requestNotificationPermissions();
    setPermissionStatus(status);

    if (status === 'granted') {
      // Refresh iOS details to check if it's provisional
      if (Platform.OS === 'ios') {
        const details = await getIOSPermissionDetails();
        setIosPermissionDetails(details);
        setIsProvisional(details?.isProvisional ?? false);
      }

      // Auto-register device when permission is granted
      await registerDevice();
      return true;
    }

    return false;
  }, []);

  // Request full (non-provisional) permission on iOS
  // Use this to upgrade from provisional to full authorization
  const requestFullPermission = useCallback(async (): Promise<boolean> => {
    if (Platform.OS !== 'ios') {
      // On Android, just call regular permission request
      return await requestPermission();
    }

    const status = await requestFullNotificationPermissions();
    setPermissionStatus(status);

    if (status === 'granted') {
      // Refresh iOS details
      const details = await getIOSPermissionDetails();
      setIosPermissionDetails(details);
      setIsProvisional(false); // Full permission is not provisional

      // Register device if not already registered
      if (!isRegistered) {
        await registerDevice();
      }
      return true;
    }

    return false;
  }, [isRegistered]);

  // Register device
  const registerDevice = useCallback(async (): Promise<boolean> => {
    const result = await registerDeviceForPushNotifications();
    setIsRegistered(result.success);
    return result.success;
  }, []);

  // Unregister device
  const unregisterDevice = useCallback(async (): Promise<boolean> => {
    const success = await unregisterDeviceFromPushNotifications();
    if (success) {
      setIsRegistered(false);
    }
    return success;
  }, []);

  // Clear badge
  const clearBadge = useCallback(async () => {
    await setBadgeCount(0);
    setBadgeCountState(0);
  }, []);

  // Initialize notifications on mount
  useEffect(() => {
    const init = async () => {
      // Initialize notification settings
      await initializeNotifications();

      // Check current permission status
      await refreshPermissionStatus();

      // Check for notification that opened the app when it was killed
      const lastResponse = await Notifications.getLastNotificationResponseAsync();
      if (lastResponse) {
        handleNotificationResponse(lastResponse);
      }
    };

    init();
  }, [refreshPermissionStatus, handleNotificationResponse]);

  // Setup notification listeners
  useEffect(() => {
    // Listener for foreground notifications
    notificationListener.current = Notifications.addNotificationReceivedListener(
      handleForegroundNotification
    );

    // Listener for notification taps
    responseListener.current = Notifications.addNotificationResponseReceivedListener(
      handleNotificationResponse
    );

    return () => {
      if (notificationListener.current) {
        notificationListener.current.remove();
      }
      if (responseListener.current) {
        responseListener.current.remove();
      }
    };
  }, [handleForegroundNotification, handleNotificationResponse]);

  // Handle app state changes
  useEffect(() => {
    const handleAppStateChange = async (nextAppState: AppStateStatus) => {
      if (appState.current.match(/inactive|background/) && nextAppState === 'active') {
        // App came to foreground - clear badge and refresh status
        await clearBadge();
        await refreshPermissionStatus();
      }
      appState.current = nextAppState;
    };

    const subscription = AppState.addEventListener('change', handleAppStateChange);
    return () => subscription.remove();
  }, [clearBadge, refreshPermissionStatus]);

  const value: NotificationContextType = {
    permissionStatus,
    isRegistered,
    isProvisional,
    iosPermissionDetails,
    requestPermission,
    requestFullPermission,
    registerDevice,
    unregisterDevice,
    refreshPermissionStatus,
    badgeCount,
    clearBadge,
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
}
