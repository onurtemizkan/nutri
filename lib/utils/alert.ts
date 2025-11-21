/**
 * Custom Alert utility that wraps the useCustomAlert hook
 * Import and call showAlert directly in your components
 *
 * Usage:
 * import { showAlert } from '@/lib/utils/alert';
 *
 * showAlert('Success', 'Your changes have been saved!');
 *
 * or with buttons:
 * showAlert('Confirm', 'Are you sure?', [
 *   { text: 'Cancel', style: 'cancel' },
 *   { text: 'Delete', style: 'destructive', onPress: handleDelete },
 * ]);
 */

// This will be set by the AlertProvider
let globalShowAlert: ((title: string, message?: string, buttons?: {
  text: string;
  onPress?: () => void;
  style?: 'default' | 'cancel' | 'destructive';
}[]) => void) | null = null;

export const setGlobalAlertHandler = (handler: typeof globalShowAlert) => {
  globalShowAlert = handler;
};

export const showAlert = (
  title: string,
  message?: string,
  buttons?: {
    text: string;
    onPress?: () => void;
    style?: 'default' | 'cancel' | 'destructive';
  }[]
) => {
  if (globalShowAlert) {
    globalShowAlert(title, message, buttons);
  } else {
    console.error('AlertProvider not initialized');
  }
};
