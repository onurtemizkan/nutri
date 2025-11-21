import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Platform,
  Modal,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, gradients, shadows, spacing, borderRadius, typography } from '@/lib/theme/colors';
import { setGlobalAlertHandler } from '@/lib/utils/alert';

interface AlertButton {
  text: string;
  onPress?: () => void;
  style?: 'default' | 'cancel' | 'destructive';
}

interface AlertOptions {
  title: string;
  message?: string;
  buttons?: AlertButton[];
}

interface AlertContextType {
  showAlert: (options: AlertOptions) => void;
}

const AlertContext = createContext<AlertContextType | undefined>(undefined);

export const useCustomAlert = () => {
  const context = useContext(AlertContext);
  if (!context) {
    throw new Error('useCustomAlert must be used within AlertProvider');
  }
  return context;
};

export const AlertProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [visible, setVisible] = useState(false);
  const [alertOptions, setAlertOptions] = useState<AlertOptions | null>(null);
  const fadeAnim = useState(new Animated.Value(0))[0];
  const scaleAnim = useState(new Animated.Value(0.9))[0];

  const showAlert = useCallback((options: AlertOptions) => {
    setAlertOptions(options);
    fadeAnim.setValue(0);
    scaleAnim.setValue(0.9);
    setVisible(true);

    // Start animations after Modal is visible
    requestAnimationFrame(() => {
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.spring(scaleAnim, {
          toValue: 1,
          tension: 100,
          friction: 10,
          useNativeDriver: true,
        }),
      ]).start();
    });
  }, [fadeAnim, scaleAnim]);

  // Set up global alert handler
  useEffect(() => {
    setGlobalAlertHandler((title, message, buttons) => {
      showAlert({ title, message, buttons });
    });
    return () => setGlobalAlertHandler(null);
  }, [showAlert]);

  const hideAlert = useCallback(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 0,
        duration: 150,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 0.9,
        duration: 150,
        useNativeDriver: true,
      }),
    ]).start(() => {
      setVisible(false);
      setAlertOptions(null);
    });
  }, [fadeAnim, scaleAnim]);

  const handleBackdropPress = useCallback(() => {
    // Find and call cancel button if present
    const cancelButton = alertOptions?.buttons?.find(btn => btn.style === 'cancel');
    if (cancelButton?.onPress) {
      hideAlert();
      setTimeout(() => {
        cancelButton.onPress?.();
      }, 200);
    } else {
      hideAlert();
    }
  }, [alertOptions, hideAlert]);

  const handleButtonPress = (button: AlertButton) => {
    hideAlert();
    // Execute button callback after animation
    setTimeout(() => {
      button.onPress?.();
    }, 200);
  };

  const buttons = alertOptions?.buttons || [{ text: 'OK', style: 'default' as const }];

  return (
    <AlertContext.Provider value={{ showAlert }}>
      {children}
      <Modal
        visible={visible}
        transparent
        animationType="none"
        statusBarTranslucent
        onRequestClose={handleBackdropPress}
        presentationStyle="overFullScreen"
        hardwareAccelerated
      >
        <Animated.View style={[styles.overlay, { opacity: fadeAnim }]}>
          <TouchableOpacity
            style={styles.overlayTouchable}
            activeOpacity={1}
            onPress={handleBackdropPress}
          >
            <Animated.View
              style={[
                styles.alertContainer,
                {
                  transform: [{ scale: scaleAnim }],
                  opacity: fadeAnim,
                },
              ]}
              onStartShouldSetResponder={() => true}
            >
              <TouchableOpacity activeOpacity={1} onPress={(e) => e.stopPropagation()}>
                <View style={styles.alert}>
                  {/* Title */}
                  <Text style={styles.title}>{alertOptions?.title}</Text>

                  {/* Message */}
                  {alertOptions?.message && (
                    <Text style={styles.message}>{alertOptions.message}</Text>
                  )}

                  {/* Buttons */}
                  <View style={[
                    styles.buttonContainer,
                    buttons.length > 2 && styles.buttonContainerVertical
                  ]}>
                    {buttons.map((button, index) => {
                      const isDestructive = button.style === 'destructive';
                      const isCancel = button.style === 'cancel';
                      const isPrimary = button.style === 'default' && buttons.length === 1;

                      return (
                        <TouchableOpacity
                          key={index}
                          style={[
                            styles.button,
                            buttons.length > 2 && styles.buttonVertical,
                            isCancel && styles.buttonCancel,
                          ]}
                          onPress={() => handleButtonPress(button)}
                          activeOpacity={0.8}
                        >
                          {isPrimary || isDestructive ? (
                            <LinearGradient
                              colors={isDestructive ? [colors.status.error, '#C62828'] : gradients.primary}
                              start={{ x: 0, y: 0 }}
                              end={{ x: 1, y: 0 }}
                              style={styles.buttonGradient}
                            >
                              <Text style={styles.buttonTextPrimary}>{button.text}</Text>
                            </LinearGradient>
                          ) : (
                            <View style={[
                              styles.buttonSecondary,
                              isCancel && styles.buttonCancelInner,
                            ]}>
                              <Text style={[
                                styles.buttonTextSecondary,
                                isCancel && styles.buttonTextCancel,
                              ]}>
                                {button.text}
                              </Text>
                            </View>
                          )}
                        </TouchableOpacity>
                      );
                    })}
                  </View>
                </View>
              </TouchableOpacity>
            </Animated.View>
          </TouchableOpacity>
        </Animated.View>
      </Modal>
    </AlertContext.Provider>
  );
};

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: colors.overlay.medium,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 9999,
    elevation: 9999,
  },
  overlayTouchable: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
  },
  alertContainer: {
    width: '85%',
    maxWidth: 400,
    zIndex: 10000,
    elevation: 10000,
  },
  alert: {
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    borderWidth: 1,
    borderColor: colors.border.primary,
    ...shadows.xl,
  },
  title: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing.sm,
    textAlign: 'center',
    letterSpacing: -0.5,
  },
  message: {
    fontSize: typography.fontSize.md,
    color: colors.text.secondary,
    lineHeight: typography.lineHeight.relaxed * typography.fontSize.md,
    marginBottom: spacing.lg,
    textAlign: 'center',
  },
  buttonContainer: {
    flexDirection: 'row',
    gap: spacing.md,
    marginTop: spacing.sm,
  },
  buttonContainerVertical: {
    flexDirection: 'column',
  },
  button: {
    flex: 1,
    borderRadius: borderRadius.md,
    overflow: 'hidden',
    minHeight: 48,
  },
  buttonVertical: {
    flex: 0,
  },
  buttonGradient: {
    paddingVertical: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 48,
  },
  buttonSecondary: {
    paddingVertical: spacing.md,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background.elevated,
    borderRadius: borderRadius.md,
    minHeight: 48,
  },
  buttonCancel: {
    borderWidth: 1.5,
    borderColor: colors.border.primary,
  },
  buttonCancelInner: {
    backgroundColor: 'transparent',
  },
  buttonTextPrimary: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    letterSpacing: 0.3,
  },
  buttonTextSecondary: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
  },
  buttonTextCancel: {
    color: colors.text.tertiary,
  },
});
