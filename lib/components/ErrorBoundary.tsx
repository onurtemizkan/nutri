/**
 * Error Boundary Component
 *
 * Catches JavaScript errors in child components and displays
 * a user-friendly fallback UI instead of crashing the entire app.
 *
 * @see https://reactjs.org/docs/error-boundaries.html
 */

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing, borderRadius, typography } from '@/lib/theme/colors';

/** Maximum number of retry attempts before showing permanent error */
const MAX_RETRY_COUNT = 3;

interface ErrorBoundaryProps {
  children: ReactNode;
  /**
   * Optional fallback component to render on error
   */
  fallback?: ReactNode;
  /**
   * Callback when an error is caught
   */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /**
   * Show technical details (for development)
   */
  showDetails?: boolean;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  retryCount: number;
}

/**
 * Error Boundary for catching and handling React component errors
 *
 * Usage:
 * ```tsx
 * <ErrorBoundary>
 *   <MyComponent />
 * </ErrorBoundary>
 *
 * // With custom fallback:
 * <ErrorBoundary fallback={<CustomErrorScreen />}>
 *   <MyComponent />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0,
    };
  }

  /**
   * Called when an error is thrown in a child component
   * Updates state to trigger fallback UI
   */
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  /**
   * Called after an error is caught
   * Used for error logging/reporting
   */
  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error details for debugging
    console.error('[ErrorBoundary] Caught error:', error);
    console.error('[ErrorBoundary] Component stack:', errorInfo.componentStack);

    // Update state with error info
    this.setState({ errorInfo });

    // Call optional onError callback
    this.props.onError?.(error, errorInfo);

    // TODO: Send to error reporting service (Sentry, Bugsnag, etc.)
    // Example: Sentry.captureException(error, { extra: { componentStack: errorInfo.componentStack } });
  }

  /**
   * Reset error state and retry rendering children
   * Limited to MAX_RETRY_COUNT attempts to prevent infinite loops
   */
  handleRetry = (): void => {
    if (this.state.retryCount >= MAX_RETRY_COUNT) {
      // Max retries reached - don't reset, keep showing error
      return;
    }

    this.setState((prevState) => ({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: prevState.retryCount + 1,
    }));
  };

  render(): ReactNode {
    const { hasError, error, errorInfo, retryCount } = this.state;
    const { children, fallback, showDetails = __DEV__ } = this.props;

    if (hasError) {
      // Use custom fallback if provided
      if (fallback) {
        return fallback;
      }

      const maxRetriesReached = retryCount >= MAX_RETRY_COUNT;

      // Default error UI
      return (
        <View
          style={styles.container}
          accessibilityRole="alert"
          accessibilityLiveRegion="assertive"
          accessibilityLabel="An error occurred in the application"
        >
          <View style={styles.content}>
            <View
              style={styles.iconContainer}
              accessibilityLabel="Error warning icon"
              accessible={true}
            >
              <Ionicons name="warning" size={64} color={colors.semantic.error} />
            </View>

            <Text style={styles.title} accessibilityRole="header">
              Something went wrong
            </Text>
            <Text style={styles.message}>
              {maxRetriesReached
                ? "We're sorry, but this error persists. Please restart the app."
                : "We're sorry, but an unexpected error occurred. Please try again or restart the app."}
            </Text>

            {!maxRetriesReached && (
              <TouchableOpacity
                style={styles.retryButton}
                onPress={this.handleRetry}
                accessibilityLabel={`Try again. Attempt ${retryCount + 1} of ${MAX_RETRY_COUNT}`}
                accessibilityRole="button"
                accessibilityHint="Attempts to recover from the error"
              >
                <Ionicons name="refresh" size={20} color={colors.text.primary} />
                <Text style={styles.retryButtonText}>Try Again</Text>
              </TouchableOpacity>
            )}

            {maxRetriesReached && (
              <View style={styles.maxRetriesContainer}>
                <Text style={styles.maxRetriesText}>
                  Maximum retry attempts reached ({MAX_RETRY_COUNT})
                </Text>
              </View>
            )}

            {showDetails && error && (
              <ScrollView style={styles.detailsContainer}>
                <Text style={styles.detailsTitle}>Error Details (Development Only)</Text>
                <Text style={styles.errorName}>{error.name}</Text>
                <Text style={styles.errorMessage}>{error.message}</Text>
                {errorInfo?.componentStack && (
                  <>
                    <Text style={styles.detailsTitle}>Component Stack</Text>
                    <Text style={styles.stackTrace}>{errorInfo.componentStack}</Text>
                  </>
                )}
              </ScrollView>
            )}
          </View>
        </View>
      );
    }

    return children;
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.secondary,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.lg,
  },
  content: {
    width: '100%',
    maxWidth: 400,
    alignItems: 'center',
  },
  iconContainer: {
    marginBottom: spacing.lg,
  },
  title: {
    ...typography.h2,
    color: colors.text.primary,
    marginBottom: spacing.md,
    textAlign: 'center',
  },
  message: {
    ...typography.body,
    color: colors.text.tertiary,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  retryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.semantic.success,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.md,
    gap: spacing.sm,
  },
  retryButtonText: {
    ...typography.button,
    color: colors.text.primary,
  },
  maxRetriesContainer: {
    padding: spacing.md,
    backgroundColor: colors.special.errorLight,
    borderRadius: borderRadius.sm,
  },
  maxRetriesText: {
    ...typography.bodySmall,
    color: colors.semantic.error,
    textAlign: 'center',
  },
  detailsContainer: {
    marginTop: spacing.xl,
    maxHeight: 200,
    width: '100%',
    backgroundColor: colors.surface.elevated,
    borderRadius: borderRadius.sm,
    padding: spacing.md,
  },
  detailsTitle: {
    ...typography.caption,
    fontWeight: typography.fontWeight.semibold,
    color: colors.semantic.warning,
    marginBottom: spacing.sm,
    marginTop: spacing.sm,
  },
  errorName: {
    ...typography.bodySmall,
    fontWeight: typography.fontWeight.semibold,
    color: colors.semantic.error,
    marginBottom: spacing.xs,
  },
  errorMessage: {
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  stackTrace: {
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
});

/**
 * Higher-order component to wrap a component with an error boundary
 *
 * Usage:
 * ```tsx
 * const SafeMyComponent = withErrorBoundary(MyComponent);
 * ```
 */
export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ErrorBoundaryProps, 'children'>
): React.FC<P> {
  const displayName = WrappedComponent.displayName || WrappedComponent.name || 'Component';

  const ComponentWithErrorBoundary: React.FC<P> = (props) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <WrappedComponent {...props} />
    </ErrorBoundary>
  );

  ComponentWithErrorBoundary.displayName = `withErrorBoundary(${displayName})`;

  return ComponentWithErrorBoundary;
}

export default ErrorBoundary;
