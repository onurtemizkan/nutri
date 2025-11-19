/**
 * Type-safe error handling utilities for React Native
 */

interface AxiosError {
  response?: {
    data?: {
      error?: string;
    };
  };
  message?: string;
}

/**
 * Type guard to check if error is an Axios error
 */
export function isAxiosError(error: unknown): error is AxiosError {
  return (
    typeof error === 'object' &&
    error !== null &&
    'response' in error
  );
}

/**
 * Extract error message from various error types
 */
export function getErrorMessage(error: unknown, fallback: string = 'An error occurred'): string {
  if (isAxiosError(error)) {
    return error.response?.data?.error || error.message || fallback;
  }

  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === 'string') {
    return error;
  }

  return fallback;
}
