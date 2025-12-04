import { Request, Response } from 'express';
import { z } from 'zod';
import { HTTP_STATUS, ERROR_MESSAGES } from '../config/constants';

/**
 * Error status configuration for controller error handling
 * Allows specifying which HTTP status to use for different error types
 */
export interface ErrorStatusConfig {
  /** Status code for Zod validation errors (default: 400) */
  zodError?: number;
  /** Status code for generic Error instances (default: 400) */
  error?: number;
  /** Status code for unknown errors (default: 500) */
  unknown?: number;
}

/**
 * Default error status configuration
 */
const DEFAULT_ERROR_STATUS: Required<ErrorStatusConfig> = {
  zodError: HTTP_STATUS.BAD_REQUEST,
  error: HTTP_STATUS.BAD_REQUEST,
  unknown: HTTP_STATUS.INTERNAL_SERVER_ERROR,
};

/**
 * Handles errors consistently across all controllers
 * Reduces code duplication by centralizing error handling logic
 *
 * @param res - Express response object
 * @param error - The error to handle
 * @param statusConfig - Optional custom status codes for different error types
 */
export function handleControllerError(
  res: Response,
  error: unknown,
  statusConfig: ErrorStatusConfig = {}
): void {
  const config = { ...DEFAULT_ERROR_STATUS, ...statusConfig };

  if (error instanceof z.ZodError) {
    res.status(config.zodError).json({ error: error.errors[0].message });
    return;
  }

  if (error instanceof Error) {
    res.status(config.error).json({ error: error.message });
    return;
  }

  res.status(config.unknown).json({ error: ERROR_MESSAGES.INTERNAL_ERROR });
}

/**
 * Type for controller handler functions
 */
export type ControllerHandler<T extends Request = Request> = (
  req: T,
  res: Response
) => Promise<void>;

/**
 * Higher-order function that wraps controller methods with error handling
 * Eliminates try-catch boilerplate from controller methods
 *
 * @param handler - The controller handler function to wrap
 * @param statusConfig - Optional custom status codes for different error types
 * @returns Wrapped handler function with error handling
 *
 * @example
 * // Instead of:
 * async createMeal(req, res) {
 *   try {
 *     // business logic
 *   } catch (error) {
 *     if (error instanceof z.ZodError) { ... }
 *     if (error instanceof Error) { ... }
 *     res.status(500).json({ error: 'Internal server error' });
 *   }
 * }
 *
 * // You can write:
 * createMeal = withErrorHandling(async (req, res) => {
 *   // business logic only
 * });
 */
export function withErrorHandling<T extends Request = Request>(
  handler: ControllerHandler<T>,
  statusConfig: ErrorStatusConfig = {}
): ControllerHandler<T> {
  return async (req: T, res: Response): Promise<void> => {
    try {
      await handler(req, res);
    } catch (error) {
      handleControllerError(res, error, statusConfig);
    }
  };
}

/**
 * Pre-configured error handlers for common scenarios
 */
export const ErrorHandlers = {
  /**
   * Handler for endpoints where Error means "not found" (404)
   */
  withNotFound: <T extends Request = Request>(handler: ControllerHandler<T>) =>
    withErrorHandling(handler, { error: HTTP_STATUS.NOT_FOUND }),

  /**
   * Handler for auth endpoints where Error means "unauthorized" (401)
   */
  withUnauthorized: <T extends Request = Request>(handler: ControllerHandler<T>) =>
    withErrorHandling(handler, { error: HTTP_STATUS.UNAUTHORIZED }),
} as const;
