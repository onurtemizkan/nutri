/**
 * Error Handler Middleware
 * Centralized error handling with structured logging and Prisma-specific handling
 */

import { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';
import {
  PrismaClientKnownRequestError,
  PrismaClientValidationError,
  PrismaClientInitializationError,
} from '@prisma/client/runtime/library';

/**
 * Map Prisma error codes to HTTP status codes and user-friendly messages
 */
const PRISMA_ERROR_MAP: Record<string, { status: number; message: string }> = {
  // Unique constraint violation
  P2002: { status: 400, message: 'A record with this data already exists' },
  // Foreign key constraint violation
  P2003: { status: 400, message: 'Referenced record does not exist' },
  // Record not found for update/delete
  P2025: { status: 404, message: 'Record not found' },
  // Value too long for column
  P2000: { status: 400, message: 'Input value is too long' },
  // Required field missing
  P2011: { status: 400, message: 'Required field is missing' },
  // Invalid data type
  P2006: { status: 400, message: 'Invalid data provided' },
  // Query timeout
  P2024: { status: 504, message: 'Database operation timed out' },
  // Connection pool timeout
  P2028: { status: 503, message: 'Database connection unavailable' },
};

/**
 * Extract user-friendly information from Prisma errors
 */
function handlePrismaError(error: PrismaClientKnownRequestError): {
  statusCode: number;
  message: string;
  details?: Record<string, unknown>;
} {
  const errorInfo = PRISMA_ERROR_MAP[error.code];

  if (errorInfo) {
    const details: Record<string, unknown> = { code: error.code };

    // Add field information for unique constraint violations
    if (error.code === 'P2002' && error.meta?.target) {
      const fields = Array.isArray(error.meta.target)
        ? (error.meta.target as string[]).join(', ')
        : String(error.meta.target);
      details.fields = fields;
      return {
        statusCode: errorInfo.status,
        message: `${errorInfo.message} (${fields})`,
        details,
      };
    }

    return {
      statusCode: errorInfo.status,
      message: errorInfo.message,
      details,
    };
  }

  // Unknown Prisma error code
  return {
    statusCode: 500,
    message: 'Database operation failed',
    details: { code: error.code },
  };
}

export const errorHandler = (
  err: Error,
  req: Request,
  res: Response,
  _next: NextFunction
): void => {
  let statusCode = res.statusCode !== 200 ? res.statusCode : 500;
  let errorMessage = err.message || 'Internal server error';
  let errorDetails: Record<string, unknown> | undefined;

  // Handle Prisma-specific errors
  if (err instanceof PrismaClientKnownRequestError) {
    const prismaError = handlePrismaError(err);
    statusCode = prismaError.statusCode;
    errorMessage = prismaError.message;
    errorDetails = prismaError.details;
  } else if (err instanceof PrismaClientValidationError) {
    statusCode = 400;
    errorMessage = 'Invalid query parameters';
  } else if (err instanceof PrismaClientInitializationError) {
    statusCode = 503;
    errorMessage = 'Database connection failed';
  }

  // Log the error with structured context
  logger.error({
    err: {
      message: err.message,
      name: err.name,
      stack: process.env.NODE_ENV !== 'production' ? err.stack : undefined,
      ...(err instanceof PrismaClientKnownRequestError && {
        prismaCode: err.code,
        prismaMeta: err.meta,
      }),
    },
    correlationId: req.id,
    statusCode,
    path: req.path,
    method: req.method,
    query: Object.keys(req.query || {}).length > 0 ? req.query : undefined,
    params: Object.keys(req.params || {}).length > 0 ? req.params : undefined,
    userId: (req as Express.Request & { userId?: string }).userId,
  }, `Request error: ${err.message}`);

  // Send error response
  res.status(statusCode).json({
    error: errorMessage,
    correlationId: req.id,
    ...(errorDetails && process.env.NODE_ENV === 'development' && { details: errorDetails }),
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
  });
};
