/**
 * Error Handler Middleware
 * Centralized error handling with structured logging
 */

import { Request, Response, NextFunction } from 'express';
import { logger } from '../config/logger';

export const errorHandler = (
  err: Error,
  req: Request,
  res: Response,
  _next: NextFunction
): void => {
  // Determine status code
  const statusCode = res.statusCode !== 200 ? res.statusCode : 500;

  // Log the error with structured context
  logger.error({
    err: {
      message: err.message,
      name: err.name,
      stack: process.env.NODE_ENV !== 'production' ? err.stack : undefined,
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
    error: err.message || 'Internal server error',
    correlationId: req.id,
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
  });
};
