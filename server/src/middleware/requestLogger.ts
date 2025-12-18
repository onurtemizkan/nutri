/**
 * Request Logger Middleware
 * Structured HTTP request/response logging with correlation IDs
 */

import pinoHttp, { HttpLogger, Options } from 'pino-http';
import { logger, generateCorrelationId } from '../config/logger';
import type { Request, Response, NextFunction } from 'express';
import type { IncomingMessage, ServerResponse } from 'http';
import type { LevelWithSilent } from 'pino';

// Extend Express Request type to include id property from pino-http
declare global {
  namespace Express {
    interface Request {
      id?: string;
    }
  }
}

// pino-http options
const httpLoggerOptions: Options = {
  logger,

  // Generate or reuse correlation ID
  genReqId: (req: IncomingMessage): string => {
    const headers = req.headers;
    const existingId = headers['x-correlation-id'] || headers['x-request-id'];
    if (typeof existingId === 'string') {
      return existingId;
    }
    if (Array.isArray(existingId) && existingId.length > 0) {
      return existingId[0];
    }
    return generateCorrelationId();
  },

  // Set log level based on response status code
  customLogLevel: (_req: IncomingMessage, res: ServerResponse, err?: Error): LevelWithSilent => {
    if (res.statusCode >= 500 || err) return 'error';
    if (res.statusCode >= 400) return 'warn';
    return 'info';
  },

  // Custom success message format
  customSuccessMessage: (req: IncomingMessage, res: ServerResponse): string => {
    return `${req.method} ${req.url} ${res.statusCode}`;
  },

  // Custom error message format
  customErrorMessage: (_req: IncomingMessage, _res: ServerResponse, err: Error): string => {
    return `Request failed: ${err.message}`;
  },

  // Customize request serialization (avoid logging sensitive/verbose data)
  serializers: {
    req: (req: IncomingMessage & { query?: Record<string, unknown>; params?: Record<string, unknown> }) => ({
      method: req.method,
      url: req.url,
      query: req.query && Object.keys(req.query).length > 0 ? req.query : undefined,
      params: req.params && Object.keys(req.params).length > 0 ? req.params : undefined,
      userAgent: req.headers?.['user-agent'],
    }),
    res: (res: ServerResponse) => ({
      statusCode: res.statusCode,
    }),
  },

  // Quiet down certain paths (health checks, etc.)
  autoLogging: {
    ignore: (req: IncomingMessage): boolean => {
      const path = req.url || '';
      // Don't log health check endpoints
      return path === '/health' || path === '/health/live' || path === '/ready';
    },
  },
};

export const requestLogger: HttpLogger = pinoHttp(httpLoggerOptions);

/**
 * Middleware to add correlation ID to response headers
 */
export const correlationIdHeader = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  // req.id from pino-http can be string, number, or object
  const correlationId = req.id;
  if (correlationId !== undefined) {
    res.setHeader('x-correlation-id', String(correlationId));
  }
  next();
};
