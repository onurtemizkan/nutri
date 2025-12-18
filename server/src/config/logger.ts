/**
 * Structured Logging Configuration
 * Uses pino for high-performance JSON logging with sensitive data redaction
 */

import pino, { Logger } from 'pino';
import { randomUUID } from 'crypto';

const isProduction = process.env.NODE_ENV === 'production';
const isTest = process.env.NODE_ENV === 'test';

// Base pino configuration
const baseConfig: pino.LoggerOptions = {
  level: process.env.LOG_LEVEL || (isProduction ? 'info' : isTest ? 'silent' : 'debug'),
  formatters: {
    level: (label) => ({ level: label }),
  },
  timestamp: pino.stdTimeFunctions.isoTime,
  redact: {
    paths: [
      'req.headers.authorization',
      'req.body.password',
      'req.body.currentPassword',
      'req.body.newPassword',
      'req.body.confirmPassword',
      'req.body.token',
      'res.headers["set-cookie"]',
      '*.password',
      '*.currentPassword',
      '*.newPassword',
      '*.token',
      '*.jwt',
      '*.secret',
      '*.apiKey',
      '*.api_key',
      '*.accessToken',
      '*.refreshToken',
    ],
    censor: '[REDACTED]',
  },
  // Add service metadata
  base: {
    service: 'nutri-backend',
    version: process.env.npm_package_version || '1.0.0',
    environment: process.env.NODE_ENV || 'development',
  },
};

// Create logger - use pino-pretty transport in development
let logger: Logger;

if (isProduction || isTest) {
  logger = pino(baseConfig);
} else {
  // Development mode with pino-pretty
  logger = pino({
    ...baseConfig,
    transport: {
      target: 'pino-pretty',
      options: {
        colorize: true,
        translateTime: 'SYS:standard',
        ignore: 'pid,hostname,service,version,environment',
      },
    },
  });
}

export { logger };

/**
 * Generate a unique correlation ID for request tracing
 */
export const generateCorrelationId = (): string => randomUUID();

/**
 * Create a child logger with additional context
 */
export const createChildLogger = (bindings: Record<string, unknown>): Logger =>
  logger.child(bindings);

export type { Logger };
