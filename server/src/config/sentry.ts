/**
 * Sentry Error Tracking Configuration
 * Captures errors, performance traces, and provides structured error reporting
 */

import * as Sentry from '@sentry/node';
import { Express, ErrorRequestHandler } from 'express';
import { config } from './env';

// Import version from package.json
// eslint-disable-next-line @typescript-eslint/no-require-imports
const packageJson = require('../../package.json') as { version: string };

/**
 * Initialize Sentry error tracking
 * Must be called before any other middleware is registered
 *
 * Note: In Sentry v10+, the Express app is not passed to expressIntegration()
 * The integration automatically instruments Express routes.
 */
export function initSentry(_app: Express): void {
  const dsn = process.env.SENTRY_DSN;

  if (!dsn) {
    console.log('Sentry DSN not configured, skipping initialization');
    return;
  }

  Sentry.init({
    dsn,
    environment: config.nodeEnv,
    release: process.env.GITHUB_SHA || packageJson.version,

    // Integrations for Express
    integrations: [
      Sentry.httpIntegration(),
      Sentry.expressIntegration(),
    ],

    // Performance Monitoring
    // Sample 10% of transactions in production, 100% in development
    tracesSampleRate: config.nodeEnv === 'production' ? 0.1 : 1.0,

    // Error filtering
    ignoreErrors: [
      // Expected authentication errors
      'Invalid credentials',
      'Token expired',
      'Unauthorized',
      // Rate limiting responses
      'Too many requests',
    ],

    // Sensitive data scrubbing - implemented in beforeSend hook
    beforeSend(event, hint) {
      return scrubSensitiveData(event, hint);
    },
  });

  console.log(`Sentry initialized for environment: ${config.nodeEnv}`);
}

/**
 * Scrub sensitive data from Sentry events before sending
 * Removes authorization tokens, cookies, passwords, and partial-masks emails
 */
function scrubSensitiveData(
  event: Sentry.ErrorEvent,
  _hint: Sentry.EventHint
): Sentry.ErrorEvent | null {
  // Scrub request headers
  if (event.request?.headers) {
    const sensitiveHeaders = ['authorization', 'cookie', 'x-api-key'];
    for (const header of sensitiveHeaders) {
      if (event.request.headers[header]) {
        delete event.request.headers[header];
      }
    }
  }

  // Scrub request body for sensitive fields
  if (event.request?.data && typeof event.request.data === 'object') {
    const sensitiveFields = [
      'password',
      'token',
      'secret',
      'apiKey',
      'api_key',
      'creditCard',
      'credit_card',
      'ssn',
      'accessToken',
      'access_token',
      'refreshToken',
      'refresh_token',
    ];

    const data = event.request.data as Record<string, unknown>;
    for (const field of sensitiveFields) {
      if (data[field]) {
        data[field] = '[REDACTED]';
      }
    }
  }

  // Partial-mask user email for debugging while preserving privacy
  // 'john.doe@example.com' becomes 'j***@example.com'
  if (event.user?.email && typeof event.user.email === 'string') {
    const email = event.user.email;
    if (email.includes('@')) {
      const [local, domain] = email.split('@');
      event.user.email = `${local[0]}***@${domain}`;
    }
  }

  return event;
}

/**
 * Setup Sentry error handler
 * Must be called after routes but before custom error handlers
 *
 * @param app - Express application instance
 */
export function setupSentryErrorHandler(app: Express): void {
  if (!process.env.SENTRY_DSN) {
    return;
  }

  // The error handler must be before any other error middleware
  app.use(Sentry.expressErrorHandler() as ErrorRequestHandler);
}

/**
 * Capture an exception with Sentry
 * Use this for manual error capture when automatic capture isn't sufficient
 *
 * @param error - The error to capture
 * @param context - Optional additional context
 */
export function captureException(
  error: Error,
  context?: Record<string, unknown>
): void {
  if (!process.env.SENTRY_DSN) {
    return;
  }

  if (context) {
    Sentry.setContext('additional', context);
  }

  Sentry.captureException(error);
}

/**
 * Set user context for Sentry events
 * Call this after user authentication to associate errors with users
 *
 * @param userId - The user's ID
 * @param email - Optional user email (will be partial-masked in events)
 */
export function setUser(userId: string, email?: string): void {
  if (!process.env.SENTRY_DSN) {
    return;
  }

  Sentry.setUser({
    id: userId,
    ...(email && { email }),
  });
}

/**
 * Clear user context
 * Call this on logout or when user context is no longer valid
 */
export function clearUser(): void {
  if (!process.env.SENTRY_DSN) {
    return;
  }

  Sentry.setUser(null);
}

// Re-export Sentry for direct access when needed
export { Sentry };
