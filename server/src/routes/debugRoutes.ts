/**
 * Debug Routes
 * Test endpoints for verifying integrations like Sentry.
 * These routes are only available in non-production environments.
 */

import { Router, Request, Response } from 'express';
import { config } from '../config/env';
import { captureException, Sentry } from '../config/sentry';

const router = Router();

/**
 * Check if debug routes should be enabled
 * Only available in development and test environments
 */
const isDebugEnabled = (): boolean => {
  return config.nodeEnv !== 'production';
};

/**
 * GET /api/debug/sentry-test
 * Trigger a test error to verify Sentry integration.
 * Only available in non-production environments.
 */
router.get('/sentry-test', (_req: Request, res: Response) => {
  if (!isDebugEnabled()) {
    res.status(404).json({ error: 'Not found' });
    return;
  }

  // Throw a test error that Sentry should capture
  const testError = new Error('Test Sentry integration - Backend');
  captureException(testError, {
    testType: 'manual-trigger',
    timestamp: new Date().toISOString(),
  });

  res.json({
    success: true,
    message: 'Test error sent to Sentry',
    environment: config.nodeEnv,
    sentryEnabled: !!process.env.SENTRY_DSN,
  });
});

/**
 * GET /api/debug/sentry-throw
 * Throw an unhandled error to verify automatic Sentry capture.
 * Only available in non-production environments.
 */
router.get('/sentry-throw', (_req: Request, res: Response) => {
  if (!isDebugEnabled()) {
    res.status(404).json({ error: 'Not found' });
    return;
  }

  // This will be caught by the global error handler and sent to Sentry
  throw new Error('Test unhandled error for Sentry - Backend');
});

/**
 * GET /api/debug/sentry-status
 * Check Sentry configuration status.
 * Only available in non-production environments.
 */
router.get('/sentry-status', (_req: Request, res: Response) => {
  if (!isDebugEnabled()) {
    res.status(404).json({ error: 'Not found' });
    return;
  }

  const client = Sentry.getClient();

  res.json({
    enabled: !!process.env.SENTRY_DSN,
    environment: config.nodeEnv,
    clientInitialized: !!client,
    dsn: process.env.SENTRY_DSN ? '[CONFIGURED]' : '[NOT SET]',
  });
});

export default router;
