/**
 * CGM Routes
 *
 * Routes for CGM (Continuous Glucose Monitor) integration.
 * Handles OAuth connection flows, connection management, and sync operations.
 */

import { Router } from 'express';
import {
  initiateConnection,
  handleOAuthCallback,
  connectLevels,
  getConnections,
  disconnectProvider,
  validateConnection,
} from '../controllers/cgmController';
import { authenticate } from '../middleware/auth';
import { rateLimiters } from '../middleware/rateLimiter';

const router = Router();

// All CGM routes require authentication
router.use(authenticate);

// ============================================================================
// CONNECTION MANAGEMENT
// ============================================================================

/**
 * Initiate OAuth connection flow
 * POST /api/cgm/connect
 * Body: { provider: 'DEXCOM' | 'LIBRE', redirectUri?: string }
 */
router.post('/connect', rateLimiters.cgmOAuth, initiateConnection);

/**
 * Handle OAuth callback (exchange code for tokens)
 * POST /api/cgm/callback
 * Body: { provider: 'DEXCOM' | 'LIBRE', code: string, state?: string }
 */
router.post('/callback', rateLimiters.cgmOAuth, handleOAuthCallback);

/**
 * Connect Levels using API key (non-OAuth)
 * POST /api/cgm/connect/levels
 * Body: { apiKey: string, externalUserId?: string }
 */
router.post('/connect/levels', rateLimiters.cgmOAuth, connectLevels);

/**
 * Get all CGM connections for the authenticated user
 * GET /api/cgm/connections
 * Query: { provider?: 'DEXCOM' | 'LIBRE' | 'LEVELS', includeInactive?: boolean }
 */
router.get('/connections', rateLimiters.cgmRead, getConnections);

/**
 * Disconnect a CGM provider
 * DELETE /api/cgm/connections/:provider
 */
router.delete('/connections/:provider', rateLimiters.cgmOAuth, disconnectProvider);

/**
 * Validate a CGM connection (check if tokens are valid)
 * GET /api/cgm/connections/:provider/validate
 */
router.get('/connections/:provider/validate', rateLimiters.cgmRead, validateConnection);

export default router;
