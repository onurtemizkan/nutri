/**
 * CGM Controller
 *
 * Handles HTTP requests for CGM integration endpoints.
 * Manages OAuth connection flows, connection status, and disconnection.
 */

import { Response } from 'express';
import { AuthenticatedRequest } from '../types';
import { cgmIntegrationService } from '../services/cgmIntegrationService';
import {
  initiateCGMConnectionSchema,
  cgmOAuthCallbackSchema,
  getCGMConnectionsQuerySchema,
  cgmProviderSchema,
} from '../validation/schemas';
import { HTTP_STATUS, CGM_ERROR_MESSAGES, CGM_SUCCESS_MESSAGES } from '../config/constants';
import { logger } from '../config/logger';
import crypto from 'crypto';

/**
 * Generate authorization URL for CGM OAuth flow
 *
 * POST /api/cgm/connect
 * Body: { provider: 'DEXCOM' | 'LIBRE', redirectUri?: string }
 */
export async function initiateConnection(req: AuthenticatedRequest, res: Response): Promise<void> {
  try {
    const userId = req.userId as string;
    const validatedData = initiateCGMConnectionSchema.parse(req.body);
    const { provider, redirectUri } = validatedData;

    // Generate a random state parameter for CSRF protection
    const state = crypto.randomBytes(32).toString('hex');

    // Store state in session or database for verification on callback
    // For simplicity, we'll encode userId in the state (in production, use a session)
    const stateWithUserId = Buffer.from(JSON.stringify({ state, userId })).toString('base64');

    const authorizationUrl = await cgmIntegrationService.generateAuthorizationUrl(
      provider,
      stateWithUserId,
      redirectUri
    );

    logger.info({ userId, provider }, 'Generated CGM authorization URL');

    res.status(HTTP_STATUS.OK).json({
      authorizationUrl,
      state: stateWithUserId,
      provider,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to initiate connection';
    logger.error({ err: error }, 'Failed to initiate CGM connection');

    if (errorMessage.includes('environment variable')) {
      res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
        error: 'CGM provider not configured',
        message: 'Please contact support to enable this CGM provider',
      });
      return;
    }

    res.status(HTTP_STATUS.BAD_REQUEST).json({
      error: 'Connection initiation failed',
      message: errorMessage,
    });
  }
}

/**
 * Handle OAuth callback from CGM provider
 *
 * POST /api/cgm/callback
 * Body: { provider: 'DEXCOM' | 'LIBRE', code: string, state?: string }
 */
export async function handleOAuthCallback(req: AuthenticatedRequest, res: Response): Promise<void> {
  try {
    const userId = req.userId as string;
    const validatedData = cgmOAuthCallbackSchema.parse(req.body);
    const { provider, code, state } = validatedData;

    // Verify state if provided (CSRF protection)
    if (state) {
      try {
        const decodedState = JSON.parse(Buffer.from(state, 'base64').toString());
        if (decodedState.userId !== userId) {
          res.status(HTTP_STATUS.BAD_REQUEST).json({
            error: 'Invalid state parameter',
            message: 'OAuth state mismatch - please try again',
          });
          return;
        }
      } catch {
        res.status(HTTP_STATUS.BAD_REQUEST).json({
          error: 'Invalid state parameter',
          message: 'Could not decode OAuth state',
        });
        return;
      }
    }

    // Exchange code for tokens
    const tokens = await cgmIntegrationService.exchangeCodeForTokens(provider, code);

    // Calculate token expiration time
    const expiresAt = new Date(Date.now() + tokens.expiresIn * 1000);

    // Create connection with encrypted tokens
    const connectionStatus = await cgmIntegrationService.createConnection(userId, {
      provider,
      accessToken: tokens.accessToken,
      refreshToken: tokens.refreshToken,
      expiresAt,
      scope: tokens.scope,
    });

    logger.info({ userId, provider }, 'CGM OAuth callback successful');

    res.status(HTTP_STATUS.OK).json({
      message: CGM_SUCCESS_MESSAGES.CONNECTED(provider),
      connection: connectionStatus,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : CGM_ERROR_MESSAGES.OAUTH_FAILED;
    logger.error({ err: error }, 'CGM OAuth callback failed');

    res.status(HTTP_STATUS.BAD_REQUEST).json({
      error: 'OAuth callback failed',
      message: errorMessage,
    });
  }
}

/**
 * Connect using Levels API key
 *
 * POST /api/cgm/connect/levels
 * Body: { apiKey: string, externalUserId?: string }
 */
export async function connectLevels(req: AuthenticatedRequest, res: Response): Promise<void> {
  try {
    const userId = req.userId as string;
    const { apiKey, externalUserId } = req.body;

    if (!apiKey || typeof apiKey !== 'string' || apiKey.length < 10) {
      res.status(HTTP_STATUS.BAD_REQUEST).json({
        error: 'Invalid API key',
        message: 'A valid Levels API key is required',
      });
      return;
    }

    const connectionStatus = await cgmIntegrationService.connectLevels(
      userId,
      apiKey,
      externalUserId
    );

    logger.info({ userId }, 'Levels API key connected');

    res.status(HTTP_STATUS.OK).json({
      message: CGM_SUCCESS_MESSAGES.CONNECTED('LEVELS'),
      connection: connectionStatus,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to connect Levels';
    logger.error({ err: error }, 'Failed to connect Levels');

    res.status(HTTP_STATUS.BAD_REQUEST).json({
      error: 'Connection failed',
      message: errorMessage,
    });
  }
}

/**
 * Get CGM connection status
 *
 * GET /api/cgm/connections
 * Query: { provider?: 'DEXCOM' | 'LIBRE' | 'LEVELS', includeInactive?: boolean }
 */
export async function getConnections(req: AuthenticatedRequest, res: Response): Promise<void> {
  try {
    const userId = req.userId as string;
    const query = getCGMConnectionsQuerySchema.parse(req.query);
    const { provider, includeInactive } = query;

    let connections = await cgmIntegrationService.getConnectionStatus(userId, provider);

    // Filter out inactive connections unless requested
    if (!includeInactive) {
      connections = connections.filter((conn) => conn.isConnected);
    }

    res.status(HTTP_STATUS.OK).json({
      connections,
      count: connections.length,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to get connections';
    logger.error({ err: error }, 'Failed to get CGM connections');

    res.status(HTTP_STATUS.INTERNAL_SERVER_ERROR).json({
      error: 'Failed to retrieve connections',
      message: errorMessage,
    });
  }
}

/**
 * Disconnect a CGM provider
 *
 * DELETE /api/cgm/connections/:provider
 */
export async function disconnectProvider(req: AuthenticatedRequest, res: Response): Promise<void> {
  try {
    const userId = req.userId as string;
    const { provider } = req.params;

    // Validate provider
    const validatedProvider = cgmProviderSchema.parse(provider.toUpperCase());

    await cgmIntegrationService.disconnectProvider(userId, validatedProvider);

    logger.info({ userId, provider: validatedProvider }, 'CGM provider disconnected');

    res.status(HTTP_STATUS.OK).json({
      message: CGM_SUCCESS_MESSAGES.DISCONNECTED(validatedProvider),
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to disconnect';
    logger.error({ err: error }, 'Failed to disconnect CGM provider');

    if (errorMessage === CGM_ERROR_MESSAGES.CONNECTION_NOT_FOUND) {
      res.status(HTTP_STATUS.NOT_FOUND).json({
        error: 'Connection not found',
        message: errorMessage,
      });
      return;
    }

    res.status(HTTP_STATUS.BAD_REQUEST).json({
      error: 'Disconnect failed',
      message: errorMessage,
    });
  }
}

/**
 * Validate a CGM connection (check if tokens are valid and API is accessible)
 *
 * GET /api/cgm/connections/:provider/validate
 */
export async function validateConnection(req: AuthenticatedRequest, res: Response): Promise<void> {
  try {
    const userId = req.userId as string;
    const { provider } = req.params;

    // Validate provider
    const validatedProvider = cgmProviderSchema.parse(provider.toUpperCase());

    const isValid = await cgmIntegrationService.validateConnection(userId, validatedProvider);

    res.status(HTTP_STATUS.OK).json({
      provider: validatedProvider,
      isValid,
      message: isValid
        ? 'Connection is healthy'
        : 'Connection needs reauthorization',
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Validation failed';
    logger.error({ err: error }, 'Failed to validate CGM connection');

    res.status(HTTP_STATUS.BAD_REQUEST).json({
      error: 'Validation failed',
      message: errorMessage,
    });
  }
}
