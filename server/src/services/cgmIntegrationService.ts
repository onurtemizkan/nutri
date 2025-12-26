/**
 * CGM Integration Service
 *
 * Handles OAuth 2.0 authentication flows for CGM platforms (Dexcom, LibreView, Levels).
 * Manages secure token storage, refresh, and connection lifecycle.
 */

import axios, { AxiosError } from 'axios';
import prisma from '../config/database';
import { encrypt, decrypt } from '../utils/encryption';
import { GlucoseSource } from '@prisma/client';
import {
  CGM_OAUTH_CONFIG,
  CGM_TOKEN_REFRESH,
  CGM_ERROR_MESSAGES,
} from '../config/constants';
import {
  CGMProvider,
  CGMOAuthTokenResponse,
  CGMConnectionStatus,
  CGMConnectionInput,
} from '../types';
import { logger } from '../config/logger';

// ============================================================================
// TYPES
// ============================================================================

interface OAuthConfig {
  clientId: string;
  clientSecret: string;
  redirectUri: string;
  authUrl: string;
  tokenUrl: string;
  scopes: readonly string[];
}

interface TokenResponse {
  access_token: string;
  refresh_token?: string;
  expires_in: number;
  token_type: string;
  scope?: string;
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * Get OAuth configuration for a provider from environment variables
 */
function getOAuthConfig(provider: CGMProvider): OAuthConfig {
  const useSandbox = process.env.NODE_ENV !== 'production';

  switch (provider) {
    case 'DEXCOM': {
      const config = CGM_OAUTH_CONFIG.DEXCOM;
      return {
        clientId: process.env.DEXCOM_CLIENT_ID || '',
        clientSecret: process.env.DEXCOM_CLIENT_SECRET || '',
        redirectUri: process.env.DEXCOM_REDIRECT_URI || '',
        authUrl: useSandbox ? config.SANDBOX_AUTH_URL : config.AUTH_URL,
        tokenUrl: useSandbox ? config.SANDBOX_TOKEN_URL : config.TOKEN_URL,
        scopes: config.SCOPES,
      };
    }
    case 'LIBRE': {
      const config = CGM_OAUTH_CONFIG.LIBRE;
      return {
        clientId: process.env.LIBRE_CLIENT_ID || '',
        clientSecret: process.env.LIBRE_CLIENT_SECRET || '',
        redirectUri: process.env.LIBRE_REDIRECT_URI || '',
        authUrl: config.AUTH_URL,
        tokenUrl: config.TOKEN_URL,
        scopes: config.SCOPES,
      };
    }
    case 'LEVELS':
      // Levels uses API key, not OAuth
      throw new Error('Levels uses API key authentication, not OAuth');
    default:
      throw new Error(CGM_ERROR_MESSAGES.PROVIDER_NOT_SUPPORTED);
  }
}

/**
 * Validate that required OAuth credentials are configured
 */
function validateOAuthConfig(config: OAuthConfig, provider: CGMProvider): void {
  if (!config.clientId) {
    throw new Error(`${provider}_CLIENT_ID environment variable is not set`);
  }
  if (!config.clientSecret) {
    throw new Error(`${provider}_CLIENT_SECRET environment variable is not set`);
  }
  if (!config.redirectUri) {
    throw new Error(`${provider}_REDIRECT_URI environment variable is not set`);
  }
}

// ============================================================================
// CGM INTEGRATION SERVICE
// ============================================================================

export class CGMIntegrationService {
  /**
   * Generate OAuth authorization URL for a CGM provider
   * User will be redirected to this URL to authorize access
   */
  async generateAuthorizationUrl(
    provider: CGMProvider,
    state: string,
    customRedirectUri?: string
  ): Promise<string> {
    if (provider === 'LEVELS') {
      throw new Error('Levels does not use OAuth. Use API key authentication instead.');
    }

    const config = getOAuthConfig(provider);
    validateOAuthConfig(config, provider);

    const redirectUri = customRedirectUri || config.redirectUri;
    const params = new URLSearchParams({
      client_id: config.clientId,
      redirect_uri: redirectUri,
      response_type: 'code',
      scope: config.scopes.join(' '),
      state,
    });

    return `${config.authUrl}?${params.toString()}`;
  }

  /**
   * Exchange authorization code for access tokens
   * Called after user authorizes access and is redirected back
   */
  async exchangeCodeForTokens(
    provider: CGMProvider,
    code: string,
    customRedirectUri?: string
  ): Promise<CGMOAuthTokenResponse> {
    if (provider === 'LEVELS') {
      throw new Error('Levels does not use OAuth');
    }

    const config = getOAuthConfig(provider);
    validateOAuthConfig(config, provider);

    const redirectUri = customRedirectUri || config.redirectUri;

    try {
      const response = await axios.post<TokenResponse>(
        config.tokenUrl,
        new URLSearchParams({
          grant_type: 'authorization_code',
          code,
          client_id: config.clientId,
          client_secret: config.clientSecret,
          redirect_uri: redirectUri,
        }),
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
        }
      );

      return {
        accessToken: response.data.access_token,
        refreshToken: response.data.refresh_token,
        expiresIn: response.data.expires_in,
        tokenType: response.data.token_type,
        scope: response.data.scope,
      };
    } catch (error) {
      logger.error({ err: error, provider }, 'Failed to exchange authorization code');

      if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError<{ error?: string; error_description?: string }>;
        const errorMessage =
          axiosError.response?.data?.error_description ||
          axiosError.response?.data?.error ||
          CGM_ERROR_MESSAGES.OAUTH_FAILED;
        throw new Error(errorMessage);
      }

      throw new Error(CGM_ERROR_MESSAGES.OAUTH_FAILED);
    }
  }

  /**
   * Refresh an expired access token using the refresh token
   */
  async refreshAccessToken(
    provider: CGMProvider,
    encryptedRefreshToken: string
  ): Promise<CGMOAuthTokenResponse> {
    if (provider === 'LEVELS') {
      throw new Error('Levels does not use OAuth');
    }

    const config = getOAuthConfig(provider);
    validateOAuthConfig(config, provider);

    const refreshToken = decrypt(encryptedRefreshToken);

    try {
      const response = await axios.post<TokenResponse>(
        config.tokenUrl,
        new URLSearchParams({
          grant_type: 'refresh_token',
          refresh_token: refreshToken,
          client_id: config.clientId,
          client_secret: config.clientSecret,
        }),
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
        }
      );

      return {
        accessToken: response.data.access_token,
        refreshToken: response.data.refresh_token,
        expiresIn: response.data.expires_in,
        tokenType: response.data.token_type,
        scope: response.data.scope,
      };
    } catch (error) {
      logger.error({ err: error, provider }, 'Failed to refresh access token');
      throw new Error(CGM_ERROR_MESSAGES.TOKEN_REFRESH_FAILED);
    }
  }

  /**
   * Create or update a CGM connection for a user
   * Encrypts and stores OAuth tokens securely
   */
  async createConnection(userId: string, data: CGMConnectionInput): Promise<CGMConnectionStatus> {
    const provider = data.provider as GlucoseSource;

    // Encrypt tokens before storing
    const encryptedAccessToken = encrypt(data.accessToken);
    const encryptedRefreshToken = data.refreshToken ? encrypt(data.refreshToken) : null;

    const connection = await prisma.cGMConnection.upsert({
      where: {
        userId_provider: {
          userId,
          provider,
        },
      },
      create: {
        userId,
        provider,
        accessToken: encryptedAccessToken,
        refreshToken: encryptedRefreshToken,
        expiresAt: data.expiresAt,
        scope: data.scope,
        isActive: true,
        externalUserId: data.externalUserId,
        metadata: data.metadata,
      },
      update: {
        accessToken: encryptedAccessToken,
        refreshToken: encryptedRefreshToken,
        expiresAt: data.expiresAt,
        scope: data.scope,
        isActive: true,
        disconnectedAt: null,
        externalUserId: data.externalUserId,
        metadata: data.metadata,
      },
    });

    logger.info({ userId, provider }, 'CGM connection created/updated');

    return {
      provider: connection.provider as CGMProvider,
      isConnected: connection.isActive,
      lastSyncAt: connection.lastSyncAt || undefined,
      lastSyncStatus: connection.lastSyncStatus || undefined,
      connectedAt: connection.connectedAt || undefined,
      externalUserId: connection.externalUserId || undefined,
    };
  }

  /**
   * Disconnect a CGM provider for a user
   */
  async disconnectProvider(userId: string, provider: CGMProvider): Promise<void> {
    const connection = await prisma.cGMConnection.findUnique({
      where: {
        userId_provider: {
          userId,
          provider: provider as GlucoseSource,
        },
      },
    });

    if (!connection) {
      throw new Error(CGM_ERROR_MESSAGES.CONNECTION_NOT_FOUND);
    }

    await prisma.cGMConnection.update({
      where: { id: connection.id },
      data: {
        isActive: false,
        disconnectedAt: new Date(),
        accessToken: '', // Clear encrypted token
        refreshToken: null,
      },
    });

    logger.info({ userId, provider }, 'CGM connection disconnected');
  }

  /**
   * Get connection status for a user's CGM providers
   */
  async getConnectionStatus(
    userId: string,
    provider?: CGMProvider
  ): Promise<CGMConnectionStatus[]> {
    const where = provider
      ? { userId, provider: provider as GlucoseSource }
      : { userId };

    const connections = await prisma.cGMConnection.findMany({
      where,
      select: {
        provider: true,
        isActive: true,
        lastSyncAt: true,
        lastSyncStatus: true,
        connectedAt: true,
        externalUserId: true,
        expiresAt: true,
      },
    });

    return connections.map((conn) => ({
      provider: conn.provider as CGMProvider,
      isConnected: conn.isActive,
      lastSyncAt: conn.lastSyncAt || undefined,
      lastSyncStatus: conn.lastSyncStatus || undefined,
      connectedAt: conn.connectedAt || undefined,
      externalUserId: conn.externalUserId || undefined,
    }));
  }

  /**
   * Get active connection with decrypted tokens (for API calls)
   * Only used internally by sync services
   */
  async getActiveConnection(
    userId: string,
    provider: CGMProvider
  ): Promise<{
    accessToken: string;
    refreshToken: string | null;
    expiresAt: Date;
    connectionId: string;
  } | null> {
    const connection = await prisma.cGMConnection.findUnique({
      where: {
        userId_provider: {
          userId,
          provider: provider as GlucoseSource,
        },
      },
    });

    if (!connection || !connection.isActive) {
      return null;
    }

    // Check if token needs refresh
    const now = new Date();
    const bufferTime = CGM_TOKEN_REFRESH.REFRESH_BUFFER_MS;
    const tokenExpiresAt = new Date(connection.expiresAt.getTime() - bufferTime);

    if (now >= tokenExpiresAt && connection.refreshToken) {
      // Token expired or about to expire - refresh it
      try {
        const newTokens = await this.refreshAccessToken(
          provider,
          connection.refreshToken
        );

        // Update stored tokens
        const encryptedAccessToken = encrypt(newTokens.accessToken);
        const encryptedRefreshToken = newTokens.refreshToken
          ? encrypt(newTokens.refreshToken)
          : connection.refreshToken;

        const expiresAt = new Date(Date.now() + newTokens.expiresIn * 1000);

        await prisma.cGMConnection.update({
          where: { id: connection.id },
          data: {
            accessToken: encryptedAccessToken,
            refreshToken: encryptedRefreshToken,
            expiresAt,
          },
        });

        return {
          accessToken: newTokens.accessToken,
          refreshToken: newTokens.refreshToken || null,
          expiresAt,
          connectionId: connection.id,
        };
      } catch (error) {
        logger.error({ err: error, userId, provider }, 'Failed to refresh token');
        // Mark connection as needing reauthorization
        await prisma.cGMConnection.update({
          where: { id: connection.id },
          data: {
            lastSyncStatus: 'token_expired',
            lastSyncError: 'Token refresh failed - reauthorization required',
          },
        });
        return null;
      }
    }

    // Token is still valid - decrypt and return
    return {
      accessToken: decrypt(connection.accessToken),
      refreshToken: connection.refreshToken ? decrypt(connection.refreshToken) : null,
      expiresAt: connection.expiresAt,
      connectionId: connection.id,
    };
  }

  /**
   * Update sync status for a connection
   */
  async updateSyncStatus(
    connectionId: string,
    status: 'success' | 'error' | 'partial',
    error?: string
  ): Promise<void> {
    await prisma.cGMConnection.update({
      where: { id: connectionId },
      data: {
        lastSyncAt: new Date(),
        lastSyncStatus: status,
        lastSyncError: error || null,
      },
    });
  }

  /**
   * Connect Levels using API key (non-OAuth flow)
   */
  async connectLevels(
    userId: string,
    apiKey: string,
    externalUserId?: string
  ): Promise<CGMConnectionStatus> {
    // For Levels, we store the API key encrypted as the "access token"
    const encryptedApiKey = encrypt(apiKey);

    const connection = await prisma.cGMConnection.upsert({
      where: {
        userId_provider: {
          userId,
          provider: GlucoseSource.LEVELS,
        },
      },
      create: {
        userId,
        provider: GlucoseSource.LEVELS,
        accessToken: encryptedApiKey,
        expiresAt: new Date('2099-12-31'), // API keys don't expire
        isActive: true,
        externalUserId,
      },
      update: {
        accessToken: encryptedApiKey,
        isActive: true,
        disconnectedAt: null,
        externalUserId,
      },
    });

    logger.info({ userId, provider: 'LEVELS' }, 'Levels API key connected');

    return {
      provider: 'LEVELS',
      isConnected: connection.isActive,
      lastSyncAt: connection.lastSyncAt || undefined,
      lastSyncStatus: connection.lastSyncStatus || undefined,
      connectedAt: connection.connectedAt || undefined,
      externalUserId: connection.externalUserId || undefined,
    };
  }

  /**
   * Validate that a connection is healthy (tokens valid, API accessible)
   */
  async validateConnection(userId: string, provider: CGMProvider): Promise<boolean> {
    const connection = await this.getActiveConnection(userId, provider);

    if (!connection) {
      return false;
    }

    // For OAuth providers, try a simple API call to verify
    try {
      if (provider === 'DEXCOM') {
        const useSandbox = process.env.NODE_ENV !== 'production';
        const baseUrl = useSandbox
          ? CGM_OAUTH_CONFIG.DEXCOM.SANDBOX_API_BASE
          : CGM_OAUTH_CONFIG.DEXCOM.API_BASE;

        await axios.get(`${baseUrl}/users/self/dataRange`, {
          headers: {
            Authorization: `Bearer ${connection.accessToken}`,
          },
        });
      } else if (provider === 'LIBRE') {
        await axios.get(`${CGM_OAUTH_CONFIG.LIBRE.API_BASE}/user`, {
          headers: {
            Authorization: `Bearer ${connection.accessToken}`,
          },
        });
      } else if (provider === 'LEVELS') {
        await axios.get(`${CGM_OAUTH_CONFIG.LEVELS.API_BASE}/user`, {
          headers: {
            Authorization: `Bearer ${connection.accessToken}`,
          },
        });
      }

      return true;
    } catch (error) {
      logger.warn({ err: error, userId, provider }, 'Connection validation failed');
      return false;
    }
  }
}

// Export singleton instance
export const cgmIntegrationService = new CGMIntegrationService();
