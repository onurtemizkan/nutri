/**
 * CGM Integration Service Tests
 *
 * Unit tests for OAuth 2.0 authentication flows, token management,
 * and connection lifecycle for CGM platforms.
 *
 * These tests mock the database to run without requiring a database connection.
 */

import axios, { AxiosError } from 'axios';
import { CGMIntegrationService } from '../../services/cgmIntegrationService';
import { encrypt } from '../../utils/encryption';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock Prisma
jest.mock('../../config/database', () => ({
  __esModule: true,
  default: {
    cGMConnection: {
      findUnique: jest.fn(),
      findMany: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      upsert: jest.fn(),
    },
  },
}));

import prisma from '../../config/database';
const mockedPrisma = prisma as jest.Mocked<typeof prisma>;

describe('CGMIntegrationService', () => {
  let service: CGMIntegrationService;
  const testUserId = 'test-user-id';

  // Store original env values
  const originalEnv = process.env;

  beforeAll(() => {
    service = new CGMIntegrationService();
  });

  beforeEach(() => {
    // Reset environment
    process.env = { ...originalEnv };
    process.env.JWT_SECRET = 'test-jwt-secret-for-cgm-tests';
    process.env.NODE_ENV = 'development';

    // Set up OAuth credentials for tests
    process.env.DEXCOM_CLIENT_ID = 'test-dexcom-client-id';
    process.env.DEXCOM_CLIENT_SECRET = 'test-dexcom-client-secret';
    process.env.DEXCOM_REDIRECT_URI = 'https://app.example.com/oauth/dexcom/callback';

    process.env.LIBRE_CLIENT_ID = 'test-libre-client-id';
    process.env.LIBRE_CLIENT_SECRET = 'test-libre-client-secret';
    process.env.LIBRE_REDIRECT_URI = 'https://app.example.com/oauth/libre/callback';

    // Reset mocks
    jest.clearAllMocks();
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  // ============================================================================
  // generateAuthorizationUrl()
  // ============================================================================

  describe('generateAuthorizationUrl()', () => {
    it('should generate Dexcom authorization URL with correct parameters', async () => {
      const state = 'test-state-123';

      const url = await service.generateAuthorizationUrl('DEXCOM', state);

      expect(url).toContain('sandbox-api.dexcom.com');
      expect(url).toContain('client_id=test-dexcom-client-id');
      expect(url).toContain('redirect_uri=' + encodeURIComponent('https://app.example.com/oauth/dexcom/callback'));
      expect(url).toContain('response_type=code');
      expect(url).toContain('state=test-state-123');
      expect(url).toContain('scope=offline_access');
    });

    it('should generate Libre authorization URL with correct parameters', async () => {
      const state = 'test-state-456';

      const url = await service.generateAuthorizationUrl('LIBRE', state);

      expect(url).toContain('libreview.io');
      expect(url).toContain('client_id=test-libre-client-id');
      expect(url).toContain('redirect_uri=' + encodeURIComponent('https://app.example.com/oauth/libre/callback'));
      expect(url).toContain('response_type=code');
      expect(url).toContain('state=test-state-456');
    });

    it('should use production URL when NODE_ENV is production', async () => {
      process.env.NODE_ENV = 'production';

      const url = await service.generateAuthorizationUrl('DEXCOM', 'state');

      expect(url).toContain('api.dexcom.com');
      expect(url).not.toContain('sandbox');
    });

    it('should throw error for Levels provider (non-OAuth)', async () => {
      await expect(service.generateAuthorizationUrl('LEVELS', 'state'))
        .rejects.toThrow('Levels does not use OAuth');
    });

    it('should throw error when client ID is not configured', async () => {
      delete process.env.DEXCOM_CLIENT_ID;

      await expect(service.generateAuthorizationUrl('DEXCOM', 'state'))
        .rejects.toThrow('DEXCOM_CLIENT_ID environment variable is not set');
    });

    it('should throw error when client secret is not configured', async () => {
      delete process.env.DEXCOM_CLIENT_SECRET;

      await expect(service.generateAuthorizationUrl('DEXCOM', 'state'))
        .rejects.toThrow('DEXCOM_CLIENT_SECRET environment variable is not set');
    });

    it('should throw error when redirect URI is not configured', async () => {
      delete process.env.DEXCOM_REDIRECT_URI;

      await expect(service.generateAuthorizationUrl('DEXCOM', 'state'))
        .rejects.toThrow('DEXCOM_REDIRECT_URI environment variable is not set');
    });

    it('should use custom redirect URI when provided', async () => {
      const customUri = 'https://custom.app/callback';

      const url = await service.generateAuthorizationUrl('DEXCOM', 'state', customUri);

      expect(url).toContain('redirect_uri=' + encodeURIComponent(customUri));
    });
  });

  // ============================================================================
  // exchangeCodeForTokens()
  // ============================================================================

  describe('exchangeCodeForTokens()', () => {
    it('should exchange authorization code for tokens successfully', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          access_token: 'test-access-token',
          refresh_token: 'test-refresh-token',
          expires_in: 7200,
          token_type: 'Bearer',
          scope: 'offline_access',
        },
      });

      const result = await service.exchangeCodeForTokens('DEXCOM', 'auth-code-123');

      expect(result.accessToken).toBe('test-access-token');
      expect(result.refreshToken).toBe('test-refresh-token');
      expect(result.expiresIn).toBe(7200);
      expect(result.tokenType).toBe('Bearer');
      expect(result.scope).toBe('offline_access');

      expect(mockedAxios.post).toHaveBeenCalledWith(
        expect.stringContaining('token'),
        expect.any(URLSearchParams),
        expect.objectContaining({
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        })
      );
    });

    it('should handle token exchange error with error_description', async () => {
      const axiosError = new Error('Request failed') as AxiosError<{ error?: string; error_description?: string }>;
      Object.assign(axiosError, {
        isAxiosError: true,
        response: {
          data: {
            error: 'invalid_grant',
            error_description: 'Authorization code has expired',
          },
        },
      });
      mockedAxios.post.mockRejectedValueOnce(axiosError);
      mockedAxios.isAxiosError.mockReturnValue(true);

      await expect(service.exchangeCodeForTokens('DEXCOM', 'expired-code'))
        .rejects.toThrow('Authorization code has expired');
    });

    it('should handle token exchange error without description', async () => {
      const axiosError = new Error('Request failed') as AxiosError<{ error?: string; error_description?: string }>;
      Object.assign(axiosError, {
        isAxiosError: true,
        response: {
          data: {
            error: 'server_error',
          },
        },
      });
      mockedAxios.post.mockRejectedValueOnce(axiosError);
      mockedAxios.isAxiosError.mockReturnValue(true);

      await expect(service.exchangeCodeForTokens('DEXCOM', 'code'))
        .rejects.toThrow('server_error');
    });

    it('should throw generic error for non-axios errors', async () => {
      mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));
      mockedAxios.isAxiosError.mockReturnValue(false);

      await expect(service.exchangeCodeForTokens('DEXCOM', 'code'))
        .rejects.toThrow('OAuth authentication failed');
    });

    it('should throw error for Levels provider', async () => {
      await expect(service.exchangeCodeForTokens('LEVELS', 'code'))
        .rejects.toThrow('Levels does not use OAuth');
    });
  });

  // ============================================================================
  // refreshAccessToken()
  // ============================================================================

  describe('refreshAccessToken()', () => {
    it('should refresh access token successfully', async () => {
      const encryptedRefreshToken = encrypt('original-refresh-token');

      mockedAxios.post.mockResolvedValueOnce({
        data: {
          access_token: 'new-access-token',
          refresh_token: 'new-refresh-token',
          expires_in: 7200,
          token_type: 'Bearer',
        },
      });

      const result = await service.refreshAccessToken('DEXCOM', encryptedRefreshToken);

      expect(result.accessToken).toBe('new-access-token');
      expect(result.refreshToken).toBe('new-refresh-token');
      expect(result.expiresIn).toBe(7200);
    });

    it('should throw error when refresh fails', async () => {
      const encryptedRefreshToken = encrypt('invalid-refresh-token');

      mockedAxios.post.mockRejectedValueOnce(new Error('Token expired'));

      await expect(service.refreshAccessToken('DEXCOM', encryptedRefreshToken))
        .rejects.toThrow('Failed to refresh CGM access token');
    });

    it('should throw error for Levels provider', async () => {
      await expect(service.refreshAccessToken('LEVELS', 'token'))
        .rejects.toThrow('Levels does not use OAuth');
    });
  });

  // ============================================================================
  // createConnection()
  // ============================================================================

  describe('createConnection()', () => {
    it('should create a new CGM connection with encrypted tokens', async () => {
      const expiresAt = new Date(Date.now() + 7200000);
      const mockConnection = {
        id: 'connection-id',
        userId: testUserId,
        provider: 'DEXCOM',
        accessToken: 'encrypted-token',
        refreshToken: 'encrypted-refresh',
        expiresAt,
        scope: 'offline_access',
        isActive: true,
        connectedAt: new Date(),
        lastSyncAt: null,
        lastSyncStatus: null,
        externalUserId: null,
      };

      (mockedPrisma.cGMConnection.upsert as jest.Mock).mockResolvedValueOnce(mockConnection);

      const result = await service.createConnection(testUserId, {
        provider: 'DEXCOM',
        accessToken: 'test-access-token',
        refreshToken: 'test-refresh-token',
        expiresAt,
        scope: 'offline_access',
      });

      expect(result.provider).toBe('DEXCOM');
      expect(result.isConnected).toBe(true);
      expect(mockedPrisma.cGMConnection.upsert).toHaveBeenCalled();
    });

    it('should store external user ID when provided', async () => {
      const expiresAt = new Date(Date.now() + 7200000);
      const mockConnection = {
        id: 'connection-id',
        userId: testUserId,
        provider: 'DEXCOM',
        accessToken: 'encrypted-token',
        expiresAt,
        isActive: true,
        connectedAt: new Date(),
        lastSyncAt: null,
        lastSyncStatus: null,
        externalUserId: 'dexcom-user-123',
      };

      (mockedPrisma.cGMConnection.upsert as jest.Mock).mockResolvedValueOnce(mockConnection);

      const result = await service.createConnection(testUserId, {
        provider: 'DEXCOM',
        accessToken: 'token',
        expiresAt,
        externalUserId: 'dexcom-user-123',
      });

      expect(result.externalUserId).toBe('dexcom-user-123');
    });
  });

  // ============================================================================
  // disconnectProvider()
  // ============================================================================

  describe('disconnectProvider()', () => {
    it('should disconnect an active provider', async () => {
      const mockConnection = {
        id: 'connection-id',
        userId: testUserId,
        provider: 'DEXCOM',
        isActive: true,
      };

      (mockedPrisma.cGMConnection.findUnique as jest.Mock).mockResolvedValueOnce(mockConnection);
      (mockedPrisma.cGMConnection.update as jest.Mock).mockResolvedValueOnce({
        ...mockConnection,
        isActive: false,
        disconnectedAt: new Date(),
        accessToken: '',
        refreshToken: null,
      });

      await service.disconnectProvider(testUserId, 'DEXCOM');

      expect(mockedPrisma.cGMConnection.update).toHaveBeenCalledWith(
        expect.objectContaining({
          where: { id: 'connection-id' },
          data: expect.objectContaining({
            isActive: false,
            accessToken: '',
            refreshToken: null,
          }),
        })
      );
    });

    it('should throw error when connection does not exist', async () => {
      (mockedPrisma.cGMConnection.findUnique as jest.Mock).mockResolvedValueOnce(null);

      await expect(service.disconnectProvider(testUserId, 'DEXCOM'))
        .rejects.toThrow('CGM connection not found');
    });
  });

  // ============================================================================
  // getConnectionStatus()
  // ============================================================================

  describe('getConnectionStatus()', () => {
    it('should return all connections for user', async () => {
      const mockConnections = [
        { provider: 'DEXCOM', isActive: true, lastSyncAt: null, lastSyncStatus: null, connectedAt: new Date(), externalUserId: null },
        { provider: 'LIBRE', isActive: false, lastSyncAt: null, lastSyncStatus: null, connectedAt: new Date(), externalUserId: null },
      ];

      (mockedPrisma.cGMConnection.findMany as jest.Mock).mockResolvedValueOnce(mockConnections);

      const connections = await service.getConnectionStatus(testUserId);

      expect(connections).toHaveLength(2);
      expect(connections.map((c) => c.provider).sort()).toEqual(['DEXCOM', 'LIBRE']);
    });

    it('should return specific provider connection', async () => {
      const mockConnections = [
        { provider: 'DEXCOM', isActive: true, lastSyncAt: null, lastSyncStatus: null, connectedAt: new Date(), externalUserId: null },
      ];

      (mockedPrisma.cGMConnection.findMany as jest.Mock).mockResolvedValueOnce(mockConnections);

      const connections = await service.getConnectionStatus(testUserId, 'DEXCOM');

      expect(connections).toHaveLength(1);
      expect(connections[0].provider).toBe('DEXCOM');
    });

    it('should return empty array when no connections exist', async () => {
      (mockedPrisma.cGMConnection.findMany as jest.Mock).mockResolvedValueOnce([]);

      const connections = await service.getConnectionStatus(testUserId);

      expect(connections).toEqual([]);
    });
  });

  // ============================================================================
  // connectLevels()
  // ============================================================================

  describe('connectLevels()', () => {
    it('should connect Levels with API key', async () => {
      const mockConnection = {
        id: 'connection-id',
        userId: testUserId,
        provider: 'LEVELS',
        accessToken: 'encrypted-api-key',
        expiresAt: new Date('2099-12-31'),
        isActive: true,
        connectedAt: new Date(),
        lastSyncAt: null,
        lastSyncStatus: null,
        externalUserId: null,
      };

      (mockedPrisma.cGMConnection.upsert as jest.Mock).mockResolvedValueOnce(mockConnection);

      const result = await service.connectLevels(testUserId, 'levels-api-key-123');

      expect(result.provider).toBe('LEVELS');
      expect(result.isConnected).toBe(true);
    });

    it('should store external user ID when provided', async () => {
      const mockConnection = {
        id: 'connection-id',
        userId: testUserId,
        provider: 'LEVELS',
        accessToken: 'encrypted-api-key',
        expiresAt: new Date('2099-12-31'),
        isActive: true,
        connectedAt: new Date(),
        lastSyncAt: null,
        lastSyncStatus: null,
        externalUserId: 'levels-user-456',
      };

      (mockedPrisma.cGMConnection.upsert as jest.Mock).mockResolvedValueOnce(mockConnection);

      const result = await service.connectLevels(testUserId, 'api-key', 'levels-user-456');

      expect(result.externalUserId).toBe('levels-user-456');
    });
  });

  // ============================================================================
  // validateConnection()
  // ============================================================================

  describe('validateConnection()', () => {
    it('should return true for valid Dexcom connection', async () => {
      const mockConnection = {
        id: 'connection-id',
        userId: testUserId,
        provider: 'DEXCOM',
        accessToken: encrypt('valid-token'),
        refreshToken: null,
        expiresAt: new Date(Date.now() + 3600000),
        isActive: true,
      };

      (mockedPrisma.cGMConnection.findUnique as jest.Mock).mockResolvedValueOnce(mockConnection);
      mockedAxios.get.mockResolvedValueOnce({ data: {} });

      const isValid = await service.validateConnection(testUserId, 'DEXCOM');

      expect(isValid).toBe(true);
      expect(mockedAxios.get).toHaveBeenCalledWith(
        expect.stringContaining('dataRange'),
        expect.objectContaining({
          headers: { Authorization: expect.stringContaining('Bearer') },
        })
      );
    });

    it('should return false when connection does not exist', async () => {
      (mockedPrisma.cGMConnection.findUnique as jest.Mock).mockResolvedValueOnce(null);

      const isValid = await service.validateConnection(testUserId, 'DEXCOM');

      expect(isValid).toBe(false);
    });

    it('should return false when API call fails', async () => {
      const mockConnection = {
        id: 'connection-id',
        userId: testUserId,
        provider: 'DEXCOM',
        accessToken: encrypt('invalid-token'),
        refreshToken: null,
        expiresAt: new Date(Date.now() + 3600000),
        isActive: true,
      };

      (mockedPrisma.cGMConnection.findUnique as jest.Mock).mockResolvedValueOnce(mockConnection);
      mockedAxios.get.mockRejectedValueOnce(new Error('Unauthorized'));

      const isValid = await service.validateConnection(testUserId, 'DEXCOM');

      expect(isValid).toBe(false);
    });
  });

  // ============================================================================
  // updateSyncStatus()
  // ============================================================================

  describe('updateSyncStatus()', () => {
    it('should update sync status to success', async () => {
      (mockedPrisma.cGMConnection.update as jest.Mock).mockResolvedValueOnce({});

      await service.updateSyncStatus('connection-id', 'success');

      expect(mockedPrisma.cGMConnection.update).toHaveBeenCalledWith({
        where: { id: 'connection-id' },
        data: {
          lastSyncAt: expect.any(Date),
          lastSyncStatus: 'success',
          lastSyncError: null,
        },
      });
    });

    it('should update sync status with error', async () => {
      (mockedPrisma.cGMConnection.update as jest.Mock).mockResolvedValueOnce({});

      await service.updateSyncStatus('connection-id', 'error', 'API rate limit exceeded');

      expect(mockedPrisma.cGMConnection.update).toHaveBeenCalledWith({
        where: { id: 'connection-id' },
        data: {
          lastSyncAt: expect.any(Date),
          lastSyncStatus: 'error',
          lastSyncError: 'API rate limit exceeded',
        },
      });
    });
  });
});
