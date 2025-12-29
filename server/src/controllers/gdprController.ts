import crypto from 'crypto';
import { gdprService } from '../services/gdprService';
import { AuthenticatedRequest } from '../types';
import { requireAuth } from '../utils/authHelpers';
import { withErrorHandling } from '../utils/controllerHelpers';
import { HTTP_STATUS } from '../config/constants';
import { z } from 'zod';
import { ConsentPurpose, ConsentSource, ExportFormat, DeletionType } from '@prisma/client';

// ============================================================================
// VALIDATION SCHEMAS
// ============================================================================

const updateConsentSchema = z.object({
  purpose: z.nativeEnum(ConsentPurpose),
  granted: z.boolean(),
  version: z.string().optional(),
});

const updateConsentsSchema = z.array(updateConsentSchema);

const updatePrivacySettingsSchema = z.object({
  dataRetentionDays: z.number().int().min(30).max(3650).optional(), // 30 days to 10 years
  autoDeleteOldData: z.boolean().optional(),
  allowAnalytics: z.boolean().optional(),
  allowCrashReporting: z.boolean().optional(),
  allowPerformanceMonitoring: z.boolean().optional(),
  shareAnonymousData: z.boolean().optional(),
  shareWithPartners: z.boolean().optional(),
  marketingEmails: z.boolean().optional(),
  productUpdates: z.boolean().optional(),
  showOnLeaderboard: z.boolean().optional(),
  healthKitDataSharing: z.boolean().optional(),
  privacyPolicyVersion: z.string().optional(),
  termsVersion: z.string().optional(),
});

const requestExportSchema = z.object({
  format: z.nativeEnum(ExportFormat).optional(),
  includeRaw: z.boolean().optional(),
});

const requestDeletionSchema = z.object({
  deletionType: z.nativeEnum(DeletionType).optional(),
  reason: z.string().max(500).optional(),
});

const cancelDeletionSchema = z.object({
  reason: z.string().max(500).optional(),
});

const verifyDeletionSchema = z.object({
  verificationCode: z.string().min(1),
});

// ============================================================================
// CONTROLLER
// ============================================================================

export class GdprController {
  // ==========================================================================
  // CONSENT MANAGEMENT
  // ==========================================================================

  /**
   * Get all consent statuses for the authenticated user
   */
  getConsentStatus = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const consents = await gdprService.getConsentStatus(userId);

    res.status(HTTP_STATUS.OK).json({
      consents,
      availablePurposes: Object.values(ConsentPurpose),
    });
  });

  /**
   * Update consent for a specific purpose
   */
  updateConsent = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const data = updateConsentSchema.parse(req.body);
    const consent = await gdprService.updateConsent(userId, {
      ...data,
      ipAddress: req.ip,
      userAgent: req.get('User-Agent'),
      source: ConsentSource.APP,
    });

    res.status(HTTP_STATUS.OK).json(consent);
  });

  /**
   * Update multiple consents at once
   */
  updateConsents = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const data = updateConsentsSchema.parse(req.body);
    const consents = await gdprService.updateConsents(
      userId,
      data.map((c) => ({
        ...c,
        ipAddress: req.ip,
        userAgent: req.get('User-Agent'),
        source: ConsentSource.APP,
      }))
    );

    res.status(HTTP_STATUS.OK).json(consents);
  });

  // ==========================================================================
  // PRIVACY SETTINGS
  // ==========================================================================

  /**
   * Get privacy settings
   */
  getPrivacySettings = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const settings = await gdprService.getOrCreatePrivacySettings(userId);

    res.status(HTTP_STATUS.OK).json(settings);
  });

  /**
   * Update privacy settings
   */
  updatePrivacySettings = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const data = updatePrivacySettingsSchema.parse(req.body);
    const settings = await gdprService.updatePrivacySettings(userId, data);

    res.status(HTTP_STATUS.OK).json(settings);
  });

  // ==========================================================================
  // DATA EXPORT
  // ==========================================================================

  /**
   * Request a data export
   */
  requestDataExport = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const data = requestExportSchema.parse(req.body);
    const request = await gdprService.requestDataExport(userId, {
      ...data,
      ipAddress: req.ip,
      userAgent: req.get('User-Agent'),
    });

    res.status(HTTP_STATUS.CREATED).json(request);
  });

  /**
   * Get export request history
   */
  getExportRequests = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const requests = await gdprService.getExportRequests(userId);

    res.status(HTTP_STATUS.OK).json(requests);
  });

  /**
   * Download exported data
   * Note: In production, this would serve from secure storage
   */
  downloadExport = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { id } = req.params;
    const { token } = req.query;

    // Verify the export belongs to the user and is ready
    const requests = await gdprService.getExportRequests(userId);
    const request = requests.find((r) => r.id === id);

    if (!request) {
      res.status(HTTP_STATUS.NOT_FOUND).json({ error: 'Export not found' });
      return;
    }

    if (request.status !== 'COMPLETED') {
      res.status(HTTP_STATUS.BAD_REQUEST).json({ error: 'Export not ready for download' });
      return;
    }

    if (request.expiresAt && new Date(request.expiresAt) < new Date()) {
      res.status(410).json({ error: 'Export link has expired' });
      return;
    }

    // Validate download token for additional security
    // Token is embedded in the stored downloadUrl as ?token=XXX
    const tokenStr = typeof token === 'string' ? token : undefined;
    if (request.downloadUrl && tokenStr) {
      const storedUrl = new URL(request.downloadUrl, 'http://localhost');
      const expectedToken = storedUrl.searchParams.get('token');
      if (expectedToken) {
        // Use timing-safe comparison to prevent timing attacks
        const tokenBuffer = Buffer.from(tokenStr);
        const expectedBuffer = Buffer.from(expectedToken);
        const tokensMatch =
          tokenBuffer.length === expectedBuffer.length &&
          crypto.timingSafeEqual(tokenBuffer, expectedBuffer);

        if (!tokensMatch) {
          res.status(HTTP_STATUS.UNAUTHORIZED).json({ error: 'Invalid download token' });
          return;
        }
      }
    }

    // Collect user data for download
    const exportData = await gdprService.collectUserData(userId, request.includeRaw);

    // Set appropriate headers
    res.setHeader('Content-Type', 'application/json');
    res.setHeader(
      'Content-Disposition',
      `attachment; filename="nutri-data-export-${new Date().toISOString().split('T')[0]}.json"`
    );

    res.status(HTTP_STATUS.OK).json(exportData);
  });

  // ==========================================================================
  // ACCOUNT DELETION
  // ==========================================================================

  /**
   * Request account deletion
   */
  requestAccountDeletion = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const data = requestDeletionSchema.parse(req.body);
    const request = await gdprService.requestAccountDeletion(userId, {
      ...data,
      ipAddress: req.ip,
      userAgent: req.get('User-Agent'),
    });

    res.status(HTTP_STATUS.CREATED).json(request);
  });

  /**
   * Verify deletion request
   * Requires authentication to ensure the request belongs to the user
   */
  verifyDeletionRequest = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { id } = req.params;
    const { verificationCode } = verifyDeletionSchema.parse(req.body);

    const result = await gdprService.verifyDeletionRequest(userId, id, verificationCode);

    res.status(HTTP_STATUS.OK).json(result);
  });

  /**
   * Cancel deletion request
   */
  cancelDeletionRequest = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const { id } = req.params;
    const data = cancelDeletionSchema.parse(req.body);

    const result = await gdprService.cancelDeletionRequest(userId, id, data.reason);

    res.status(HTTP_STATUS.OK).json(result);
  });

  /**
   * Get deletion request history
   */
  getDeletionRequests = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const requests = await gdprService.getDeletionRequests(userId);

    res.status(HTTP_STATUS.OK).json(requests);
  });

  // ==========================================================================
  // DATA ACCESS LOGS
  // ==========================================================================

  /**
   * Get data access logs for the user
   */
  getDataAccessLogs = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const limit = req.query.limit ? parseInt(req.query.limit as string, 10) : 50;
    const logs = await gdprService.getDataAccessLogs(userId, Math.min(limit, 100));

    res.status(HTTP_STATUS.OK).json(logs);
  });

  // ==========================================================================
  // PRIVACY SUMMARY
  // ==========================================================================

  /**
   * Get complete privacy summary (dashboard endpoint)
   */
  getPrivacySummary = withErrorHandling<AuthenticatedRequest>(async (req, res) => {
    const userId = requireAuth(req, res);
    if (!userId) return;

    const [consents, privacySettings, exportRequests, deletionRequests] = await Promise.all([
      gdprService.getConsentStatus(userId),
      gdprService.getOrCreatePrivacySettings(userId),
      gdprService.getExportRequests(userId),
      gdprService.getDeletionRequests(userId),
    ]);

    res.status(HTTP_STATUS.OK).json({
      consents,
      privacySettings,
      exportRequests: exportRequests.slice(0, 5),
      deletionRequests: deletionRequests.slice(0, 5),
      pendingDeletionRequest: deletionRequests.find((r) => r.status === 'PENDING'),
    });
  });
}

export const gdprController = new GdprController();
