import prisma from '../config/database';
import { logger } from '../config/logger';
import {
  ConsentPurpose,
  ConsentSource,
  DataRequestStatus,
  ExportFormat,
  DeletionType,
  DataAccessAction,
  Prisma,
} from '@prisma/client';
import crypto from 'crypto';

// ============================================================================
// TYPES
// ============================================================================

interface ConsentUpdate {
  purpose: ConsentPurpose;
  granted: boolean;
  ipAddress?: string;
  userAgent?: string;
  source?: ConsentSource;
  version?: string;
}

interface PrivacySettingsUpdate {
  dataRetentionDays?: number;
  autoDeleteOldData?: boolean;
  allowAnalytics?: boolean;
  allowCrashReporting?: boolean;
  allowPerformanceMonitoring?: boolean;
  shareAnonymousData?: boolean;
  shareWithPartners?: boolean;
  marketingEmails?: boolean;
  productUpdates?: boolean;
  showOnLeaderboard?: boolean;
  healthKitDataSharing?: boolean;
  privacyPolicyVersion?: string;
  termsVersion?: string;
}

interface DataExportOptions {
  format?: ExportFormat;
  includeRaw?: boolean;
  ipAddress?: string;
  userAgent?: string;
}

interface DeletionRequestOptions {
  deletionType?: DeletionType;
  reason?: string;
  ipAddress?: string;
  userAgent?: string;
}

interface DataAccessLogEntry {
  userId?: string;
  action: DataAccessAction;
  resource: string;
  resourceId?: string;
  accessedBy?: string;
  accessorId?: string;
  ipAddress?: string;
  userAgent?: string;
  metadata?: Prisma.InputJsonValue;
}

// Grace period for account deletion (days)
const DELETION_GRACE_PERIOD_DAYS = 30;
// Export download link expiration (hours)
const EXPORT_LINK_EXPIRATION_HOURS = 72;

// ============================================================================
// SERVICE
// ============================================================================

class GdprService {
  // ==========================================================================
  // CONSENT MANAGEMENT
  // ==========================================================================

  /**
   * Get all consent statuses for a user
   */
  async getConsentStatus(userId: string) {
    const consents = await prisma.userConsent.findMany({
      where: { userId },
      orderBy: { purpose: 'asc' },
    });

    // Create a map with all purposes and their status
    const consentMap: Record<
      string,
      { granted: boolean; grantedAt: Date | null; revokedAt: Date | null; version: string }
    > = {};

    // Initialize all purposes as not granted
    for (const purpose of Object.values(ConsentPurpose)) {
      consentMap[purpose] = { granted: false, grantedAt: null, revokedAt: null, version: '1.0' };
    }

    // Update with actual consent records
    for (const consent of consents) {
      consentMap[consent.purpose] = {
        granted: consent.granted,
        grantedAt: consent.grantedAt,
        revokedAt: consent.revokedAt,
        version: consent.version,
      };
    }

    return consentMap;
  }

  /**
   * Update consent for a specific purpose
   */
  async updateConsent(userId: string, update: ConsentUpdate) {
    const {
      purpose,
      granted,
      ipAddress,
      userAgent,
      source = ConsentSource.APP,
      version = '1.0',
    } = update;

    // Essential consent cannot be revoked
    if (purpose === ConsentPurpose.ESSENTIAL && !granted) {
      throw new Error('Essential consent cannot be revoked while using the service');
    }

    const consent = await prisma.userConsent.upsert({
      where: {
        userId_purpose: { userId, purpose },
      },
      create: {
        userId,
        purpose,
        granted,
        grantedAt: granted ? new Date() : null,
        revokedAt: granted ? null : new Date(),
        ipAddress,
        userAgent,
        source,
        version,
      },
      update: {
        granted,
        grantedAt: granted ? new Date() : undefined,
        revokedAt: granted ? null : new Date(),
        ipAddress,
        userAgent,
        source,
        version,
      },
    });

    // Log the consent change
    await this.logDataAccess({
      userId,
      action: DataAccessAction.MODIFY,
      resource: 'consent',
      resourceId: consent.id,
      accessedBy: 'user',
      accessorId: userId,
      ipAddress,
      userAgent,
      metadata: { purpose, granted },
    });

    logger.info({ userId, purpose, granted }, 'Consent updated');

    return consent;
  }

  /**
   * Update multiple consents at once
   */
  async updateConsents(userId: string, consents: ConsentUpdate[]) {
    const results = [];
    for (const consent of consents) {
      const result = await this.updateConsent(userId, consent);
      results.push(result);
    }
    return results;
  }

  // ==========================================================================
  // PRIVACY SETTINGS
  // ==========================================================================

  /**
   * Get or create privacy settings for a user
   */
  async getOrCreatePrivacySettings(userId: string) {
    let settings = await prisma.privacySettings.findUnique({
      where: { userId },
    });

    if (!settings) {
      settings = await prisma.privacySettings.create({
        data: { userId },
      });
    }

    return settings;
  }

  /**
   * Update privacy settings
   */
  async updatePrivacySettings(userId: string, update: PrivacySettingsUpdate) {
    // Handle policy acceptance timestamps
    const data: Record<string, unknown> = { ...update };

    if (update.privacyPolicyVersion) {
      data.privacyPolicyAcceptedAt = new Date();
    }
    if (update.termsVersion) {
      data.termsAcceptedAt = new Date();
    }

    const settings = await prisma.privacySettings.upsert({
      where: { userId },
      create: {
        userId,
        ...data,
      },
      update: data,
    });

    logger.info({ userId }, 'Privacy settings updated');

    return settings;
  }

  // ==========================================================================
  // DATA EXPORT (Right to Portability - GDPR Article 20)
  // ==========================================================================

  /**
   * Request a data export
   */
  async requestDataExport(userId: string, options: DataExportOptions = {}) {
    const { format = ExportFormat.JSON, includeRaw = false, ipAddress, userAgent } = options;

    // Check for existing pending/processing request
    const existingRequest = await prisma.dataExportRequest.findFirst({
      where: {
        userId,
        status: { in: [DataRequestStatus.PENDING, DataRequestStatus.PROCESSING] },
      },
    });

    if (existingRequest) {
      throw new Error('You already have a pending data export request');
    }

    const request = await prisma.dataExportRequest.create({
      data: {
        userId,
        format,
        includeRaw,
        ipAddress,
        userAgent,
      },
    });

    // Log the request
    await this.logDataAccess({
      userId,
      action: DataAccessAction.EXPORT,
      resource: 'data_export_request',
      resourceId: request.id,
      accessedBy: 'user',
      accessorId: userId,
      ipAddress,
      userAgent,
    });

    logger.info({ userId, requestId: request.id, format }, 'Data export requested');

    // In a real implementation, this would queue a background job
    // For now, we'll process it immediately in a simplified way
    await this.processDataExport(request.id);

    return request;
  }

  /**
   * Process a data export request
   */
  async processDataExport(requestId: string) {
    const request = await prisma.dataExportRequest.findUnique({
      where: { id: requestId },
      include: { user: true },
    });

    if (!request) {
      throw new Error('Export request not found');
    }

    // Update status to processing
    await prisma.dataExportRequest.update({
      where: { id: requestId },
      data: { status: DataRequestStatus.PROCESSING, processedAt: new Date() },
    });

    try {
      // Collect all user data
      const exportData = await this.collectUserData(request.userId, request.includeRaw);

      // Generate download URL (in production, this would upload to secure storage)
      const downloadToken = crypto.randomBytes(32).toString('hex');
      const downloadUrl = `/api/privacy/exports/${requestId}/download?token=${downloadToken}`;

      // Calculate file size (approximate)
      const jsonData = JSON.stringify(exportData, null, 2);
      const fileSize = Buffer.byteLength(jsonData, 'utf8');

      // Update request with completion details
      await prisma.dataExportRequest.update({
        where: { id: requestId },
        data: {
          status: DataRequestStatus.COMPLETED,
          completedAt: new Date(),
          expiresAt: new Date(Date.now() + EXPORT_LINK_EXPIRATION_HOURS * 60 * 60 * 1000),
          downloadUrl,
          fileSize,
        },
      });

      logger.info({ userId: request.userId, requestId }, 'Data export completed');

      return {
        downloadUrl,
        expiresAt: new Date(Date.now() + EXPORT_LINK_EXPIRATION_HOURS * 60 * 60 * 1000),
      };
    } catch (error) {
      await prisma.dataExportRequest.update({
        where: { id: requestId },
        data: {
          status: DataRequestStatus.FAILED,
          errorMessage: error instanceof Error ? error.message : 'Unknown error',
          retryCount: { increment: 1 },
        },
      });

      logger.error({ requestId, error }, 'Data export failed');
      throw error;
    }
  }

  /**
   * Collect all user data for export
   */
  async collectUserData(userId: string, includeRaw: boolean) {
    const [
      user,
      meals,
      waterIntakes,
      weightRecords,
      healthMetrics,
      activities,
      supplements,
      supplementLogs,
      consents,
      privacySettings,
    ] = await Promise.all([
      prisma.user.findUnique({
        where: { id: userId },
        select: {
          id: true,
          email: true,
          name: true,
          createdAt: true,
          dateOfBirth: true,
          biologicalSex: true,
          goalCalories: true,
          goalProtein: true,
          goalCarbs: true,
          goalFat: true,
          goalWater: true,
          currentWeight: true,
          goalWeight: true,
          height: true,
          activityLevel: true,
          primaryGoal: true,
          dietaryPreferences: true,
          timezone: true,
        },
      }),
      prisma.meal.findMany({
        where: { userId },
        orderBy: { consumedAt: 'desc' },
      }),
      prisma.waterIntake.findMany({
        where: { userId },
        orderBy: { recordedAt: 'desc' },
      }),
      prisma.weightRecord.findMany({
        where: { userId },
        orderBy: { recordedAt: 'desc' },
      }),
      prisma.healthMetric.findMany({
        where: { userId },
        orderBy: { recordedAt: 'desc' },
      }),
      prisma.activity.findMany({
        where: { userId },
        orderBy: { startedAt: 'desc' },
      }),
      prisma.supplement.findMany({
        where: { userId },
        orderBy: { createdAt: 'desc' },
      }),
      prisma.supplementLog.findMany({
        where: { userId },
        orderBy: { takenAt: 'desc' },
      }),
      prisma.userConsent.findMany({
        where: { userId },
      }),
      prisma.privacySettings.findUnique({
        where: { userId },
      }),
    ]);

    const exportData: Record<string, unknown> = {
      exportDate: new Date().toISOString(),
      exportFormat: 'Nutri Data Export v1.0',
      user,
      nutrition: {
        meals,
        waterIntakes,
        supplements,
        supplementLogs,
      },
      health: {
        weightRecords,
        healthMetrics,
        activities,
      },
      privacy: {
        consents,
        privacySettings,
      },
    };

    // Include ML data if requested
    if (includeRaw) {
      const [mlFeatures, mlPredictions, mlInsights, mlProfile] = await Promise.all([
        prisma.mLFeature.findMany({ where: { userId } }),
        prisma.mLPrediction.findMany({ where: { userId } }),
        prisma.mLInsight.findMany({ where: { userId } }),
        prisma.userMLProfile.findUnique({ where: { userId } }),
      ]);

      exportData.mlData = {
        features: mlFeatures,
        predictions: mlPredictions,
        insights: mlInsights,
        profile: mlProfile,
      };
    }

    return exportData;
  }

  /**
   * Get export request status
   */
  async getExportRequests(userId: string) {
    return prisma.dataExportRequest.findMany({
      where: { userId },
      orderBy: { requestedAt: 'desc' },
      take: 10,
    });
  }

  // ==========================================================================
  // ACCOUNT DELETION (Right to Erasure - GDPR Article 17)
  // ==========================================================================

  /**
   * Request account deletion
   */
  async requestAccountDeletion(userId: string, options: DeletionRequestOptions = {}) {
    const { deletionType = DeletionType.FULL, reason, ipAddress, userAgent } = options;

    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { email: true },
    });

    if (!user) {
      throw new Error('User not found');
    }

    // Check for existing pending request
    const existingRequest = await prisma.dataDeletionRequest.findFirst({
      where: {
        userId,
        status: { in: [DataRequestStatus.PENDING, DataRequestStatus.PROCESSING] },
      },
    });

    if (existingRequest) {
      throw new Error('You already have a pending account deletion request');
    }

    // Generate verification code
    const verificationCode = crypto.randomBytes(32).toString('hex');

    // Schedule deletion after grace period
    const scheduledAt = new Date();
    scheduledAt.setDate(scheduledAt.getDate() + DELETION_GRACE_PERIOD_DAYS);

    const request = await prisma.dataDeletionRequest.create({
      data: {
        userId,
        userEmail: user.email,
        deletionType,
        verificationCode,
        scheduledAt,
        reason,
        ipAddress,
        userAgent,
      },
    });

    // Log the request
    await this.logDataAccess({
      userId,
      action: DataAccessAction.DELETE,
      resource: 'deletion_request',
      resourceId: request.id,
      accessedBy: 'user',
      accessorId: userId,
      ipAddress,
      userAgent,
      metadata: { deletionType, scheduledAt },
    });

    logger.info({ userId, requestId: request.id, scheduledAt }, 'Account deletion requested');

    // In production, send verification email here

    return {
      id: request.id,
      scheduledAt,
      gracePeriodDays: DELETION_GRACE_PERIOD_DAYS,
      message: `Account deletion scheduled for ${scheduledAt.toISOString()}. You can cancel this request within the grace period.`,
    };
  }

  /**
   * Verify deletion request
   */
  async verifyDeletionRequest(requestId: string, verificationCode: string) {
    const request = await prisma.dataDeletionRequest.findUnique({
      where: { id: requestId },
    });

    if (!request) {
      throw new Error('Deletion request not found');
    }

    if (request.verificationCode !== verificationCode) {
      throw new Error('Invalid verification code');
    }

    if (request.status !== DataRequestStatus.PENDING) {
      throw new Error('Request is no longer pending');
    }

    await prisma.dataDeletionRequest.update({
      where: { id: requestId },
      data: { verifiedAt: new Date() },
    });

    logger.info({ requestId }, 'Deletion request verified');

    return { message: 'Deletion request verified successfully' };
  }

  /**
   * Cancel deletion request
   */
  async cancelDeletionRequest(userId: string, requestId: string, reason?: string) {
    const request = await prisma.dataDeletionRequest.findFirst({
      where: {
        id: requestId,
        userId,
        status: DataRequestStatus.PENDING,
      },
    });

    if (!request) {
      throw new Error('No pending deletion request found');
    }

    await prisma.dataDeletionRequest.update({
      where: { id: requestId },
      data: {
        status: DataRequestStatus.CANCELLED,
        cancelledAt: new Date(),
        cancellationReason: reason,
      },
    });

    logger.info({ userId, requestId }, 'Deletion request cancelled');

    return { message: 'Account deletion request cancelled successfully' };
  }

  /**
   * Process scheduled deletions (to be called by a cron job)
   */
  async processScheduledDeletions() {
    const now = new Date();

    const pendingDeletions = await prisma.dataDeletionRequest.findMany({
      where: {
        status: DataRequestStatus.PENDING,
        verifiedAt: { not: null },
        scheduledAt: { lte: now },
      },
    });

    for (const request of pendingDeletions) {
      try {
        await this.executeAccountDeletion(request.id);
      } catch (error) {
        logger.error({ requestId: request.id, error }, 'Failed to process scheduled deletion');
      }
    }

    return { processed: pendingDeletions.length };
  }

  /**
   * Execute account deletion
   */
  async executeAccountDeletion(requestId: string) {
    const request = await prisma.dataDeletionRequest.findUnique({
      where: { id: requestId },
    });

    if (!request || !request.userId) {
      throw new Error('Deletion request not found');
    }

    // Update status
    await prisma.dataDeletionRequest.update({
      where: { id: requestId },
      data: { status: DataRequestStatus.PROCESSING, processedAt: new Date() },
    });

    try {
      // Collect deletion summary before deleting
      const summary = await this.collectDeletionSummary(request.userId);

      if (request.deletionType === DeletionType.ANONYMIZE) {
        // Anonymize instead of delete
        await this.anonymizeUserData(request.userId);
      } else {
        // Full deletion - cascading deletes will handle related records
        await prisma.user.delete({
          where: { id: request.userId },
        });
      }

      // Update request with completion
      await prisma.dataDeletionRequest.update({
        where: { id: requestId },
        data: {
          status: DataRequestStatus.COMPLETED,
          completedAt: new Date(),
          deletionSummary: summary,
        },
      });

      logger.info({ requestId, userId: request.userId }, 'Account deletion completed');

      return { message: 'Account deletion completed' };
    } catch (error) {
      await prisma.dataDeletionRequest.update({
        where: { id: requestId },
        data: {
          status: DataRequestStatus.FAILED,
          errorMessage: error instanceof Error ? error.message : 'Unknown error',
          retryCount: { increment: 1 },
        },
      });

      logger.error({ requestId, error }, 'Account deletion failed');
      throw error;
    }
  }

  /**
   * Collect summary of data to be deleted
   */
  async collectDeletionSummary(userId: string) {
    const [
      mealCount,
      waterIntakeCount,
      weightRecordCount,
      healthMetricCount,
      activityCount,
      supplementCount,
      supplementLogCount,
    ] = await Promise.all([
      prisma.meal.count({ where: { userId } }),
      prisma.waterIntake.count({ where: { userId } }),
      prisma.weightRecord.count({ where: { userId } }),
      prisma.healthMetric.count({ where: { userId } }),
      prisma.activity.count({ where: { userId } }),
      prisma.supplement.count({ where: { userId } }),
      prisma.supplementLog.count({ where: { userId } }),
    ]);

    return {
      meals: mealCount,
      waterIntakes: waterIntakeCount,
      weightRecords: weightRecordCount,
      healthMetrics: healthMetricCount,
      activities: activityCount,
      supplements: supplementCount,
      supplementLogs: supplementLogCount,
    };
  }

  /**
   * Anonymize user data instead of deleting
   */
  async anonymizeUserData(userId: string) {
    const anonymousEmail = `deleted-${crypto.randomBytes(8).toString('hex')}@anonymized.local`;

    await prisma.user.update({
      where: { id: userId },
      data: {
        email: anonymousEmail,
        name: 'Deleted User',
        password: null,
        appleId: null,
        profilePicture: null,
        dateOfBirth: null,
        resetToken: null,
        resetTokenExpiresAt: null,
        isActive: false,
      },
    });

    logger.info({ userId }, 'User data anonymized');
  }

  /**
   * Get deletion request status
   */
  async getDeletionRequests(userId: string) {
    return prisma.dataDeletionRequest.findMany({
      where: { userId },
      orderBy: { requestedAt: 'desc' },
      select: {
        id: true,
        status: true,
        deletionType: true,
        requestedAt: true,
        scheduledAt: true,
        verifiedAt: true,
        cancelledAt: true,
        completedAt: true,
      },
    });
  }

  // ==========================================================================
  // DATA ACCESS LOGGING
  // ==========================================================================

  /**
   * Log data access for audit purposes
   */
  async logDataAccess(entry: DataAccessLogEntry) {
    try {
      await prisma.dataAccessLog.create({
        data: {
          userId: entry.userId,
          action: entry.action,
          resource: entry.resource,
          resourceId: entry.resourceId,
          accessedBy: entry.accessedBy,
          accessorId: entry.accessorId,
          ipAddress: entry.ipAddress,
          userAgent: entry.userAgent,
          metadata: entry.metadata,
        },
      });
    } catch (error) {
      // Log errors but don't fail the main operation
      logger.error({ error, entry }, 'Failed to log data access');
    }
  }

  /**
   * Get data access logs for a user
   */
  async getDataAccessLogs(userId: string, limit = 50) {
    return prisma.dataAccessLog.findMany({
      where: { userId },
      orderBy: { accessedAt: 'desc' },
      take: limit,
    });
  }

  // ==========================================================================
  // DATA RETENTION
  // ==========================================================================

  /**
   * Apply data retention policies (to be called by a cron job)
   */
  async applyDataRetention() {
    const usersWithAutoDelete = await prisma.privacySettings.findMany({
      where: { autoDeleteOldData: true },
      select: { userId: true, dataRetentionDays: true },
    });

    let deletedCount = 0;

    for (const settings of usersWithAutoDelete) {
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - settings.dataRetentionDays);

      // Delete old data beyond retention period
      const [meals, waterIntakes, healthMetrics, activities] = await Promise.all([
        prisma.meal.deleteMany({
          where: { userId: settings.userId, consumedAt: { lt: cutoffDate } },
        }),
        prisma.waterIntake.deleteMany({
          where: { userId: settings.userId, recordedAt: { lt: cutoffDate } },
        }),
        prisma.healthMetric.deleteMany({
          where: { userId: settings.userId, recordedAt: { lt: cutoffDate } },
        }),
        prisma.activity.deleteMany({
          where: { userId: settings.userId, startedAt: { lt: cutoffDate } },
        }),
      ]);

      deletedCount += meals.count + waterIntakes.count + healthMetrics.count + activities.count;

      if (meals.count + waterIntakes.count + healthMetrics.count + activities.count > 0) {
        logger.info(
          {
            userId: settings.userId,
            deleted: {
              meals: meals.count,
              waterIntakes: waterIntakes.count,
              healthMetrics: healthMetrics.count,
              activities: activities.count,
            },
          },
          'Applied data retention policy'
        );
      }
    }

    return { deletedRecords: deletedCount };
  }
}

export const gdprService = new GdprService();
