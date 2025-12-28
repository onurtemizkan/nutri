import { Router } from 'express';
import { gdprController } from '../controllers/gdprController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All GDPR routes are protected
router.use(authenticate);

// ============================================================================
// CONSENT MANAGEMENT
// ============================================================================

/**
 * @swagger
 * /api/privacy/consents:
 *   get:
 *     summary: Get all consent statuses
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Current consent statuses for all purposes
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 consents:
 *                   type: object
 *                   additionalProperties:
 *                     type: object
 *                     properties:
 *                       granted:
 *                         type: boolean
 *                       grantedAt:
 *                         type: string
 *                         format: date-time
 *                       revokedAt:
 *                         type: string
 *                         format: date-time
 *                       version:
 *                         type: string
 *                 availablePurposes:
 *                   type: array
 *                   items:
 *                     type: string
 */
router.get('/consents', (req, res) => gdprController.getConsentStatus(req, res));

/**
 * @swagger
 * /api/privacy/consents:
 *   put:
 *     summary: Update a single consent
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - purpose
 *               - granted
 *             properties:
 *               purpose:
 *                 type: string
 *                 enum: [ESSENTIAL, ANALYTICS, MARKETING, RESEARCH, THIRD_PARTY, PERSONALIZATION, HEALTH_DATA_SHARING]
 *               granted:
 *                 type: boolean
 *               version:
 *                 type: string
 *     responses:
 *       200:
 *         description: Consent updated successfully
 */
router.put('/consents', (req, res) => gdprController.updateConsent(req, res));

/**
 * @swagger
 * /api/privacy/consents/batch:
 *   put:
 *     summary: Update multiple consents at once
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: array
 *             items:
 *               type: object
 *               required:
 *                 - purpose
 *                 - granted
 *               properties:
 *                 purpose:
 *                   type: string
 *                   enum: [ESSENTIAL, ANALYTICS, MARKETING, RESEARCH, THIRD_PARTY, PERSONALIZATION, HEALTH_DATA_SHARING]
 *                 granted:
 *                   type: boolean
 *                 version:
 *                   type: string
 *     responses:
 *       200:
 *         description: Consents updated successfully
 */
router.put('/consents/batch', (req, res) => gdprController.updateConsents(req, res));

// ============================================================================
// PRIVACY SETTINGS
// ============================================================================

/**
 * @swagger
 * /api/privacy/settings:
 *   get:
 *     summary: Get privacy settings
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: User privacy settings
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 dataRetentionDays:
 *                   type: integer
 *                 autoDeleteOldData:
 *                   type: boolean
 *                 allowAnalytics:
 *                   type: boolean
 *                 allowCrashReporting:
 *                   type: boolean
 *                 shareAnonymousData:
 *                   type: boolean
 *                 marketingEmails:
 *                   type: boolean
 *                 showOnLeaderboard:
 *                   type: boolean
 */
router.get('/settings', (req, res) => gdprController.getPrivacySettings(req, res));

/**
 * @swagger
 * /api/privacy/settings:
 *   put:
 *     summary: Update privacy settings
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               dataRetentionDays:
 *                 type: integer
 *                 minimum: 30
 *                 maximum: 3650
 *               autoDeleteOldData:
 *                 type: boolean
 *               allowAnalytics:
 *                 type: boolean
 *               allowCrashReporting:
 *                 type: boolean
 *               allowPerformanceMonitoring:
 *                 type: boolean
 *               shareAnonymousData:
 *                 type: boolean
 *               shareWithPartners:
 *                 type: boolean
 *               marketingEmails:
 *                 type: boolean
 *               productUpdates:
 *                 type: boolean
 *               showOnLeaderboard:
 *                 type: boolean
 *               healthKitDataSharing:
 *                 type: boolean
 *               privacyPolicyVersion:
 *                 type: string
 *               termsVersion:
 *                 type: string
 *     responses:
 *       200:
 *         description: Privacy settings updated
 */
router.put('/settings', (req, res) => gdprController.updatePrivacySettings(req, res));

// ============================================================================
// DATA EXPORT (Right to Portability)
// ============================================================================

/**
 * @swagger
 * /api/privacy/exports:
 *   post:
 *     summary: Request a data export
 *     description: Request a copy of all your personal data (GDPR Article 20 - Right to data portability)
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               format:
 *                 type: string
 *                 enum: [JSON, CSV, PDF]
 *                 default: JSON
 *               includeRaw:
 *                 type: boolean
 *                 default: false
 *                 description: Include raw ML data in export
 *     responses:
 *       201:
 *         description: Export request created
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 id:
 *                   type: string
 *                 status:
 *                   type: string
 *                 format:
 *                   type: string
 *                 requestedAt:
 *                   type: string
 *                   format: date-time
 */
router.post('/exports', (req, res) => gdprController.requestDataExport(req, res));

/**
 * @swagger
 * /api/privacy/exports:
 *   get:
 *     summary: Get export request history
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: List of export requests
 */
router.get('/exports', (req, res) => gdprController.getExportRequests(req, res));

/**
 * @swagger
 * /api/privacy/exports/{id}/download:
 *   get:
 *     summary: Download exported data
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Exported data file
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *       404:
 *         description: Export not found
 *       410:
 *         description: Export link expired
 */
router.get('/exports/:id/download', (req, res) => gdprController.downloadExport(req, res));

// ============================================================================
// ACCOUNT DELETION (Right to Erasure)
// ============================================================================

/**
 * @swagger
 * /api/privacy/deletion:
 *   post:
 *     summary: Request account deletion
 *     description: Request deletion of your account and all data (GDPR Article 17 - Right to erasure). There is a 30-day grace period during which you can cancel the request.
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               deletionType:
 *                 type: string
 *                 enum: [FULL, PARTIAL, ANONYMIZE]
 *                 default: FULL
 *               reason:
 *                 type: string
 *                 maxLength: 500
 *     responses:
 *       201:
 *         description: Deletion request created
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 id:
 *                   type: string
 *                 scheduledAt:
 *                   type: string
 *                   format: date-time
 *                 gracePeriodDays:
 *                   type: integer
 *                 message:
 *                   type: string
 */
router.post('/deletion', (req, res) => gdprController.requestAccountDeletion(req, res));

/**
 * @swagger
 * /api/privacy/deletion:
 *   get:
 *     summary: Get deletion request history
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: List of deletion requests
 */
router.get('/deletion', (req, res) => gdprController.getDeletionRequests(req, res));

/**
 * @swagger
 * /api/privacy/deletion/{id}/verify:
 *   post:
 *     summary: Verify deletion request
 *     description: Verify the deletion request with the code sent via email
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - verificationCode
 *             properties:
 *               verificationCode:
 *                 type: string
 *     responses:
 *       200:
 *         description: Deletion request verified
 */
router.post('/deletion/:id/verify', (req, res) => gdprController.verifyDeletionRequest(req, res));

/**
 * @swagger
 * /api/privacy/deletion/{id}/cancel:
 *   post:
 *     summary: Cancel deletion request
 *     description: Cancel a pending deletion request within the grace period
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *     requestBody:
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               reason:
 *                 type: string
 *                 maxLength: 500
 *     responses:
 *       200:
 *         description: Deletion request cancelled
 */
router.post('/deletion/:id/cancel', (req, res) => gdprController.cancelDeletionRequest(req, res));

// ============================================================================
// DATA ACCESS LOGS
// ============================================================================

/**
 * @swagger
 * /api/privacy/access-logs:
 *   get:
 *     summary: Get data access logs
 *     description: View logs of who accessed your data and when
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 50
 *           maximum: 100
 *     responses:
 *       200:
 *         description: Data access logs
 */
router.get('/access-logs', (req, res) => gdprController.getDataAccessLogs(req, res));

// ============================================================================
// PRIVACY DASHBOARD
// ============================================================================

/**
 * @swagger
 * /api/privacy/summary:
 *   get:
 *     summary: Get privacy summary
 *     description: Get a complete overview of your privacy settings, consents, and pending requests
 *     tags: [Privacy & GDPR]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Privacy summary
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 consents:
 *                   type: object
 *                 privacySettings:
 *                   type: object
 *                 exportRequests:
 *                   type: array
 *                 deletionRequests:
 *                   type: array
 *                 pendingDeletionRequest:
 *                   type: object
 *                   nullable: true
 */
router.get('/summary', (req, res) => gdprController.getPrivacySummary(req, res));

export default router;
