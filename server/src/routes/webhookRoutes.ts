/**
 * Webhook Routes
 *
 * Routes for handling external webhooks (App Store, etc.)
 */

import { Router } from 'express';
import { handleAppStoreWebhook } from '../controllers/webhookController';
import { appStoreWebhookAuth } from '../middleware/webhookAuth';

const router = Router();

/**
 * POST /api/webhooks/app-store
 *
 * Receives App Store Server Notifications V2.
 * Signature is verified by appStoreWebhookAuth middleware.
 *
 * @see https://developer.apple.com/documentation/appstoreservernotifications
 */
router.post('/app-store', appStoreWebhookAuth, handleAppStoreWebhook);

export default router;
