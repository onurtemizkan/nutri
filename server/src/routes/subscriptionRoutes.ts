/**
 * Subscription Routes
 *
 * Routes for client-side subscription operations:
 * - POST /api/subscription/verify - Verify a purchase
 * - POST /api/subscription/restore - Restore purchases
 * - GET /api/subscription/status - Get subscription status
 * - GET /api/subscription/products - Get available products
 */

import { Router } from 'express';
import { authenticate } from '../middleware/auth';
import {
  verifyPurchase,
  restorePurchases,
  getSubscriptionStatus,
  getProducts,
} from '../controllers/subscriptionController';

const router = Router();

// Products endpoint is public (for displaying in paywall before auth)
router.get('/products', getProducts);

// Protected endpoints require authentication
router.post('/verify', authenticate, verifyPurchase);
router.post('/restore', authenticate, restorePurchases);
router.get('/status', authenticate, getSubscriptionStatus);

export default router;
