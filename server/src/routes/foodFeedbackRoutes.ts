/**
 * Food Feedback Routes
 *
 * Routes for food classification feedback:
 * - Submit feedback (public with optional auth)
 * - Get statistics (public)
 * - Admin routes for managing feedback
 */

import { Router } from 'express';
import { foodFeedbackController } from '../controllers/foodFeedbackController';
import { authenticate, optionalAuthenticate } from '../middleware/auth';

const router = Router();

// =============================================================================
// Public Routes (with optional auth for userId tracking)
// =============================================================================

// Submit feedback for food classification
// POST /api/foods/feedback
router.post('/', optionalAuthenticate, (req, res) =>
  foodFeedbackController.submitFeedback(req, res)
);

// Get feedback statistics
// GET /api/foods/feedback/stats
router.get('/stats', (req, res) =>
  foodFeedbackController.getStats(req, res)
);

// Get patterns needing review
// GET /api/foods/feedback/patterns-needing-review
router.get('/patterns-needing-review', (req, res) =>
  foodFeedbackController.getPatternsNeedingReview(req, res)
);

// Get misclassifications for a specific food
// GET /api/foods/feedback/misclassifications/:foodKey
router.get('/misclassifications/:foodKey', (req, res) =>
  foodFeedbackController.getMisclassificationsForFood(req, res)
);

// =============================================================================
// Protected Routes (admin)
// =============================================================================

// Get feedback list with pagination
// GET /api/foods/feedback/list?page=1&limit=20&status=pending
router.get('/list', authenticate, (req, res) =>
  foodFeedbackController.getFeedbackList(req, res)
);

// Update feedback status
// PATCH /api/foods/feedback/:feedbackId/status
router.patch('/:feedbackId/status', authenticate, (req, res) =>
  foodFeedbackController.updateFeedbackStatus(req, res)
);

// Mark pattern as reviewed
// POST /api/foods/feedback/patterns/:patternId/review
router.post('/patterns/:patternId/review', authenticate, (req, res) =>
  foodFeedbackController.markPatternReviewed(req, res)
);

// Sync approved feedback to ML service
// POST /api/foods/feedback/sync
router.post('/sync', authenticate, (req, res) =>
  foodFeedbackController.syncToMLService(req, res)
);

export default router;
