import { Router } from 'express';
import { supplementController } from '../controllers/supplementController';
import { authenticate } from '../middleware/auth';

const router = Router();

// All user supplement routes are protected
router.use(authenticate);

// ==========================================================================
// USER SUPPLEMENT SCHEDULES
// ==========================================================================

// POST /user-supplements - Create new supplement schedule
router.post('/', (req, res) => supplementController.createUserSupplement(req, res));

// GET /user-supplements/scheduled - Get supplements scheduled for a specific date
router.get('/scheduled', (req, res) => supplementController.getScheduledSupplements(req, res));

// GET /user-supplements - Get all user supplement schedules
router.get('/', (req, res) => supplementController.getUserSupplements(req, res));

// GET /user-supplements/:id - Get single user supplement schedule
router.get('/:id', (req, res) => supplementController.getUserSupplementById(req, res));

// PUT /user-supplements/:id - Update user supplement schedule
router.put('/:id', (req, res) => supplementController.updateUserSupplement(req, res));

// DELETE /user-supplements/:id - Deactivate user supplement schedule
router.delete('/:id', (req, res) => supplementController.deleteUserSupplement(req, res));

export default router;
