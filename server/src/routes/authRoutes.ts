import { Router } from 'express';
import { authController } from '../controllers/authController';
import { authenticate } from '../middleware/auth';

const router = Router();

// Public routes
router.post('/register', (req, res) => authController.register(req, res));
router.post('/login', (req, res) => authController.login(req, res));
router.post('/forgot-password', (req, res) => authController.forgotPassword(req, res));
router.post('/reset-password', (req, res) => authController.resetPassword(req, res));
router.post('/verify-reset-token', (req, res) => authController.verifyResetToken(req, res));

// Protected routes
router.get('/profile', authenticate, (req, res) => authController.getProfile(req, res));
router.put('/profile', authenticate, (req, res) => authController.updateProfile(req, res));

export default router;
